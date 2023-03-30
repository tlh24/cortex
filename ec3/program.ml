open Lexer
open Lexing
open Printf
open Logo
open Ast
open Torch
open Graf

let pi = 3.1415926
(*let image_count = ref 0 *)
let image_alloc = 2048 (*6*2048*2*) (* need to make this a parameter *)
let image_res = 30
let batch_size = ref (256*3)
let toklen = 30
let poslen = 6
let p_indim = toklen + 1 + poslen*2 (* 31 + 12 = 43 *)
let e_indim = 5 + toklen + poslen*2
let p_ctx = 64
let nreplace = ref 0 (* number of repalcements during hallucinations *)
let glive = ref true 
let g_debug = ref false 
let gparallel = ref false
let gdisptime = ref false
let listen_address = Unix.inet_addr_loopback
let port = 4340
let backlog = 10

module Dtask = Domainslib.Task
	
let nulimg = Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout 1 1

type dreamt = [
	(* int = index; bool = dosub *)
	| `Train  (* supervised edit application *)
	| `Verify (* decode edit appl. *)
	| `Mnist
	]
	
type batche = (* batch edit structure, aka 'be' variable*)
	{ a_pid : int (* index to graf *)
	; b_pid : int (* index to graf or MNIST *)
	; a_progenc : string (* from *)
	; b_progenc : string (* to; something if supervised; blank if dream *)
	; c_progenc : string (* program being edited *)
	; a_imgi : int
	; b_imgi : int
	; edits : (string*int*char) list (* supervised *)
	; edited : float array (* tell python which chars have been changed*)
	; count : int (* count & cap total # of edits *)
	; dt : dreamt
	}
	
let nulbatche = 
	{ a_pid = 0 (* indexes steak.gs.g *)
	; b_pid = 0
	; a_progenc = ""
	; b_progenc = ""
	; c_progenc = ""
	; a_imgi = 0
	; b_imgi = 0
	; edits = []
	; edited = [| 0.0 |]
	; count = 0 
	; dt = `Train
	}
	
type batchd = (* batch data structure *)
	{ bpro : (float, Bigarray.float32_elt, Bigarray.c_layout)
				Bigarray.Array3.t (* btch , p_ctx , p_indim *)
	; bimg : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array3.t (* btch*3, image_res, image_res *)
	; bedts : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array2.t (* btch , e_indim - supervised *)
	; bedtd : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array2.t (* btch , e_indim - decode *)
	; posenc : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array2.t (* p_ctx , poslen*2 *)
	; posencn : Tensor.t (* normalized position encoding *)
	; bea : batche array
	; fresh : bool array
	}

type dreamcheck = 
	{ be : batche
	; mutable decode : string list
	; mutable cossim : float list
	; mutable correct_cnt : int
	}
	
let nuldream = 
	{ be = nulbatche
	; decode = []
	; cossim = []
	; correct_cnt = 0
	}
	
type tsteak = (* thread state *)
	{ device : Torch.Device.t
	; gs : Graf.gstat
	; dbf : Tensor.t (* on the gpu, so we can run fast comparisons *)
	; dbf_cpu : Tensor.t
	; dbf_enc : Tensor.t
	; mnist : Tensor.t (* on the CPU *)
	; mnist_enc : Tensor.t (* on the GPU *)
	; vae : Vae.VAE.t (* GPU *)
	; db_mutex : Mutex.t
	; superv : bool (* supervised or generative *)
	; sockno : int
	; fid : out_channel (* log for e.g. replacements *)
	; fid_verify : out_channel (* log for verification *)
	; mutable batchno : int (* for e.g doing garbage collection *)
	; mutable pool : Domainslib.Task.pool (* needs to be replaced for dream *)
	}

let read_lines name : string list =
	let ic = open_in name in
	let try_read () =
		try Some (input_line ic) with End_of_file -> None in
	let rec loop acc = match try_read () with
		| Some s -> loop (s :: acc)
		| None -> close_in ic; List.rev acc in
	loop []

let print_position lexbuf = 
	let pos = lexbuf.lex_curr_p in
	let bf = Buffer.create 64 in
	Printf.bprintf bf "%s:%d:%d" pos.pos_fname
		pos.pos_lnum (pos.pos_cnum - pos.pos_bol + 1); 
	(Buffer.contents bf)

let parse_with_error lexbuf =
	let prog = try Some (Parser.parse_prog Lexer.read lexbuf) with
	| SyntaxError _msg ->
		(*Logs.debug (fun m -> m "%s: %s" 
			(print_position lexbuf) msg);*)  (* these errors overwhelm while dreaming *)
		None
	| Parser.Error ->
		(*Logs.debug (fun m -> m "%s: syntax error" 
			(print_position lexbuf));*)
		None in
	prog

let bigarray_to_bytes arr = 
	(* convert a bigarray to a list of bytes *)
	(* this is not efficient.. *)
	let len = Bigarray.Array1.dim arr in
	Bytes.init len 
		(fun i -> Bigarray.Array1.get arr i |> Char.chr)

let run_prog prog res fname =
	(*Logs.debug(fun m -> m "enter run_prog");*)
	match prog with
	| Some(prog) -> (
		let (_,_,segs) = Logo.eval (Logo.start_state ()) prog in
		(*Logs.debug(fun m -> m "%s" (Logo.output_program_pstr prog)); 
		Logs.debug(fun m -> m "%s" (Logo.output_segments_str segs));*) 
		Logo.segs_to_png segs res fname; 
		(*let _arr,scost = Logo.segs_to_array_and_cost segs res in*)
		(*Logs.debug(fun m -> m "run_prog cost %f" scost);*) 
			(* another good way of doing it*)
		(*Logs.debug(fun m -> m  "run_prog done");*)
		true)
	| None -> ( false )

let parse_logo_string s = 
	let lexbuf = Lexing.from_string s in
	lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = "from string" };
	parse_with_error lexbuf
	
let run_logo_string s res fname = 
	let pro = parse_logo_string s in
	run_prog pro res fname 
	
let parse_logo_file fname = 
	let ic = open_in fname in
	let s = really_input_string ic (in_channel_length ic) in
	close_in ic;
	parse_logo_string s

let run_logo_file fname =
	let prog = parse_logo_file fname in
	ignore(run_prog prog 256 (fname^".png") )
	
(*let program_to_gdata pro res = 
	let progenc = Logo.encode_program_str pro in
	let (_,_,segs) = Logo.eval (Logo.start_state ()) pro in
	let scost = segs_to_cost segs in
	let pcost = Logo.progenc_cost progenc in
	let lx,hx,ly,hy = segs_bbx segs in
	let dx = hx-.lx in
	let dy = hy-.ly in
	let maxd = max dx dy in
	if maxd >= 2. && maxd <= 9. && scost >= 4. && scost <= 64. && List.length segs < 8 && String.length progenc < 24 then (
		let img, _ = Logo.segs_to_array_and_cost segs res in
		let progt = `Uniq in
		Some ({Graf.nulgdata with 
			progt; pro; progenc; scost; pcost; segs}, img)
	) else None*)

let progenc2str progenc =
	progenc |>
	Logo.string_to_intlist |>
	Logo.decode_program

let progenc_to_edata progenc =
	let progstr = progenc2str progenc in
	let g = parse_logo_string progstr in
	match g with
	| Some g2 -> (
		let pd = Graf.pro_to_edata_opt g2 image_res in
		match pd with
		| Some(data,img) -> (true,data,img)
		| _ -> (false,nuledata,nulimg) )
	| _ -> (false,nuledata,nulimg)
	
(*let edata_to_gdata d imgf_indx = 
	{ progt = `Uniq
	; pro = d.epro
	; progenc = d.eprogenc
	; proaddr = []
	; scost = d.escost
	; pcost = d.epcost
	; img = imgf_indx
	; outgoing = []
	; equivalents = []
	}*)

let render_simplest steak =
	(* render the shortest 1024 programs in the database.*)
	let g = Graf.sort_graph steak.gs.g in
	let dbl = Vector.length g in
	let res = 48 in
	let lg = open_out "/tmp/png/db_init.txt" in
	for id = 0 to min (1024-1) (dbl-1) do (
		let data = Vector.get g id in
		Logo.segs_to_png data.ed.segs res (Printf.sprintf "/tmp/png/test%d.png" id);
		let bf = Buffer.create 30 in
		Logo.output_program_p bf data.ed.pro;
		fprintf lg "%d %s\n" id (Buffer.contents bf);
	) done;
	close_out lg
	
(* because we include position encodings, need to be float32 *)
let mmap_bigarray2 fname rows cols = 
	(let open Bigarray in
	let fd = Unix.openfile fname
		[Unix.O_RDWR; Unix.O_TRUNC; Unix.O_CREAT] 0o666 in
	let a = Bigarray.array2_of_genarray 
		(Unix.map_file fd float32 c_layout true [|rows; cols|]) in (* true = shared *)
	(* return the file descriptor and array. 
	will have to close the fd later *)
	fd,a )

let mmap_bigarray3 fname batches rows cols = 
	(let open Bigarray in
	let fd = Unix.openfile fname
		[Unix.O_RDWR; Unix.O_TRUNC; Unix.O_CREAT] 0o666 in
	let a = Bigarray.array3_of_genarray 
		(Unix.map_file fd float32 c_layout true [|batches;rows;cols|]) in (* true = shared *)
	(* return the file descriptor and array. 
	will have to close the fd later *)
	fd,a )

(* both these are exclusively float32 *)
let bigarray2_of_tensor m =
	let c = Tensor.to_bigarray ~kind:Bigarray.float32 m in
	Bigarray.array2_of_genarray c
	
let tensor_of_bigarray2 m device =
	let o = Tensor.of_bigarray (Bigarray.genarray_of_array2 m) in
	o |> Tensor.to_device ~device
	
(*let tensor_of_bigarray1img img device = 
	let stride = (Bigarray.Array1.dim img) / image_res in
	let len = Bigarray.Array1.dim img in
	assert (len >= image_res * image_res); 
	let o = Tensor.(zeros [image_res; image_res]) in
	let l = image_res - 1 in
	for i = 0 to l do (
		for j = 0 to l do (
			let c = Bigarray.Array1.get img ((i*stride)+j) in
			if c <> 0 then (
				let cf = foi c in
				(let open Tensor in
				o.%.{[i;j]} <- (cf /. 255.0); )
			)
		) done 
	) done;
	o |> Tensor.to_device ~device*)
		
let db_get steak i = 
	Mutex.lock steak.db_mutex ; 
	let r = Vector.get steak.gs.g i in
	Mutex.unlock steak.db_mutex ; 
	r

let imgf_to_enc steak imgf =
	Vae.encode1_ext steak.vae
		(Tensor.view imgf ~size:[image_res*image_res])
		

let db_add_uniq ?(doenc=true) steak ed imgf imgf_cpu =
	(*imgf is a tensor on same device as dbf*)
	let added = ref false in
	let enc = if doenc then
		imgf_to_enc steak imgf  else Tensor.( zeros [2] ) in
	Mutex.lock steak.db_mutex; 
	let indx,imgi = Graf.add_uniq steak.gs ed in
	if imgi >= 0 then (
		Tensor.copy_ ~src:imgf (Tensor.narrow steak.dbf ~dim:0 ~start:imgi ~length:1);
		(*let imgf = Tensor.to_device ~device:steak.device imgf in*)
		Tensor.copy_ ~src:imgf (Tensor.narrow steak.dbf ~dim:0 ~start:imgi ~length:1);
		Tensor.copy_ ~src:imgf_cpu (Tensor.narrow steak.dbf_cpu ~dim:0 ~start:imgi ~length:1);
		if doenc then
			Tensor.copy_ ~src:enc (Tensor.narrow steak.dbf_enc ~dim:0 ~start:imgi ~length:1) ;
		added := true
	); 
	Mutex.unlock steak.db_mutex; 
	!added,indx
	
(* not sure how to ?functor? compress this *)
let db_replace_equiv steak indx d2 = 
	Mutex.lock steak.db_mutex ; 
	let r = Graf.replace_equiv steak.gs indx d2 in
	Mutex.unlock steak.db_mutex ; 
	r
	
let db_add_equiv steak indx d2 = 
	Mutex.lock steak.db_mutex ; 
	let r = Graf.add_equiv steak.gs indx d2 in
	Mutex.unlock steak.db_mutex ; 
	r
	
let db_len steak = 
	Mutex.lock steak.db_mutex ; 
	let r = Vector.length steak.gs.g in
	Mutex.unlock steak.db_mutex ; 
	r

let dbf_dist steak img = 
	(* cosine distance between two images *)
	(* first check that there is an image *)
	let sum = Tensor.sum img |> Tensor.float_value in
	let sumt = float_of_int (image_res * image_res) in
	if (sum > 1.0) && (sum < (0.99 *. sumt)) then (
		(* using Cosine Similarity *)
		let imgcnt = steak.gs.num_uniq in 
		assert (imgcnt > 0); (* torch calls fail ow *)
		let ir = image_res*image_res in
		let a = Tensor.narrow steak.dbf ~dim:0 ~start:0 ~length:imgcnt
			|> Tensor.view ~size:[-1;ir] in
		let b = Tensor.view img ~size:[1;-1] 
			|> Tensor.expand ~implicit:false ~size:[imgcnt; ir] in
		let d = Tensor.cosine_similarity ~x1:a ~x2:b ~dim:1 ~eps:1e-7 in
		let maxdex = Tensor.argmax d ~dim:0 ~keepdim:true 
			|> Tensor.int_value in
		let dist = Tensor.get d maxdex |> Tensor.float_value in
		true, dist, maxdex	
	) else (
		false, 0.0, 0 
	)

let dbf_to_png bigt i filename =
	let dbfim = Tensor.narrow bigt ~dim:0 ~start:i ~length:1 in
	Torch_vision.Image.write_image Tensor.((f 1. - dbfim) * f 255.) ~filename
	
let normalize_tensor m = 
	(* normalizes length along dimension 1 *)
	let len = Tensor.einsum ~equation:"ij,ij -> i" [m;m] ~path:None 
			|> Tensor.sqrt in
	let len = Tensor.(f 1. / len) in
	Tensor.einsum ~equation:"ij,i -> ij" [m;len] ~path:None 
	
let reset_bea dreaming =
	(* dt sets the 'mode' of batchd, which persists thru update_bea*)
	let dt = if dreaming then `Verify else `Train in
	let bea = Array.init !batch_size
		(fun _i -> {nulbatche with dt}) in
	let fresh = Array.init !batch_size (fun _i -> true) in
	bea,fresh
	
let init_batchd filnum =
	Logs.debug (fun m -> m "init_batchd"); 
	let dreaming = (filnum = 1) in
	let mkfnam s = 
		Printf.sprintf "%s_%d.mmap" s filnum in
	let fd_bpro,bpro = mmap_bigarray3 (mkfnam "bpro") 
			!batch_size p_ctx p_indim in
	let fd_bimg,bimg = mmap_bigarray3 (mkfnam "bimg") 
			(!batch_size*3) image_res image_res in
			(* note: needs a reshape on python side!! *)
	(* supervised batch of edits: ocaml to python *)
	let fd_bedts,bedts = mmap_bigarray2 (mkfnam "bedts") 
			!batch_size e_indim in
	(* hallucinated batch of edits: python to ocaml *)
	let fd_bedtd,bedtd = mmap_bigarray2 (mkfnam "bedtd") 
			!batch_size e_indim in
	let fd_posenc,posenc = mmap_bigarray2 (mkfnam "posenc")
			p_ctx (poslen*2) in
	let bea,fresh = reset_bea dreaming in
	(* fill out the posenc matrix *)
	let scl = 2.0 *. pi /. (foi p_ctx) in
	for i = 0 to (p_ctx-1) do (
		(* i is the position *)
		let fi = foi i in
		for j = 0 to (poslen-1) do (
			(* j is the frequency; j=0 means once cycle in p_ctx *)
			(* posenc[i,j*2+0] = sin((2*pi*i / p_ctx) * (j+1)) *)
			let fj = foi (j+1) in
			posenc.{i, j*2+0} <- sin(scl *. fi *. fj); 
			posenc.{i, j*2+1} <- cos(scl *. fi *. fj)
		) done
	) done ;
	let device = Torch.Device.Cpu in
	let posencn = tensor_of_bigarray2 posenc device |> normalize_tensor in
	Bigarray.Array3.fill bpro 0.0 ;
	Bigarray.Array3.fill bimg 0.0 ;
	Bigarray.Array2.fill bedts 0.0 ;
	Bigarray.Array2.fill bedtd 0.0 ; 
	(* return batchd struct & list of files to close later *)
	{bpro; bimg; bedts; bedtd; posenc; posencn; bea; fresh}, 
	[fd_bpro; fd_bimg; fd_bedts; fd_bedtd; fd_posenc]
	
(*let pdata_to_edits a b = 
	let dist, edits = Levenshtein.distance a.progenc b.progenc true in
	let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
	(* verify .. a bit of overhead *)
	(*let re = Levenshtein.apply_edits a.progenc edits in
	if re <> b.progenc then (
		Logs.err(fun m -> m  
			"error! %s edits should be %s was %s"
			a.progenc b.progenc re)
	);*)
	(* edits are applied in reverse *)
	(* & add a 'done' edit/indicator *)
	let edits = ("fin",0,'0') :: edits in
	dist, (List.rev edits)*)
	
(*let edit_criteria edits dosub = 
	let count_type typ = 
		List.fold_left 
		(fun a (t,_p,_c) -> if t = typ then a+1 else a) 0 edits 
	in
	let nsub = count_type "sub" in
	let ndel = count_type "del" in
	let nins = count_type "ins" in
	let r = ref false in
	if dosub then (
		if nsub = 1 && ndel = 0 && nins = 0 then r := true 
	) else (
		if nsub = 0 && ndel <= 6 && nins = 0 then r := true; 
		if nsub = 0 && ndel = 0 && nins <= 6 then r := true
	);
	!r*)
	
let make_batche_train a ai b bi = 
	let _,edits = Graf.get_edits a.ed.progenc b.ed.progenc in
	let edited = Array.make p_ctx 0.0 in
	{ a_pid = ai 
	; b_pid = bi
	; a_progenc = a.ed.progenc
	; b_progenc = b.ed.progenc
	; c_progenc = a.ed.progenc
	; a_imgi = a.imgi
	; b_imgi = b.imgi
	; edits
	; edited
	; count = 0
	; dt = `Train
	}
	
let new_batche_train steak = 
	(* supervised mode, any 'A' -- including eqiv *)
	let rec selector () = 
		let di = Random.int (Vector.length steak.gs.g) in 
		let d = Vector.get steak.gs.g di in
		match d.progt with
		| `Uniq -> (
			let o = SI.elements d.outgoing in
			let ol = List.length o in
			if ol > 0 then (
				let k = Random.int ol in
				let ei = List.nth o k in
				let e = Vector.get steak.gs.g ei in
				if e.progt = `Uniq then (
					make_batche_train d di e ei
				) else selector ()
			) else selector () )
		| `Equiv -> (
			(* always simplify to the minimum desc *)
			let ei = d.equivroot in
			let e = Vector.get steak.gs.g ei in
			if e.progt = `Uniq then (
				make_batche_train d di e ei
			) else ( assert (0 <> 0); nulbatche ) )
		| _ -> assert (0 <> 0); nulbatche
	in
	selector ()
	
(*let new_batche_train steak =
	(* only supervised mode, only uniq 'A'*)
	let rec selector () = 
		(* only target unique nodes, e.g. with an image *)
		let i = Random.int (steak.gs.num_uniq) in 
		let di = steak.gs.img_inv.(i) in
		(*let di = 0 in*)
		let d = Vector.get steak.gs.g di in 
		let o = SI.elements d.outgoing in
		let ol = List.length o in 
		if ol > 0 then (
			let k = Random.int ol in
			let ei = List.nth o k in
			let e = Vector.get steak.gs.g ei in
			if e.progt = `Uniq then (
				make_batche_train d di e ei
			) else selector ()
		) else selector ()
	in
	selector ()*)

let new_batche_mnist steak =
	(* for now, just set the target B to a sample from MNIST; ultimately will need to have longer interactions & intermediate starting points *)
	(*let len = match steak.dreams with
		| Some dream -> Array.length dream
		| _ -> 1 in*)
	let mid = Random.int 128 in
	(* select a starting point closer to the target, w/threshold.
		goal is conflated with longer interactions, guess ? *)
	let imgcnt = steak.gs.num_uniq in (* may be updated in other threads *)
	let _dbfn,cols = Tensor.shape2_exn steak.dbf_enc in
	let a = Tensor.narrow steak.dbf_enc ~dim:0 ~start:0 ~length:imgcnt in
	let b = Tensor.narrow steak.mnist_enc ~dim:0 ~start:mid ~length:1 
			|> Tensor.expand ~implicit:false ~size:[imgcnt;cols] in
	let d = Tensor.cosine_similarity ~x1:a ~x2:b ~dim:1 ~eps:1e-7 in
	(* add a bit of noise .. ?? *)
	let d = Tensor.( d + (f 0.05 * (randn [imgcnt;]))) in
	assert ((Tensor.shape1_exn d) = imgcnt) ; (* sanity.. *)
	let ind = Tensor.argmax d ~dim:0 ~keepdim:true |> Tensor.int_value in
	let indx = steak.gs.img_inv.(ind) in 
	let a = db_get steak indx in
	assert (ind = a.imgi) ; 
	let edited = Array.make p_ctx 0.0 in
	(* output these subs for inspection *)
	(*let filename = Printf.sprintf "/tmp/png/b%05d_target.png" (Atomic.get inspect_counter) in
	dbf_to_png steak.mnist mid filename; 
	Logo.segs_to_png a.segs 64
		(Printf.sprintf "/tmp/png/b%05d_closest.png" (Atomic.get inspect_counter));
	Atomic.incr inspect_counter ; *)
	{ a_pid = indx; b_pid = mid;
		a_progenc = a.ed.progenc; 
		b_progenc = ""; 
		c_progenc = a.ed.progenc; 
		a_imgi = a.imgi; 
		b_imgi = mid; 
		edits = []; edited; count=0; dt=`Mnist}
	(* note: bd.fresh is set in the calling function (for consistency) *)
		
let new_batche_unsup steak =
	if (Random.int 10) < 5 then (
		new_batche_train steak
	) else (
		new_batche_mnist steak
	)
	
(*let new_batche_dream_x steak _bn =
	(* not threaded !! *)
	match steak.dreams with
	| Some dreams -> (
		let i = steak.dreamn in
		let d = dreams.(i) in
		steak.dreamn <- (i+1) mod (Array.length dreams); 
		let edited = Array.make p_ctx 0.0 in (* memory thrash *)
		let count = 0 in
		{d.be with edited; count} )
	| None -> ( nulbatche )*)
	
let update_edited be ed = 
	(* update the 'edited' array, which indicates what has changed in the program string *)
	let typ,pp,_chr = ed in 
	let la = min (String.length be.a_progenc) (p_ctx/2) in
	let lc = min (String.length be.c_progenc) (p_ctx/2) in
	(* in-place array modification *)
	(match typ with 
	| "sub" -> (
		let pp = if pp > lc-1 then lc-1 else pp in
		let pp = if pp < 0 then 0 else pp in
		be.edited.(la+pp) <- 0.5 )
	| "del" -> (
		(* lc is already one less at this point -- c has been edited *)
		let pp = if pp > lc-1 then lc-1 else pp in
		let pp = if pp < 0 then 0 else pp in
		(* shift left *)
		for i = la+pp to p_ctx-2 do (
			be.edited.(i) <- be.edited.(i+1)
		) done; 
		assert ((la+pp) < p_ctx); 
		be.edited.(la+pp) <- ( -1.0 ) ) 
	| "ins" -> (
		if lc < p_ctx/2 && la+pp < p_ctx then (
			let pp = if pp > lc then lc else pp in
			let pp = if pp < 0 then 0 else pp in
			(* shift right one *)
			for i = p_ctx-2 downto la+pp do (
				be.edited.(i+1) <- be.edited.(i)
			) done; 
			be.edited.(la+pp) <- 1.0 ) )
	| _ -> () )

let sample_dist x = 
	(* sample a discrete decoding from the weightings along dim 1 *)
	(* assumes batch is dim 0 *)
	let bs,n = Tensor.shape2_exn x in
	let y = Tensor.clamp x ~min:(Scalar.float 0.0) ~max:(Scalar.float 1e6) in
	let mx = Tensor.amax y ~dim:[1] ~keepdim:true in
	let z = Tensor.( exp(y / mx) - (f 1.0) ) (*could be another nonlinearity*)
			|> Tensor.cumsum  ~dim:1 ~dtype:(T Float) in
	(* this will de facto be sorted, so lo and hi are easy *)
	let hi = Tensor.narrow z ~dim:1 ~start:(n-1) ~length:1 in
	let r = Tensor.( (rand_like hi) * hi ) 
			|> Tensor.expand ~implicit:true ~size:[bs;n] in
	let msk = Tensor.( (gt_tensor z r) + (f 0.001) ) in (* cast to float *)
	Tensor.argmax msk ~dim:1 ~keepdim:false
	(* argmax returns the first index if all are equal (as here) *)

let sample_dist_dum x = 
	Tensor.argmax x ~dim:1 ~keepdim:false
	
let decode_edit bd ba_edit = 
	(* decode model output (from python) *)
	let sta = Unix.gettimeofday () in
	let device = Torch.Device.Cpu in
	let m = tensor_of_bigarray2 ba_edit device in
	(* typ = th.argmax(y[:,0:4], 1)  (0 is the batch dim) *)
	(* stochastic decoding through sample_dist *)
	let typ = sample_dist (Tensor.narrow m ~dim:1 ~start:0 ~length:4) in 
	let chr = sample_dist (Tensor.narrow m ~dim:1 ~start:4 ~length:toklen) in
	(* now need to compute cosine distance.  normalize vectors first *)
	let pos = Tensor.narrow m ~dim:1 ~start:(5+toklen) ~length:(poslen*2)
			|> normalize_tensor in
	(*let pos = Tensor.narrow bd.posencn ~dim:0 ~start:4 ~length:1 in
	let pos = Tensor.expand pos ~size:[!batch_size;poslen*2] ~implicit:true in
	Logs.debug (fun m -> m "decode_edit: normalized pos\n"); 
	Tensor.print pos; *)
	(* ^^ why 5? 4 edit types, toklen char options, and one for shakes *)
	(* add a leading p_ctx dimension *)
	let pos = Tensor.expand pos ~size:[p_ctx;!batch_size;poslen*2] ~implicit:true in
	let posenc = Tensor.expand bd.posencn ~size:[!batch_size;p_ctx;poslen*2] ~implicit:true in
	(*Logs.debug (fun m -> m "decode_edit: bd.posencn\n"); 
	Tensor.print bd.posencn;*)
	let sim = Tensor.einsum ~equation:"cbp,bcp -> bc" [pos;posenc] ~path:None in
	(*Logs.debug (fun m -> m "decode_edit: sim\n"); 
	Tensor.print sim;*)
	let loc = sample_dist_dum sim in 
	(* cosine similarity is permissive; take top *)
	(* location must be clipped to within the program. *)
	let edit_arr = Array.init !batch_size (fun i -> 
		let etyp = match Tensor.get_int1 typ i with
			| 0 -> "sub"
			| 1 -> "del" 
			| 2 -> "ins"
			| _ -> "fin" in
		let echr = (Tensor.get_int1 chr i) + Char.code('0') |> Char.chr in
		let eloc = Tensor.get_int1 loc i in
		(etyp,eloc,echr) ) in
	(* debug *)
	(*for i = 0 to (min 1 !batch_size) do (
		let typ,loc,chr = edit_arr.(i) in
		Logs.debug (fun m -> m "decode_edit %d: %s,%c,%d" i typ chr loc)
	) done;*)
	let fin = Unix.gettimeofday () in
	if !gdisptime then Logs.debug (fun m -> m "decode_edit time %f" (fin-.sta)); 
	edit_arr
	
let apply_edits be = 
	(* apply after (of course) sending to python. *)
	(* edit length must be > 0 *)
	(* updates batche; pops first edit *)
	let ed = List.hd be.edits in
	(* note: Levenshtein clips the edit positions *)
	let c = Levenshtein.apply_edits be.c_progenc [ed] in
	let be2 = { be with c_progenc=c; edits=(List.tl be.edits) } in
	update_edited be2 ed; 
	be2
	
(*let pdata_to_edata p = 
	{ epro = p.pro
	; eprogenc = p.progenc
	; eimg = p.img
	; escost = p.scost
	; epcost = p.pcost
	; esegs = p.segs }*)
	
let better_counter = Atomic.make 0

let img_to_imgf steak img = 
	let imgf_cpu = tensor_of_bigarray2 img Torch.Device.Cpu in
	let imgf = Tensor.to_device ~device:steak.device imgf_cpu in
	imgf_cpu,imgf
	
let try_add_program steak data img be = 
	(*Logs.info (fun m -> m "Parsed! [%d]: %s \"%s\"" bi progenc progstr);*)
	let imgf_cpu,imgf = img_to_imgf steak img in
	let good2,dist,minde = dbf_dist steak imgf in
	if good2 then (
	let mindex = steak.gs.img_inv.(minde) in
	(* idea: if it's similar to a currently stored program, but has been hallucinated by the network, and is not too much more costly than what's there,
	then replace the entry! *)
	if dist > 0.994 then (
		let data2 = db_get steak mindex in
		let c1 = data.pcost in
		let c2 = data2.ed.pcost in
		if c1 < c2 then (
			let progstr = progenc2str data.progenc in
			let progstr2 = progenc2str data2.ed.progenc in
			let r = db_replace_equiv steak mindex data in
			if r >= 0 then (
				Logs.info (fun m -> m "#%d b:%d replacing equivalents [%d] %s with %s" !nreplace steak.batchno mindex progstr2 progstr);
				Printf.fprintf steak.fid
					"(%d) [%d] d:%f %s --> %s | pcost %d -> %d | scost %f -> %f\n"
					!nreplace mindex dist progstr2 progstr c2 c1
					data2.ed.scost data.scost;
				flush steak.fid;
				Logo.segs_to_png data2.ed.segs 64
					(Printf.sprintf "/tmp/png/%05d_old.png" !nreplace);
				Logo.segs_to_png data.segs 64
					(Printf.sprintf "/tmp/png/%05d_new.png" !nreplace);
				(* get the dbf entry too, to check *)
				let filename = Printf.sprintf
					"/tmp/png/%05d_dbf.png" !nreplace in
				dbf_to_png steak.dbf mindex filename;
				incr nreplace
			)
			(* those two operations are in-place, so subsequent batches should contain the new program :-) *)
		)
	) ;
	if dist < 0.7 then (
		let added,l = db_add_uniq steak data imgf imgf_cpu in
		if added then (
			Logs.info(fun m -> m
				"try_add_program: adding new [%d] = %s" l
				(Logo.output_program_pstr data.pro) )
		) else (
			Logs.debug(fun m -> m
				"try_add_program: could not add new, db full. [%d]" l )
		)
	) ;
	if be.dt = `Mnist then (
		(*let cpu = Torch.Device.Cpu in*)
		let a = db_get steak be.a_pid in
		let mid = be.b_pid in
		let aimg = Tensor.narrow steak.dbf ~dim:0 ~start:a.imgi ~length:1 in
		let bimg = Tensor.narrow steak.mnist ~dim:0 ~start:mid ~length:1
			|> Tensor.to_device ~device:steak.device in
		let cimg = imgf in
		let encode v =
			Tensor.view v ~size:[image_res*image_res;]
			|> Vae.encode1_ext steak.vae in
		let aenc = encode aimg in
		let benc = encode bimg in
		let cenc = encode cimg in
		let cos_ab = Tensor.cosine_similarity ~x1:aenc ~x2:benc ~dim:0 ~eps:1e-7 |> Tensor.float_value in
		let cos_cb = Tensor.cosine_similarity ~x1:cenc ~x2:benc ~dim:0 ~eps:1e-7 |> Tensor.float_value in
		let ab = Tensor.( mean((aimg - bimg) * (aimg - bimg)) )
			|> Tensor.float_value in
		let cb = Tensor.( mean((cimg - bimg) * (cimg - bimg)) )
			|> Tensor.float_value in
		if cos_cb > cos_ab then (
			let q = Atomic.get better_counter in
			Logs.info (fun m -> m "Made an improvement! see %d; cos: %f --> %f ; mse: %f --> %f" q cos_ab cos_cb ab cb);
			let filename = Printf.sprintf "/tmp/png/b%05d_a_target.png" q in
			dbf_to_png steak.mnist mid filename;
			Logo.segs_to_png a.ed.segs 64
				(Printf.sprintf "/tmp/png/b%05d_b_old.png" q);
			Logo.segs_to_png data.segs 64
				(Printf.sprintf "/tmp/png/b%05d_c_new.png" q);
			Atomic.incr better_counter
		)
	);
	(*if dist >= 0.6 && dist <= 5.0 then (
		let data2 = db_get steak mindex in
		Logs.info(fun m -> m
				"try_add_program: new \n\t%s sim existing\n\t%s"
				(Logo.output_program_pstr data.pro)
				(Logo.output_program_pstr data2.pro) )
	)*)
	)
	(* TODO not sure if we need to call GC here *)

let update_bea steak bd =
	let sta = Unix.gettimeofday () in
	let edit_arr = decode_edit bd bd.bedtd in
	let innerloop_update_bea bi =
		let be = bd.bea.(bi) in
		(* check edited array allocation *)
		let be1 = if Array.length be.edited <> p_ctx
			then ( let edited = Array.make p_ctx 0.0 in
				{be with edited} ) else be in
		let cnt = be1.count in
		let typ,loc,chr = edit_arr.(bi) in
		(*assert (Array.length be.edited = p_ctx) ;*)
		let bnew = match be1.dt with
		| `Train -> (
			(* last edit is 'fin', in which case: new batche *)
			if List.length be1.edits <= 1 then (
				bd.fresh.(bi) <- true; (* update image flag *)
				let bn = new_batche_train steak in
				if bi = 0 then (
					Logs.debug (fun m -> m "|b0 new_batche_train; edit length %d"
						(List.length bn.edits))
				); 
				bn
			) else (
				bd.fresh.(bi) <- false;
				apply_edits be1
			)
			(* NOTE : add decoding verification here? 
				would require second accumulator string *)
			)
		| `Verify -> (
			let be2 = {be1 with edits=[(typ,loc,chr)];count=cnt+1} in
			let be3 = apply_edits be2 in
			if typ = "fin" || be3.count >= p_ctx/2 then (
				(* write this to file for now; stats later *)
				let c = if be3.c_progenc = be3.b_progenc then '+' else '-' in
				Printf.fprintf steak.fid_verify "[%c] %s -> %s ; decode %s\n" c
					(progenc2str be3.a_progenc)
					(progenc2str be3.b_progenc)
					(progenc2str be3.c_progenc); 
				if c = '+' then (
					Graf.incr_good steak.gs be3.a_pid; 
					Graf.incr_good steak.gs be3.b_pid 
				); 
				bd.fresh.(bi) <- true;
				new_batche_unsup steak 
			) else (
				bd.fresh.(bi) <- false;
				be3
			) )
		| `Mnist -> (
			let be2 = {be1 with edits=[(typ,loc,chr)];count=cnt+1} in
			let be3 = apply_edits be2 in
			if typ = "fin" || be3.count >= p_ctx/2 then (
				let good,data,img = progenc_to_edata be3.c_progenc in
				if good then (
					(* fix this later: keep around programs that improve fit *)
					(*let imgf = tensor_of_bigarray2 img steak.device in
					let enc = imgf_to_enc steak imgf in
					let s = progenc2str data.progenc in
					let j = be3.dt.indx in
					let a = Tensor.narrow steak.mnist_enc ~dim:0 ~start:j ~length:1 in
					let d = Tensor.cosine_similarity ~x1:a ~x2:enc ~dim:1 ~eps:1e-7
						|> Tensor.float_value in
					(match steak.dreams with
					| Some dreams -> (
						dreams.(j).decode <- (s :: dreams.(j).decode);
						dreams.(j).cossim <- (d :: dreams.(j).cossim)
						)
					| None -> assert (0 <> 0); () );*)
					try_add_program steak data img be3
				);
				bd.fresh.(bi) <- true;
				new_batche_unsup steak
			) else (
				bd.fresh.(bi) <- false;
				be3
			) ) in
		bd.bea.(bi) <- bnew;
	in (* /innerloop_update_bea *)

	if !gparallel then
		Dtask.parallel_for steak.pool ~start:0 ~finish:(!batch_size-1)
			~body:innerloop_update_bea
	else
		for i=0 to (!batch_size-1) do
			innerloop_update_bea i done;

	let fin = Unix.gettimeofday () in
	if !gdisptime then Logs.debug (fun m -> m "update_bea time %f" (fin-.sta));
	if (steak.batchno mod 10) = 9 then (
		Mutex.lock steak.db_mutex;
		Caml.Gc.major (); (* clean up torch variables (slow..) *)
		Mutex.unlock steak.db_mutex
	);
	steak.batchno <- steak.batchno + 1

let bigfill_batchd steak bd = 
	let sta = Unix.gettimeofday () in
	(* convert bea to the 3 mmaped bigarrays *)
	(* first clear them *)
	(*Bigarray.Array3.fill bd.bimg 0.0 ;*)
	(* fill one-hots *)
	Bigarray.Array3.fill bd.bpro 0.0 ;
	Bigarray.Array2.fill bd.bedts 0.0 ; 
	for u = 0 to !batch_size-1 do (
		bd.bedts.{u,0} <- -1. (* TEST (checked via min, python side) *)
	) done; 
	let innerloop_bigfill_batchd u = 
		let be = bd.bea.(u) in
	  (* - bpro - *)
		let offs = Char.code '0' in
		let llim = (p_ctx/2)-2 in
		let l = String.length be.a_progenc in
		if l > llim then 
			Logs.debug(fun m -> m  "a_progenc too long(%d):%s" l be.a_progenc);
		String.iteri (fun i c -> 
			let j = (Char.code c) - offs in
			if i < llim then bd.bpro.{u,i,j} <- 1.0 ) be.a_progenc ; 
		let lc = String.length be.c_progenc in
		if lc > llim then 
			Logs.debug(fun m -> m  "c_progenc too long(%d):%s" lc be.c_progenc);
		String.iteri (fun i c -> 
			let j = (Char.code c) - offs in
			if (i+l) < 2*llim then (
				bd.bpro.{u,i+l,j} <- 1.0; 
				(* indicate this is c, to be edited *)
				bd.bpro.{u,i+l,toklen-1} <- 1.0 ) ) be.c_progenc ;
		(* copy over the edited tags (set in apply_edits) *)
		assert (Array.length be.edited = p_ctx) ; 
		for i = 0 to p_ctx-1 do (
			bd.bpro.{u,i,toklen} <- be.edited.(i)
		) done; 
		(* position encoding *)
		let l = if l > llim then llim else l in 
		let lc = if lc > llim then llim else lc in
		for i = 0 to l-1 do (
			for j = 0 to poslen*2-1 do (
				bd.bpro.{u,i,toklen+1+j} <- bd.posenc.{i,j}
			) done
		) done;
		for i = 0 to lc-1 do (
			for j = 0 to poslen*2-1 do (
				bd.bpro.{u,i+l,toklen+1+j} <- bd.posenc.{i,j}
			) done
		) done ;
	  (* - bimg - *)
		if bd.fresh.(u) then (
			let pid_to_ba2 imgi dt = 
				assert (imgi >= 0) ; 
				if dt = `Mnist then (
					assert (imgi < 60000) ; 
					Tensor.narrow steak.mnist ~dim:0 ~start:imgi ~length:1 
					|> Tensor.squeeze |> bigarray2_of_tensor 
				) else (
					if imgi >= steak.gs.num_uniq then 
						Logs.debug (fun m -> m 
							"%s \n\t>>bigfill_batchd imgi %d gs.num_uniq %d"
							__LOC__ imgi steak.gs.num_uniq); 
					assert (imgi < steak.gs.num_uniq) ; 
					Tensor.narrow steak.dbf_cpu ~dim:0 ~start:imgi ~length:1 
					|> Tensor.squeeze |> bigarray2_of_tensor 
				) 
			in
			(* would be good if this didn't require two copy ops *)
			(* tensor to bigarray, bigarray to bigarray *)
			if u = 0 then (
				Logs.debug (fun m -> m "bigfill_batchd a g:%d i:%d b g:%d i:%d" 
				be.a_pid be.a_imgi be.b_pid be.b_imgi)); 
			let aimg = pid_to_ba2 be.a_imgi `Train in
			let bimg = pid_to_ba2 be.b_imgi be.dt in
			for i=0 to (image_res-1) do (
				for j=0 to (image_res-1) do (
					let aif = aimg.{i,j} in
					let bif = bimg.{i,j} in
					bd.bimg.{3*u+0,i,j} <- aif; 
					bd.bimg.{3*u+1,i,j} <- bif;
					bd.bimg.{3*u+2,i,j} <- aif -. bif;
				) done
			) done ; 
			bd.fresh.(u) <- false
		); 
	  (* - bedts - *)
		bd.bedts.{u,0} <- 0.0; (* TEST python synchronization *)
		let (typ,pp,c) = if (List.length be.edits) > 0 
			then List.hd be.edits
			else ("fin",0,'0') in
		(* during dreaming, the edit list is drained dry: 
		we apply the edits (thereby emptying the 1-element list),
		update the program encodings, 
		and ask the model to generate a new edit. *)
		(match typ with
		| "sub" -> bd.bedts.{u,0} <- 1.0
		| "del" -> bd.bedts.{u,1} <- 1.0
		| "ins" -> bd.bedts.{u,2} <- 1.0
		| "fin" -> bd.bedts.{u,3} <- 1.0
		| _ -> () ); 
		let ci = (Char.code c) - offs in
		if ci >= 0 && ci < toklen then 
			bd.bedts.{u,4+ci} <- 1.0 ; 
		(* position encoding *)
		if pp >= 0 && pp < p_ctx then (
			for i = 0 to poslen*2-1 do (
				bd.bedts.{u,5+toklen+i} <- bd.posenc.{pp,i}
			) done
		) in (* /il_bigfill_batchd *)
	if !gparallel then
		Dtask.parallel_for steak.pool ~start:0 ~finish:(!batch_size-1)
			~body:innerloop_bigfill_batchd
	else
		for i=0 to (!batch_size-1) do
			innerloop_bigfill_batchd i done; 
	let fin = Unix.gettimeofday () in
	if !gdisptime then 
		Logs.debug (fun m -> m "bigfill_batchd time %f" (fin-.sta))
	
let truncate_list n l = 
	let n = min n (List.length l) in
	let a = Array.of_list l in
	Array.sub a 0 n |> Array.to_list
	
let render_database steak g = 
	(* TODO: make the iter parallel *)
	let dbf_cpu = Tensor.( 
		( ones [image_alloc; image_res; image_res] ) * (f (-1.0))) in
	let k = ref 0 in
	Vector.iter (fun d -> 
		match d.progt with 
		| `Uniq -> (
			let _,img = Graf.pro_to_edata d.ed.pro image_res in
			let imgf_cpu,_ = img_to_imgf steak img in
			Tensor.copy_ ~src:imgf_cpu 
				(Tensor.narrow dbf_cpu ~dim:0 ~start:!k ~length:1); 
			incr k)
			
		| _ -> () ) g ; 
	let dbf = Tensor.to_device ~device:steak.device dbf_cpu in
	(* checksies *)
	for k = 0 to (Vector.length g) -1 do (
		dbf_to_png dbf_cpu k (Printf.sprintf "/tmp/png/dbf_check_%d.png" k)
	) done; 
	dbf,dbf_cpu

let sort_database steak =
	(* sort the graph 
		well, sort the Vector underlying the graph -- since graphs don't really have an order. 
		leave imgf alone, since now we don't have a dup *)
	let g = Graf.sort_graph steak.gs.g in
	let gs = {steak.gs with g} in
	let dbf,dbf_cpu = render_database steak g in
	{steak with gs; dbf; dbf_cpu} 
	
let rec generate_random_logo res =
	let actvar = Array.init 5 (fun _i -> false) in
	let prog = gen_ast false (3,1,actvar) in
	let pro = compress_ast prog in
	if has_pen_nop pro then assert (0 <> 0) ; 
	let ed = Graf.pro_to_edata_opt pro res in
	match ed with
	| Some q -> q
	| _ -> generate_random_logo res

let generate_empty_logo res =
	(* the first program needs to be empty, for diffing *)
	let pro = `Nop in
	Graf.pro_to_edata pro res

let init_database steak count = 
	(* generate 'count' initial program & image pairs *)
	let dbf_cpu = Tensor.( 
		( ones [image_alloc; image_res; image_res] ) * (f (-1.0))) in
	let dbf = Tensor.to_device ~device:steak.device dbf_cpu in
	let steak = {steak with dbf; dbf_cpu} in
	Logs.info(fun m -> m  "init_database %d" count); 
	let fid = open_out "/tmp/png/newdbg.txt" in
	let i = ref 0 in
	let iters = ref 0 in
	while !i < count do (
		let data,img = if !i = 0 then
			generate_empty_logo image_res else
			generate_random_logo image_res in
		let imgf_cpu,imgf = img_to_imgf steak img in
		let good,dist,minde = if !i = 0 then true,0.5,0 
			else dbf_dist steak imgf in
		let mindex = steak.gs.img_inv.(minde) in
		if (good || !i < 2) then (
		(* bug: white image sets distance to 1.0 to [0] *)
		(*Logs.debug (fun m -> m "dbf_dist %f %d" dist mindex);*) 
		if dist < 0.7 then ( 
			let s = Logo.output_program_pstr data.pro in
			Logs.debug(fun m -> m 
				"%d: adding [%d] = %s" !iters !i s); 
			ignore( db_add_uniq steak data imgf imgf_cpu ~doenc:false );
			Logo.segs_to_png data.segs 64
				(Printf.sprintf "/tmp/png/db%05d_.png" !i); 
			dbf_to_png steak.dbf !i
				(Printf.sprintf "/tmp/png/db%05d_f.png" !i);
			Printf.fprintf fid "[%d] %s (dist:%f to:%d)\n" !i s dist mindex; 
			incr i;
		) ; 
		if dist > 0.99 then (
			(* see if there's a replacement *)
			let data2 = db_get steak mindex in
			let c1 = data.pcost in (* progenc_cost  *)
			let c2 = data2.ed.pcost in
			if c1 < c2 then (
				let r = db_replace_equiv steak mindex data in 
				if r >= 0 then 
					Logs.debug(fun m -> m 
					"%d: replacing [%d] = %s ( was %s)" 
					!iters mindex
					(Logo.output_program_pstr data.pro) 
					(Logo.output_program_pstr data2.ed.pro));
			); 
			if c1 > c2 then (
				if (SI.cardinal data2.equivalents) < 32 then (
					let r = db_add_equiv steak mindex data in
					if r >= 0 then 
						Logs.debug(fun m -> m 
						"iter %d: added equiv [loc %d] = %s [now loc %d] ( was %s) %s %s %b " 
						!iters mindex
						(Logo.output_program_pstr data.pro) r
						(Logo.output_program_pstr data2.ed.pro)
						data.progenc data2.ed.progenc
						(data.progenc = data2.ed.progenc) );
				)
			); 
		) );  
		if !iters mod 40 = 39 then 
			(* needed to clean up torch allocations *)
			Caml.Gc.major (); 
		incr iters
	) done; 
	close_out fid; 
	let steak = sort_database steak in
	Logs.info(fun m -> m  "%d done; %d sampled; %d replacements; %d equivalents" !i !iters steak.gs.num_uniq steak.gs.num_equiv); 
	steak
	
let load_database steak fname = 
	(* returns a new steak, freshly sizzled *)
	let gs0 = create image_alloc in
	let gs = Graf.load gs0 fname in
	let dbf,dbf_cpu = render_database steak gs.g in
	Logs.info (fun m -> m "Loaded %d: %d uniq and %d equivalent"
		(Vector.length gs.g) gs.num_uniq gs.num_equiv ); 
	{steak with gs; dbf; dbf_cpu} 
	
let save_database steak fname = 
	(*let g = Graf.sort_graph steak.gs.g in*)
	Graf.save fname steak.gs.g 
	(*(* saves in the current state -- not sorted. *)
	let dba = Vector.to_array db in (* inefficient *)
	let fil = open_out fname in
	Printf.fprintf fil "%d\n" (Array.length dba); 
	Array.iteri (fun i d -> 
		let pstr = Logo.output_program_pstr d.pro in
		Printf.fprintf fil "[%d] %s "  i pstr; 
		List.iter (fun ep -> 
			let ps = Logo.output_program_pstr ep.epro in
			if String.length ps > 0 then 
				Printf.fprintf fil "| %s " ps) d.equiv; 
		Printf.fprintf fil "\n") dba ; 
	close_out fil; 
	Logs.info(fun m -> m  "saved %d to %s" (Array.length dba) fname); 
	
	(* verification .. human readable*)
	let fil = open_out "db_human_log.txt" in
	Printf.fprintf fil "%d\n" (Array.length dba); 
	Array.iteri (fun i d -> 
		Printf.fprintf fil "\n[%d]\t" i ; 
		Logo.output_program_h d.pro fil) dba ; 
	close_out fil; 
	Logs.debug(fun m -> m  "saved %d to db_human_log.txt" (Array.length dba))
	(* save the equivalents too? *)*)

(*let load_database_line steak s pid equivalent =
	let sl = Pcre.split ~pat:"[\\[\\]]+" s in
	let _h,ids,ps = match sl with
		| h::m::t -> h,m,t
		| h::[] -> h,"",[]
		| [] -> "0","0",[] in
	let pid2 = int_of_string ids in
	if pid <> pid2 then 
		Logs.err (fun m -> m "load pid %d != line %d" pid2 pid); 
	let pr,prl = if pid = 0 || (List.length ps) < 1
		then Some `Nop, []
		else (
			let a = String.split_on_char '|' (List.hd ps) in
			parse_logo_string (List.hd a), (List.tl a) ) in
	(match pr with 
	| Some pro -> 
		let progenc = Logo.encode_program_str pro in
		let (_,_,segs) = Logo.eval (Logo.start_state ()) pro in
		let scost = segs_to_cost segs in
		let pcost = Logo.progenc_cost progenc in
		let img,_ = Logo.segs_to_array_and_cost segs image_res in
		let equiv = List.map (fun s -> 
			let g = parse_logo_string s in
			match g with 
			| Some epro -> ( 
				let eprogenc = Logo.encode_program_str epro in
				let (_,_,esegs) = Logo.eval (Logo.start_state ()) epro in
				let escost = segs_to_cost esegs in
				let epcost = Logo.progenc_cost eprogenc in
				let eimg,_ = Logo.segs_to_array_and_cost esegs image_res in
				Atomic.incr equivalent; 
				{epro; eprogenc; eimg; escost; epcost; esegs}
				)
			| _ -> ( 
				Logs.err(fun m -> m  "could not parse equiv program %d %s" pid s); 
				nulpequiv ) ) prl in
		let data = {pid; pro; progenc; img; 
						scost; pcost; segs; equiv} in
		let imgf = tensor_of_bigarray2 img steak.device in
		ignore( db_push steak data imgf ~doenc:false);  (* mutex protected *)
	
	| _ -> Logs.err(fun m -> m 
				"could not parse program %d %s" pid s ))*)
	
(*let load_database steak = 
	let equivalent = Atomic.make 0 in
	let fname = "db_prog.txt" in
	let lines = read_lines fname in
	let a,progs = match lines with
		| h::r -> h,r
		| [] -> "0",[] in
	let ai = int_of_string a in
	if ai > image_alloc then (
		Logs.err(fun m -> m "%s image_count too large, %d > %d" fname ai image_alloc) 
	) else (
		(* db_prog format: [$id] program | eq. prog | eq. prog ... \n *)
		let progsa = Array.of_list progs in
		let pl = Array.length progsa in
		(* this needs to be called within Dtask.run to be parallel *)
		(* actually ... running in parallel seems to screw things up! *)
		if !gparallel && false then (
			Dtask.parallel_for steak.pool ~start:0 ~finish:(pl-1) 
				~body:( fun i -> load_database_line steak progsa.(i) i equivalent )
		) else (
			for i = 0 to (pl-1) do 
				load_database_line steak progsa.(i) i equivalent 
			done
		); 
		Logs.info (fun m -> m "Loaded %d programs (%d max) and %d equivalents" 
			(List.length progs) image_alloc (Atomic.get equivalent)); 
	)*)

(*let save_dreams steak =
	(* iterate over the training data.. *)
	let write_dreamcheck dca fname =
		let fid = open_out (fname^".txt") in
		Array.iteri (fun i dc ->
			let d = if List.length dc.decode >= 1 then
					List.hd dc.decode else "" in
			let a = progenc2progstr dc.be.a_progenc in
			let b = progenc2progstr dc.be.b_progenc in
			Printf.fprintf fid "[%d] ncorrect:%d is:%s->%s decode:%s\n"
					i dc.correct_cnt a b d;
			if i < 2048 then (
				ignore(run_logo_string b 64
					(Printf.sprintf "/tmp/png/%s%05d_real.png" fname i));
				ignore(run_logo_string d 64
					(Printf.sprintf "/tmp/png/%s%05d_decode.png" fname i)) );
		) dca;
		close_out fid;
		Logs.debug (fun m -> m "Saved %d to %s" (Array.length dca) fname) in
	(match steak.trains_sub with
	| Some trains -> write_dreamcheck trains "dreamcheck_sub"
	| _ -> () );
	(match steak.trains_insdel with
	| Some trains -> write_dreamcheck trains "dreamcheck_insdel"
	| _ -> () );
	(* mnist doesn't need a file, just examples *)
	let cnt = ref 0 in
	(match steak.dreams with
	| Some mnist -> (
		Array.iteri (fun i dc ->
			(* take the best (highest cosine sim) match & output that *)
			if List.length dc.decode > 1 then (
			let e = List.map2 (fun q w -> q,w) dc.cossim dc.decode
				|> List.sort (fun (a,_) (b,_) -> compare b a) in
			let f,d = List.hd e in
			let f' = int_of_float (f *. 100.) in (* percent *)
			if (run_logo_string d 64
				(Printf.sprintf "/tmp/png/mnist%05d_%d_decode.png" f' i)) then (
				dbf_to_png steak.mnist dc.be.dt.indx
					(Printf.sprintf "/tmp/png/mnist%05d_%d_real.png" f' i);
				incr cnt)
			) ) mnist )
	| _ -> () );
	Logs.debug (fun m -> m "Saved %d mnist decodes to /tmp/png" !cnt)*)

(*let hidden_nodes = 128
let epochs = 10000
let learning_rate = 1e-3*)
	
let handle_message steak bd msg =
	(*Logs.debug (fun m -> m "handle_message %s" msg);*)
	let msgl = String.split_on_char ':' msg in
	let cmd = if List.length msgl = 1 then msg 
		else List.hd msgl in
	match cmd with
	| "update_batch" -> (
		(* supervised: sent when python has a working copy of data *)
		(* dreaming : sent when the model has produced edits (maybe old) *)
		update_bea steak bd;
		bigfill_batchd steak bd; 
		(*Logs.debug(fun m -> m "new batch %d" steak.batchno);*)
		bd,(Printf.sprintf "ok %d" !nreplace)
		)
	| "decode_edit" -> (
		(* python sets bd.bedtd -- the dreamed edits *)
		(* read this independently of updating bd.bedts; so as to determine acuracy. *)
		let decode_edit_accuracy edit_arr = 
			let typright, chrright, posright = ref 0, ref 0, ref 0 in
			let print = ref true in
			Array.iteri (fun i (typ,pos,chr) -> 
				if List.length (bd.bea.(i).edits) > 0 then (
					let styp,spos,schr = List.hd (bd.bea.(i).edits) in (* supervised *)
					if typ = styp then incr typright; 
					if pos = spos then incr posright; 
					if chr = schr then incr chrright; 
					if !print then (
						Logs.info (fun m -> m "|i%d |b%d true: %s [%d] %c ; decode: %s [%d] %c \"%s\""
							steak.batchno i styp spos schr typ pos chr bd.bea.(i).c_progenc); 
						print := false
					)
				) ) edit_arr ; 
			let pctg v = (foi !v) /. (foi !batch_size) in
			Logs.info (fun m -> m (* TODO: debug mode *)
				"decode_edit: correct typ %0.3f chr %0.3f pos %0.3f " 
				(pctg typright) (pctg chrright) (pctg posright) );
		in
		(*let edit_sup = decode_edit bd bd.bedts in *)(* verification *)
		let edit_dream = decode_edit bd bd.bedtd in
		(*decode_edit_accuracy edit_sup "superv"; *)
		decode_edit_accuracy edit_dream; 
		bd, "ok" )
	| _ -> bd,"Unknown command"

let read_socket client_sock = 
	let maxlen = 256 in
	let data_read = Bytes.create maxlen in
	let data_length =
		try
			Unix.recv client_sock data_read 0 maxlen []
		with Unix.Unix_error (e, _, _) ->
			Logs.err (fun m -> m "Sock can't receive: %s ; shutting down"
				(Unix.error_message e));
			( try Unix.shutdown client_sock SHUTDOWN_ALL
			with Unix.Unix_error (e, _, _) ->
				Logs.err (fun m -> m "Sock can't shutdown: %s"
					(Unix.error_message e)) ); 
			0 in
	if data_length > 0 then 
		Some (Bytes.sub data_read 0 data_length |> Bytes.to_string )
	else None

let servthread steak () = (* steak = thread state *)
	(let open Unix in
	Logs.info (fun m -> m "Starting server on %d" steak.sockno); 
	let server_sock = socket PF_INET SOCK_STREAM 0 in
	let listen_address = inet_addr_loopback in
	setsockopt server_sock SO_REUSEADDR true ; (* for faster restarts *)
	bind server_sock (ADDR_INET (listen_address, steak.sockno)) ;
	listen server_sock 2 ; (* 2 max connections *)
	while !glive do (
		let bd,fdlist = init_batchd (steak.sockno-4340) in
		let (client_sock, _client_addr) = accept server_sock in
		Logs.info (fun m -> m "new connection on %d" steak.sockno); 
		(* make new mmap files & batchd for each connection *)
		let rec message_loop bd =
			let msg = read_socket client_sock in
			(match msg with 
			| Some msg -> (
				let sta = Unix.gettimeofday () in
				let bd,resp = handle_message steak bd msg in 
				let fin = Unix.gettimeofday () in
				if !gdisptime then 
					Logs.debug (fun m -> m "handle_message %s time %f" msg (fin-.sta));
				ignore ( send client_sock (Bytes.of_string resp) 
					0 (String.length resp) [] ) ; 
				message_loop bd )
			| _ -> (
				save_database steak "db_prog.txt"; 
			) ) in
		if !gparallel then (
			Dtask.run steak.pool (fun () -> message_loop bd )
		) else (
			message_loop bd ); 
		(* if client disconnects, close the files and reopen *)
		Logs.debug (fun m -> m "disconnect %d" steak.sockno); 
		List.iter (fun fd -> close fd) fdlist; 
	) done; 
	) (* )open Unix *)
	
let measure_torch_copy_speed device = 
	let start = Unix.gettimeofday () in
	let nimg = 6*2048*2 in
	let dbf = Tensor.( zeros [nimg; image_res; image_res] ) in
	for i = 0 to nimg/2-1 do (
		let k = Tensor.ones [image_res; image_res] in
		Tensor.copy_ (Tensor.narrow dbf ~dim:0 ~start:i ~length:1) ~src:k; 
	) done; 
	let y = Tensor.to_device dbf ~device in
	let z = Tensor.sum y in
	let stop = Unix.gettimeofday () in
	Printf.printf "%d image_copy time: %fs\n%!" (nimg/2) (stop -. start);
	Printf.printf "%f\n%!" (Tensor.float_value z) 
	(* this is working just as fast or faster than python.*)
	(* something else must be going on in the larger program *)

(*let make_trains steak = 
	(* pre-compute training data; enumeration better than search. *)
	let trains_sub = Vector.create ~dummy:nuldream in
	let trains_insdel = Vector.create ~dummy:nuldream in
	let edited = Array.make p_ctx 0.0 in (* needs replacement! *)
	let dba = Vector.to_array steak.db in
	let dbn = Vector.length steak.db in
	(*let dba2 = Array.sub dba 0 8192 in*)
	let make_trains_innerloop i = 
		let a = dba.(i) in
		Array.iteri (fun j b -> 
			let dist,edits = pdata_to_edits a b in
			let admit dosub = 
				if dist > 0 && (edit_criteria edits dosub) then (
					{a_pid = i; 
					b_pid = j;
					a_progenc = a.progenc; 
					b_progenc = b.progenc; 
					c_progenc = a.progenc; (* starting point *)
					edits; 
					edited; 
					count = 0; 
					dt=nuldreamtd},true
				) else (nulbatche,false)
			in
			let doadd be2 f train =
				if f then (
					let indx = Vector.length train in
					if (indx mod 1000) = 0 then
						Logs.debug (fun m -> m "trains_ size %d" indx);
					let dt = {indx; dosub=true; dtyp=`Train} in
					let be = {be2 with dt} in
					Mutex.lock steak.db_mutex;
					Vector.push train {be; decode=[]; cossim=[]; correct_cnt=0};
					Mutex.unlock steak.db_mutex
				)
			in
			let be2,f = admit true in
			doadd be2 f trains_sub ;
			let be2,f = admit false in
			doadd be2 f trains_insdel ;
		) dba in (* /make_trains_innerloop *)
	if !gparallel then ( 
		Dtask.run steak.pool (fun () -> 
			Dtask.parallel_for steak.pool ~start:0 ~finish:(dbn-1) 
				~body:make_trains_innerloop )
	) else (
		for i = 0 to (dbn-1) do (* non-parallel version *)
			make_trains_innerloop i done );
	Logs.debug (fun m -> m "Generated %d sub and %d insdel training examples" 
		(Vector.length trains_sub) (Vector.length trains_insdel)); 
	(* save them! (can recreate the edits later) *)
	let fid = open_out "trains_sub.txt" in
	Vector.iteri (fun i a -> 
		Printf.fprintf fid "%d\t%d\t%d\n" i a.be.a_pid a.be.b_pid) trains_sub;
	close_out fid; 
	let fid = open_out "trains_insdel.txt" in
	Vector.iteri (fun i a -> 
		Printf.fprintf fid "%d\t%d\t%d\n" i a.be.a_pid a.be.b_pid) trains_insdel;
	close_out fid;
	(Vector.to_array trains_sub),(Vector.to_array trains_insdel)
	
let load_trains steak = 
	let edited = Array.make p_ctx 0.0 in (* needs replacement! *)
	let i = ref 0 in
	let readfile fname dosub =
		let lines = read_lines fname in
		let linesa = Array.of_list lines in
		let r = Array.map (fun l -> 
			let sl = String.split_on_char '\t' l in
			match sl with 
			| _::aq::bq -> (
				let ai = int_of_string aq in
				let bi = int_of_string (List.hd bq) in
				let a = db_get steak ai in
				let b = db_get steak bi in
				let _dist,edits = pdata_to_edits a b in
				let dt = {indx = !i; dosub; dtyp = `Train} in
				let be = {a_pid = ai; 
						b_pid = bi;
						a_progenc = a.progenc; 
						b_progenc = b.progenc; 
						c_progenc = a.progenc; (* starting point *)
						edits; 
						edited; 
						count = 0; 
						dt} in
				incr i; 
				{be; decode=[]; cossim=[]; correct_cnt=0}
				)
			| _ -> (
				Logs.err (fun m -> m "%s could not parse %s" fname l); 
				nuldream
				) ) linesa in
		Logs.debug (fun m -> m "%s loaded %d" fname (Array.length r)); 
		r
	in
	(readfile "trains_sub.txt" true),(readfile "trains_insdel.txt" false)
	
let make_dreams () =
	(* array to accumulate mnist approximations *)
	let dreams = Vector.create ~dummy:nuldream in
	for i=0 to 160-1 do (
		let edited = Array.make p_ctx 0.0 in
			(* don't forget this has to be replaced later *)
		let dt = {indx = i; dosub = false; dtyp = `Mnist} in
		let be = {a_pid = 0; 
					b_pid = i;
					a_progenc = "";
					b_progenc = ""; 
					c_progenc = ""; 
					edits = []; 
					edited; 
					count=0; dt} in
		Vector.push dreams {be; decode=[]; cossim=[]; correct_cnt=0}
	) done; 
	let nall = Vector.length dreams in
	Logs.debug (fun m -> m "Generated %d dreams" nall);
	Vector.to_array dreams*)

let usage_msg = "program.exe -b <batch_size>"
let input_files = ref []
let output_file = ref ""
let anon_fun filename = (* just incase we need later *)
  input_files := filename :: !input_files
let speclist =
  [("-b", Arg.Set_int batch_size, "Training batch size");
   ("-o", Arg.Set_string output_file, "Set output file name"); 
   ("-g", Arg.Set g_debug, "Turn on debug");
   ("-p", Arg.Set gparallel, "Turn on parallel");
   ("-t", Arg.Set gdisptime, "Turn on timing instrumentation");]
   
let test_logo () = 
	let s = "( pen ua ; move 4 - ua , ua ; move 2 - 4 , ul / 2 )" in
	Printf.printf "%s\n" s ; 
	let g = parse_logo_string s in
	match g with
	| Some g2 -> ( 
		let ss,qq = encode_ast g2 in
		Logo.print_encoded_ast ss qq; 
		(* check the decoding *)
		let g3 = Logo.decode_ast_struct ss qq in
		Printf.printf "recon:\n%s\n" (Logo.output_program_pstr g3) )
	| _ -> ()

let () = 
	Arg.parse speclist anon_fun usage_msg;
	Random.self_init (); 
	Logs_threaded.enable ();
	let () = Logs.set_reporter (Logs.format_reporter ()) in
	let () = Logs.set_level 
		(if !g_debug then Some Logs.Debug else Some Logs.Info) in
	if !g_debug then Logs.debug (fun m -> m "Debug logging enabled.")
	else Logs.info (fun m -> m "Debug logging disabled.") ; 
	if !gparallel then 
		Logs.info (fun m -> m "Parallel enabled.")
	else 
		Logs.info (fun m -> m "Parallel disabled.") ; 
	(* Logs levels: App, Error, Warning, Info, Debug *)

	Logs.info(fun m -> m "batch_size:%d" !batch_size);
	Logs.info(fun m -> m "cuda available: %b%!" 
				(Cuda.is_available ()));
	Logs.info(fun m -> m "cudnn available: %b%!"
				(Cuda.cudnn_is_available ()));
	(*for _i = 0 to 4 do 
		measure_torch_copy_speed device
	done; *)
	test_logo (); 
	
	let mnistd = Mnist_helper.read_files ~prefix:"../otorch-test/data" () in
	let mimg = Tensor.reshape mnistd.train_images 
		~shape:[60000; 28; 28] in
	(* need to pad to 30 x 30, one pixel on each side *)
	let mnist = Tensor.zeros [60000; 30; 30] in
	Tensor.copy_ (
		Tensor.narrow mnist ~dim:1 ~start:1 ~length:28 |> 
		Tensor.narrow ~dim:2 ~start:1 ~length:28) ~src:mimg ;
		(* Tensor.narrow returns a pointer/view; copy_ is in-place. *)
		(* keep on CPU? *)
	
	(*let device = Torch.Device.Cpu in*) (* slower *)
	let device = Torch.Device.cuda_if_available () in
	let gs = Graf.create image_alloc in
	let dbf = Tensor.zeros [2;2] in
	let dbf_cpu = Tensor.zeros [2;2] in 
	let dbf_enc = Tensor.zeros [2;2] in
	let mnist_enc = Tensor.zeros [2;2] in
	let vae = Vae.dummy_ext () in
	let db_mutex = Mutex.create () in
	let pool = Dtask.setup_pool ~num_domains:12 () in 
		(* tune this -- 8-12 seems ok *)
	let supfid = open_out "/tmp/png/replacements_sup.txt" in
	let dreamfid = open_out "/tmp/png/replacements_dream.txt" in
	let fid_verify = open_out "/tmp/png/verify.txt" in
	
	let supstak = 
		{device; gs; dbf; dbf_cpu; dbf_enc; mnist; mnist_enc; vae; db_mutex;
		superv=true; sockno=4340; fid=supfid; fid_verify; batchno=0; pool } in
	
	let supsteak = if Sys.file_exists "db_sorted.S" then ( 
		(*Dtask.run supsteak.pool (fun () -> load_database supsteak )*)
		load_database supstak "db_sorted.S"
	) else ( 
		Logs.app(fun m -> m "Generating %d programs" (image_alloc/2));
		let start = Unix.gettimeofday () in
		let stk = init_database supstak (image_alloc/2) in
		(* init also sorts. *)
		let stop = Unix.gettimeofday () in
		Logs.app(fun m -> m "Execution time: %fs\n%!" (stop -. start)); 
		Logs.info(fun m -> m ":: first 10 programs");
		for i = 0 to 7 do (
			let p = db_get stk i in
			Logs.info(fun m -> m "%d: %s" i
					(Logo.output_program_pstr p.ed.pro)); 
		) done; 
		save_database stk "db_prog.S"; 
		let stk = sort_database stk in
		save_database stk "db_sorted.S";
		stk
	) in
	render_simplest supsteak; 
	
	(* try to train the vae? *)
	let dbfs = Tensor.narrow supsteak.dbf ~dim:0 ~start:0 ~length:(supsteak.gs.num_uniq) in
	let vae,dbf_enc',mnist_enc = Vae.train_ext dbfs mnist device !batch_size in
	(* need to re-expand for future allocation *)
	let encl,cols = Tensor.shape2_exn dbf_enc' in
	let dbf_enc = Tensor.( (ones [image_alloc;cols]) * (f (-1.0) ) ) in
	Tensor.copy_ (Tensor.narrow dbf_enc ~dim:0 ~start:0 ~length:encl) ~src:dbf_enc' ;
	
	(*(* dreams test structure *)
	let trains_sub, trains_insdel = 
		if Sys.file_exists "trains_sub.txt" 
			&& Sys.file_exists "trains_insdel.txt" 
		then load_trains supsteak 
		else make_trains supsteak in
		
	let dreams = make_dreams () in*)
	
	(* update the thread state *)
	let supsteak = {supsteak with dbf_enc; mnist_enc; vae } in 
	let pool2 = Dtask.setup_pool ~num_domains:8 () in
	let dreamsteak = 
		{device; gs; dbf; dbf_cpu; dbf_enc; mnist; mnist_enc; vae; db_mutex;
		superv=false; sockno=4341; fid=dreamfid; fid_verify; batchno=0; pool=pool2 } in
			
	(* extra bit of complexity!! if Cuda hangs in one of the domains, e.g. for an out-of-memory error, you won't see it on stdout -- it will just stop. 
	to properly debug, will need to strip down to one thread, no domainslib *)
	(*let d = Domain.spawn (fun _ ->
		let pool2 = Dtask.setup_pool ~num_domains:8 () in
		dreamsteak.pool <- pool2; 
		servthread dreamsteak () ) in*)
	servthread supsteak () ;
(* 	servthread dreamsteak () ; *)
	(*Domain.join d;*)
	close_out supfid; 
	close_out dreamfid; 
	Dtask.teardown_pool supsteak.pool ; 
	Dtask.teardown_pool dreamsteak.pool ;
