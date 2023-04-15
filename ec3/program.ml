open Lexer
open Lexing
open Printf
open Logo
open Ast
open Torch
open Graf

let pi = 3.1415926
(*let image_count = ref 0 *)
let image_alloc = 4*2048 (*6*2048*2*) (* need to make this a parameter *)
let image_res = 30
let batch_size = ref 512
let toklen = 30
let p_ctx = 64
let poslen = p_ctx / 2 (* onehot; two programs need to fit into the context *)
let p_indim = toklen + 1 + poslen (* 31 + 12 = 43 *)
let e_indim = 5 + toklen + poslen
let nreplace = ref 0 (* number of replacements during hallucinations *)
let glive = ref true 
let gdebug = ref false 
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
	| `Mnist of (Editree.t * int list)
		(* root node * address being evaled *)
	| `Nodream
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
				Bigarray.Array2.t (* p_ctx , poslen *)
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
	
type decode_tensors = 
	{ shape : int list
	; typ : Tensor.t
	; chr : Tensor.t
	; pos : Tensor.t
	}
	
type tsteak = (* thread state *)
	{ device : Torch.Device.t
	; gs : Graf.gstat
	; dbf : Tensor.t (* on the gpu, so we can run fast comparisons *)
	; dbf_cpu : Tensor.t
	; dbf_enc : Tensor.t
	; mnist : Tensor.t 
	; mnist_cpu : Tensor.t 
	; mnist_enc : Tensor.t (* on the GPU *)
	(*; vae : Vae.VAE.t (* GPU *)*)
	; db_mutex : Mutex.t
	; superv : bool (* supervised or dreaming *)
	(*; sockno : int 4340 for supervised, 4341 dreaming *)
	; fid : out_channel (* log for e.g. replacements *)
	; fid_verify : out_channel (* log for verification *)
	; mutable batchno : int (* for e.g doing garbage collection *)
	; mutable pool : Domainslib.Task.pool (* needs to be replaced for dream *)
	; de : decode_tensors
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

let render_simplest steak =
	(* render the shortest 1024 programs in the database.*)
	let g = Graf.sort_graph steak.gs.g in
	let dbl = Vector.length g in
	let res = 48 in
	let lg = open_out "/tmp/ec3/render_simplest.txt" in
	for id = 0 to min (1024-1) (dbl-1) do (
		let data = Vector.get g id in
		Logo.segs_to_png data.ed.segs res (Printf.sprintf "/tmp/ec3/render_simplest/s%04d.png" id);
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

let imgf_to_enc _steak _imgf =
	Tensor.randn [12;]
	(*Vae.encode1_ext steak.vae
		(Tensor.view imgf ~size:[image_res*image_res])*)

let db_add_uniq ?(doenc=true) steak ed imgf imgf_cpu =
	(*imgf is a tensor on same device as dbf*)
	let added = ref false in
	let enc = if doenc then
		imgf_to_enc steak imgf  else Tensor.( zeros [2] ) in
	Mutex.lock steak.db_mutex; 
	let indx,imgi = Graf.add_uniq steak.gs ed in
	if imgi >= 0 then (
		Tensor.copy_ ~src:imgf (Tensor.narrow steak.dbf ~dim:0 ~start:imgi ~length:1);
		Tensor.copy_ ~src:imgf_cpu (Tensor.narrow steak.dbf_cpu ~dim:0 ~start:imgi ~length:1);
		if doenc && false then
			Tensor.copy_ ~src:enc (Tensor.narrow steak.dbf_enc ~dim:0 ~start:imgi ~length:1) ;
		added := true
	); 
	Mutex.unlock steak.db_mutex; 
	!added,indx
	
let db_replace_equiv steak indx d2 imgf imgf_cpu = 
	Mutex.lock steak.db_mutex ; 
	let r,imgi = Graf.replace_equiv steak.gs indx d2 in
	if r > 0 then (
		Tensor.copy_ ~src:imgf (Tensor.narrow steak.dbf ~dim:0 ~start:imgi ~length:1);
		Tensor.copy_ ~src:imgf_cpu (Tensor.narrow steak.dbf_cpu ~dim:0 ~start:imgi ~length:1);
	); 
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

(*let dbf_dist steak img = 
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
	)*)
	
let dbf_dist steak img = 
	(* MSE between two images *)
	(* first check that there is an image *)
	let sum = Tensor.sum img |> Tensor.float_value in
	let sumt = float_of_int (image_res * image_res) in
	if (sum > 1.0) && (sum < (0.99 *. sumt)) then (
		let imgcnt = steak.gs.num_uniq in 
		assert (imgcnt > 0); (* torch calls fail ow *)
		(*let ir = image_res*image_res in*)
		let a = Tensor.narrow steak.dbf ~dim:0 ~start:0 ~length:imgcnt in
		let b = Tensor.expand img ~implicit:true ~size:[imgcnt; image_res; image_res] in
		let d = Tensor.(sum_dim_intlist (square(a - b)) ~dim:(Some [1;2]) ~keepdim:false ~dtype:(T Float)) in
		let mindex = Tensor.argmin d ~dim:(Some 0) ~keepdim:true 
			|> Tensor.int_value in
		let dist = Tensor.get d mindex |> Tensor.float_value in
		true, (dist /. sumt), mindex	
	) else (
		false, 1.0, 0 
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
	
let make_batche_train a ai b bi dt = 
	let _,edits = Graf.get_edits a.ed.progenc b.ed.progenc in
	let edited = Array.make (p_ctx/2) 0.0 in
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
	; dt (* dreamt *)
	}
	
let new_batche_train steak dt = 
	(* supervised mode, any 'A' -- including eqiv *)
	let rec selector () = 
		let di = Random.int (Vector.length steak.gs.g) in 
		let d = Vector.get steak.gs.g di in
		match d.progt with
		| `Uniq -> (
			if String.length d.ed.progenc < (p_ctx/2-2) then (
				let o = SI.elements d.outgoing in
				let ol = List.length o in
				if ol > 0 then (
					let k = Random.int ol in
					let ei = List.nth o k in
					let e = Vector.get steak.gs.g ei in
					if e.progt = `Uniq then (
						make_batche_train d di e ei dt
					) else selector ()
				) else selector ()
			) else selector () )
		| `Equiv -> (
			(* always simplify to the minimum desc *)
			if String.length d.ed.progenc < (p_ctx/2-2) && false then (
				let ei = d.equivroot in
				let e = Vector.get steak.gs.g ei in
				if e.progt = `Uniq then (
					make_batche_train d di e ei dt
				) else ( assert (0 <> 0); nulbatche ) 
			) else selector () )
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

let new_batche_mnist_cos steak =
	(* for now, just set the target B to a sample from MNIST; ultimately will need to have longer interactions & intermediate starting points *)
	(*let len = match steak.dreams with
		| Some dream -> Array.length dream
		| _ -> 1 in*)
	let mid = 1 + (Random.int 2000) in
	(* select a starting point closer to the target, w/threshold.
		goal is conflated with longer interactions, guess ? *)
	let imgcnt = steak.gs.num_uniq in (* may be updated in other threads *)
	let _dbfn,cols = Tensor.shape2_exn steak.dbf_enc in
	let a = Tensor.narrow steak.dbf_enc ~dim:0 ~start:0 ~length:imgcnt in
	let b = Tensor.narrow steak.mnist_enc ~dim:0 ~start:mid ~length:1 
			|> Tensor.expand ~implicit:true ~size:[imgcnt;cols] in
	let d = Tensor.cosine_similarity ~x1:a ~x2:b ~dim:1 ~eps:1e-7 in
	(* add a bit of noise .. ?? *)
	let d = Tensor.( d + (f 0.0005 * (randn [imgcnt;]))) in
	assert ((Tensor.shape1_exn d) = imgcnt) ; (* sanity.. *)
	let _,ind = Tensor.sort d ~dim:0 ~descending:true in
	(* select the best match that is short enough *)
	let rec selector k = 
		assert (k < imgcnt); 
		let h = Tensor.get_int1 ind k in
		let indx = steak.gs.img_inv.(h) in 
		let a = db_get steak indx in
		if String.length a.ed.progenc < (p_ctx/2-2) then (
			assert (h = a.imgi) ; 
			let edited = Array.make (p_ctx/2) 0.0 in
			let root = Editree.make_root a.ed.progenc in
			let dt = `Mnist(root, []) in
			{  a_pid = indx; b_pid = mid;
				a_progenc = a.ed.progenc; 
				b_progenc = ""; 
				c_progenc = a.ed.progenc; 
				a_imgi = a.imgi; 
				b_imgi = mid; 
				edits = []; edited; count=0; dt}
		) else selector (k+1) in
	selector 0
	(* note: bd.fresh is set in the calling function (for consistency) *)
	
let distance_array = Array.make 512 0.0 

let rec new_batche_mnist_mse steak bi =
	(* for now, just set the target B to a sample from MNIST; ultimately will need to have longer interactions & intermediate starting points *)
	(*let len = match steak.dreams with
		| Some dream -> Array.length dream
		| _ -> 1 in*)
	let mid = 1 + (Random.int 1000) in
	(* select a starting point closer to the target, w/threshold.
		goal is conflated with longer interactions, guess ? *)
	let imgcnt = steak.gs.num_uniq in (* may be updated in other threads *)
	let a = Tensor.narrow steak.dbf ~dim:0 ~start:0 ~length:imgcnt 
			|> Tensor.reshape ~shape:[imgcnt; image_res*image_res] in
	let b = Tensor.narrow steak.mnist ~dim:0 ~start:mid ~length:1
			|> Tensor.expand ~implicit:true ~size:[imgcnt;image_res;image_res]
			|> Tensor.reshape ~shape:[imgcnt; image_res*image_res] in
	let d = Tensor.(sum_dim_intlist (square(a - b)) 
			~dim:(Some [1]) ~keepdim:false ~dtype:(T Float) ) in
	assert ((Tensor.shape1_exn d) = imgcnt) ; (* sanity.. *)
	let d,ind = Tensor.sort d ~dim:0 ~descending:false in
	(* if the problem is solved, select a new digit *)
	let h = (Tensor.get_float1 d 0) /. (foi (image_res * image_res)) in
	(*Logs.debug (fun m->m "new_batche_mnist_mse %d %f" bi h);*) 
	if h > 0.025 then (
	(* select the best match that is short enough *)
	let rec selector k = 
		assert (k < imgcnt); 
		let h = Tensor.get_int1 ind k in
		let indx = steak.gs.img_inv.(h) in 
		let a = db_get steak indx in
		if String.length a.ed.progenc < (p_ctx/2-2) then (
			assert (h = a.imgi) ; 
			distance_array.(bi) <- Tensor.get_float1 d h ; 
			let edited = Array.make (p_ctx/2) 0.0 in
			let root = Editree.make_root a.ed.progenc in
			let dt = `Mnist(root, []) in
			{  a_pid = indx; b_pid = mid;
				a_progenc = a.ed.progenc; 
				b_progenc = ""; 
				c_progenc = a.ed.progenc; 
				a_imgi = a.imgi; 
				b_imgi = mid; 
				edits = []; edited; count=0; dt}
		) else selector (k+1) in
	selector 0
	) else (
		Caml.Gc.major(); 
		new_batche_mnist_mse steak bi
	)
	(* note: bd.fresh is set in the calling function (for consistency) *)


let new_batche_mnist_mse_b steak bi =
	(* for now, just set the target B to a sample from MNIST; ultimately will need to have longer interactions & intermediate starting points *)
	(*let len = match steak.dreams with
		| Some dream -> Array.length dream
		| _ -> 1 in*)
	let mid = 1 + (Random.int 1000) in
	(* select a starting point closer to the target, w/threshold.
		goal is conflated with longer interactions, guess ? *)
	let imgcnt = steak.gs.num_uniq in (* may be updated in other threads *)
	let d = Tensor.( 
			sum_dim_intlist
				(square
					(narrow steak.dbf ~dim:0 ~start:0 ~length:imgcnt) - 
					(narrow steak.mnist ~dim:0 ~start:mid ~length:1) )
				~dim:(Some [1;2]) ~keepdim:false ~dtype:(T Float) ) in
	assert ((Tensor.shape1_exn d) = imgcnt) ; (* sanity.. *)
	let _,ind = Tensor.sort d ~dim:0 ~descending:false in
	(* select the best match that is short enough *)
	let rec selector k = 
		assert (k < imgcnt); 
		let h = Tensor.get_int1 ind k in
		let indx = steak.gs.img_inv.(h) in 
		let a = db_get steak indx in
		if String.length a.ed.progenc < (p_ctx/2-2) then (
			assert (h = a.imgi) ; 
			distance_array.(bi) <- Tensor.get_float1 d h ; 
			let edited = Array.make (p_ctx/2) 0.0 in
			let root = Editree.make_root a.ed.progenc in
			let dt = `Mnist(root, []) in
			{  a_pid = indx; b_pid = mid;
				a_progenc = a.ed.progenc; 
				b_progenc = ""; 
				c_progenc = a.ed.progenc; 
				a_imgi = a.imgi; 
				b_imgi = mid; 
				edits = []; edited; count=0; dt}
		) else selector (k+1) in
	selector 0
	(* note: bd.fresh is set in the calling function (for consistency) *)
		
let new_batche_unsup steak bi =
	if (Random.int 10) < 6 then ( (* FIXME 5 *)
		new_batche_train steak `Verify
	) else (
		new_batche_mnist_mse steak bi
	)

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
	
(* fixed tensors for enumeration decoding *)
let decode_edit_tensors batchsiz = 
	let size = [batchsiz; 4; toklen; poslen] in
	let shape = [batchsiz; 4 * toklen * poslen] in
	let ityp = 
		Tensor.range ~start:(Scalar.int 0) ~end_:(Scalar.int 3) 
				~options:(T Float,Torch.Device.Cpu)
		|> Tensor.unsqueeze ~dim:0
		|> Tensor.unsqueeze ~dim:2
		|> Tensor.unsqueeze ~dim:3
		|> Tensor.expand ~size ~implicit:true in
	let ichr = 
		Tensor.range ~start:(Scalar.int 0) ~end_:(Scalar.int (toklen-1)) 
				~options:(T Float,Torch.Device.Cpu)
		|> Tensor.unsqueeze ~dim:0
		|> Tensor.unsqueeze ~dim:1
		|> Tensor.unsqueeze ~dim:3
		|> Tensor.expand ~size ~implicit:true in
	let ipos = 
		Tensor.range ~start:(Scalar.int 0) ~end_:(Scalar.int (poslen-1)) 
				~options:(T Float,Torch.Device.Cpu)
		|> Tensor.unsqueeze ~dim:0
		|> Tensor.unsqueeze ~dim:1
		|> Tensor.unsqueeze ~dim:2
		|> Tensor.expand ~size ~implicit:true in
		
	let typ = Tensor.reshape ityp ~shape in
	let chr = Tensor.reshape ichr ~shape in
	let pos = Tensor.reshape ipos ~shape in
	(* index these tensors with index from argsort *)
	(* they only need to be created once! *)
	{shape;typ;chr;pos}

let decode_edit_enumerate steak ba_edit = 
	(* Rather than sampling the pseudo-probabilities emitted by the model, 
		make 3 matrices of them.  
		sorted by probability, descending.
		This is used like beam search, which DreamCoder
		& many other models use *)
	let sta = Unix.gettimeofday () in
	let device = Torch.Device.Cpu in
	let m = tensor_of_bigarray2 ba_edit device in
	let select start length = 
		let o = Tensor.narrow m ~dim:1 ~start ~length in
		Tensor.clamp_ o ~min:(Scalar.float 1e-5) ~max:(Scalar.float 1.0)
		(* clamp above zero prob, so log works *)
	in
	let typ = select 0 4 in
	let chr = select 4 toklen in
	let pos = select (5+toklen) poslen in
	(*Printf.printf "decode_edit_enumerate typ:\n"; 
	Tensor.print typ;*) 
	(* outer-product this *)
	let x = Tensor.einsum ~equation:"bt,bc,bp -> btcp" [typ;chr;pos] 
				~path:None in
	let x2 = Tensor.reshape x ~shape:steak.de.shape in
	let index = Tensor.argsort x2 ~dim:1 ~descending:true 
		|> Tensor.narrow ~dim:1 ~start:0 ~length:32 in
	(*let typ2 = Tensor.gather steak.de.typ ~dim:1 ~index ~sparse_grad:false in*)
	(*Tensor.print typ2;*) 
	let convert w = 
		Tensor.gather w ~dim:1 ~index ~sparse_grad:false
		|> Tensor.to_bigarray ~kind:Bigarray.float32 
		|> Bigarray.array2_of_genarray 
		(* output is batch_size x 10 *)
	in
	let ba_prob = convert x2 in
	let ba_typ = convert steak.de.typ in (* float -> in conv later *)
	let ba_chr = convert steak.de.chr in
	let ba_pos = convert steak.de.pos in
	
	let fin = Unix.gettimeofday () in
	if !gdisptime then 
		Logs.debug (fun m -> m "decode_edit_enumerate time %f" (fin-.sta)); 
	ba_prob, ba_typ, ba_chr, ba_pos
	(* now we need to iterate over these tensors
		-- for each batch element
		& decide what to do with them. 
		Maybe another function ? *)
	
let decode_edit ba_edit = 
	(* decode model output (from python) *)
	let sta = Unix.gettimeofday () in
	let device = Torch.Device.Cpu in
	let m = tensor_of_bigarray2 ba_edit device in
	(* typ = th.argmax(y[:,0:4], 1)  (0 is the batch dim) *)
	(* stochastic decoding through sample_dist *)
	let typ = sample_dist (Tensor.narrow m ~dim:1 ~start:0 ~length:4) in 
	let chr = sample_dist (Tensor.narrow m ~dim:1 ~start:4 ~length:toklen) in
	let pos = sample_dist 
		(Tensor.narrow m ~dim:1 ~start:(5+toklen) ~length:poslen) in
	let edit_arr = Array.init !batch_size (fun i -> 
		let etyp = match Tensor.get_int1 typ i with
			| 0 -> "sub"
			| 1 -> "del" 
			| 2 -> "ins"
			| _ -> "fin" in
		let echr = (Tensor.get_int1 chr i) + Char.code('0') |> Char.chr in
		let eloc = Tensor.get_int1 pos i in
		(etyp,eloc,echr) ) in
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
	ignore(Editree.update_edited ~inplace:true be2.edited ed (String.length c)); 
	be2
	
let better_counter = Atomic.make 0

let img_to_imgf steak img = 
	let imgf_cpu = tensor_of_bigarray2 img Torch.Device.Cpu in
	let imgf = Tensor.to_device ~device:steak.device imgf_cpu in
	imgf_cpu,imgf
	
let try_add_program steak data img be bi = 
	if false then (
		let progstr = Logo.output_program_pstr data.pro in
		Logs.info (fun m -> m "try_add_program [%d]: %s \"%s\""
			steak.batchno data.progenc progstr) );
	let imgf_cpu,imgf = img_to_imgf steak img in
	let good2,dist,minde = dbf_dist steak imgf in
	let success = ref false in
	if good2 then (
	let mindex = steak.gs.img_inv.(minde) in
	(* idea: if it's similar to a currently stored program, 
		but has been hallucinated by the network, 
		and is not too much more costly than what's there,
		then replace the entry! *)
	(* if dist > 0.02 then (
		let added,l = db_add_uniq steak data imgf imgf_cpu in
		if added then (
			Logs.info(fun m -> m
				"try_add_program: adding new [%d] = %s" l
				(Logo.output_program_pstr data.pro) ); 
			(*success := true*)
		) else (
			Logs.debug(fun m -> m
				"try_add_program: could not add new, db full. [%d]" l )
		)
	) ; *)
	if dist < 0.0005 then (
		let data2 = db_get steak mindex in
		let c1 = data.pcost in
		let c2 = data2.ed.pcost in
		if c1 < c2 then (
			let progstr = progenc2str data.progenc in
			let progstr2 = progenc2str data2.ed.progenc in
			let r = db_replace_equiv steak mindex data imgf imgf_cpu in
			if r >= 0 then (
				let root = "/tmp/ec3/replace_verify" in
				Logs.info (fun m -> m "#%d b:%d replacing equivalents [%d] %s with %s" !nreplace steak.batchno mindex progstr2 progstr);
				Printf.fprintf steak.fid
					"(%d) [%d] d:%f %s --> %s | pcost %.2f -> %.2f\n"
					!nreplace mindex dist progstr2 progstr c2 c1 ;
				flush steak.fid;
				Logo.segs_to_png data2.ed.segs 64
					(Printf.sprintf "%s/%05d_old.png" root !nreplace);
				Logo.segs_to_png data.segs 64
					(Printf.sprintf "%s/%05d_new.png" root !nreplace);
				(* get the dbf entry too, to check *)
				let filename = Printf.sprintf
					"%s/%05d_dbf.png" root !nreplace in
				dbf_to_png steak.dbf minde filename;
				incr nreplace; 
				(*success := true*)
			)
			(* those two operations are in-place, so subsequent batches should contain the new program :-) *)
		)
	) ;
	match be.dt with 
	| `Mnist(_,_) -> ( (* sigh, redundant info *)
		(*let cpu = Torch.Device.Cpu in*)
		let a = db_get steak be.a_pid in
		let mid = be.b_pid in
		let aimg = Tensor.narrow steak.dbf ~dim:0 ~start:a.imgi ~length:1 in
		let bimg = Tensor.narrow steak.mnist ~dim:0 ~start:mid ~length:1 in
		let cimg = imgf in
		let encode v = imgf_to_enc steak v in
		let aenc = imgf_to_enc steak aimg in
		let benc = encode bimg in
		let cenc = encode cimg in
		let cos_ab = Tensor.cosine_similarity ~x1:aenc ~x2:benc ~dim:0 ~eps:1e-7 |> Tensor.float_value in
		let cos_cb = Tensor.cosine_similarity ~x1:cenc ~x2:benc ~dim:0 ~eps:1e-7 |> Tensor.float_value in
		let ab = Tensor.( mean((aimg - bimg) * (aimg - bimg)) )
			|> Tensor.float_value in
		let cb = Tensor.( mean((cimg - bimg) * (cimg - bimg)) )
			|> Tensor.float_value in
		if cb < ab then (
		(*if cos_cb > cos_ab then ( *)
			let data2 = db_get steak mindex in
			let progstr = progenc2str data.progenc in
			let progstr2 = progenc2str data2.ed.progenc in
			let progstr3 = progenc2str be.a_progenc in
			let q = Atomic.fetch_and_add better_counter 1 in
			let root = "/tmp/ec3/mnist_improve" in
			Logs.info (fun m -> m "Made an improvement! see %d; cos: %f --> %f ; mse: %f --> %f (init %f); dist %f" q cos_ab cos_cb ab cb (distance_array.(bi) /. 900.0) dist);
			Logs.info (fun m -> m "closest :%s [%d]" progstr2 mindex );
			Logs.info (fun m -> m "new     :%s " progstr );
			Logs.info (fun m -> m "original:%s [%d]" progstr3 be.a_pid);
			assert (be.a_progenc = a.ed.progenc); 
			let filename = Printf.sprintf "%s/b%05d_a_target.png" root q in
			dbf_to_png steak.mnist_cpu mid filename;
			Logo.segs_to_png a.ed.segs 64
				(Printf.sprintf "%s/b%05d_b_old.png" root q);
			Logo.segs_to_png data.segs 64
				(Printf.sprintf "%s/b%05d_c_new.png" root q);
			
			if dist > 0.005 then (
				let added,l = db_add_uniq steak data imgf imgf_cpu in
				if added then (
					Logs.info(fun m -> m
						"try_add_program: adding new [%d] = %s" l
						(Logo.output_program_pstr data.pro) ); 
					(*success := true*)
				) else (
					Logs.debug(fun m -> m
						"try_add_program: could not add new, db full. [%d]" l )
				)
			); 
			success := true
		))
	| _ -> ()
	); 
	!success
	(* NOTE not sure if we need to call GC here *)
	(* apparently not? *)
	
let update_bea_parallel body steak description =
	let sta = Unix.gettimeofday () in
	
	if !gparallel then
		Dtask.parallel_for steak.pool 
			~start:0 ~finish:(!batch_size-1) ~body
	else
		for i=0 to (!batch_size-1) do
			body i done;
			
	let fin = Unix.gettimeofday () in
	if !gdisptime then Logs.debug (fun m -> m "%s time %f" 
		description (fin-.sta));
	steak.batchno <- steak.batchno + 1
	
let update_bea_train steak bd = 
	let innerloop_bea_train bi =
		let be = bd.bea.(bi) in
		(* check edited array allocation *)
		let be1 = if Array.length be.edited <> (p_ctx/2)
			then ( let edited = Array.make (p_ctx/2) 0.0 in
				{be with edited} ) else be in
		(* last edit is 'fin', in which case: new batche *)
		let bnew = 
		 if List.length be1.edits <= 1 then (
			bd.fresh.(bi) <- true; (* update image flag *)
			new_batche_train steak `Train
		) else (
			bd.fresh.(bi) <- false;
			apply_edits be1
		) in
		bd.bea.(bi) <- bnew
	in
	update_bea_parallel innerloop_bea_train steak "update_bea_train"
	
let update_bea_verify steak bd = 
	let edit_arr = decode_edit bd.bedtd in
	let innerloop_bea_verify bi =
		let be = bd.bea.(bi) in
		let typ,loc,chr = edit_arr.(bi) in
		(* check edited array allocation *)
		let be1 = if Array.length be.edited <> (p_ctx/2)
			then ( let edited = Array.make (p_ctx/2) 0.0 in
				{be with edited} ) else be in
		let cnt = be1.count in
		let be2 = {be1 with edits=[(typ,loc,chr)];count=cnt+1} in
		let be3 = apply_edits be2 in
		let bnew = 
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
			) else (
				(* might be a simplification ? *)
				let good,data,img = progenc_to_edata be3.c_progenc in
				if good then ignore( try_add_program steak data img be3 bi)
			); 
			bd.fresh.(bi) <- true;
			new_batche_unsup steak bi
		) else (
			bd.fresh.(bi) <- false;
			be3
		) in
		bd.bea.(bi) <- bnew
	in
	update_bea_parallel innerloop_bea_verify steak "update_bea_verify"

module SE = Set.Make(
	struct
		let compare (_,at,ap,ac) (_,bt,bp,bc) = 
			if at=bt && ap=bp && ac=bc then 0 else 1 
		type t = float * string * int * char
	end )
	
let update_bea_mnist steak bd = 
	(*Logs.debug (fun m -> m "entering update_bea_mnist");*) 
	let ba_prob, ba_typ, ba_chr, ba_pos = 
		decode_edit_enumerate steak bd.bedtd in
		
	(*for j = 0 to 9 do (
		for i = 0 to 9 do (
			Logs.debug (fun m -> m "[%d,%d] pr%0.3f t%0.3f c%0.3f p%0.3f"
				i j ba_prob.{i,j} ba_typ.{i,j} ba_chr.{i,j} ba_pos.{i,j})
		) done
	) done; *) (* looks ok *)
		
	let decode bi j lc = 
		let prob = ba_prob.{bi,j} in
		let chr = (iof ba_chr.{bi,j}) + Char.code('0') |> Char.chr in
		let typ,lim,chr = match iof ba_typ.{bi,j} with
			| 0 -> "sub", lc-1, chr
			| 1 -> "del", lc-1, '0'
			| 2 -> "ins", lc, chr
			| _ -> "fin", 0, '0' in
		(* pos has to exist within the string *)
		let pos = iof ba_pos.{bi,j} in
		let pos = if pos > lim then lim else pos in
		prob,typ,pos,chr
	in
	
	let rec getedits eset lc bi j = 
		if (j < 32) && (SE.cardinal eset < 10) then (
			getedits (SE.add ( decode bi j lc ) eset) lc bi (j+1)
		) else ( eset )
	in
	let normedits elist = 
		let sum = List.fold_left 
			(fun a (p,_,_,_) -> a +. p) 0.0001 elist in
		List.map (fun (pr,t,p,c) -> (pr /. sum, t,p,c)) elist
	in
	
	let innerloop_bea_mnist bi = 
		let newdream () = 
			bd.fresh.(bi) <- true;
			new_batche_mnist_mse steak bi
		in
		let be = bd.bea.(bi) in
		let bnew = match be.dt with 
		| `Mnist(root,adr) -> (
			(* model has just been queried about adr *)
			let lc = String.length be.c_progenc in
			let eset = getedits SE.empty lc bi 0 
					|> SE.elements 
					|> normedits in
			(*if bi = 0 then (
				(* these will be sorted, descending *)
				Editree.print root [] "" 0.0 ;
				Logs.debug (fun m -> m "update_bea_mnist batch 0 edits:");
				List.iteri (fun i (pr,t,p,c) -> 
					Logs.debug(fun m -> m "  [%d] p:%0.3f %s %d %c"
						i pr t p c)) eset; 
			);*) 
			let eds = List.map (fun (_,t,p,c) -> (t,p,c)) eset in
			let pr = List.map (fun (p,_,_,_) -> log p) eset in
			Editree.model_update root adr eds pr; 
			let rec selec () = 
				let nadr,edit,k_progenc,k_edited = Editree.model_select root in
				let typ,pos,chr = edit in
				if bi = -1 then ( 
					Logs.debug (fun m -> m "[%d] selec'ted %s %d %c  @%s" 
						bi typ pos chr (Editree.adr2str nadr));
					Logs.debug (fun m -> m 
						"[%d] editree \n\tc_ %s \n\tk_ %s\n\tr_ %s\n\ta_ %s count %d"
						bi be.c_progenc k_progenc
						(Editree.getprogenc root) be.a_progenc be.count); 
					Editree.print root [] "" 0.0
				); 
				let toolong = be.count > 256 in
				if toolong then 
					Logs.debug (fun m -> m "[%d] search limit exceeded." bi); 
				match toolong,edit with
				| true,_ 
				| false,("con",_,_) -> (
					(* we did not find a suitable edit -- give up *)
					newdream () )
				| false,("fin",_,_) -> (
					(* label 'done' this leaf *)
					Editree.model_done root nadr;
					let good,data,img = progenc_to_edata k_progenc in
					if good then (
						if try_add_program steak data img be bi then (
							newdream ()
						) else selec ()
					) else selec () )
				| false,(typ,pos,chr) -> (
					(* apply the edit *)
					assert (typ <> "fin"); 
					if bi = -1 then (
						Logs.debug (fun m -> m 
							"[%d] applied %s %d %c @%s count %d\n\tr_ %s \n\tk_ %s"
							bi typ pos chr (Editree.adr2str nadr) be.count 
							(Editree.getprogenc root) k_progenc)
					); 
					let count = be.count+1 in
					let dt = `Mnist(root,nadr) in (* so we can eval its leaves *)
					{be with c_progenc=k_progenc; edited=k_edited; count; dt})
			in
			selec ()
			)
		| _ -> newdream () in
		bd.bea.(bi) <- bnew
	in (* / innerloop_ *)
	update_bea_parallel innerloop_bea_mnist steak "update_bea_mnist"; 
	(* mnist comparisons generate torch variables that need cleanup *)
	Mutex.lock steak.db_mutex;
	Caml.Gc.major (); (* clean up torch variables (slow..) *)
	Mutex.unlock steak.db_mutex

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
		let la = String.length be.a_progenc in
		let lc = String.length be.c_progenc in
		if lc > llim then 
			Logs.debug(fun m -> m  "c_progenc too long(%d):%s" lc be.c_progenc);
		String.iteri (fun i c -> 
			let j = (Char.code c) - offs in
			if (i+l) < 2*llim then (
				bd.bpro.{u,i+l,j} <- 1.0; 
				(* indicate this is c, to be edited *)
				bd.bpro.{u,i+l,toklen-1} <- 1.0 ) ) be.c_progenc ;
		(* copy over the edited tags (set in update_edited) *)
		assert (Array.length be.edited = (p_ctx/2)) ; 
		let lim = if la+lc > p_ctx then p_ctx else la+lc in
		for i = la to lim-1 do (
			bd.bpro.{u,i,toklen} <- be.edited.(i-la)
			(* a_progenc is zeroed via fill above *)
		) done; 
		(* position encoding *)
		let l = if l > llim then llim else l in 
		let lc = if lc > llim then llim else lc in
		for i = 0 to l-1 do (
			for j = 0 to poslen-1 do (
				bd.bpro.{u,i,toklen+1+j} <- bd.posenc.{i,j}
			) done
		) done;
		for i = 0 to lc-1 do (
			for j = 0 to poslen-1 do (
				bd.bpro.{u,i+l,toklen+1+j} <- bd.posenc.{i,j}
			) done
		) done ;
	  (* - bimg - *)
		if bd.fresh.(u) then (
			let pid_to_ba2 imgi dt = 
				assert (imgi >= 0) ; 
				match dt with
				| `Mnist(_,_) -> (
					assert (imgi < 60000) ; 
					Tensor.narrow steak.mnist_cpu ~dim:0 ~start:imgi ~length:1 
					|> Tensor.squeeze |> bigarray2_of_tensor ) 
				| _ -> (
					assert (imgi < steak.gs.num_uniq) ; 
					Tensor.narrow steak.dbf_cpu ~dim:0 ~start:imgi ~length:1 
					|> Tensor.squeeze |> bigarray2_of_tensor ) 
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
			for i = 0 to poslen-1 do (
				bd.bedts.{u,5+toklen+i} <- bd.posenc.{pp,i}
			) done
		) in (* /innerloop_bigfill_batchd *)
	if !gparallel then
		Dtask.parallel_for steak.pool ~start:0 ~finish:(!batch_size-1)
			~body:innerloop_bigfill_batchd
	else
		for i=0 to (!batch_size-1) do
			innerloop_bigfill_batchd i done; 
	let fin = Unix.gettimeofday () in
	if !gdisptime then 
		Logs.debug (fun m -> m "bigfill_batchd time %f" (fin-.sta))
		
let reset_bea steak dreaming =
	(* dt sets the 'mode' of batchd, which persists thru update_bea*)
	let dt = `Train in
	(*Printf.printf "\n------- reset_bea:\n"; 
	Gc.print_stat stdout; *)
	let bea = Array.init !batch_size
		(fun i -> 
			if dreaming then new_batche_mnist_mse steak i
			else new_batche_train steak dt) in
	let fresh = Array.init !batch_size (fun _i -> true) in
	(*Printf.printf "\n------- after new_batche:\n"; 
	Gc.print_stat stdout; 
	Caml.Gc.full_major(); 
	Printf.printf "\n------- after Gc.full_major ():\n"; 
	Gc.print_stat stdout; 
	flush stdout; *)
	bea,fresh
	
let init_batchd steak superv =
	Logs.debug (fun m -> m "init_batchd"); 
	let dreaming = not superv in
	let filnum = if superv then 0 else 1 in
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
			p_ctx poslen in
	let bea,fresh = reset_bea steak dreaming in
	(* fill out the posenc matrix *)
	(* just a diagonal matrix w one-hot! *)
	for i = 0 to (poslen-1) do (
		for j = 0 to (poslen-1) do (
			posenc.{i,j} <- if i = j then 1.0 else 0.0; 
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
	(*for k = 0 to (Vector.length g) -1 do (
		dbf_to_png dbf_cpu k (Printf.sprintf "/tmp/ec3/dbf_check_%d.png" k)
	) done; *)
	dbf,dbf_cpu

let sort_database steak =
	(* sort the graph 
		well, sort the Vector underlying the graph -- since graphs don't really have an order. 
		redo dbf as well. *)
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
	let root = "/tmp/ec3/init_database" in
	let fid = open_out (Printf.sprintf "%s/newdbg.txt" root) in
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
		if dist > 0.02 then ( 
			let s = Logo.output_program_pstr data.pro in
			Logs.debug(fun m -> m 
				"%d: adding [%d] = %s" !iters !i s); 
			ignore( db_add_uniq steak data imgf imgf_cpu ~doenc:false );
			Logo.segs_to_png data.segs 64
				(Printf.sprintf "%s/db%05d_.png" root !i); 
			dbf_to_png steak.dbf !i
				(Printf.sprintf "%s/db%05d_f.png" root !i);
			Printf.fprintf fid "[%d] %s (dist:%f to:%d)\n" !i s dist mindex; 
			incr i;
		) ; 
		if dist < 0.0005 then (
			(* see if there's a replacement *)
			let data2 = db_get steak mindex in
			let c1 = data.pcost in (* progenc_cost  *)
			let c2 = data2.ed.pcost in
			if c1 < c2 then (
				let r = db_replace_equiv steak mindex data imgf imgf_cpu in 
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
						"iter %d: added equiv [loc %d] = %s [new loc %d] ( simpler %s) %s %s same:%b " 
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

let verify_database steak =
	(* render everything again for sanity *)
	let root = "/tmp/ec3/verify_database" in
	let n = Vector.length steak.gs.g in
	let imf = Tensor.zeros [n; image_res; image_res] in
	Vector.iteri (fun i d -> 
		let _,img = Graf.pro_to_edata d.ed.pro image_res in
		let imgf_cpu,_ = img_to_imgf steak img in
		Tensor.copy_ (Tensor.narrow imf ~dim:0 ~start:i ~length:1) ~src:imgf_cpu; 
		Logo.segs_to_png d.ed.segs 64
			(Printf.sprintf "%s/db%05d_.png" root i); 
		) steak.gs.g; 
	let good = ref true in
	let verify_equivalent i d j e =
		let u = Tensor.narrow imf ~dim:0 ~start:i ~length:1 in
		let v = Tensor.narrow imf ~dim:0 ~start:j ~length:1 in
		let w = Tensor.( sum ( square (u - v) ) ) |> Tensor.float_value in
		let sumt = float_of_int (image_res * image_res) in
		let w = w /. sumt in
		if w > 0.002 then (
			Logs.err (fun m -> m "%d %d: %s not equiv %s (%f)"
				i j (Logo.output_program_pstr d.ed.pro) (Logo.output_program_pstr e.ed.pro) w ); 
			good := false
		)
	in
	let gi = Vector.to_list steak.gs.g |> List.mapi (fun i d -> i,d) in
	Vector.iteri (fun i d -> 
		match d.progt with
		| `Uniq -> (
			(* verify that all equivalents are actually that. *)
			SI.iter (fun j -> 
				let e = Vector.get steak.gs.g j in
				verify_equivalent i d j e
				) d.equivalents ; 
			(* re-create outgoing to verify *)
			let ii,_ = List.filter (fun (j,e) -> 
				if j <> i then 
				let _,edits = Graf.get_edits d.ed.progenc e.ed.progenc in
				let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
				Graf.edit_criteria edits else false ) gi
				|> List.split in
			let og = SI.of_list ii in
			if og <> d.outgoing then (
				let ogl = SI.cardinal og in
				let doutl = SI.cardinal d.outgoing in
				Printf.printf "%d %s outgoing wrong! len %d should be %d\n"
					i (Logo.output_program_pstr d.ed.pro) doutl ogl ; 
				let print_diff a b =
					let df = SI.diff a b in
					Printf.printf "diff:\n"; 
					SI.iter (fun k -> 
						let e = Vector.get steak.gs.g k in
						let dist,edits = Graf.get_edits d.ed.progenc e.ed.progenc in
						Printf.printf "fwd\t%d, %s -> %s dist %d\n" k d.ed.progenc e.ed.progenc dist; 
						Levenshtein.print_edits edits; 
						(* flip it and do it again *)
						let dist,edits = Graf.get_edits e.ed.progenc d.ed.progenc in
						Printf.printf "rev\t%d, %s -> %s dist %d\n" k e.ed.progenc d.ed.progenc dist; 
						Levenshtein.print_edits edits; 
						) df ;  
				in
				if ogl > doutl then print_diff og d.outgoing;
				if doutl > ogl then print_diff d.outgoing og ; 
				good := false )
			)
		| `Equiv -> (
			let j = d.equivroot in
			let e = Vector.get steak.gs.g j in
			verify_equivalent i d j e)
		| _ -> () 
		) steak.gs.g; 
	if !good then 
		Logs.info (fun m -> m "database verified.")

let save_database steak fname = 
	(*let g = Graf.sort_graph steak.gs.g in*)
	Graf.save fname steak.gs.g; 
	Logs.info (fun m -> m "saved %d (%d uniq, %d equiv) to %s" 
		(Vector.length steak.gs.g) 
		steak.gs.num_uniq steak.gs.num_equiv fname )
		
let load_database steak fname = 
	(* returns a new steak, freshly sizzled *)
	let gs0 = create image_alloc in
	let gs = Graf.load gs0 fname in
	let dbf,dbf_cpu = render_database steak gs.g in
	Logs.info (fun m -> m "Loaded %d: %d uniq and %d equivalent"
		(Vector.length gs.g) gs.num_uniq gs.num_equiv ); 
	let steak = {steak with gs; dbf; dbf_cpu} in
	steak
	
let handle_message steak bd msg =
	(*Logs.debug (fun m -> m "handle_message %s" msg);*)
	let msgl = String.split_on_char ':' msg in
	let cmd = if List.length msgl = 1 then msg 
		else List.hd msgl in
	match cmd with
	| "update_batch" -> (
		(* supervised: sent when python has a working copy of data *)
		(* dreaming : sent when the model has produced edits (maybe old) *)
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
		if steak.superv then (
			let edit_dream = decode_edit bd.bedtd in
			decode_edit_accuracy edit_dream
		); 
		
		if steak.superv then 
			update_bea_train steak bd
		else
			update_bea_mnist steak bd ; 
		
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
	
let superv2sockno = function
	| true -> 4340
	| false -> 4341

let servthread steak () = (* steak = thread state *)
	(let open Unix in
	let sockno = superv2sockno steak.superv in
	Logs.info (fun m -> m "Starting server on %d" sockno ); 
	let server_sock = socket PF_INET SOCK_STREAM 0 in
	let listen_address = inet_addr_loopback in
	setsockopt server_sock SO_REUSEADDR true ; (* for faster restarts *)
	bind server_sock (ADDR_INET (listen_address, sockno)) ;
	listen server_sock 2 ; (* 2 max connections *)
	while !glive do (
		let bd,fdlist = init_batchd steak steak.superv in
		let (client_sock, _client_addr) = accept server_sock in
		Logs.info (fun m -> m "new connection on %d" sockno); 
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
		Logs.debug (fun m -> m "disconnect %d" sockno); 
		List.iter (fun fd -> close fd) fdlist; 
		save_database steak "db_improved.S"; 
		Mutex.lock steak.db_mutex;
		Caml.Gc.full_major (); (* clean up torch variables (slow..) *)
		Mutex.unlock steak.db_mutex
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

