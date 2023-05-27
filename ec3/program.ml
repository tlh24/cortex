open Printf
open Logo
open Ast
open Torch
open Graf

let pi = 3.1415926
(*let image_count = ref 0 *)
let image_alloc = 24*1024 (*6*2048*2*) (* need to make this a parameter *)
let all_alloc = 25*1024 (* including equivalents *)
let image_res = 30
let batch_size = ref 512
let toklen = 30
let p_ctx = 96
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
	; b_progenc : string (* to; for checking when superv; blank if dream *)
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
	(*; posencn : Tensor.t*) (* normalized position encoding *)
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
	; sdb : Simdb.imgdb
	; mnist : Tensor.t 
	; mnist_cpu : Tensor.t 
	(*; vae : Vae.VAE.t (* GPU *)*)
	; mutex : Mutex.t
	; superv : bool (* supervised or dreaming *)
	(*; sockno : int 4340 for supervised, 4341 dreaming *)
	; fid : out_channel (* log for e.g. replacements *)
	; fid_verify : out_channel (* log for verification *)
	; mutable batchno : int (* for e.g doing garbage collection *)
	; mutable pool : Domainslib.Task.pool (* needs to be replaced for dream *)
	; de : decode_tensors
	; training : (int*int*int) array
	}

let read_lines name : string list =
	let ic = open_in name in
	let try_read () =
		try Some (input_line ic) with End_of_file -> None in
	let rec loop acc = match try_read () with
		| Some s -> loop (s :: acc)
		| None -> close_in ic; List.rev acc in
	loop []

let bigarray_to_bytes arr = 
	(* convert a bigarray to a list of bytes *)
	(* this is not efficient.. *)
	let len = Bigarray.Array1.dim arr in
	Bytes.init len 
		(fun i -> Bigarray.Array1.get arr i |> Char.chr)

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
		| _ -> (false,nuledata,Logo.nulimg) )
	| _ -> (false,nuledata,Logo.nulimg)

let render_simplest steak =
	(* render the shortest 1024 programs in the database.*)
	let g = Graf.sort_graph steak.gs.g in
	let dbl = Array.length g in
	let res = 48 in
	let lg = open_out "/tmp/ec3/render_simplest.txt" in
	for id = 0 to min (1024-1) (dbl-1) do (
		let data = g.(id) in
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
	
let tensor_of_bigarray1 img = 
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
				o.%.{[i;j]} <- cf ; )
			)
		) done 
	) done;
	o 
	
let bigarray1_of_tensor m = 
	let l = image_res - 1 in
	let ba = Simdb.newrow () in
	for i = 0 to l do (
		for j = 0 to l do (
			ba.{i*image_res + j} <- int_of_float 
				( Tensor.get_float2 m i j  )
		) done ; 
	) done; 
	ba
		
let db_get steak i = 
	Mutex.lock steak.mutex ; 
	let r = steak.gs.g.(i) in
	Mutex.unlock steak.mutex ; 
	r

let db_add_uniq steak ed img =
	(* img is a bigarray uchar *)
	let added = ref false in
	let indx,imgi = Graf.add_uniq steak.mutex steak.pool steak.gs ed in
	if imgi >= 0 then (
		Mutex.lock steak.mutex ; 
		Simdb.rowset steak.sdb imgi img; 
		Mutex.unlock steak.mutex ; 
		added := true
	); 
	!added,indx
	
let db_replace_equiv steak indx d2 img = 
	(*let d = db_get steak indx in
	Logs.debug (fun m->m "replace_eqiv [%d] imgi %d" indx d.imgi);*)
	let r,imgi = Graf.replace_equiv steak.mutex steak.pool steak.gs indx d2 in
	if r > 0 then (
		Mutex.lock steak.mutex ; 
		Simdb.rowset steak.sdb imgi img; 
		Mutex.unlock steak.mutex ; 
	); 
	r
	
let db_add_equiv steak indx d2 = 
	let r = Graf.add_equiv steak.mutex steak.pool steak.gs indx d2 in
	r
	
let db_len steak = 
	Mutex.lock steak.mutex ; 
	let r = Array.length steak.gs.g in
	Mutex.unlock steak.mutex ; 
	r
	
let simdb_dist steak img = 
	(* MSE between two images *)
	let dist,mindex = Simdb.query steak.sdb img in
	(* Simdb.query returns -1.0 if you pass a blank image *)
	(dist >= 0.0), dist, mindex

let simdb_to_png steak i filename =
	let ba = Simdb.rowget steak.sdb i in
	let im = tensor_of_bigarray1 ba 
		|> Tensor.unsqueeze ~dim:2 
		|> Tensor.expand ~size:[-1;-1;3] ~implicit:true in
	Torch_vision.Image.write_image 
		Tensor.((f 1. - im) * f 255.) ~filename
	
let normalize_tensor m = 
	(* normalizes length along dimension 1 *)
	let len = Tensor.einsum ~equation:"ij,ij -> i" [m;m] ~path:None 
			|> Tensor.sqrt in
	let len = Tensor.(f 1. / len) in
	Tensor.einsum ~equation:"ij,i -> ij" [m;len] ~path:None 
	
let render_database steak = 
	(* TODO: make the iter parallel *)
	(* imgi index is set by load_database / graph ops *)
	let sum = Simdb.checksum steak.sdb in
	Logs.debug (fun m->m "render_database start; sum %f" sum); 
	Simdb.clear steak.sdb; 
	Array.iteri (fun i _ -> steak.gs.img_inv.(i) <- -2) steak.gs.img_inv; 
	Array.iteri (fun i d -> 
		match d.progt with 
		| `Uniq -> (
			let _,img = Graf.pro_to_edata d.ed.pro image_res in
			Simdb.rowset steak.sdb d.imgi img; 
			steak.gs.img_inv.(d.imgi) <- i; 
			)
		| _ -> () ) steak.gs.g ; 
	
	let sum = Simdb.checksum steak.sdb in
	Logs.debug (fun m->m "render_database done; sum %f" sum); 
	;;
	
(* new training: we need to select the target image from
	*anywhere* 'downstream' of the current edit.
	for example, if an edit chain (from dijkstra) is
	(..) -> a -> b -> c -> d
	and we select edit a -> b
		to gen the supervised edits
	then possible context for the image is b, c, or d.

	It probably makes the most sense to store this datastructure
	a bit differently, so we can sample a, b, c.
		a = start (anywhere along causal chain except end)
		b = start+1 (one edit from start)
		c = context (select an image)
	*)

module SX = Set.Make(Int);;
	
let make_training steak = 
	(* make/remake the training array *)
	let nuniq = steak.gs.num_uniq in
	Logs.debug (fun m->m "entering make_training. num_uniq %d" nuniq); 
	let fid = open_out "training.txt" in
	
	Graf.remove_unreachable steak.mutex steak.gs; 
	(* removes terminal `Equiv nodes *)
	render_database steak; 
	
	let dist,prev = Graf.dijkstra steak.gs 0 false in

	let rec make_devlist l r =
		(* assemble a list, reverse order, for the chain of programs
		leading to a terminal r *)
		if r >= 0 then (
			let q = prev.(r) in
			if q >= 0 then (
				make_devlist ( r :: l ) q
			) else l
		) else l
	in

	let print_devlist k l =
		List.iteri (fun i j ->
			let d = db_get steak j in
			let s = progenc2str d.ed.progenc in
			Printf.fprintf fid "terminal [%d] step %d node [%d] %s\n" k i j s;
		) l
	in

	(* only start from terminal nodes, to avoid redundancy *)
	let terminal = Array.make (Array.length dist) false in
	Array.iteri (fun i d -> if d > 0 then terminal.(i) <- true) dist;
	Array.iter (fun p -> if p >= 0 then terminal.(p) <- false) prev;

	let (_,training) = Array.fold_left (fun (k,lst) term ->
		if term then (
			let l = make_devlist [] k in
			print_devlist k l ;

			(* this would probably be simpler with nested for-loops
				but ohwell, let's do it the ocaml way *)
			let rec select_c acc ll a b =
				if List.length ll > 0 then (
					let c = List.hd ll in
					select_c ((a,b,c) :: acc) (List.tl ll) a b
				) else acc
			in

			let rec select_bc acc ll a =
				if List.length ll > 0 then (
					let b = List.hd ll in
					let bacc = select_c acc ll a b in
					select_bc bacc (List.tl ll) a
				) else acc
			in

			let rec select_abc acc ll =
				if List.length ll >= 2 then (
					let a = List.hd ll in
					let aacc = select_bc acc (List.tl ll) a in
					select_abc aacc (List.tl ll)
				) else acc
			in

			let acc = select_abc [] l in
			(k+1, List.rev_append acc lst)
		) else (
			(k+1, lst)
		) ) (0, []) terminal in
	
	(*(* add in everything to the training database *)
	let s = Array.fold_left (fun a b -> SX.add b a) SX.empty prev in
	List.iter ( fun target -> 
		let rec loop r = 
			if r >= 0 then (
				let q = prev.(r) in
				if q >= 0 then (
					training := (q,r) :: !training; 
					loop q
				)
			)
		in
		loop target (* if there are no routes to root, don't add *)
		) (SX.elements s); *)
	
	List.iteri (fun i (pre,post,context) ->
		Printf.fprintf fid "[%d] %d %d %d\n" i pre post context) training;
	close_out fid; 
	
	Graf.gexf_out steak.gs ;
	let ta = Array.of_list training in
	{steak with training = ta}
	;;
	
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

let new_batche_train steak dt = 
	let n = Array.length steak.training in
	let i = Random.int n in
	let (pre,post,context) = steak.training.(i) in
	let a = steak.gs.g.(pre) in
	let b = steak.gs.g.(post) in
	let c = steak.gs.g.(context) in
	let _,edits = Graf.get_edits a.ed.progenc b.ed.progenc in
	let edited = Array.make (p_ctx/2) 0.0 in
	{ a_pid = pre
	; b_pid = post
	; a_progenc = a.ed.progenc
	; b_progenc = b.ed.progenc
	; c_progenc = a.ed.progenc
	; a_imgi = a.imgi
	; b_imgi = c.imgi (* harder! :-) *)
	; edits
	; edited
	; count = 0
	; dt (* dreamt *)
	}

let rec new_batche_mnist_mse steak bi =
	(* select a random mnist image *)
	let mid = 1 + (Random.int 1000) in
	let b = Tensor.narrow steak.mnist ~dim:0 ~start:mid ~length:1 
		|> Tensor.squeeze in
	(* add a bit of (uniform) noise *)
	let c = Tensor.(b + (rand_like b) * (f 30.0)) in
	(* convert to bigarray *)
	let cba = bigarray1_of_tensor c in
	(* select the closest in the database *)
	let dist,ind = Simdb.query steak.sdb cba in
	let indx = steak.gs.img_inv.(ind) in
	let a = db_get steak indx in
		if ind <> a.imgi then (
			Logs.err (fun q->q "consistency failure: mindex %d maps back to gs.g.(%d); gs.g.(%d).imgi = %d progenc=%s" ind indx indx a.imgi a.ed.progenc); 
			assert (0 <> 0) 
		);
	let short = String.length a.ed.progenc < (p_ctx/2-2) in
	if dist > 768.0 && short then ( (* three pixels thresh *)
		if bi = 0 then 
			Logs.debug (fun m->m "new_batche_mnist dist %f indx %d" dist indx); 
		let edited = Array.make (p_ctx/2) 0.0 in
		let root = Editree.make_root a.ed.progenc in
		let dt = `Mnist(root, []) in
		{  a_pid = indx; b_pid = mid;
			a_progenc = a.ed.progenc; 
			b_progenc = ""; (* only used for checking *)
			c_progenc = a.ed.progenc; 
			a_imgi = a.imgi; 
			b_imgi = mid; 
			edits = []; edited; count=0; dt }
	) else (
		if bi = 0 then 
			Logs.debug (fun m->m "new_batche_mnist h %f" dist); 
		new_batche_mnist_mse steak bi
	)
	(* note: bd.fresh is set in the calling function (for consistency) *)
		
let new_batche_unsup steak bi =
	if (Random.int 10) < 10 then ( (* FIXME 5 *)
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
	let ba_typ = convert steak.de.typ in (* float -> int conv later *)
	let ba_chr = convert steak.de.chr in
	let ba_pos = convert steak.de.pos in
	
	let fin = Unix.gettimeofday () in
	if !gdisptime then 
		Logs.debug (fun m -> m "decode_edit_enumerate time %f" (fin-.sta)); 
	ba_prob, ba_typ, ba_chr, ba_pos
	(* now we need to iterate over these tensors
		-- for each batch element
		& decide what to do with them. *)
	
let decode_edit ?(sample=true) ba_edit = 
	(* decode model output (from python) *)
	(* default is to sample the distribution; 
		set sample to false for argmax decoding. *)
	let sta = Unix.gettimeofday () in
	let device = Torch.Device.Cpu in
	let m = tensor_of_bigarray2 ba_edit device in
	(* typ = th.argmax(y[:,0:4], 1)  (0 is the batch dim) *)
	(* stochastic decoding through sample_dist *)
	let sfun = if sample then sample_dist else sample_dist_dum in
	let typ = sfun (Tensor.narrow m ~dim:1 ~start:0 ~length:4) in 
	let chr = sfun (Tensor.narrow m ~dim:1 ~start:4 ~length:toklen) in
	let pos = sfun 
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
	
let apply_edits be ed = 
	(* apply after (of course) sending to python. *)
	(* edit length must be > 0 *)
	(* does not change the edit list! *)
	(* note: Levenshtein clips the edit positions *)
	let cnt = be.count in
	let c = Levenshtein.apply_edits be.c_progenc [ed] in
	let be2 = { be with c_progenc=c; count=cnt+1 } in
	ignore(Editree.update_edited 
		~inplace:true be2.edited ed (String.length c)); 
	be2
	
let better_counter = Atomic.make 0
	
let try_add_program steak data img be = 
	if false then (
		let progstr = Logo.output_program_pstr data.pro in
		Logs.info (fun m -> m "try_add_program [%d]: %s \"%s\""
			steak.batchno data.progenc progstr) );
	let good2,dist,minde = simdb_dist steak img in
	(* dist is in uchar units; used to be floats. *)
	let distnorm = dist /. (foi (image_res * image_res * 255)) in
	let success = ref false in
	if good2 then (
	let mindex = steak.gs.img_inv.(minde) in
	
	if distnorm < 0.0002 then (
		let data2 = db_get steak mindex in
		let c1 = data.pcost in
		let c2 = data2.ed.pcost in
		if c1 < c2 then (
			let progstr = progenc2str data.progenc in
			let progstr2 = progenc2str data2.ed.progenc in
			let r = db_replace_equiv steak mindex data img in
			if r >= 0 then (
				let root = "/tmp/ec3/replace_verify" in
				Logs.info (fun m -> m "#%d b:%d replacing equivalents [%d] %f %s with %s" !nreplace steak.batchno mindex distnorm progstr2 progstr);
				Printf.fprintf steak.fid
					"(%d) [%d] d:%f %s --> %s | pcost %.2f -> %.2f\n"
					!nreplace mindex distnorm progstr2 progstr c2 c1 ;
				flush steak.fid;
				Logo.segs_to_png data2.ed.segs 64
					(Printf.sprintf "%s/%05d_old.png" root !nreplace);
				Logo.segs_to_png data.segs 64
					(Printf.sprintf "%s/%05d_new.png" root !nreplace);
				(* get the simdb entry too, to check *)
				let filename = Printf.sprintf
					"%s/%05d_simdb.png" root !nreplace in
				simdb_to_png steak minde filename;
				incr nreplace; 
				success := true
			)
			(* those two operations are in-place, so subsequent batches should contain the new program :-) *)
		)
	) ;
	
	if distnorm > 0.002 then (
	match be.dt with 
	| `Mnist(_,_) -> (
		(*let cpu = Torch.Device.Cpu in*)
		let a = db_get steak be.a_pid in
		let mid = be.b_pid in
		let aimg = Simdb.rowget steak.sdb a.imgi |> tensor_of_bigarray1 in
		let bimg = Tensor.narrow steak.mnist_cpu ~dim:0 ~start:mid ~length:1 in
		let cimg = tensor_of_bigarray1 img in
		
		let ab = Tensor.( mean((aimg - bimg) * (aimg - bimg)) )
			|> Tensor.float_value in
		let cb = Tensor.( mean((cimg - bimg) * (cimg - bimg)) )
			|> Tensor.float_value in
		if cb < ab then (
			let data2 = db_get steak mindex in
			let progstr = progenc2str data.progenc in
			let progstr2 = progenc2str data2.ed.progenc in
			let progstr3 = progenc2str be.a_progenc in
			let q = Atomic.fetch_and_add better_counter 1 in
			let root = "/tmp/ec3/mnist_improve" in
			Logs.info (fun m -> m 
				"Made an improvement! see %d ; mse: %f --> %f; db dist %f" 
				q ab cb distnorm);
			Logs.info (fun m -> m "closest :%s [%d]" progstr2 mindex );
			Logs.info (fun m -> m "new     :%s " progstr );
			Logs.info (fun m -> m "original:%s [%d]" progstr3 be.a_pid);
			assert (be.a_progenc = a.ed.progenc); 
			let filename = Printf.sprintf "%s/b%05d_a_target.png" root q in
			simdb_to_png steak mid filename;
			Logo.segs_to_png a.ed.segs 64
				(Printf.sprintf "%s/b%05d_b_old.png" root q);
			Logo.segs_to_png data.segs 64
				(Printf.sprintf "%s/b%05d_c_new.png" root q);
			
			if dist > 0.005 then (
				let added,l = db_add_uniq steak data img in
				if added then (
					Logs.info(fun m -> m
						"try_add_program: adding new [%d] = %s" l
						(Logo.output_program_pstr data.pro) ); 
					success := true
				) else (
					Logs.debug(fun m -> m
						"try_add_program: could not add new, db full. [%d]" l )
				)
			); 
			success := true
		))
	| _ -> ()
	) ); 
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
		let be1 = bd.bea.(bi) in
		(* check edited array allocation *)
		(*let be1 = if Array.length be.edited <> (p_ctx/2)
			then ( let edited = Array.make (p_ctx/2) 0.0 in
				{be with edited} ) else be in*)
		(* last edit is 'fin', in which case: new batche *)
		let bnew = 
		 if List.length be1.edits <= 1 then (
			bd.fresh.(bi) <- true; (* update image flag *)
			new_batche_train steak `Train
		) else (
			bd.fresh.(bi) <- false;
			let be2 = apply_edits be1 (List.hd be1.edits) in
			{be2 with edits = (List.tl be2.edits)}
		) in
		bd.bea.(bi) <- bnew
	in
	update_bea_parallel innerloop_bea_train steak "update_bea_train"

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
		
	(* deterministic decoding, for `Verify *)
	let edit_arr = decode_edit ~sample:false bd.bedtd in
		
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
	
	(* make a set of the top 10 edits at this branch *)
	let rec getedits eset lc bi j = 
		if (j < 32) && (SE.cardinal eset < 10) then (
			getedits (SE.add ( decode bi j lc ) eset) lc bi (j+1)
		) else ( eset )
	in
	(* make the probabilities sum to 1 *)
	let normedits elist = 
		let sum = List.fold_left 
			(fun a (p,_,_,_) -> a +. p) 0.0001 elist in
		List.map (fun (pr,t,p,c) -> (pr /. sum, t,p,c)) elist
	in
	
	let innerloop_bea_mnist bi = 
		let newdream () = 
			bd.fresh.(bi) <- true;
			new_batche_unsup steak bi
		in
		let be = bd.bea.(bi) in
		let lc = String.length be.c_progenc in
		let eset = getedits SE.empty lc bi 0 
					|> SE.elements 
					|> normedits in
		let eds = List.map (fun (_,t,p,c) -> (t,p,c)) eset in
		let pr = List.map (fun (p,_,_,_) -> log p) eset in
		let bnew = match be.dt with 
		| `Mnist(root,adr) -> (
			(* model has just been queried about adr *)
			(*if bi = 0 then (
				(* these will be sorted, descending *)
				Editree.print root [] "" 0.0 ;
				Logs.debug (fun m -> m "update_bea_mnist batch 0 edits:");
				List.iteri (fun i (pr,t,p,c) -> 
					Logs.debug(fun m -> m "  [%d] p:%0.3f %s %d %c"
						i pr t p c)) eset; 
			);*) 
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
				let toolong = be.count > 100 in
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
						if try_add_program steak data img be then (
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
		| `Verify -> (
			let dtyp,dloc,dchr = edit_arr.(bi) in (* deterministic *)
			let etyp,eloc,echr = List.hd eds in (* editree; compare *)
			let styp,sloc,schr = if List.length be.edits > 0 then 
				List.hd be.edits else ("fin",0,'0') in (* supervised *)
			(* check edited array allocation *)
			(* let be1 = if Array.length be.edited <> (p_ctx/2)
				then ( let edited = Array.make (p_ctx/2) 0.0 in
					{be with edited} ) else be in *)
			let ed = (dtyp,dloc,dchr) in
			let be2 = apply_edits be ed in
			let be3 = if List.length be2.edits > 1 
				then {be2 with edits = (List.tl be2.edits)} else be2 in
			if bi = 0 then (
				let ea,eb,ec = be3.a_progenc, 
									be3.b_progenc,
									be3.c_progenc in
				let sa,sb,sc = (progenc2str ea), 
									(progenc2str eb), 
									(progenc2str ec) in
				let cnt = be3.count in
				Logs.debug (fun m -> m "\n\ttrue: \027[34m %s [%d] %c\027[0m ; ( i%d b%d ) \n\tdetrm:\027[31m %s [%d] %c\027[0m \n\teditr:\027[32m %s [%d] %c\027[0m cnt:%d \n|a:%s \027[34m %s \027[0m \n|b:%s \027[31m %s \027[0m \n|c:%s \027[32m %s \027[0m"
							styp sloc schr   steak.batchno bi
							dtyp dloc dchr
							etyp eloc echr
							cnt ea sa eb sb ec sc)
			);
			if dtyp = "fin" || be3.count >= p_ctx/2 then (
				(* write result to verify.txt; stats later *)
				let c = if be3.c_progenc = be3.b_progenc then '+' else '-' in
				Printf.fprintf steak.fid_verify "[%c] from:%s \n      to:%s \n  decode:%s\n" c
					(progenc2str be3.a_progenc)
					(progenc2str be3.b_progenc)
					(progenc2str be3.c_progenc); 
				flush steak.fid_verify; 
				if c = '+' then (
					Graf.incr_good steak.gs be3.a_pid; 
					Graf.incr_good steak.gs be3.b_pid 
				);  
				(* else (
					(* might be a simplification ? *)
					let good,data,img = progenc_to_edata be3.c_progenc in
					if good then ignore( try_add_program steak data img be3 bi)
				) *)
				newdream ()
			) else ( 
				bd.fresh.(bi) <- false;
				be3 
			)
			)
		| _ -> newdream () in
		bd.bea.(bi) <- bnew
	in (* / innerloop_ *)
	update_bea_parallel innerloop_bea_mnist steak "update_bea_mnist" 
	(* mnist comparisons generate torch variables that need cleanup *)
	(*Mutex.lock steak.mutex;
	Caml.Gc.major (); (* clean up torch variables (slow..) *)
	Mutex.unlock steak.mutex*)
	;;

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
		let lim = if lc > p_ctx/2 then p_ctx/2 else lc in
		assert (la < p_ctx/2); 
		for i = 0 to lim-1 do (
			bd.bpro.{u,la+i,toklen} <- be.edited.(i)
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
			let pid_to_ba1 imgi dt = 
				if imgi < 0 then (
					Logs.err (fun m->m "imgi:%d be.a_pid:%d be.b_pid:%d be.a_imgi:%d be.b_imgi:%d"
					imgi be.a_pid be.b_pid be.a_imgi be.b_imgi)
				); 
				assert (imgi >= 0) ; 
				match dt with
				| `Mnist(_,_) -> (
					assert (imgi < 60000) ; 
					Tensor.narrow steak.mnist_cpu ~dim:0 ~start:imgi ~length:1 
					|> Tensor.squeeze |> bigarray1_of_tensor ) 
				| _ -> (
					assert (imgi < steak.gs.image_alloc) ; 
					Simdb.rowget steak.sdb imgi ) 
			in
			(* would be good if this didn't require two copy ops *)
			(* tensor to bigarray, bigarray to bigarray *)
			if u = 0 then (
				Logs.debug (fun m -> m "bigfill_batchd a g:%d i:%d b g:%d i:%d" 
				be.a_pid be.a_imgi be.b_pid be.b_imgi)); 
			let aimg = pid_to_ba1 be.a_imgi `Train in
			let bimg = pid_to_ba1 be.b_imgi be.dt in
			for i=0 to (image_res-1) do (
				for j=0 to (image_res-1) do (
					let aif = (foi aimg.{i*image_res+j}) /. 255.0 in
					let bif = (foi bimg.{i*image_res+j}) /. 255.0 in
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
			else ("sub",0,'0') in
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
	;;
		
let reset_bea steak dreaming =
	(* dt sets the 'mode' of batchd, which persists thru update_bea*)
	let dt = `Train in
	(*Printf.printf "\n------- reset_bea:\n"; 
	Gc.print_stat stdout; *)
	let bea = Array.init !batch_size
		(fun i -> 
			if dreaming then new_batche_unsup steak i
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
	(*let device = Torch.Device.Cpu in
	let posencn = tensor_of_bigarray2 posenc device |> normalize_tensor in*)
	Bigarray.Array3.fill bpro 0.0 ;
	Bigarray.Array3.fill bimg 0.0 ;
	Bigarray.Array2.fill bedts 0.0 ;
	Bigarray.Array2.fill bedtd 0.0 ; 
	(* return batchd struct & list of files to close later *)
	{bpro; bimg; bedts; bedtd; posenc; bea; fresh}, 
	[fd_bpro; fd_bimg; fd_bedts; fd_bedtd; fd_posenc]
	
let truncate_list n l = 
	let n = min n (List.length l) in
	let a = Array.of_list l in
	Array.sub a 0 n |> Array.to_list

let sort_database steak =
	(* sort the graph 
		well, sort the array underlying the graph -- since graphs don't quite have an order. 
		redo simdb as well. *)
	let g = Graf.sort_graph steak.gs.g in
	let gs = {steak.gs with g} in
	render_database {steak with gs};
	{steak with gs}
	;;
	
let rec generate_random_logo res =
	let actvar = Array.init 5 (fun _i -> false) in
	let prog = gen_ast false (3,1,actvar) in
	let pro = compress_ast prog in
	if has_pen_nop pro then assert (0 <> 0) ; 
	let ed = Graf.pro_to_edata_opt pro res in
	match ed with
	| Some q -> q
	| _ -> generate_random_logo res

let generate_logo_fromstr str = 
	match parse_logo_string str with
	| Some pro -> Graf.pro_to_edata pro image_res
	| _ -> Graf.pro_to_edata `Nop image_res
	
let permute_array a = 
	let n = Array.length a in
	let b = Array.init n (fun i -> i,(Random.float 10.0)) in
	Array.sort (fun (_,d) (_,e) -> compare d e) b;
	Array.map (fun (i,_) -> a.(i)) b

let renderpng n s fid = 
	let fname = Printf.sprintf "/tmp/ec3/init_database/%05d.png" !n in
	ignore( Logoext.run_logo_string s 64 fname false); 
	Printf.fprintf fid "[%d] %s\n" !n s; 
	incr n
	;;
	
let make_moves fid n = 
	let lenopts = [|"1";"2";"3";"2*2";"1/2";"2/3"|] in
	let angopts = [|"1/5";"2/5";"3/5";"4/5";"1/4";"3/4";"1/3";"2/3";"1/2";
		"1";"2";"3";"4";"5";"6";"ua/5";"ua/4";"ua/3";"ua/2";"ua"|] in
	let preopts = [| ""; "0 - "|] in
	let r = ref [] in
	for i = 0 to (Array.length lenopts)-1 do (
		for j = 0 to (Array.length angopts)-1 do (
			for k = 0 to (Array.length preopts)-1 do (
				let s = "move "^ lenopts.(i)^
						" , "^preopts.(k)^angopts.(j) in
				r := s :: !r; 
				renderpng n s fid
			) done; 
		) done;
	) done;
	List.rev !r
	;;

let init_database steak count = 
	(* generate 'count' initial program & image pairs *)
	Logs.info(fun m -> m  "init_database %d" count); 
	let root = "/tmp/ec3/init_database" in
	let fid = open_out (Printf.sprintf "%s/enumerate.txt" root) in
	let u = ref 0 in
	let iters = ref 0 in

	let tryadd stk data img override =
		let good,dist,minde = if override then true,10000.0,0
			else simdb_dist stk img in
		let mindex = stk.gs.img_inv.(minde) in
		if mindex < 0 then (
			simdb_to_png stk minde "tryadd_error.png"; 
			Logs.debug (fun m->m "tryadd imgi:%d i:%d" minde mindex); 
		); 
		let s = Logo.output_program_pstr data.pro in
		if good then (
		(* bug: white image sets distance to 1.0 to [0] *)
		if dist > 4096.0 then (
			(*Logs.debug(fun m -> m
				"%d: adding [%d] = %s" !iters !u s);*)
			let added,y = db_add_uniq stk data img in
			if not added then Logs.err (fun m->m "did not add %s to db" s);
			if added then (
				Logo.segs_to_png data.segs 64
					(Printf.sprintf "%s/db%05d_.png" root y);
				simdb_to_png stk stk.gs.g.(y).imgi
					(Printf.sprintf "%s/db%05d_f.png" root y);
			); 
			Printf.fprintf fid "[%d] %s (dist:%f to:%d)\n" y s dist mindex;
			incr u;
		) ;
		if dist < 256.0 then (
			(* see if there's a replacement *)
			let data2 = db_get stk mindex in
			let c1 = data.pcost in (* progenc_cost  *)
			let c2 = data2.ed.pcost in
			if c1 < c2 then (
				let r = db_replace_equiv stk mindex data img in
				if r >= 0 then (
					(*Logs.debug (fun m->m "   distance %f" dist);*)
					(*Logs.debug(fun m -> m
					"%d: replacing [%d] = %s ( was %s) dist:%f new [%d]"
					!iters mindex
					(Logo.output_program_pstr data.pro)
					(Logo.output_program_pstr data2.ed.pro) dist r);*)
					incr u;
				)
			);
			if c1 > c2 then (
				if (SI.cardinal data2.equivalents) < 16 then (
					let r = db_add_equiv stk mindex data in
					(*Logs.debug (fun m->m "   distance %f" dist); *)
					if r >= 0 then (
						(*Logs.debug(fun m -> m
						"iter %d: added equiv [loc %d] = %s [new loc %d] ( simpler= %s) %s %s same:%b dist:%f"
						!iters mindex
						(Logo.output_program_pstr data.pro) r
						(Logo.output_program_pstr data2.ed.pro)
						data.progenc data2.ed.progenc
						(data.progenc = data2.ed.progenc) dist);*)
						incr u;
					)
				)
			);
		) ; 
		if dist <= 4096.0 && dist >= 256.0 then (
			(*Logs.debug (fun m->m "%d: %s reject, dist: %f to: %d" 
				!iters s dist mindex)*)
		)
		) else (
			(*Logs.debug (fun m->m "%d: dist %f not good: %s %f" !iters dist s data.scost)*)
		) ; 
		(*if !iters mod 40 = 39 then
			(* needed to clean up torch allocations 
				-- not anymore w cuda! *)
			Caml.Gc.major ();*)
		incr iters
	in

	let tryadd_fromstr stk str override =
		(*Logs.debug (fun m->m "tryadd %s" str); *)
		let data,img = generate_logo_fromstr str in
		tryadd stk data img override
	in

	tryadd_fromstr steak "" true;
	tryadd_fromstr steak "move 1, 1" false;
	tryadd_fromstr steak "move 1, 0 - 1" false;
	tryadd_fromstr steak "move 2, 1" false;
	tryadd_fromstr steak "move 2, 0 - 1" false;
	tryadd_fromstr steak "move 3, 1" false;
	tryadd_fromstr steak "move 3, 0 - 1" false;
	tryadd_fromstr steak "move 4, 1" false;
	tryadd_fromstr steak "move 4, 0 - 1" false;
	tryadd_fromstr steak "move 1, 2" false;
	tryadd_fromstr steak "move 1, 0 - 2" false;
	(*let y = db_get steak 1 in
	Logs.debug (fun m->m "y %s" y.ed.progenc);*) 
	let n = ref 3 in
	let r = make_moves fid n in
	let rp = List.map (fun s -> "( "^s^")") r in
	(* now outer-prod them in a sequence + pen *)
	let penopts = [|"1";"2";"3";"4"|] 
		|> Array.map (fun s -> "pen "^s^"; ") in
	let r2 = ref [] in
	for h = 0 to (Array.length penopts)-1 do (
		let pen = penopts.(h) in
		let t = List.map (fun s -> "("^pen^s^")") r in
		List.iter (fun s -> renderpng n s fid) t; 
		r2 := List.rev_append t !r2; 
	) done; 
	
	List.iter (fun s -> 
		tryadd_fromstr steak s false)
		(r @ rp @ !r2) ; 
	
	(* for longer sequences, we need to sub-sample *)
	let sub_sample () = 
		let penopts = [|"0";"1";"2";"3";"4"|] 
			|> Array.map (fun s -> "pen "^s) in
		let r3 = ref [] in
		let ra = Array.of_list r in
		let nra = List.length r in
		n := 0; 
		while !n < (iof ((foi count) *. 0.2)) do (
			let i = Random.int nra in
			let j = Random.int nra in
			let k = Random.int (Array.length penopts) in
			let y = [| ra.(i); ra.(j); penopts.(k) |] |> permute_array in
			let s,_ = Array.fold_left (fun (a,m) b -> 
				(if m<2 then a^b^"; " else a^b),m+1) ("",0) y in
			let s2 = "("^s^")" in
			r3 := s2 :: !r3; 
			renderpng n s2 fid
		) done; 
		while !n < (iof ((foi count) *. 0.4)) do (
			let i = Random.int nra in
			let j = Random.int nra in
			let k = Random.int (Array.length penopts) in
			let y = [| ra.(i); penopts.(k); ra.(j) |] in
			let s,_ = Array.fold_left (fun (a,m) b -> 
				(if m<2 then a^b^"; " else a^b),m+1) ("",0) y in
			let s2 = "(pen 0 ;"^s^")" in
			r3 := s2 :: !r3; 
			renderpng n s2 fid
		) done;
		while !n < (iof ((foi count) *. 0.7)) do (
			let i = Random.int nra in
			let j = Random.int nra in
			let k = Random.int nra in
			let l = Random.int (Array.length penopts) in
			let y = [| ra.(i); penopts.(l); ra.(j); ra.(k); |] in
			let s,_ = Array.fold_left (fun (a,m) b -> 
				(if m<3 then a^b^"; " else a^b),m+1) ("",0) y in
			let s2 = "(pen 0 ;"^s^")" in
			r3 := s2 :: !r3; 
			renderpng n s2 fid
		) done;
		while !n < (iof ((foi count) *. 1.0)) do (
			let i = Random.int nra in
			let j = Random.int nra in
			let k = Random.int nra in
			let l = Random.int (Array.length penopts) in
			let m = Random.int (Array.length penopts) in
			let y = [| ra.(i); penopts.(l); ra.(j); penopts.(m); ra.(k); |] in
			let s,_ = Array.fold_left (fun (a,m) b -> 
				(if m<4 then a^b^"; " else a^b),m+1) ("",0) y in
			let s2 = "(pen 0 ;"^s^")" in
			r3 := s2 :: !r3; 
			renderpng n s2 fid
		) done;
		!r3
	in
	
	let rec runbatch stk n = 
		if n > 6 then stk
		else (
			let ra = sub_sample () |> Array.of_list |> permute_array in
			let ran = Array.length ra in
			let body i = tryadd_fromstr stk ra.(i) false in
			
			(*if !gparallel then
				Dtask.parallel_for stk.pool 
					~start:0 ~finish:(ran-1) ~body
			else*)
				(* moved parallelism into graf.ml *)
				for i=0 to (ran-1) do
					body i done;
			(* remove unreachable nodes *)
			Graf.remove_unreachable stk.mutex stk.gs;
			Graf.save "db_remove_unused.S" stk.gs.g; 
			render_database stk; 
			let sum = Simdb.checksum steak.sdb in
			Logs.debug (fun m->m "remove_unused; simdb sum %f" sum);
			runbatch stk (n+1)
		)
	in
	let steak = runbatch steak 0 in
	(*while !i < count do (
		(*Logs.debug (fun m->m "init_database %d" !i);*) 
		let data,img = generate_random_logo image_res in
		tryadd data img false
	) done; *)
	close_out fid; 
	(*let steak = sort_database steak in*)
	Logs.info(fun m -> m  "%d done; %d sampled; %d replacements; %d equivalents" !u !iters steak.gs.num_uniq steak.gs.num_equiv); 
	steak

let verify_database steak =
	(* render everything again for sanity *)
	let root = "/tmp/ec3/verify_database" in
	let n = Array.length steak.gs.g in
	let imf = Tensor.zeros [n; image_res; image_res] in
	Printf.printf "rendering all programs to /tmp/ec3/verify_database\n"; 
	Array.iteri (fun i d -> 
		let _,img = Graf.pro_to_edata d.ed.pro image_res in
		(* do not use Simdb here -- just torch. *)
		let imgf_cpu = tensor_of_bigarray1 img in
		Tensor.copy_ (Tensor.narrow imf ~dim:0 ~start:i ~length:1) ~src:imgf_cpu;
		Logo.segs_to_png d.ed.segs 64
			(Printf.sprintf "%s/db%05d_.png" root i); 
		) steak.gs.g; 
	Printf.printf "done rendering.\n";
	
	let errors = Atomic.make 0 in
	
	let verify_equivalent i d j e =
		let u = Tensor.narrow imf ~dim:0 ~start:i ~length:1 in
		let v = Tensor.narrow imf ~dim:0 ~start:j ~length:1 in
		let w = Tensor.( sum ( square (u - v) ) ) |> Tensor.float_value in
		if w > 256.0 then (
			Printf.printf "not equiv (%f) %d %d:\n\t%s \n\t%s \n"
				w i j 
				(Logo.output_program_pstr d.ed.pro) 
				(Logo.output_program_pstr e.ed.pro) ; 
			Atomic.incr errors
		)
	in
	let gi = Array.to_list steak.gs.g 
		|> List.mapi (fun i d -> i,d) 
		|> List.filter (fun (_,d) -> d.progt = `Uniq || d.progt = `Equiv) in
	
	let body i = 
		let d = steak.gs.g.(i) in
		match d.progt with
		| `Uniq -> (
			(* verify that all equivalents are actually that. *)
			SI.iter (fun (j,typ,cnt,_) -> 
				let e = steak.gs.g.(j) in
				verify_equivalent i d j e; 
				let cnt',edits = get_edits d.ed.progenc e.ed.progenc in
				let _,typ' = edit_criteria edits in
				if typ <> typ' then 
					Printf.printf 
						"%d %d equiv edit type wrong is %s should be %s\n" i j typ typ';
				if cnt <> cnt' then 
					Printf.printf 
						"%d %d equiv edit count wrong is %d should be %d\n" i j cnt cnt'; 
				) d.equivalents ; 
			(* re-create outgoing to verify *)
			let ii = List.map (fun (j,e) -> 
				if j <> i then (
					let cnt,edits = Graf.get_edits d.ed.progenc e.ed.progenc in
					let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
					let b,typ = Graf.edit_criteria edits in
					(j,b,typ,cnt) 
				) else (0,false,"",0)
				) gi
				|> List.filter (fun (_,b,_,_) -> b) 
				|> List.map (fun (j,_,typ,cnt) -> j,typ,cnt,0) in
			let og = SI.of_list ii in
			if og <> d.outgoing then (
				let ogl = SI.cardinal og in
				let doutl = SI.cardinal d.outgoing in
				Printf.printf "%d %s outgoing wrong! len %d should be %d\n"
					i (Logo.output_program_pstr d.ed.pro) doutl ogl ; 
				let print_diff a b sorder =
					let df = SI.diff a b in
					(* diff s1 s2 contains the elements of s1 that are not in s2. *)
					Printf.printf "diff %s:\n" sorder; 
					SI.iter (fun (k,_,_,_) -> 
						let e = steak.gs.g.(k) in
						let dist,_edits = Graf.get_edits d.ed.progenc e.ed.progenc in
						Printf.printf " fwd %d -> %d, dist %d \n |%s -> \n |%s\n" 
							i k dist
							(progenc2str d.ed.progenc) 
							(progenc2str e.ed.progenc) ; 
						(*Levenshtein.print_edits edits; -- working*) 
						(* flip it and do it again *)
						(*let dist,_edits = Graf.get_edits e.ed.progenc d.ed.progenc in
						Printf.printf "rev\t%d, %s -> %s dist %d\n" k e.ed.progenc d.ed.progenc dist;*) 
						(*Levenshtein.print_edits edits; -- working *)
						) df ;  
				in
				print_diff og d.outgoing "missing";
				print_diff d.outgoing og "extra"; 
				Atomic.incr errors )
			)
		| `Equiv -> (
			let j = d.equivroot in
			let e = steak.gs.g.(j) in
			verify_equivalent i d j e)
		| _ -> () 
	in 
	
	Dtask.parallel_for steak.pool 
		~start:0 ~finish:((Array.length steak.gs.g)-1) ~body; 
	
	if (Atomic.get errors) = 0 then 
		Printf.printf "database verified.\n"

let save_database steak fname = 
	(*let g = Graf.sort_graph steak.gs.g in*)
	Graf.save fname steak.gs.g; 
	Logs.info (fun m -> m "saved %d (%d uniq, %d equiv) to %s" 
		(Array.length steak.gs.g) 
		steak.gs.num_uniq steak.gs.num_equiv fname )
		
let load_database steak fname = 
	(* returns a new steak, freshly sizzled *)
	let gs = Graf.load steak.gs fname in
	Logs.debug (fun m->m "Loaded %s" (Graf.get_stats gs)); 
	render_database {steak with gs} ; 
	{steak with gs} (* and butter *)
	;;
	
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
			let typright, chrright, posright, totsup = 
					ref 0, ref 0, ref 0, ref 0 in
			Array.iteri (fun i (typ,pos,chr) -> 
				if List.length (bd.bea.(i).edits) > 0 then (
					let styp,spos,schr = List.hd (bd.bea.(i).edits) in (* supervised *)
					if typ = styp then incr typright; 
					if pos = spos then incr posright; 
					if chr = schr then incr chrright; 
					incr totsup; 
				) ) edit_arr ; 
			if !totsup > 0 then (
				let pctg v = (foi !v) /. (foi !totsup) in
				Logs.debug (fun m -> m (* TODO: debug mode *)
					"decode_edit: correct typ %0.3f chr %0.3f pos %0.3f " 
					(pctg typright) (pctg chrright) (pctg posright) )
			);
		in
		(* always check! *)
		let edit_dream = decode_edit ~sample:false bd.bedtd in
		decode_edit_accuracy edit_dream; 
		
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
		Mutex.lock steak.mutex;
		Caml.Gc.full_major (); (* clean up torch variables (slow..) *)
		Mutex.unlock steak.mutex
	) done; 
	) (* )open Unix *)
	
let measure_torch_copy_speed device = 
	let start = Unix.gettimeofday () in
	let nimg = 6*2048*2 in
	let tn = Tensor.( zeros [nimg; image_res; image_res] ) in
	for i = 0 to nimg/2-1 do (
		let k = Tensor.ones [image_res; image_res] in
		Tensor.copy_ (Tensor.narrow tn ~dim:0 ~start:i ~length:1) ~src:k; 
	) done; 
	let y = Tensor.to_device tn ~device in
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

