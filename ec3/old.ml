
let read_protobuf ic pfunc = 
	(* polymorphic! so cool *)
	let buf = Bytes.create 256 in
	let len = input ic buf 0 256 in
	if len > 0 then (
		Logs.info (fun m -> m "read_protobuf: got %d bytes from stdin" len )); 
	if len > 0 then (
		let lp = try 
			Some ( pfunc
				(Pbrt.Decoder.of_bytes (Bytes.sub buf 0 len)))
		with _ -> ( 
			Logs.err(fun m -> m "Could not decode protobuf");
			None
		) in
		lp
	) else (
		Unix.sleepf 0.01; 
		None
	)
	
let write_protobuf sout pfunc r = 
	let encoder = Pbrt.Encoder.create () in 
	pfunc r encoder; 
	output_bytes sout (Pbrt.Encoder.to_bytes encoder);
	flush sout

	
let tensor_of_bigarray1img img device = 
	(* i think this was slower ... *)
	let stride = (Bigarray.Array1.dim img) / image_res in
	let len = Bigarray.Array1.dim img in
	assert (len >= image_res * image_res); 
	let o = Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout image_res image_res in
	for i = 0 to image_res-1 do (
		for j = 0 to image_res-1 do (
			let c = Bigarray.Array1.get img ((i*stride)+j) |> foi in
			o.{i,j} <- c /. 255.0; 
		) done; 
	) done;
	o	|> Bigarray.genarray_of_array2 (* generic array; doesn't copy data? *)
		|> Tensor.of_bigarray  
		|> Torch.Tensor.to_device ~device

		
let dbf_dist2 steak img = 
	(*Mutex.lock steak.db_mutex;*) 
	let d = Tensor.( (steak.dbf - img) ) in (* broadcast *)
	(* this is not good: allocates a ton of memory!! *)
	(* what we need to do is substract, square, sum. at the same time *)
	(*Mutex.unlock steak.db_mutex;*) (* we have a copy *)
	let d = Tensor.einsum ~equation:"ijk,ijk -> i" [d;d] ~path:None in
	let mindex = Tensor.argmin d ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d mindex |> Tensor.float_value in
	dist,mindex

	
let rec new_batche steak bn dosub = 
	(* only supervised mode! *)
	let ndb = db_len steak in
	let nb = (Random.int (ndb-1)) + 1 in
	(*let na = if doedit then (Random.int (ndb-1)) + 1 else 0 in*)
	let na = (Random.int ndb) in
	(* small portion of the time na is the empty program *)
	let a = db_get steak na in
	let b = db_get steak nb in
	let a_ns = List.length a.segs in
	let b_ns = List.length b.segs in
	let a_np = String.length a.progenc in
	let b_np = String.length b.progenc in
	let lim = (p_ctx/2)-4 in 
	(* check this .. really should be -2 as in bigfill_batchd *)
	if a_ns <= 8 && b_ns <= 8 && a_np < lim && b_np < lim then (
		let dist,edits = pdata_to_edits a b in
		(*Logs.debug(fun m -> m  
			"trying [%d] [%d] for batch; dist %d" na nb dist);*)
		if edit_criteria edits dosub (not dosub) && dist > 0 then (
			(*Logs.debug(fun m -> m 
				"|%d adding [%d] %s [%d] %s to batch; dist:%d"
				bn na a.progenc nb b.progenc dist);*)
			let edited = Array.make p_ctx 0.0 in
			{a_pid=na; b_pid=nb; 
				a_progenc = a.progenc; 
				b_progenc = b.progenc; 
				c_progenc = a.progenc; 
				edits; edited; count=0; indx=na} (* FIXME indx *)
		) else new_batche steak bn dosub
	) else new_batche steak	bn dosub

let rec new_batche doedit db = 
	(* only supervised mode! *)
	let ndb = (Vector.length db) in
	let nb = (Random.int (ndb-1)) + 1 in
	let na = if doedit then (Random.int (ndb-1)) + 1 else 0 in
	let distthresh = if doedit then 5 else 15 in (* FIXME: 4,12 -> 5,15 *)
	(*let na = if (Random.int 10) = 0 then 0 else (Random.int nb) in*)
	(* small portion of the time na is the empty program *)
	let a = Vector.get db na in
	let b = Vector.get db nb in
	let a_ns = List.length a.segs in
	let b_ns = List.length b.segs in
	let a_np = String.length a.progenc in
	let b_np = String.length b.progenc in
	let lim = (p_ctx/2)-4 in 
	(* check this .. really should be -2 as in bigfill_batchd *)
	if a_ns <= 8 && b_ns <= 8 && a_np < lim && b_np < lim then (
		let dist,edits = Levenshtein.distance a.progenc b.progenc false in
		(*Logs.debug(fun m -> m  
			"trying [%d] [%d] for batch; dist %d" na nb dist);*)
		if dist > 0 && dist < distthresh then (
			let edits = pdata_to_edits a b in
			(*Logs.debug(fun m -> m 
				"adding [%d] %s [%d] %s to batch (unsorted pids: %d %d)"
				na a.progenc nb b.progenc a.pid b.pid);*)
			let edited = Array.make p_ctx 0.0 in
			{a_pid=na; b_pid=nb; 
				a_progenc = a.progenc; 
				b_progenc = b.progenc; 
				c_progenc = a.progenc; 
				edits; edited; count=0; indx=na} (* FIXME indx *)
		) else new_batche doedit db
	) else new_batche doedit db

	
(* -- various toplevels -- *)



let () = 
	Unix.clear_nonblock stdin; 
	(*run_logo_file lg sout serr "semicircle.logo" ;*)
	Random.self_init (); 
	let lg = open_out "logo_Logs.txt" in
	let sout = out_channel_of_descr stdout in
	let serr = out_channel_of_descr stderr in 
	run_logo_file lg sout serr "badvar.logo" ;
	(*let data = generate_random_logo lg 0 48 in
	print_prog data.pro; *)
	close_out lg; 

 
let () = 
	(* this for checking program encoding & decoding *)
	Unix.clear_nonblock stdin; 
	let lg = open_out "logo_Logs.txt" in
	let serr = out_channel_of_descr stderr in
	let sout = out_channel_of_descr stdout in 
	let g = parse_logo_file lg serr "semicircle.logo" in
	(match g with 
	| Some gg -> (
		Logo.output_program_h sout gg; 
		flush sout; 
		let progenc = Logo.encode_program gg in
		Printf.printf "encoded program:\n";
		Printf.printf "%s\n" (intlist_to_string progenc); 
		let progstr = Logo.decode_program progenc in
		Printf.printf "decoded program:\n"; 
		Printf.printf "%s\n" progstr ; 
		let g2 = parse_logo_string lg serr progstr in
		match g2 with
		| Some g3 -> 
			Printf.printf "%s\n" (Logo.output_program_pstr g3); 
		| _ -> () )
	| _ -> () ); 
	close_out lg; 
	close_out serr


let () = 
	(* this tests change_ast *)
	Unix.clear_nonblock stdin; (* this might not be needed *)
	Random.self_init (); 
	let lg = open_out "logo_Logs.txt" in
	(*let ic = in_channel_of_descr stdin in*)
	(*let sout = out_channel_of_descr stdout in*)
	let serr = out_channel_of_descr stderr in 
	(*let channels = (sout, serr, ic, lg) in*)
	Printf.fprintf lg "hello\n";
	let origp = parse_logo_string lg serr "move ul, ul" in
	match origp with
	| Some prog -> (
		let newp = change_ast prog lg in
		let newp = compress_ast newp in
		print_prog prog ; 
		printf "\n"; 
		print_prog newp ; 
		printf "----\n"; 
		printf "%s\n" (encode_program_str prog); 
		printf "%s\n" (encode_program_str newp); )
	| _ -> printf "failed to parse\n";
	Printf.fprintf lg "\n"; 
	close_out serr;
	close_out lg; 

	
let test_torch () =
	(* This should reach ~97% accuracy. *)
	Stdio.printf "cuda available: %b%!" (Cuda.is_available ());
	Stdio.printf "cudnn available: %b%!" (Cuda.cudnn_is_available ());
	let device = Torch.Device.cuda_if_available () in
	let mnist = Mnist_helper.read_files () in
	let { Dataset_helper.train_images; train_labels; _ } = mnist in
	let vs = Var_store.create ~name:"nn" ~device () in
	let linear1 =
		Layer.linear vs hidden_nodes ~activation:Relu ~input_dim:Mnist_helper.image_dim
	in
	let linear2 = Layer.linear vs Mnist_helper.label_count ~input_dim:hidden_nodes in
	let adam = Optimizer.adam vs ~learning_rate ~weight_decay:5e-5 in
	let model xs =
		Layer.forward linear1 xs
		|> Layer.forward linear2 in
	let img = train_images |> Torch.Tensor.to_device ~device in
	let lab = train_labels |> Torch.Tensor.to_device ~device in
	for index = 1 to epochs-1 do
		(* Compute the cross-entropy loss. *)
		let loss =
			Tensor.cross_entropy_for_logits (model img) ~targets:lab
		in
		Optimizer.backward_step adam ~loss;
		if index mod 50 = 0
		then (
			(* Compute the validation error. *)
			let test_accuracy =
			Dataset_helper.batch_accuracy ~device mnist `test ~batch_size:1000 ~predict:model
			in
			Stdio.printf
			"%d %f %.2f%%\n%!"
			index
			(Tensor.float_value loss)
			(100. *. test_accuracy));
		Caml.Gc.full_major ()
	done


let image_dist dbf img = 
	let d = Tensor.( (dbf - img) ) in
	(* per-element square and sum *)
	let d = Tensor.einsum ~equation:"ijk,ijk -> i" [d;d] ~path:None in
	let mindex = Tensor.argmin d ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d mindex |> Tensor.float_value in
	dist,mindex

let () = 
	Unix.clear_nonblock stdin; 
	Printf.printf "cuda available: %b\n%!" (Cuda.is_available ());
	let device = Torch.Device.cuda_if_available () in
	(* dbf is a tensor of images to be compared (MSE) against *)
	let dbf = Tensor.( 
		( ones [image_count; image_res; image_res] ) * (f (-1.0))) 
		|> Tensor.to_device ~device in
	let start = Unix.gettimeofday () in
	for i = 0 to 100_000 do (
		(* generate a random image *)
		let img = Tensor.(randn [image_res; image_res] ) 
			|> Tensor.to_device ~device in
		ignore( image_dist dbf img ); 
		if i mod 30 = 29 then 
			Caml.Gc.full_major()
		(* in the actual program, we do something with dist,mindex *)
	) done; 
	let stop = Unix.gettimeofday () in
	Printf.printf "100k image_dist calc time: %fs\n%!" 
		(stop -. start);


let () = 
	Unix.clear_nonblock stdin; 
	Random.self_init (); 

	let lg = open_out "logo_Logs.txt" in
	let ic = in_channel_of_descr stdin in
	let sout = out_channel_of_descr stdout in
	let serr = out_channel_of_descr stderr in 
	let channels = (sout, serr, ic, lg) in
	Printf.fprintf lg "hello\n";
	
	
	let db = Vector.create ~dummy:nulpdata in
	let dba = Array.make 1 nulpdata in
	(*
	Printf.printf "cuda available: %b\n%!" (Cuda.is_available ());
	Printf.printf "cudnn available: %b\n%!" (Cuda.cudnn_is_available ());
	let device = Torch.Device.cuda_if_available () in
	(*let device = Torch.Device.Cpu in*) (* slower *)
	let dbf = Tensor.( 
		( ones [image_count; image_res; image_res] ) * (f (-1.0))) 
		|> Tensor.to_device ~device in
	init_database channels device db dbf;*) 
	loop_random channels 0 db dba;
	(*let _b = make_batch lg dba 1 in*)

	close_out sout; 
	close_out serr;
	close_out lg; 


let update_bea_sup steak bd =
	let sta = Unix.gettimeofday () in
	(* need to run through twice, first to apply the edits, second to replace the finished elements. *)
	let innerloop i =
		let be = bd.bea.(i) in
		let be2 = if List.length be.edits > 0 then (
			bd.fresh.(i) <- false;
			apply_edits be
			) else be in
		let be3 = if List.length be.edits = 0 then (
			bd.fresh.(i) <- true; (* update image flag *)
			new_batche_sup steak i
			) else be2 in
		bd.bea.(i) <- be3 in (* /innerloop *)
	if !gparallel then
		Dtask.parallel_for steak.pool ~start:0 ~finish:(!batch_size-1)
			~body:innerloop
	else
		for i=0 to (!batch_size-1) do
			innerloop i done;
	let fin = Unix.gettimeofday () in
	Logs.debug (fun m -> m "update_bea_sup time %f" (fin-.sta));
	(* array update was in-place, so just return bd. *)
	bd


let update_bea steak bd =
	let sta = Unix.gettimeofday () in
	Logs.debug (fun m -> m "entering update_bea");
	(* this one only needs one run-through. *)
	let edit_arr = decode_edit bd bd.bedtd in
	let innerloop i =
		let be = bd.bea.(i) in
		match be.dt with
		| `Train_sub j -> (
		let cnt = be.count in
		let typ,loc,chr = edit_arr.(i) in
		(*Logs.debug (fun m -> m "update_bea_dream.innerloop %d %s %d %c"
						i typ loc chr );*)
		(* edited is initialized in update_be_sup/new_batche_sup; same here *)
		let edited = if Array.length be.edited <> p_ctx
			then Array.make p_ctx 0.0 else be.edited in
		let be2 = {be with edits=[(typ,loc,chr)];count=cnt+1;edited} in
		let be3 = apply_edits be2 in
		let be4 = if typ = "fin" || be2.count >= p_ctx/2 then (
			(* log it! *)
			(match steak.dreams with
			| Some dreams -> (
				let a = be.a_pid in
				let b = be.b_pid in
				let i = be.indx in
				let s = progenc2progstr be.c_progenc in
				if b < !image_count then (
					if be.c_progenc = be.b_progenc then (
						dreams.(i).decode <- (s :: dreams.(i).decode);
						dreams.(i).correct_cnt <- dreams.(i).correct_cnt+1;
						Logs.debug (fun m -> m "dream:%d [%d]->[%d] %s decoded correctly." i a b s)
					) else (
						(* if wrong, save one example decode *)
						if (List.length dreams.(i).decode) = 0 then
						dreams.(i).decode <- (s :: dreams.(i).decode);
					)
				) else (
					let mid = b - image_alloc in
					if mid < 60000 && mid >= 0 then (
						dreams.(i).decode <- (s :: dreams.(i).decode);
					)
				) )
			| None -> () );
			try_add_program steak be3.c_progenc be3;
			bd.fresh.(i) <- true;
			new_batche_dream steak i
		) else (
			bd.fresh.(i) <- false;
			be3
		) in
		bd.bea.(i) <- be4;
	in (* /innerloop *)
	if !gparallel then (* this might work again? *)
		Dtask.parallel_for steak.pool ~start:0 ~finish:(!batch_size-1)
			~body:innerloop
	else
		for i = 0 to (!batch_size-1) do (* non-parallel version *)
			innerloop i done;
	let fin = Unix.gettimeofday () in
	Logs.debug (fun m -> m "update_bea_dream time %f" (fin-.sta));
	Mutex.lock steak.db_mutex;
(* 	Caml.Gc.major (); (* clean up torch variables *) *)
	Mutex.unlock steak.db_mutex;
	(* let fin2 = Unix.gettimeofday () in
	Logs.debug (fun m -> m "update_bea_dream: Caml.Gc.major time %f;"
			(fin2 -. fin)); *)
	(* cannot do this within a parallel for loop!!! *)
	(* in-place update of bea *)
	bd
