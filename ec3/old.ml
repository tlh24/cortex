
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

	
let make_trains steak = 
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
	Vector.to_array dreams

	
let save_database steak fname = 
	(* saves in the current state -- not sorted. *)
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
	(* save the equivalents too? *)

let load_database_line steak s pid equivalent =
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
				"could not parse program %d %s" pid s ))
	
let load_database steak = 
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
	)

let save_dreams steak =
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
	Logs.debug (fun m -> m "Saved %d mnist decodes to /tmp/png" !cnt)

let hidden_nodes = 128
let epochs = 10000
let learning_rate = 1e-3
