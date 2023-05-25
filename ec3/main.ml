(*open Lexer
open Lexing
open Printf
open Logo
open Ast
open Graf*)
open Torch
open Program

let usage_msg = "program.exe -b <batch_size>"
let input_files = ref []
let output_file = ref ""
let anon_fun filename = (* just incase we need later *)
  input_files := filename :: !input_files
let speclist =
  [("-b", Arg.Set_int batch_size, "Training batch size");
   ("-o", Arg.Set_string output_file, "Set output file name"); 
   ("-g", Arg.Set gdebug, "Turn on debug");
   ("-p", Arg.Set gparallel, "Turn on parallel");
   ("-t", Arg.Set gdisptime, "Turn on timing instrumentation");]

let () = 
	Arg.parse speclist anon_fun usage_msg;
	Random.self_init (); 
	Logs_threaded.enable ();
	let () = Logs.set_reporter (Logs.format_reporter ()) in
	let () = Logs.set_level 
		(if !gdebug then Some Logs.Debug else Some Logs.Info) in
	if !gdebug then Logs.debug (fun m -> m "Debug logging enabled.")
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
				
	(* check simdb *)
	Simdb.test (); 
	(*for _i = 0 to 4 do 
		measure_torch_copy_speed device
	done; *)
	test_logo (); 
	
	(*let device = Torch.Device.Cpu in*) (* slower *)
	let device = Torch.Device.cuda_if_available () in
	
	let mnistd = Mnist_helper.read_files ~prefix:"../otorch-test/data" () in
	let mimg = Tensor.reshape mnistd.train_images 
		~shape:[60000; 28; 28] in
	(* need to pad to 30 x 30, one pixel on each side *)
	let mnist_cpu = Tensor.zeros [60000; 30; 30] in
	Tensor.copy_ (
		Tensor.narrow mnist_cpu ~dim:1 ~start:1 ~length:28 |> 
		Tensor.narrow ~dim:2 ~start:1 ~length:28) ~src:mimg ;
		(* Tensor.narrow returns a pointer/view; copy_ is in-place. *)
	let mnist = Tensor.to_device mnist_cpu ~device in

	let gs = Graf.create all_alloc image_alloc in
	let sdb = Simdb.init image_alloc in
	(*let vae = Vae.dummy_ext () in*)
	let db_mutex = Mutex.create () in
	let pool = Dtask.setup_pool ~num_domains:12 () in 
		(* tune this -- 8-12 seems ok *)
	let de = decode_edit_tensors !batch_size in
	let training = [| |] in
	let supfid = open_out "/tmp/ec3/replacements_sup.txt" in
	let dreamfid = open_out "/tmp/ec3/replacements_dream.txt" in
	let fid_verify = open_out "/tmp/ec3/verify.txt" in
	
	let supstak = 
		{device; gs; sdb; mnist; mnist_cpu; db_mutex;
		superv=true; fid=supfid; fid_verify; batchno=0; pool; de; training} in
	
	let supsteak = if Sys.file_exists "db_sorted.S" then ( 
		(*Dtask.run supsteak.pool (fun () -> load_database supsteak )*)
		load_database supstak "db_sorted.S"
	) else ( 
		let nprogs = 4*2048 (*image_alloc*) in
		Logs.app(fun m -> m "Generating %d programs" nprogs);
		let start = Unix.gettimeofday () in
		let stk = Dtask.run supstak.pool 
				(fun () -> init_database supstak nprogs) in
		(*let stk = init_database supstak nprogs in*)
		(* init also sorts. *)
		let stop = Unix.gettimeofday () in
		Logs.app(fun m -> m "Execution time: %fs\n%!" (stop -. start)); 
		Logs.info(fun m -> m ":: first 8 programs");
		for i = 0 to 7 do (
			let p = db_get stk i in
			Logs.info(fun m -> m "%d: %s" i
					(Logo.output_program_pstr p.ed.pro)); 
		) done; 
		save_database stk "db_prog.S"; 
		(*let stk = sort_database stk in*)
		let dist,_prev = Graf.dijkstra stk.gs 0 false in
		Graf.dist_to_good stk.gs dist; 
		save_database stk "db_sorted.S";
		stk
	) in
	
	render_simplest supsteak; 
	
	Logs.info (fun m->m "generating training dataset.."); 
	let supsteak = make_training supsteak in
	Logs.info (fun m->m "training size: %d" (Array.length supsteak.training)); 
	save_database supsteak "db_sorted_.S";
	
	(* try to train the vae? *)
	(*let dbfs = Tensor.narrow supsteak.dbf ~dim:0 ~start:0 ~length:(supsteak.gs.num_uniq) in
	let vae,dbf_enc',mnist_enc = Vae.train_ext dbfs mnist device !batch_size in
	(* need to re-expand for future allocation *)
	let encl,cols = Tensor.shape2_exn dbf_enc' in
	let dbf_enc = Tensor.( (ones [image_alloc;cols]) * (f (-1.0) ) ) in
	Tensor.copy_ (Tensor.narrow dbf_enc ~dim:0 ~start:0 ~length:encl) ~src:dbf_enc' ;
	let supsteak = {supsteak with dbf_enc; mnist_enc; vae} in*)

	
	(* PSA: extra bit of complexity!! 
		if Cuda hangs in one of the domains, 
		e.g. for an out-of-memory error, 
		you won't see it on stdout -- it will just stop. 
		to properly debug, will need to strip down to one thread, 
		no Domains.spawn *)
	(* note there are two forms of parallelism here: 
		Domains (train and dream) 
		and pools (parfor, basically) *)
	
	let threadmode = 0 in
	
	(match threadmode with
	| 0 -> ( (* train only *)
		servthread supsteak () ; 
		Dtask.teardown_pool supsteak.pool)
	| 1 -> ( (* dream only *)
		let dreamsteak = {supsteak with
			superv=false; fid=dreamfid;} in
		servthread dreamsteak (); 
		Dtask.teardown_pool dreamsteak.pool )
	| 2 -> ( (* both *)
		let d = Domain.spawn (fun _ ->
			let pool2 = Dtask.setup_pool ~num_domains:12 () in
			let dreamsteak = {supsteak with
				superv=false; fid=dreamfid; batchno=0; pool=pool2 } in
			servthread dreamsteak (); 
			Dtask.teardown_pool dreamsteak.pool) in
		servthread supsteak () ;
		Domain.join d; 
		Dtask.teardown_pool supsteak.pool )
	| _ -> ()
	); 
	
	close_out supfid; 
	close_out dreamfid;
	close_out fid_verify; 
	
