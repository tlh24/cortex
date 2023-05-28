(*open Lexer
open Lexing
open Printf
open Logo
open Ast
open Torch
open Graf*)
(*open Torch*)
open Program

let usage_msg = "verify.exe <db_file>"
let input_files = ref []
let anon_fun filename = (* just incase we need later *)
  input_files := filename :: !input_files

let speclist =
  [("-g", Arg.Set gdebug, "Turn on debug");]

let () = 
	Arg.parse speclist anon_fun usage_msg;
	Random.self_init (); 
	Logs_threaded.enable ();
	let () = Logs.set_reporter (Logs.format_reporter ()) in
	let () = Logs.set_level (Some Logs.Debug) in
	
	Logs.info(fun m -> m "verifying the program graph database."); 

	Logs.info(fun m -> m "cuda available: %b%!" 
				(Torch.Cuda.is_available ()));
	Logs.info(fun m -> m "cudnn available: %b%!"
				(Torch.Cuda.cudnn_is_available ()));
	
	let device = Torch.Device.cuda_if_available () in
	let gs = Graf.create Program.all_alloc Program.image_alloc in
	let sdb = Simdb.init image_alloc in
	let mnist = Torch.Tensor.zeros [2;2] in
	let mnist_cpu = Torch.Tensor.zeros [2;2] in
	(*let vae = Vae.dummy_ext () in*)
	let mutex = Mutex.create () in
	let pool = Dtask.setup_pool ~num_domains:12 () in 
		(* tune this -- 8-12 seems ok *)
	let de = decode_edit_tensors 4 in (* dummy *)
	let training = [| |] in
	let supfid = open_out "/tmp/ec3/replacements_sup.txt" in
	let dreamfid = open_out "/tmp/ec3/replacements_dream.txt" in
	let fid_verify = open_out "/tmp/ec3/verify.txt" in
	
	let supstak = 
		{device; gs; sdb; mnist; mnist_cpu; mutex;
		superv=true; fid=supfid; fid_verify; batchno=0; pool; de; training} in
	
	let fname = if List.length !input_files > 0 then
		List.hd !input_files else "db_sorted.s" in

	let supsteak = load_database supstak fname in
	(* test dijsktra *)
	ignore( Graf.dijkstra supsteak.gs 0 false); 
	Dtask.run supsteak.pool 
		(fun () -> verify_database supsteak) ;
	save_database supsteak "db_rewrite.S"; 
	
	Graf.gexf_out supsteak.gs;
	
	close_out supfid; 
	close_out dreamfid;
	close_out fid_verify; 
	
