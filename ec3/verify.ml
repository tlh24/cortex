(*open Lexer
open Lexing
open Printf
open Logo
open Ast
open Torch
open Graf*)
(*open Torch*)
open Program

let () = 
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
	let gs = Graf.create Program.image_alloc in
	let dbf = Torch.Tensor.zeros [2;2] in
	let dbf_cpu = Torch.Tensor.zeros [2;2] in 
	let dbf_enc = Torch.Tensor.zeros [2;2] in
	let mnist = Torch.Tensor.zeros [2;2] in
	let mnist_enc = Torch.Tensor.zeros [2;2] in
	let vae = Vae.dummy_ext () in
	let db_mutex = Mutex.create () in
	let pool = Dtask.setup_pool ~num_domains:12 () in 
		(* tune this -- 8-12 seems ok *)
	let de = decode_edit_tensors 4 in (* dummy *)
	let supfid = open_out "/tmp/ec3/replacements_sup.txt" in
	let dreamfid = open_out "/tmp/ec3/replacements_dream.txt" in
	let fid_verify = open_out "/tmp/ec3/verify.txt" in
	
	let supstak = 
		{device; gs; dbf; dbf_cpu; dbf_enc; mnist; mnist_enc; vae; db_mutex;
		superv=true; fid=supfid; fid_verify; batchno=0; pool; de} in
	
	let supsteak = load_database supstak "db_sorted.S" in
	verify_database supsteak; 
	save_database supsteak "db_rewrite.S"; 
	
	close_out supfid; 
	close_out dreamfid;
	close_out fid_verify; 
	
