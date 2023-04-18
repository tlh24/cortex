(*open Lexer
open Lexing
open Printf
open Logo
open Ast
open Torch
open Graf*)
(*open Torch*)
open Program
open Graf

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
	let mnist_cpu = Torch.Tensor.zeros [2;2] in
	let mnist_enc = Torch.Tensor.zeros [2;2] in
	(*let vae = Vae.dummy_ext () in*)
	let db_mutex = Mutex.create () in
	let pool = Dtask.setup_pool ~num_domains:12 () in 
		(* tune this -- 8-12 seems ok *)
	let de = decode_edit_tensors 4 in (* dummy *)
	let supfid = open_out "/tmp/ec3/replacements_sup.txt" in
	let dreamfid = open_out "/tmp/ec3/replacements_dream.txt" in
	let fid_verify = open_out "/tmp/ec3/verify.txt" in
	
	let supstak = 
		{device; gs; dbf; dbf_cpu; dbf_enc; mnist; mnist_cpu; mnist_enc; (*vae;*) db_mutex;
		superv=true; fid=supfid; fid_verify; batchno=0; pool; de} in
	
	let supsteak = load_database supstak "db_sorted.S" in
	verify_database supsteak; 
	save_database supsteak "db_rewrite.S"; 
	
	(* also save GEXF file for gephi visualization *)
	let fid = open_out "../prog-gephi-viz/db.gexf" in
	Printf.fprintf fid "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"; 
	Printf.fprintf fid "<gexf xmlns=\"http://gexf.net/1.3\" version=\"1.3\">\n"; 
	Printf.fprintf fid "<graph mode=\"static\" defaultedgetype=\"directed\">\n"; 
	Printf.fprintf fid "<attributes class=\"node\">\n"; 
	Printf.fprintf fid "<attribute id=\"0\" title=\"progt\" type=\"string\"/>\n"; 
	Printf.fprintf fid "</attributes>\n";
	Printf.fprintf fid "<attributes class=\"edge\">\n"; 
	Printf.fprintf fid "<attribute id=\"0\" title=\"typ\" type=\"string\"/>\n"; 
	Printf.fprintf fid "</attributes>\n"; 
	Printf.fprintf fid "<nodes>\n"; 
	Vector.iteri (fun i d -> 
		Printf.fprintf fid "<node id=\"%d\" label=\"%s\" >\n" i
			(Logo.output_program_pstr d.ed.pro); 
		let pts = match d.progt with
			| `Uniq -> "uniq"
			| `Equiv -> "equiv"
			| _ -> "nul" in
		Printf.fprintf fid 
		"<attvalues><attvalue for=\"0\" value=\"%s\"/></attvalues>\n" pts; 
		Printf.fprintf fid "</node>\n"
		) supsteak.gs.g ; 
	Printf.fprintf fid "</nodes>\n";
	
	Printf.fprintf fid "<edges>\n"; 
	Vector.iteri (fun i d -> 
		Graf.SI.iter (fun (j,typ,_cnt) -> 
			Printf.fprintf fid "<edge source=\"%d\" target=\"%d\">" i j; 
			Printf.fprintf fid 
			"<attvalues><attvalue for=\"0\" value=\"%s\"/></attvalues>\n" typ; 
			Printf.fprintf fid "</edge>\n"
		) d.outgoing
		) supsteak.gs.g ; 
	Printf.fprintf fid "</edges>\n";
	Printf.fprintf fid "</graph>\n";
	Printf.fprintf fid "</gexf>\n";
	close_out fid; 
	Printf.printf "saved ../prog-gephi-viz/db.gexf\n"; 
	
	close_out supfid; 
	close_out dreamfid;
	close_out fid_verify; 
	
