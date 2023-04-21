(* interacting with logo programs / write to png *)
open Lexer
open Lexing

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
			(print_position lexbuf) msg);  (* these errors overwhelm while dreaming *)*)
		None
	| Parser.Error ->
		(*Logs.debug (fun m -> m "%s: syntax error" 
			(print_position lexbuf));*)
		None in
	prog

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
