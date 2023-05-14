(* interacting with logo programs / write to png *)
open Lexer
open Lexing

let print_position lexbuf = 
	let pos = lexbuf.lex_curr_p in
	let bf = Buffer.create 64 in
	Printf.bprintf bf "%s:%d:%d" pos.pos_fname
		pos.pos_lnum (pos.pos_cnum - pos.pos_bol + 1); 
	(Buffer.contents bf)

let parse_with_error lexbuf dbg =
	let prog = try Some (Parser.parse_prog Lexer.read lexbuf) with
	| SyntaxError msg ->
		if dbg then Logs.debug (fun m -> m "%s: %s" 
			(print_position lexbuf) msg);  (* these errors overwhelm while dreaming *)
		None
	| Parser.Error ->
		if dbg then Logs.debug (fun m -> m "%s: syntax error" 
			(print_position lexbuf));
		None in
	prog

let run_prog prog res fname dbg =
	(*Logs.debug(fun m -> m "enter run_prog");*)
	match prog with
	| Some(prog) -> (
		let (_,_,segs) = Logo.eval (Logo.start_state ()) prog in
		Logo.segs_to_png segs res fname; 
		if dbg then (
			Logs.debug(fun m -> m "%s" (Logo.output_program_pstr prog)); 
			Logs.debug(fun m -> m "%s" (Logo.output_segments_str segs)) 
		); 
		true)
	| None -> ( false )

let parse_logo_string s dbg = 
	let lexbuf = Lexing.from_string s in
	lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = "from string" };
	parse_with_error lexbuf dbg
	
let run_logo_string s res fname dbg = 
	let pro = parse_logo_string s dbg in
	run_prog pro res fname dbg 
	
let parse_logo_file fname dbg = 
	let ic = open_in fname in
	let s = really_input_string ic (in_channel_length ic) in
	close_in ic;
	parse_logo_string s dbg

let run_logo_file fname res dbg =
	let prog = parse_logo_file fname dbg in
	ignore(run_prog prog res (fname^".png") dbg )
