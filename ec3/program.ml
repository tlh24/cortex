(*open Core :-X core doesnt yet support byte output, necessary for protobufs*)
open Lexer
open Lexing
open Printf
open Unix

let print_position outx lexbuf = (*that's a cool tryck!*)
  let pos = lexbuf.lex_curr_p in
  fprintf outx "%s:%d:%d" pos.pos_fname
    pos.pos_lnum (pos.pos_cnum - pos.pos_bol + 1)

let parse_with_error lg serr lexbuf =
	let prog = try Some (Parser.parse_prog Lexer.read lexbuf) with
	| SyntaxError msg ->
		Printf.fprintf serr "%a: %s\n" print_position lexbuf msg;
		Printf.fprintf lg "%a: %s\n" print_position lexbuf msg;
		None
	| Parser.Error ->
		Printf.fprintf serr "%a: syntax error\n" print_position lexbuf;
		Printf.fprintf lg "%a: syntax error\n" print_position lexbuf;
		None in
	prog
    

let bigarray_to_bytes arr = 
	(* convert a bigarray to a list of bytes *)
	(* this is not efficient.. *)
	let len = Bigarray.Array1.dim arr in
	Bytes.init len 
		(fun i -> Bigarray.Array1.get arr i |> Char.chr)
    
let parse_and_render lg sout serr lexbuf id res =
	Printf.fprintf lg "enter parse_and_render\n"; flush lg; 
	let prog = parse_with_error lg serr lexbuf in
	match prog with
	| Some(prog) -> (
		let (_,_,segs) = Logo.eval Logo.start_state prog in
		Logo.output_program lg prog; 
		Printf.fprintf lg "\n"; 
		Logo.output_segments lg segs; 
		Printf.fprintf lg "\n"; 
		flush lg; 
		(* Logo.segs_to_png segs res "test.png"; *)
		let arr,cost = Logo.segs_to_array_and_cost segs res in
		let r = Logo_types.({
			id = id; 
			stride = res; 
			width = res; 
			height = res; 
			segs = List.map (fun (x0,y0,x1,y1) -> 
				Logo_types.({x0=x0; y0=y0; x1=x1; y1=y1;})) segs; 
			cost = cost; 
		}) in
		(* write the data to stderr, message to stdout *)
		let arr = bigarray_to_bytes arr in
		output_bytes serr arr ;

		(* Create a Protobuf encoder and encode value *)
		let encoder = Pbrt.Encoder.create () in 
		Logo_pb.encode_logo_result r encoder; 
		output_bytes sout (Pbrt.Encoder.to_bytes encoder);

		Printf.fprintf lg "result %s\n" (Format.asprintf "%a" Logo_pp.pp_logo_result r);
		Printf.fprintf lg "parse_and_render done\n";flush lg; 
		flush sout; 
		flush serr; 
		true)
	| None -> ( 
		output_bytes sout (Bytes.create 4) ;  (* keep things moving *)
		flush sout; 
		flush serr; 
		false )

  
let rec loop_input lg sout serr ic buf cnt = 
	Unix.clear_nonblock stdin; (* this probably is not necessary *)
(*	let (readch, _, _) = select [stdin] [] [] 1.0 in
	(* select doesn't seem to work .. *)
	if List.length readch > 0 && cnt < 1000 then (  *)
	Printf.fprintf lg "waiting for command\n" ;
	let len = input ic buf 0 4096 in
	Printf.fprintf lg "got %d bytes from stdin\n" len ; flush lg; 
	if len > 0 then (
		let lp = try 
			Some (Logo_pb.decode_logo_program
				(Pbrt.Decoder.of_bytes (Bytes.sub buf 0 len)))
		with _ -> ( 
			Printf.fprintf lg "Could not decode protobuf\n";
			None
		) in
			
		match lp with 
		| Some lp -> (
			Printf.fprintf lg "%s" (Format.asprintf "logo: %a" Logo_pp.pp_logo_program lp);
			Printf.fprintf lg "\n";
			flush lg;

			let lexbuf = Lexing.from_string lp.prog in
			lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = "from stdin" };
			ignore(parse_and_render lg sout serr lexbuf lp.id lp.resolution ) )
		| None -> ()
	) ;
	if cnt < 10000000 then (
		loop_input lg sout serr ic buf (cnt+1)
	) else true
  
let () = 
	let ic = in_channel_of_descr stdin in
	let lg = open_out "logo_log.txt" in
	let sout = out_channel_of_descr stdout in
	let serr = out_channel_of_descr stderr in
	Printf.fprintf lg "hello\n"; 
	let buf = Bytes.create 4096 in
	ignore(loop_input lg sout serr ic buf 0); 
	close_out lg; 
	close_out sout; 
	close_out serr; 

(*
(* this relies on Core *)
let () =
  Command.basic_spec ~summary:"Parse and display Logo"
    Command.Spec.(empty +> anon ("filename" %: string))
    parse
  |> Command_unix.run

*)
