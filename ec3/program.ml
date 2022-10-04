(*open Core :-X core doesnt yet support byte output*)
open Lexer
open Lexing
open Printf
open Unix

let print_position outx lexbuf =
  let pos = lexbuf.lex_curr_p in
  fprintf outx "%s:%d:%d" pos.pos_fname
    pos.pos_lnum (pos.pos_cnum - pos.pos_bol + 1)

let parse_with_error lexbuf =
  try Parser.parse_prog Lexer.read lexbuf with
  | SyntaxError msg ->
    Printf.eprintf "%a: %s\n" print_position lexbuf msg;
    exit (-1)
  | Parser.Error ->
    Printf.eprintf "%a: syntax error\n" print_position lexbuf;
    exit (-1)

let bigarray_to_bytes arr = 
	(* convert a bigarray to a list of bytes *)
	(* this is not efficient.. *)
	let len = Bigarray.Array1.dim arr in
	Bytes.init len 
		(fun i -> Bigarray.Array1.get arr i |> Char.chr)
    
let parse_and_render lexbuf id res =
	let prog = parse_with_error lexbuf in
	let (_,_,segs) = Logo.eval Logo.start_state prog in
	(*Logo.output_program prog; 
	printf "\n"; 
	Logo.output_segments segs; 
	printf "\n"; *)
	Logo.segs_to_png segs res "test.png";
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
	let oc = out_channel_of_descr stderr in
	output_bytes oc arr ;
	flush oc ; 
	close_out oc; 
	
	(* Create a Protobuf encoder and encode value *)
	let encoder = Pbrt.Encoder.create () in 
	Logo_pb.encode_logo_result r encoder; 

	let oc = out_channel_of_descr stdout in
	output_bytes oc (Pbrt.Encoder.to_bytes encoder);
	flush oc; 
	close_out oc


(*let parse filename () =
  let inx = open_in filename in
  let lexbuf = Lexing.from_channel inx in
  lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = filename };
  parse_and_render lexbuf;
  close_in inx*)
  
let () = 
	let ic = in_channel_of_descr stdin in
	let buf = Bytes.create 4096 in
	let _len = input ic buf 0 4096 in 
(* 	Printf.printf "zgot %d bytes\n" len ;   *)
	
	let lp = Logo_pb.decode_logo_program 
		(Pbrt.Decoder.of_bytes buf) in
	
(* 	print_endline (Format.asprintf "logo: %a" Logo_pp.pp_logo_program lp);  *)
	
	let lexbuf = Lexing.from_string lp.prog in
	lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = "from stdin" };
	parse_and_render lexbuf lp.id lp.resolution;

(*
(* this relies on Core *)
let () =
  Command.basic_spec ~summary:"Parse and display Logo"
    Command.Spec.(empty +> anon ("filename" %: string))
    parse
  |> Command_unix.run

*)
