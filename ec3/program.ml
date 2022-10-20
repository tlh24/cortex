(*open Core :-X core doesnt yet support byte output, necessary for protobufs*)
open Lexer
open Lexing
open Printf
open Unix
open Logo

let g_logEn = ref true (* yes global but ... *)

let print_position outx lexbuf = (*that's a cool tryck!*)
  let pos = lexbuf.lex_curr_p in
  fprintf outx "%s:%d:%d" pos.pos_fname
    pos.pos_lnum (pos.pos_cnum - pos.pos_bol + 1)
    
let print_log lg str = 
	if !g_logEn then (
		Printf.fprintf lg "%s" str; 
		flush lg;
	) 

let parse_with_error lg serr lexbuf =
	let prog = try Some (Parser.parse_prog Lexer.read lexbuf) with
	| SyntaxError msg ->
		Printf.fprintf serr "%a: %s\n" print_position lexbuf msg;
		if !g_logEn then (
			(* %a doesn't work with sprintf .. *)
			Printf.fprintf lg "%a: %s\n" print_position lexbuf msg;
		);  None
	| Parser.Error ->
		Printf.fprintf serr "%a: syntax error\n" print_position lexbuf;
		if !g_logEn then (
			Printf.fprintf lg "%a: syntax error\n" print_position lexbuf;
		);  None in
	prog
    

let bigarray_to_bytes arr = 
	(* convert a bigarray to a list of bytes *)
	(* this is not efficient.. *)
	let len = Bigarray.Array1.dim arr in
	Bytes.init len 
		(fun i -> Bigarray.Array1.get arr i |> Char.chr)
    
let parse_and_render lg sout serr lexbuf id res =
	print_log lg "enter parse_and_render\n";
	let prog = parse_with_error lg serr lexbuf in
	match prog with
	| Some(prog) -> (
		let (_,_,segs) = Logo.eval Logo.start_state prog in
		if !g_logEn then Logo.output_program_h lg prog; 
		print_log lg "\n"; 
		if !g_logEn then Logo.output_segments lg segs; 
		print_log lg "\n"; 
		Logo.segs_to_png segs res "test.png";
		let arr,cost = Logo.segs_to_array_and_cost segs res in
		let stride = (Bigarray.Array1.dim arr) / res in
		let r = Logo_types.({
			id = id; 
			stride = stride; 
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

		if !g_logEn then (
			Printf.fprintf lg "result %s\n" (Format.asprintf "%a" Logo_pp.pp_logo_result r);
			print_log lg "parse_and_render done\n"
		); 
		flush sout; 
		flush serr; 
		true)
	| None -> ( 
		output_bytes sout (Bytes.create 4) ;  (* keep things moving *)
		flush sout; 
		flush serr; 
		false )

			 
(* now, wait a second here ... we can evaluate programs much more efficiently than in python, since we have the ASTs here *)
(* will return a list of list of progs. *)
let select_loopvar loopvar = 
	let len = Array.length loopvar in
	let sum = Array.fold_left (fun a b -> if b then a+1 else a) 0 loopvar in
	let avail = ref [] in
	Array.iteri (fun i a -> if not a then avail := i :: !avail) loopvar ; 
	if sum = len then (
		Random.int len
	) else (
		List.nth !avail (Random.int (List.length !avail))
	)

let select_actvar actvar = 
	let len = Array.length actvar in
	let sum = Array.fold_left (fun a b -> if b then a+1 else a) 0 actvar in
	let avail = ref [] in
	Array.iteri (fun i a -> if a then avail := i :: !avail) actvar ;
	if sum = 0 then (
		Random.int len (* degenerate *)
	) else (
		List.nth !avail (Random.int (List.length !avail))
	)

let rec enumerate_ast return_val state = 
	(* stochastic AST generation ! *)
	let r, br, av, lv = state in 
	(* recurs lim, binop recursion limit, active variables, loop variables *)
	if not return_val then (
		match Random.int 5 with 
		| 0 -> 
			let i = Random.int (Array.length av) in
			av.(i) <- true; 
			`Save(i, enumerate_ast true (r-1,br,av,lv))
		| 1 -> 
			`Move(enumerate_ast true (r-1,br,av,lv), enumerate_ast true (r-1,br,av,lv))
		| 2 -> 
			let len = (Random.int 3) + 1 in
			let ls = List.init len (fun _i -> enumerate_ast false (r-1,br,av,lv)) in
			`Seq(ls)
		| 3 -> 
			let i = select_loopvar lv in
			av.(i) <- true; lv.(i) <- true; 
			let n = `Const(foi(Random.int 12)) in
			`Loop(i, n, enumerate_ast false (r-1,br,av,lv))
		| _ -> `Nop
	) else (
		(* if recursion limit, emit a constant / terminal*)
		let q = if br > 0 then 
			Random.int 3 else Random.int 2 in
		match q with 
		| 0 -> 
			`Var(select_actvar av)
		| 1 -> (
			match Random.int 3 with
			| 0 -> `Const(8.0 *. atan 1.0) (* unit angle *)
			| 1 -> `Const(1.0) (* unit length *)
			| _ -> `Const(Random.int 5 |> foi)
			)
		| 2 -> 
			let a = enumerate_ast true (r-1,br-1,av,lv) in
			let b = enumerate_ast true (r-1,br-1,av,lv) in
			(match (Random.int 4) with 
				| 0 -> `Binop( a, "+", ( +. ), b)
				| 1 -> `Binop( a, "-", ( -. ), b)
				| 2 -> `Binop( a, "*", ( *. ), b)
				| _ -> `Binop( a, "/", ( /. ), b) )
		| _ -> `Nop
	)
(* see also: https://stackoverflow.com/questions/71718527/can-you-pattern-match-integers-to-ranges-in-ocaml *)
let rec compress ast change = 
	(* recusively copy an ast, removing obvious nops *)
	match ast with
	| `Var(i) -> `Var(i)
	| `Save(i, a) -> (
		match a with 
		| `Nop -> ( change := true; `Nop )
		| `Var(j) -> 
			if i = j then( change := true; `Nop )
			else `Save(i, a)
		| _ -> `Save(i, compress a change))
	| `Move(a,b) -> (
		match a,b with
		| `Nop, _ -> change := true; `Nop
		| _,`Nop -> change := true; `Nop
		| _ -> `Move(compress a change, compress b change) )
	| `Binop(a,s,f,b) -> (
		match a,b with 
		| `Nop, _ -> ( change := true; `Nop )
		| _, `Nop -> ( change := true; `Nop )
		| `Const(aa), `Const(bb) -> (
			if aa = bb then (
				match s with 
				| "/" -> (change := true; `Const(1.0))
				| "-" -> (change := true; `Const(0.0))
				| _ -> `Binop(a,s,f,b) 
			) else (`Binop(a,s,f,b) ))
		| `Var(aa), `Var(bb) -> (
			if aa = bb then (
				match s with 
				| "/" -> (change := true; `Const(1.0))
				| "-" -> (change := true; `Const(0.0))
				| _ -> `Binop(a,s,f,b) 
			) else (`Binop(a,s,f,b) ))
		| _ -> `Binop(compress a change, s,f, compress b change) )
	| `Const(i) -> `Const(i)
	| `Loop(indx, niter, body) -> (
		match body with 
		| `Nop -> (change := true; `Nop)
		| _ -> `Loop(indx, niter, compress body change) )
	| `Seq(l) -> (
		let l2 = List.filter (fun a -> match a with
			| `Nop -> (change := true; false)
			| _ -> true) l in
		if List.length l2 > 0 then 
		`Seq( List.map (fun a -> compress a change) l2 )
		else( change := true; `Nop ) )
	| `Call(i,l) -> `Call(i,l)
	| `Def(i,body) -> `Def(i,body)
	| `Nop -> `Nop

let rec has_move ast = 
	match ast with
	| `Save(_,a) -> has_move a
	| `Move(_,_) -> true
	| `Binop(a,_,_,b) -> ( has_move a || has_move b )
	| `Seq(l) -> List.exists (fun q -> has_move q) l
	| `Loop(_,a,b) -> ( has_move a || has_move b )
	| _ -> false
	
let rec compress_ast ast = 
	let change = ref false in
	let ast2 = compress ast change in
	let ast3 = if !change then compress_ast ast2 else ast2 in
	if has_move ast3 then ast3 else `Nop
  
  
let rec loop_input lg sout serr ic buf cnt = 
	Unix.clear_nonblock stdin; (* this probably is not necessary *)
(*	let (readch, _, _) = select [stdin] [] [] 1.0 in
	(* select doesn't seem to work .. *)
	if List.length readch > 0 && cnt < 1000 then (  *)
	print_log lg "waiting for command\n" ;
	let len = input ic buf 0 4096 in
	if !g_logEn then (
		Printf.fprintf lg "got %d bytes from stdin\n" len 
	); 
	if len > 0 then (
		let lp = try 
			Some (Logo_pb.decode_logo_program
				(Pbrt.Decoder.of_bytes (Bytes.sub buf 0 len)))
		with _ -> ( 
			print_log lg "Could not decode protobuf\n";
			None
		) in
			
		match lp with 
		| Some lp -> (
			g_logEn := lp.log_en ; 
			if lp.log_en then (
				Printf.fprintf lg "%s" (Format.asprintf "logo: %a" Logo_pp.pp_logo_program lp);
				print_log lg "\n"
			); 

			let lexbuf = Lexing.from_string lp.prog in
			lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = "from stdin" };
			ignore(parse_and_render lg sout serr lexbuf lp.id lp.resolution ) )
		| None -> ()
	) ;
	if cnt < 10000000 then (
		loop_input lg sout serr ic buf (cnt+1)
	) else true

let run_logo_file lg sout serr fname =
	let ic = open_in fname in
	let s = really_input_string ic (in_channel_length ic) in
	close_in ic;
	let lexbuf = Lexing.from_string s in
	lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = fname };
	ignore(parse_and_render lg sout serr lexbuf 0 256 )

let generate_random_logo lg n =
	for i = 0 to n do (
		Printf.fprintf lg "program %d : \n" i;
		let actvar = Array.init 5 (fun _i -> false) in
		let loopvar = Array.init 5 (fun _i -> false) in
		let prog = enumerate_ast false (3,3,actvar,loopvar) in
		Logo.output_program_p lg prog;
		Printf.fprintf lg "\ncompressed %d : \n" i;
		let progc = compress_ast prog in
		Logo.output_program_p lg progc;
		Printf.fprintf lg "\n\n";
		let (_,_,segs) = Logo.eval Logo.start_state progc in
		Logo.segs_to_png segs 128 (Printf.sprintf "png/test%d.png" i);
	) done

let () = 
	let lg = open_out "logo_log.txt" in
(* 	let ic = in_channel_of_descr stdin in *)
	let sout = out_channel_of_descr stdout in
	let serr = out_channel_of_descr stderr in
	Random.self_init (); 
	Printf.fprintf lg "hello\n";

	run_logo_file lg sout serr "poly.logo" ;

	(*let buf = Bytes.create 4096 in
	ignore(loop_input lg sout serr ic buf 0); *)

	close_out sout; 
(* 	close_out serr; *)
	close_out lg; 


