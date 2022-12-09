(*open Core :-X core doesnt yet support byte output, necessary for protobufs*)
open Lexer
open Lexing
open Printf
open Unix
open Logo

type pdata = 
	{ pid : int
	; pro  : Logo.prog
	; progenc : string
	; img  : (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
	; cost : float
	; segs : Logo.segment list
	}
	
let nulpdata = 
	{ pid = -1
	; pro  = `Nop 
	; progenc = ""
	; img  = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout 1
	; cost = -1.0 
	; segs = [] 
	}
	

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
	
let print_prog p = 
	let bf = Buffer.create 30 in
	Logo.output_program_p bf p; 
	Printf.printf "%s\n" (Buffer.contents bf)

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

let encode_program_str prog = 
	Logo.encode_program prog |> intlist_to_string 
	
let read_protobuf lg ic pfunc = 
	(* polymorphic! so cool *)
	let buf = Bytes.create 256 in
	let len = input ic buf 0 256 in
	if !g_logEn && len > 0 then (
		Printf.fprintf lg "read_protobuf: got %d bytes from stdin\n" len ); 
	if len > 0 then (
		let lp = try 
			Some ( pfunc
				(Pbrt.Decoder.of_bytes (Bytes.sub buf 0 len)))
		with _ -> ( 
			print_log lg "Could not decode protobuf\n";
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
    
let run_prog lg sout serr prog id res =
	print_log lg "enter run_prog\n";
	match prog with
	| Some(prog) -> (
		let (_,_,segs) = Logo.eval (Logo.start_state ()) prog in
		if !g_logEn then ( 
			Logo.output_program_h lg prog; 
			print_log lg "\n"; 
			Logo.output_segments lg segs; 
			print_log lg "\n"; 
			Logo.segs_to_png segs res "test.png"
		); 
		let arr,cost = Logo.segs_to_array_and_cost segs res in
		let stride = (Bigarray.Array1.dim arr) / res in
		let bf = Buffer.create 30 in
		Logo.output_program_p bf prog; 
		let progenc = Logo.encode_program prog
			|> intlist_to_string in
		let progstr = Buffer.contents bf in
		let r = Logo_types.({
			id = id; 
			stride = stride; 
			width = res; 
			height = res; 
			segs = List.map (fun (x0,y0,x1,y1) -> 
				Logo_types.({x0=x0; y0=y0; x1=x1; y1=y1;})) segs; 
			cost = cost; 
			prog = progstr; 
			progenc = progenc;
		}) in
		(* write the data to stderr, message to stdout *)
		let arr = bigarray_to_bytes arr in
		output_bytes serr arr ;

		(* Create a Protobuf encoder and encode value *)
		write_protobuf sout Logo_pb.encode_logo_result r; 

		if !g_logEn then (
			Printf.fprintf lg "run_prog result %s\n" (Format.asprintf "%a" Logo_pp.pp_logo_result r);
			print_log lg "run_prog done\n"
		); 
		flush sout; 
		flush serr; 
		true)
	| None -> ( 
		output_bytes sout (Bytes.create 4) ;  (* keep things moving *)
		flush sout; 
		flush serr; 
		false )

let parse_logo_string lg serr s = 
	let lexbuf = Lexing.from_string s in
	lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = "from string" };
	parse_with_error lg serr lexbuf
	
let parse_logo_file lg serr fname = 
	let ic = open_in fname in
	let s = really_input_string ic (in_channel_length ic) in
	close_in ic;
	parse_logo_string lg serr s

let run_logo_file lg sout serr fname =
	let prog = parse_logo_file lg serr fname in
	ignore(run_prog lg sout serr prog 0 256 )


let num_actvar actvar =
	Array.fold_left (fun a b -> if b then a+1 else a) 0 actvar

let select_writevar actvar =
	let len = Array.length actvar in
	let sum = num_actvar actvar in
	if sum = len then (
		Random.int len
	) else (
		let sel = ref (len-1) in
		for i = len-1 downto 0 do (
			if not actvar.(i) then sel := i;
		) done;
		!sel
	)

let select_readvar actvar =
	let len = Array.length actvar in
	let sum = num_actvar actvar in
	let avail = ref [] in
	Array.iteri (fun i a -> if a then (
		avail := i :: !avail) ) actvar ;
	if sum = 0 then (
		Random.int len (* degenerate *)
	) else (
		List.nth !avail (Random.int (List.length !avail))
	)


let rec gen_ast return_val state = 
	(* stochastic AST generation ! *)
	let r, br, av = state in
	(* recurs lim, binop recursion limit, active variables, loop variables *)
	if not return_val then (
		match Random.int 5 with 
		| 0 -> 
			(* generate the rhs before updating available variables *)
			let b = gen_ast true (r-1,br,av) in
			let i = select_writevar av in
			av.(i) <- true; 
			`Save(i, b, nulptag)
		| 1 -> 
			`Move(gen_ast true (r-1,br,av), gen_ast true (r-1,br,av), nulptag)
		| 2 -> 
			let len = (Random.int 3) + 1 in
			let ls = List.init len (fun _i -> gen_ast false (r-1,br,av)) in
			`Seq(ls,nulptag)
		| 3 -> 
			let i = select_writevar av in
			av.(i) <- true;
			let n = `Const(foi(Random.int 12),nulptag) in
			`Loop(i, n, gen_ast false (r-1,br,av),nulptag)
		| _ -> `Nop
	) else (
		(* if recursion limit, emit a constant / terminal*)
		let uplim = if br > 0 then 3 else 2 in
		let q = if (num_actvar av = 0) then (
			(Random.int (uplim-1)) + 1
		) else Random.int uplim in
		match q with 
		| 0 -> 
			`Var(select_readvar av, nulptag)
		| 1 -> (
			match Random.int 3 with
			| 0 -> `Const(8.0 *. atan 1.0,nulptag) (* unit angle *)
			| 1 -> `Const(1.0,nulptag) (* unit length *)
			| _ -> `Const(Random.int 5 |> foi,nulptag)
			)
		| 2 -> 
			let a = gen_ast true (r-1,br-1,av) in
			let b = gen_ast true (r-1,br-1,av) in
			(match (Random.int 4) with 
				| 0 -> `Binop( a, "+", ( +. ), b,nulptag)
				| 1 -> `Binop( a, "-", ( -. ), b,nulptag)
				| 2 -> `Binop( a, "*", ( *. ), b,nulptag)
				| _ -> `Binop( a, "/", ( /. ), b,nulptag) )
		| _ -> `Nop
	)

let rec count_ast ast = 
	match ast with
	| `Var(_,_) -> 1
	| `Save(_,a,_) -> 1 + count_ast a
	| `Move(a,b,_) -> 1 + (count_ast a) + (count_ast b)
	| `Binop(a,_,_,b,_) -> 1 + (count_ast a) + (count_ast b)
	| `Const(_,_) -> 1
	| `Seq(l,_) -> List.fold_left (fun a e -> a + count_ast e) 0 l
	| `Loop(_,a,b,_) -> 1 + (count_ast a) + (count_ast b)
	| `Call(_,l,_) -> List.fold_left (fun a e -> a + count_ast e) 1 l
	| `Def(_,a,_) -> 1 + count_ast a
	| `Nop -> 0
	
	
(* We need to tag the AST -- counting won't work b/c the tree is changing. *)
let rec mark_ast ast n sel = 
	match ast with
	| `Var(i,_) -> ( if !n = sel then 
		(incr n; `Var(i,1) ) else 
		(incr n; `Var(i,nulptag) ) )
	| `Save(i,a,_) -> ( if !n = sel then
		(incr n; `Save(i,(mark_ast a n sel),1) ) else 
		(incr n; `Save(i,(mark_ast a n sel),nulptag) ) )
	| `Move(a,b,_) -> ( if !n = sel then
		(incr n; `Move((mark_ast a n sel),(mark_ast b n sel),1) ) else 
		(incr n; `Move((mark_ast a n sel),(mark_ast b n sel),nulptag) ) )
	| `Binop(a,s,f,b,_) -> ( if !n = sel then
		(incr n; `Binop((mark_ast a n sel),s,f,(mark_ast b n sel),1) ) else 
		(incr n; `Binop((mark_ast a n sel),s,f,(mark_ast a n sel),nulptag) ) )
	| `Const(f,_) -> ( if !n = sel then 
		(incr n; `Const(f,1) ) else
		(incr n; `Const(f,nulptag) ) ) 
	| `Seq(l,_) -> ( if !n = sel then 
		(incr n; `Seq(List.map (fun a -> mark_ast a n sel) l, 1) ) else
		(incr n; `Seq(List.map (fun a -> mark_ast a n sel) l, nulptag) ) )
	| `Loop(i,a,b,_) -> ( if !n = sel then 
		(incr n; `Loop(i,(mark_ast a n sel),(mark_ast b n sel),1) ) else 
		(incr n; `Loop(i,(mark_ast a n sel),(mark_ast a n sel),nulptag) ) )
	| `Call(i,l,_) -> ( if !n = sel then 
		(incr n; `Call(i,List.map (fun a -> mark_ast a n sel) l, 1) ) else
		(incr n; `Call(i,List.map (fun a -> mark_ast a n sel) l, nulptag) ) )
	| `Def(i,a,_) -> ( if !n = sel then
		(incr n; `Def(i,(mark_ast a n sel),1) ) else 
		(incr n; `Def(i,(mark_ast a n sel),nulptag) ) )
	| `Nop -> `Nop

let rec chng_ast ast st = 
	let r, _, av = st in
	match ast with
	| `Var(i,w) -> (if w > 0 then 
		( gen_ast true st ) else 
		( `Var(i,nulptag) ) )
	| `Save(i,a,w) -> ( if w > 0 then
		( let ii = Random.int 5 in
		  av.(i) <- false; 
		  av.(ii) <- true; 
		  `Save(ii, chng_ast a st, nulptag) ) else 
		( av.(i) <- true; 
		  `Save(i, chng_ast a st, nulptag) ) )
	| `Move(a,b,w) -> ( if w > 0 then
		( gen_ast false st ) else 
		( `Move(chng_ast a st, chng_ast b st, nulptag) ) )
	| `Binop(a,s,f,b,w) -> ( if w > 0 then 
		( let st2 = r, 2, av in
		  gen_ast true st2 ) else
		( `Binop(chng_ast a st, s, f, chng_ast b st, nulptag) ) )
	| `Const(f,w) -> (if w > 0 then 
		( gen_ast true st ) else 
		( `Const(f, nulptag) ) )
	| `Seq(l,w) -> (if w > 0 then 
		( gen_ast false st ) else
		( `Seq(List.map (fun a -> chng_ast a st) l, nulptag) ) ) 
	| `Loop(i,a,b,w) -> (if w > 0 then 
		( gen_ast false st ) else
		( av.(i) <- true ;
		  `Loop(i, chng_ast a st, chng_ast b st, nulptag) ) )
	| `Call(i,l,w) -> (if w > 0 then 
		( gen_ast false st ) else 
		( `Call(i, List.map (fun a -> chng_ast a st) l, nulptag) ) )
	| `Def(i,a,w) -> (if w > 0 then 
		( gen_ast false st ) else 
		( `Def(i, chng_ast a st, nulptag ) ) )
	| `Nop -> `Nop
	
let change_ast ast lg = 
	let cnt = count_ast ast in
	let n = ref 0 in
	let sel = Random.int cnt in
	printf "count_ast %d labelling %d\n" cnt sel;
	let marked = mark_ast ast n sel in
	Logo.output_program_h lg marked ; 
	let actvar = Array.init 5 (fun _i -> false) in
	let st = (2,3,actvar) in
	chng_ast marked st
(* I wonder if there is a better, learning-aware means of doing above... something that operates directly at the character level ( like a LLM) and learns the constraints / syntax implicitly. *)
	
	
(* see also: https://stackoverflow.com/questions/71718527/can-you-pattern-match-integers-to-ranges-in-ocaml *)
let rec compress ast change = 
	(* recusively copy an ast, removing obvious nops *)
	let eps = 0.0001 in
	let neps = -1.0 *. eps in
	match ast with
	| `Var(i,w) -> `Var(i,w)
	| `Save(i,a,w) -> (
		match a with 
		| `Nop -> ( change := true; `Nop )
		| `Var(j,_) -> 
			if i = j then( change := true; `Nop )
			else `Save(i,a,w)
		| _ -> `Save(i, compress a change,w))
	| `Move(a,b,w) -> (
		match a,b with
		| `Nop, _ -> change := true; `Nop
		| _,`Nop -> change := true; `Nop
		| _ -> `Move(compress a change, compress b change,w) )
	| `Binop(a,s,f,b,w) -> (
		match a,b with 
		| `Nop, _ -> ( change := true; `Nop )
		| _, `Nop -> ( change := true; `Nop )
		| `Const(aa,aw), `Const(bb,bw) -> (
			match s with
			| "+" -> (
				match aa,bb with
				| a3,_ when a3 > neps && a3 < eps -> 
					(change := true;`Const(bb,nulptag))
				| _,b3 when b3 > neps && b3 < eps -> 
					(change := true;`Const(aa,nulptag))
				| a3,b3 when b3 > a3 -> 
					(change := true; `Binop(`Const(b3,bw),s,f,`Const(a3,aw),w))
				| _ -> `Binop(compress a change, s,f, compress b change,w))
			| "-" -> (
				match aa,bb with
				| _,b3 when b3 > neps && b3 < eps -> 
					(change := true;`Const(aa,nulptag))
				| a3,b3 when a3-.b3 > neps && a3-.b3 < eps -> 
					(change := true;`Const(0.0,nulptag))
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| "*" -> (
				match aa,bb with
				| a3,_ when a3 > neps && a3 < eps -> 
					(change := true;`Nop)
				| _,b3 when b3 > neps && b3 < eps -> 
					(change := true;`Nop)
				| a3,_ when a3 > 1.0-.eps && a3 < 1.0+.eps ->
					(change := true;`Const(bb,nulptag))
				| _,b3 when b3 > 1.0-.eps && b3 < 1.0+.eps ->
					(change := true;`Const(aa,nulptag))
				| a3,b3 when b3 > a3 -> 
					(change := true; `Binop(`Const(b3,bw),s,f,`Const(a3,aw),w))
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| "/" -> (
				match aa,bb with
				| a3,_ when a3 > neps && a3 < eps -> 
					(change := true;`Nop)
				| _,b3 when b3 > neps && b3 < eps -> 
					(change := true;`Nop)
				| _,b3 when b3 > 1.0-.eps && b3 <1.0+.eps -> 
					(change := true;`Const(aa,nulptag))
				| a3,b3 when a3-.b3 > neps && a3-.b3 < eps -> 
					(change := true;`Const(1.0,nulptag))
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| _ -> `Nop )
		| `Var(aa,_), `Var(bb,_) -> (
			if aa = bb then (
				match s with 
				| "/" -> (change := true; `Const(1.0,nulptag))
				| "-" -> (change := true; `Const(0.0,nulptag))
				| _ -> `Binop(compress a change, s,f, compress b change,w) 
			) else ( `Binop(compress a change, s,f, compress b change,w) ))
		| _, `Const(bb,_) -> (
			match s with
			| "+" -> (
				match bb with
				| b3 when b3 > neps && b3 < eps -> 
					(change := true; a)
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| "-" -> (
				match bb with
				| b3 when b3 > neps && b3 < eps -> 
					(change := true; a)
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| "*" -> (
				match bb with
				| b3 when b3 > neps && b3 < eps -> 
					(change := true;`Const(0.0,nulptag))
				| b3 when b3 > 1.0-.eps && b3 < 1.0+.eps ->
					(change := true; a)
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| "/" -> (
				match bb with
				| b3 when b3 > neps && b3 < eps ->
					(change := true;`Nop)
				| b3 when b3 > 1.0-.eps && b3 < 1.0+.eps ->
					(change := true; a)
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| _ -> `Nop )
		| `Const(aa,_), _ -> (
			match s with
			| "+" -> (
				match aa with
				| a3 when a3 > neps && a3 < eps -> 
					(change := true; b)
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| "-" -> (
				match aa with
				| a3 when a3 > neps && a3 < eps -> 
					(change := true; b)
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| "*" -> (
				match aa with
				| a3 when a3 > neps && a3 < eps -> 
					(change := true;`Nop)
				| a3 when a3 > 1.0-.eps && a3 < 1.0+.eps -> 
					(change := true; b)
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| "/" -> (
				match aa with
				| a3 when a3 > neps && a3 < eps -> 
					(change := true;`Nop)
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| _ -> `Nop )
		| _ -> `Binop(compress a change, s,f, compress b change,w) )
	| `Const(i,w) -> `Const(i,w)
	| `Loop(indx, niter, body,w) -> (
		match body with 
		| `Nop -> (change := true; `Nop)
		| _ -> `Loop(indx, niter, compress body change,w) )
	| `Seq(l,w) -> (
		let l2 = List.filter (fun a -> match a with
			| `Nop -> (change := true; false)
			| _ -> true) l in
		match List.length l2 with
		| 0 -> ( change := true; `Nop )
		| 1 -> ( change := true; 
			compress (List.hd l2) change ) (*dont need seq*)
		| _ -> `Seq( List.map (fun a -> compress a change) l2, w )
		)
	| `Call(i,l,w) -> `Call(i,l,w)
	| `Def(i,body,w) -> `Def(i,body,w)
	| `Nop -> `Nop

let rec has_move ast = 
	match ast with
	| `Save(_,a,_) -> has_move a
	| `Move(_,_,_) -> true
	| `Binop(a,_,_,b,_) -> ( has_move a || has_move b )
	| `Seq(l,_) -> List.exists (fun q -> has_move q) l
	| `Loop(_,a,b,_) -> ( has_move a || has_move b )
	| _ -> false
	
let rec compress_ast ast = 
	let change = ref false in
	let ast2 = compress ast change in
	let ast3 = if !change then compress_ast ast2 else ast2 in
	if has_move ast3 then ast3 else `Nop
  
(*let rec loop_input lg sout serr ic buf cnt = 
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

			let prog = parse_logo_string lg serr lp.prog in 
			ignore(run_prog lg sout serr prog lp.id lp.resolution ) )
		| None -> ()
	) ;
	if cnt < 10000000 then (
		loop_input lg sout serr ic buf (cnt+1)
	) else true*)

let segs_bbx segs = 
	let minx = List.fold_left 
		(fun c (x0,_,x1,_) -> min (min x0 x1) c) 0.0 segs in
	let maxx = List.fold_left 
		(fun c (x0,_,x1,_) -> max (max x0 x1) c) 0.0 segs in
	let miny = List.fold_left 
		(fun c (_,y0,_,y1) -> min (min y0 y1) c) 0.0 segs in
	let maxy = List.fold_left 
		(fun c (_,y0,_,y1) -> max (max y0 y1) c) 0.0 segs in
	(minx,maxx,miny,maxy)
	
let rec generate_random_logo lg id res =
	(*if !g_logEn then Printf.fprintf lg "program %d : \n" id;*)
	let actvar = Array.init 5 (fun _i -> false) in
	(*Printf.printf "=======\n";*) 
	let prog = gen_ast false (3,1,actvar) in
	(*if !g_logEn then Logo.output_program_plg lg prog;*)
	(*if !g_logEn then Printf.fprintf lg "\ncompressed %d : \n" id;*)
	let pro = compress_ast prog in
	let progenc = Logo.encode_program pro |> intlist_to_string in
	(*if !g_logEn then Logo.output_program_plg lg pro;
	if !g_logEn then Printf.fprintf lg "\n\n";*)
	
	let (_,_,segs) = Logo.eval (Logo.start_state ()) pro in
	(* heuristics for selecting programs... *)
	let seglen (x0,y0,x1,y1) = 
		Float.hypot (x0-.x1) (y0-.y1) in
	let cost = List.fold_left 
		(fun c s -> c +. (seglen s)) 0.0 segs in
	let lx,hx,ly,hy = segs_bbx segs in
	let dx = hx-.lx in
	let dy = hy-.ly in
	let maxd = max dx dy in
	if maxd >= 2. && maxd <= 9. && cost >= 4. && cost <= 64. && List.length segs < 8 && String.length progenc < 24 then (
		let img, _ = Logo.segs_to_array_and_cost segs res in
		{pid=id; pro; progenc; img; cost; segs}
	) else ( 
		generate_random_logo lg id res
	)

let generate_empty_logo lg id res =
	(* the first program needs to be empty, for diffing *)
	let pro = `Nop in
	let progenc = Logo.encode_program pro |> intlist_to_string in
	Printf.fprintf lg "empty program encoding: \"%s\"\n" progenc;
	let segs = [] in
	let cost = 0.0 in
	let img, _ = Logo.segs_to_array_and_cost segs res in
	{pid=id; pro; progenc; img; cost; segs}

let transmit_result channels data res = 
	let sout, serr, _ic, lg = channels in
	let stride = (Bigarray.Array1.dim data.img) / res in
	let bf = Buffer.create 30 in
	Logo.output_program_p bf data.pro; 
	let r = Logo_types.({
		id = data.pid; 
		stride = stride; 
		width = res; 
		height = res; 
		segs = List.map (fun (x0,y0,x1,y1) -> 
			Logo_types.({x0=x0; y0=y0; x1=x1; y1=y1;})) data.segs; 
		cost = data.cost; 
		prog = Buffer.contents bf; 
		progenc = data.progenc
	}) in
	(* write the data to stderr, message to stdout *)
	let arr = bigarray_to_bytes data.img in
	output_bytes serr arr ;
	flush serr; 
	
	write_protobuf sout Logo_pb.encode_logo_result r; 

	if !g_logEn then (
		Printf.fprintf lg "transmit_result %s\n" (Format.asprintf "%a" Logo_pp.pp_logo_result r);
		print_log lg "transmit_result done\n"
	); 
	flush sout
	
let transmit_ack channels id = 
	let sout, _serr, _ic, _lg = channels in
	let r = Logo_types.({ going=true; ackid=id;}) in
	write_protobuf sout Logo_pb.encode_logo_ack r

let render_simplest db dosort =
	(* render the shortest 4*1024 programs in the database.*)
	let dba = Vector.to_array db in
	let dbal = Array.length dba in
	if dosort then Array.sort (fun a b ->
		let la = List.length a.segs in
		let lb = List.length b.segs in
		let na = count_ast a.pro in
		let nb = count_ast b.pro in
		if la = lb then compare na nb else compare la lb ) dba; 
		(* in-place sorting *)
	let res = 48 in
	let lg = open_out "/tmp/png/log.txt" in
	for id = 0 to min (4*1024-1) (dbal-1) do (
		let data = dba.(id) in
		let (_,_,segs) = Logo.eval (Logo.start_state ()) data.pro in
		Logo.segs_to_png segs res (Printf.sprintf "/tmp/png/test%d.png" id);
		let bf = Buffer.create 30 in
		Logo.output_program_p bf data.pro;
		fprintf lg "%d %s\n" id (Buffer.contents bf);
	) done;
	close_out lg; 
	dba (* return the sorted array *)
	
let make_batch lg dba nbatch = 
	(* make a batch of pre, post, edits *)
	(* image is saved in python, no need to duplicate *)
	let ndba = min (Array.length dba) 1024 in (* FIXME 2048 *)
	if !g_logEn then Printf.fprintf lg "entering make_batch, req %d of %d\n" nbatch ndba; 
	(* sorting is done in render_simplest *)
	let batch = ref [] in
	while List.length !batch < nbatch do (
		let nb = (Random.int (ndba-1)) + 1 in
		let na = 0 in (* FIXME *)
		(*let na = if (Random.int 10) = 0 then 0 else (Random.int nb) in*)
		(* small portion of the time na is the empty program *)
		let a = dba.(na) in
		let b = dba.(nb) in
		let a_ns = List.length a.segs in
		let b_ns = List.length b.segs in
		let a_np = String.length a.progenc in
		let b_np = String.length b.progenc in
		if a_ns < 4 && b_ns < 4 && a_np < 16 && b_np < 16 then (
			let dist,_ = Levenshtein.distance a.progenc b.progenc false in
			if !g_logEn then
			Printf.fprintf lg "trying [%d] [%d] for batch; dist %d\n" na nb dist;
			if dist > 0 && dist < 16 then ( (* FIXME: dist < 7 *)
				(* "move a , b ;" is 5 insertions; need to allow *)
				let _, edits = Levenshtein.distance a.progenc b.progenc true in
				let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
				(* emphasize insertion and only a little substitution or deletion -- this mirrors how a human programs *)
				(*let nsub,ndel,nins = List.fold_left (fun (sub,del,ins) (s,_,_) ->
					match s with
					| "sub" -> (sub+1,del,ins)
					| "del" -> (sub,del+1,ins)
					| "ins" -> (sub,del,ins+1)
					| _ -> (sub,del,ins)
						) (0,0,0) edits in
				if (nsub <= 0 && ndel <= 0 && nins <= 0) || (nsub = 0 && ndel = 0 && nins <= 6) then ( (*fixme edit distance *)*)
				(* verify ..*)
				let re = Levenshtein.apply_edits a.progenc edits in
				if re <> b.progenc then (
					Printf.fprintf lg "error! %s edits should be %s was %s\n"
						a.progenc b.progenc re
				);
				let a_progstr = Logo.output_program_pstr a.pro in
				let b_progstr = Logo.output_program_pstr b.pro in
				(* edits are applied in reverse, do it here not py *)
				(* also add a 'done' edit/indicator *)
				let edits = ("fin",0,'0') :: edits in
				let edits = List.rev edits in
				batch := (a.pid, b.pid, a.progenc, b.progenc,
					a_progstr, b_progstr, edits) :: !batch;
				if !g_logEn then Printf.fprintf lg "adding [%d] %s [%d] %s to batch (unsorted pids: %d %d)\n"
					na a.progenc nb b.progenc a.pid b.pid;
				(*Levenshtein.print_edits edits*)
				(* ) *)
			)
		)
	) done;
	flush lg; 
	!batch

let rec loop_random channels cnt db dba = 
	(* unlike loop_input, we generate the program here *)
	if cnt > (100 * 20000) then () else (
	let sout, _serr, ic, lg = channels in
	if !g_logEn then ( print_log lg "`" );
	let lp = read_protobuf lg ic Logo_pb.decode_logo_request in
	let dba2 = match lp with 
	| Some lp -> (
		g_logEn := lp.log_en ; 
		if lp.batch > 0 then (
			(* generate a batch of data instead *)
			let batch = make_batch lg dba lp.batch in
			let r = Logo_types.({
				count = lp.batch; 
				btch = List.map (fun (aid,bid,ape,bpe,aps,bps,editz) -> 
					Logo_types.({
						a_pid = aid; 
						b_pid = bid;
						a_progenc = ape; 
						b_progenc = bpe; 
						a_progstr = aps; 
						b_progstr = bps;
						edits = List.map (fun (s,ri,c) -> 
							Logo_types.({
								typ = s; 
								pos = ri; 
								chr = String.make 1 c ; }) ) editz; 
						}) ) batch ; 
				}) in
			write_protobuf sout Logo_pb.encode_logo_batch r; 
			dba
		) 
		else (
			let data = if lp.id = 0 then
				generate_empty_logo lg lp.id lp.res else
				generate_random_logo lg lp.id lp.res in
			transmit_result channels data lp.res ; 
			let lp2 = read_protobuf lg ic Logo_pb.decode_logo_last in
			match lp2 with 
			| Some lp2 -> (
				if lp2.keep then (
					let data2 = {data with pid=lp2.where } in
					if lp2.where = Vector.length db then (
						if !g_logEn then Printf.fprintf lg "db saving %d push\n" lp2.where; 
						Vector.push db data2
					) else (
						if !g_logEn then Printf.fprintf lg "db saving %d set\n" lp2.where; 
						Vector.set db lp2.where data2
					)
				); 
				let dba3 = if lp2.render_simplest then (
					Printf.fprintf lg "render_simplest: first 10 programs\n";
					for i = 0 to 9 do (
						let p = Vector.get db i in
						Printf.fprintf lg "%d: %s\n" i
								(Logo.output_program_pstr p.pro); 
					) done; 
					render_simplest db true 
				) else dba in
				transmit_ack channels lp.id; 
				dba3 )
			| _ -> dba
		) )
	| _ -> dba in
	loop_random channels (cnt+1) db dba2
	)

(*
let () = 
	Unix.clear_nonblock stdin; 
	(*run_logo_file lg sout serr "semicircle.logo" ;*)
	Random.self_init (); 
	let lg = open_out "logo_log.txt" in
	let sout = out_channel_of_descr stdout in
	let serr = out_channel_of_descr stderr in 
	run_logo_file lg sout serr "badvar.logo" ;
	(*let data = generate_random_logo lg 0 48 in
	print_prog data.pro; *)
	close_out lg; 
*)
(* 
let () = 
	(* this for checking program encoding & decoding *)
	Unix.clear_nonblock stdin; 
	let lg = open_out "logo_log.txt" in
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
*)
(*
let () = 
	(* this tests change_ast *)
	Unix.clear_nonblock stdin; (* this might not be needed *)
	Random.self_init (); 
	let lg = open_out "logo_log.txt" in
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
*)

let () = 
	Unix.clear_nonblock stdin; 
	Random.self_init (); 
	let lg = open_out "logo_log.txt" in
	let ic = in_channel_of_descr stdin in
	let sout = out_channel_of_descr stdout in
	let serr = out_channel_of_descr stderr in 
	let channels = (sout, serr, ic, lg) in
	Printf.fprintf lg "hello\n";

	let db = Vector.create ~dummy:nulpdata in
	let dba = Array.make 1 nulpdata in
	loop_random channels 0 db dba;
	(*let _b = make_batch lg dba 1 in*)

	close_out sout; 
	close_out serr;
	close_out lg; 

