(*open Core :-X core doesnt yet support byte output, necessary for protobufs*)
open Lexer
open Lexing
open Printf
(*open Unix*)
open Logo
open Torch
open Lwt (* I have my resevations *)

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

type batche = 
	{ a_pid : int 
	; b_pid : int
	; a_progenc : string (* these are redundant, but useful *)
	; b_progenc : string
	; c_progenc : string
	; edits : (string*int*char) list
	; edited : float array
	}
	
let nulbatche = 
	{ a_pid = 0
	; b_pid = 0
	; a_progenc = ""
	; b_progenc = ""
	; c_progenc = ""
	; edits = []
	; edited = [| 0.0 |]
	}
	
type batchd = 
	{ bpro : (float, Bigarray.float32_elt, Bigarray.c_layout)
				Bigarray.Array3.t (* btch , p_ctx , p_indim *)
	; bimg : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array3.t (* btch*3, image_res, image_res *)
	; bedt : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array2.t (* btch , e_indim *)
	; posenc : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array2.t (* p_ctx , poslen*2 *)
	; bea : batche array
	; fresh : bool array
	}

let pi = 3.1415926
let image_count = 6*2048 
let image_res = 30
let batch_size = ref (256*3)
let toklen = 30
let poslen = 6
let p_indim = toklen + 1 + poslen*2 (* 31 + 12 = 43 *)
let e_indim = 5 + toklen + poslen*2
let p_ctx = 64
let nreplace = ref 0 (* number of repalcements durng hallucinations *)

let listen_address = Unix.inet_addr_loopback
let port = 4340
let backlog = 10

let read_lines name : string list =
	let ic = open_in name in
	let try_read () =
		try Some (input_line ic) with End_of_file -> None in
	let rec loop acc = match try_read () with
		| Some s -> loop (s :: acc)
		| None -> close_in ic; List.rev acc in
	loop []

let print_position lexbuf = 
	let pos = lexbuf.lex_curr_p in
	let bf = Buffer.create 64 in
	Printf.bprintf bf "%s:%d:%d" pos.pos_fname
		pos.pos_lnum (pos.pos_cnum - pos.pos_bol + 1); 
	(Buffer.contents bf)

let parse_with_error lexbuf =
	let prog = try Some (Parser.parse_prog Lexer.read lexbuf) with
	| SyntaxError msg ->
		Logs.debug (fun m -> m "%s: %s" 
			(print_position lexbuf) msg); 
		None
	| Parser.Error ->
		Logs.debug (fun m -> m "%s: syntax error" 
			(print_position lexbuf));
		None in
	prog

let bigarray_to_bytes arr = 
	(* convert a bigarray to a list of bytes *)
	(* this is not efficient.. *)
	let len = Bigarray.Array1.dim arr in
	Bytes.init len 
		(fun i -> Bigarray.Array1.get arr i |> Char.chr)
	
let read_protobuf ic pfunc = 
	(* polymorphic! so cool *)
	let buf = Bytes.create 256 in
	let len = input ic buf 0 256 in
	if len > 0 then (
		Logs.info (fun m -> m "read_protobuf: got %d bytes from stdin" len )); 
	if len > 0 then (
		let lp = try 
			Some ( pfunc
				(Pbrt.Decoder.of_bytes (Bytes.sub buf 0 len)))
		with _ -> ( 
			Logs.err(fun m -> m "Could not decode protobuf");
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

	
let run_prog prog id res =
	Logs.debug(fun m -> m "enter run_prog");
	match prog with
	| Some(prog) -> (
		let (_,_,segs) = Logo.eval (Logo.start_state ()) prog in
		Logs.debug(fun m -> m "%s" (Logo.output_program_pstr prog)); 
		Logs.debug(fun m -> m "%s" (Logo.output_segments_str segs)); 
		Logo.segs_to_png segs res "test.png"; 
		let arr,cost = Logo.segs_to_array_and_cost segs res in
		let stride = (Bigarray.Array1.dim arr) / res in
		let bf = Buffer.create 30 in
		Logo.output_program_p bf prog; 
		let progenc = Logo.encode_program_str prog in
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

		Logs.debug(fun m -> m "run_prog result %s" (Format.asprintf "%a" Logo_pp.pp_logo_result r)); 
			(* another good way of doing it*)
		Logs.debug(fun m -> m  "run_prog done");
		true)
	| None -> ( false )

let parse_logo_string s = 
	let lexbuf = Lexing.from_string s in
	lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = "from string" };
	parse_with_error lexbuf
	
let parse_logo_file fname = 
	let ic = open_in fname in
	let s = really_input_string ic (in_channel_length ic) in
	close_in ic;
	parse_logo_string s

let run_logo_file fname =
	let prog = parse_logo_file fname in
	ignore(run_prog prog 0 256 )


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
	
let change_ast ast = 
	let cnt = count_ast ast in
	let n = ref 0 in
	let sel = Random.int cnt in
	Logs.debug(fun m -> m  "count_ast %d labelling %d" cnt sel);
	let marked = mark_ast ast n sel in
	Logs.debug(fun m -> m  "%s" (Logo.output_program_pstr marked) ); 
	let actvar = Array.init 5 (fun _i -> false) in
	let st = (2,3,actvar) in
	chng_ast marked st
	
	
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
	
let segs_to_cost segs = 
	let seglen (x0,y0,x1,y1) = 
		Float.hypot (x0-.x1) (y0-.y1) in
	let cost = List.fold_left 
		(fun c s -> c +. (seglen s)) 0.0 segs in
	cost
	
let program_to_pdata pro id res = 
	let progenc = Logo.encode_program_str pro in
	let (_,_,segs) = Logo.eval (Logo.start_state ()) pro in
	let cost = segs_to_cost segs in
	let lx,hx,ly,hy = segs_bbx segs in
	let dx = hx-.lx in
	let dy = hy-.ly in
	let maxd = max dx dy in
	if maxd >= 2. && maxd <= 9. && cost >= 4. && cost <= 64. && List.length segs < 8 && String.length progenc < 24 then (
		let img, _ = Logo.segs_to_array_and_cost segs res in
		Some {pid=id; pro; progenc; img; cost; segs}
	) else None

let rec generate_random_logo id res =
	let actvar = Array.init 5 (fun _i -> false) in
	let prog = gen_ast false (3,1,actvar) in
	let pro = compress_ast prog in
	let pd = program_to_pdata pro id res in
	match pd with
	| Some q -> q
	| _ -> generate_random_logo id res

let generate_empty_logo id res =
	(* the first program needs to be empty, for diffing *)
	let pro = `Nop in
	let progenc = Logo.encode_program_str pro in
	Logs.debug(fun m -> m  "empty program encoding: \"%s\"" progenc);
	let segs = [] in
	let cost = 0.0 in
	let img, _ = Logo.segs_to_array_and_cost segs res in
	{pid=id; pro; progenc; img; cost; segs}

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
	let lg = open_out "/tmp/png/db_init.txt" in
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
	
(* because we include position encodings, need to be float32 *)
let mmap_bigarray2 fname rows cols = 
	(let open Bigarray in
	let fd = Unix.openfile fname
		[Unix.O_RDWR; Unix.O_TRUNC; Unix.O_CREAT] 0o666 in
	let a = Bigarray.array2_of_genarray 
		(Unix.map_file fd float32 c_layout true [|rows; cols|]) in (* true = shared *)
	(* return the file descriptor and array. 
	will have to close the fd later *)
	fd,a )
	
let mmap_bigarray3 fname batches rows cols = 
	(let open Bigarray in
	let fd = Unix.openfile fname
		[Unix.O_RDWR; Unix.O_TRUNC; Unix.O_CREAT] 0o666 in
	let a = Bigarray.array3_of_genarray 
		(Unix.map_file fd float32 c_layout true [|batches;rows;cols|]) in (* true = shared *)
	(* return the file descriptor and array. 
	will have to close the fd later *)
	fd,a )
	
let reset_bea () = 
	let bea = Array.init !batch_size (fun _i -> nulbatche) in
	let fresh = Array.init !batch_size (fun _i -> true) in
	bea,fresh
	
let init_batchd () =
	let fd_bpro,bpro = mmap_bigarray3 "bpro.mmap" 
			!batch_size p_ctx p_indim in
	let fd_bimg,bimg = mmap_bigarray3 "bimg.mmap" 
			(!batch_size*3) image_res image_res in
			(* note needs a reshape on python side!! *)
	let fd_bedt,bedt = mmap_bigarray2 "bedt.mmap" 
			!batch_size e_indim in
	let fd_posenc,posenc = mmap_bigarray2 "posenc.mmap"
			p_ctx (poslen*2) in
	let bea,fresh = reset_bea () in
	(* fill out the posenc matrix *)
	let scl = 2.0 *. pi /. (foi p_ctx) in
	for i = 0 to (p_ctx-1) do (
		(* i is the position *)
		let fi = foi i in
		for j = 0 to (poslen-1) do (
			(* j is the frequency; j=0 means once cycle in p_ctx *)
			(* posenc[i,j*2+0] = sin((2*pi*i / p_ctx) * (j+1)) *)
			let fj = foi (j+1) in
			posenc.{i, j*2+0} <- sin(scl *. fi *. fj); 
			posenc.{i, j*2+1} <- cos(scl *. fi *. fj)
		) done
	) done ;
	Bigarray.Array3.fill bpro 0.0 ;
	Bigarray.Array3.fill bimg 0.0 ;
	Bigarray.Array2.fill bedt 0.0 ; 
	(* return batchd struct & list of files to close later *)
	{bpro; bimg; bedt; posenc; bea; fresh}, 
	[fd_bpro; fd_bimg; fd_bedt; fd_posenc]
	
let rec new_batche dba = 
	let ndba = min (Array.length dba) (1024*8) in (* FIXME *)
	let nb = (Random.int (ndba-1)) + 1 in
	let na = (Random.int (ndba-1)) + 1 in
	(*let na = 0 in (* FIXME *) *)
	(*let na = if (Random.int 10) = 0 then 0 else (Random.int nb) in*)
	(* small portion of the time na is the empty program *)
	let a = dba.(na) in
	let b = dba.(nb) in
	let a_ns = List.length a.segs in
	let b_ns = List.length b.segs in
	let a_np = String.length a.progenc in
	let b_np = String.length b.progenc in
	let lim = (p_ctx/2)-4 in 
	(* check this .. really should be -2 as in bigfill_batchd *)
	if a_ns <= 4 && b_ns <= 4 && a_np < lim && b_np < lim then (
		let dist,_ = Levenshtein.distance a.progenc b.progenc false in
		Logs.debug(fun m -> m  
			"trying [%d] [%d] for batch; dist %d" na nb dist);
		if dist > 0 && dist < 8 then ( (* FIXME: dist < 7 *)
			(* "move a , b ;" is 5 insertions; need to allow *)
			let _, edits = Levenshtein.distance a.progenc b.progenc true in
			let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
			(* verify ..*)
			let re = Levenshtein.apply_edits a.progenc edits in
			if re <> b.progenc then (
				Logs.err(fun m -> m  
					"error! %s edits should be %s was %s"
					a.progenc b.progenc re)
			);
			(* edits are applied in reverse, do it here not py *)
			(* also add a 'done' edit/indicator *)
			let edits = ("fin",0,'0') :: edits in
			let edits = List.rev edits in
			Logs.debug(fun m -> m 
				"adding [%d] %s [%d] %s to batch (unsorted pids: %d %d)"
				na a.progenc nb b.progenc a.pid b.pid);
			let edited = Array.make p_ctx 0.0 in
			{a_pid=na; b_pid=nb; 
				a_progenc = a.progenc; 
				b_progenc = b.progenc; 
				c_progenc = a.progenc; 
				edits; edited }
		) else new_batche dba
	) else new_batche dba

let apply_edits be = 
	(* apply after (of course) sending to python. *)
	(* edit length must be > 0 *)
	let ed = List.hd be.edits in
	let c = Levenshtein.apply_edits be.c_progenc [ed] in
	(* in-place modify the 'edited' array, too *)
	let la = String.length be.a_progenc in
	let lc = String.length be.c_progenc in
	let typ,pp,chr = ed in (* chr already used above *)
	Logs.debug (fun m -> m "apply_edits %s %d %c" typ pp chr );
	(match typ with 
	| "sub" -> (
		let pp = if pp > lc-1 then lc-1 else pp in
		let pp = if pp < 0 then 0 else pp in
		be.edited.(la+pp) <- 0.6 )
	| "del" -> (
		(* lc is already one less at this point -- c has been edited *)
		let pp = if pp >= lc-1 then lc-1 else pp in
		let pp = if pp < 0 then 0 else pp in
		(* shift left *)
		for i = la+pp to p_ctx-2 do (
			be.edited.(i) <- be.edited.(i+1)
		) done; 
		be.edited.(la+pp) <- ( -1.0 ) )
	| "ins" -> (
		let pp = if pp > lc then lc else pp in
		let pp = if pp < 0 then 0 else pp in
		(* shift right one *)
		for i = p_ctx-2 downto la+pp+1 do (
			be.edited.(i) <- be.edited.(i+1)
		) done; 
		be.edited.(la+pp) <- 1.0 )
	| _ -> () ); 
	{be with c_progenc=c; edits=(List.tl be.edits) }
	
	
let update_bea dba bd = 
	(* iterate over batchd struct, make new program pairs when edits are empty *)
	Logs.debug (fun m -> m "entering update_bea");
	(* need to run through twice, first to apply the edits, second to replace the finished elements. *)
	let bea2 = Array.mapi (fun i be -> 
		if List.length be.edits > 0 then (
			bd.fresh.(i) <- false; 
			apply_edits be 
		) else be ) bd.bea in
	let bea3 = Array.mapi (fun i be -> 
		if List.length be.edits = 0 then (
			bd.fresh.(i) <- true; 
			new_batche dba 
		) else be ) bea2 in
	{bd with bea=bea3}
	
let bigfill_batchd dba bd = 
	(* convert bea to the 3 mmaped bigarrays *)
	(* first clear them *)
	(*Bigarray.Array3.fill bd.bimg 0.0 ;*)
	Logs.debug (fun m -> m "entering bigfill_batchd"); 
	Bigarray.Array3.fill bd.bpro 0.0 ;
	Bigarray.Array2.fill bd.bedt 0.0 ; 
	(* fill one-hots *)
	Array.iteri (fun u be -> 
		let a = dba.(be.a_pid) in
		let b = dba.(be.b_pid) in
	  (* - bpro - *)
		let offs = Char.code '0' in
		let llim = (p_ctx/2)-2 in
		let l = String.length be.a_progenc in
		if l > llim then 
			Logs.err(fun m -> m  "too long(%d):%s" l be.a_progenc);
		String.iteri (fun i c -> 
			let j = (Char.code c) - offs in
			if i < llim then bd.bpro.{u,i,j} <- 1.0 ) be.a_progenc ; 
		let lc = String.length be.c_progenc in
		if lc > llim then 
			Logs.err(fun m -> m  "too long(%d):%s" lc be.c_progenc);
		String.iteri (fun i c -> 
			let j = (Char.code c) - offs in
			if (i+l) < 2*llim then (
				bd.bpro.{u,i+l,j} <- 1.0; 
				(* inidicate this is c, to be edited *)
				bd.bpro.{u,i+l,toklen-1} <- 1.0 ) ) be.c_progenc ;
		(* copy over the edited tags (set in apply_edits) *)
		for i = 0 to p_ctx-1 do (
			bd.bpro.{u,i,toklen} <- be.edited.(i)
		) done; 
		(* position encoding *)
		let l = if l > llim then llim else l in 
		let lc = if lc > llim then llim else lc in
		for i = 0 to l-1 do (
			for j = 0 to poslen*2-1 do (
				bd.bpro.{u,i,toklen+1+j} <- bd.posenc.{i,j}
			) done
		) done;
		for i = 0 to lc-1 do (
			for j = 0 to poslen*2-1 do (
				bd.bpro.{u,i+l,toklen+1+j} <- bd.posenc.{i,j}
			) done
		) done ;
	  (* - bimg - *)
		if bd.fresh.(u) then (
			(* could do this with blit, but need int -> float conv *)
			let stride = (Bigarray.Array1.dim a.img) / image_res in
			for i=0 to (image_res-1) do (
				for j=0 to (image_res-1) do (
					let aif = a.img.{i*stride+j} |> foi in
					let bif = b.img.{i*stride+j} |> foi in
					bd.bimg.{3*u+0,i,j} <- aif; 
					bd.bimg.{3*u+1,i,j} <- bif;
					bd.bimg.{3*u+2,i,j} <- aif -. bif;
				) done
			) done; 
			bd.fresh.(u) <- false
		); 
	  (* - bedt - *)
		let (typ,pp,c) = if (List.length be.edits) > 0 
			then List.hd be.edits
			else ("fin",0,'0') in
		(* during hallucination, the edit list is be drained dry: 
		we apply the edits (thereby emptying the 1-element list),
		update the program encodings, 
		and ask the model to generate a new edit. *)
		(match typ with
		| "sub" -> bd.bedt.{u,0} <- 1.0
		| "del" -> bd.bedt.{u,1} <- 1.0
		| "ins" -> bd.bedt.{u,2} <- 1.0
		| "fin" -> bd.bedt.{u,3} <- 1.0
		| _ -> () ); 
		let ci = (Char.code c) - offs in
		if ci >= 0 && ci < toklen then 
			bd.bedt.{u,4+ci} <- 1.0 ; 
		(* position encoding *)
		if pp >= 0 && pp < p_ctx then (
			for i = 0 to poslen*2-1 do (
				bd.bedt.{u,5+toklen+i} <- bd.posenc.{pp,i}
			) done
		); 
	) bd.bea 

let bigarray_img2tensor img device = 
	let stride = (Bigarray.Array1.dim img) / image_res in
	let len = Bigarray.Array1.dim img in
	assert (len >= image_res * image_res); 
	let o = Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout image_res image_res in
	for i = 0 to image_res-1 do (
		for j = 0 to image_res-1 do (
			let c = Bigarray.Array1.get img ((i*stride)+j) |> foi in
			o.{i,j} <- c /. 255.0; 
		) done; 
	) done;
	o	|> Bigarray.genarray_of_array2 (* generic array; doesn't copy data? *)
		|> Tensor.of_bigarray  
		|> Torch.Tensor.to_device ~device
		
		
let bigarray_img2tensor_2 img device = 
	let stride = (Bigarray.Array1.dim img) / image_res in
	let len = Bigarray.Array1.dim img in
	assert (len >= image_res * image_res); 
	let o = Tensor.(zeros [image_res; image_res]) in
	let l = image_res - 1 in
	for i = 0 to l do (
		for j = 0 to l do (
			let c = Bigarray.Array1.get img ((i*stride)+j) in
			if c <> 0 then (
				let cf = foi c in
				(let open Tensor in
				o.%.{[i;j]} <- (cf /. 255.0); )
			)
		) done 
	) done;
	o |> Tensor.to_device ~device
	
	
let progenc_cost s = 
	String.fold_left (fun a b -> a + (Char.code b)) 0 s
	
let dbf_dist dbf img = 
	let d = Tensor.( (dbf - img) ) in
	let d = Tensor.einsum ~equation:"ijk, ijk -> i" [d;d] ~path:None in
	let mindex = Tensor.argmin d ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d mindex |> Tensor.float_value in
	dist,mindex
	
let init_database device db (dbf : Tensor.t) = 
	(* generate the initial program, image pairs *)
	Logs.info(fun m -> m  "init_database %d" image_count); 
	let i = ref 0 in
	let iters = ref 0 in
	let replace = ref 0 in
	while !i < image_count do (
		let data = if !i = 0 then
			generate_empty_logo !i image_res else
			generate_random_logo !i image_res in
		let img = bigarray_img2tensor data.img device in
		let dist,mindex = dbf_dist dbf img in
		if dist > 5. then (
			Logs.debug(fun m -> m 
				"%d: adding [%d] = %s" !iters !i 
				(Logo.output_program_pstr data.pro) ); 
			Vector.push db data; 
			Tensor.copy_ (Tensor.narrow dbf ~dim:0 ~start:!i ~length:1) ~src:img;
			incr i;
		) else (
			(* see if there's a replacement *)
			let data2 = Vector.get db mindex in
			let c1 = progenc_cost data.progenc in
			let c2 = progenc_cost data2.progenc in
			if c1 < c2 then (
				Logs.debug(fun m -> m 
					"%d: replacing [%d] = %s ( was %s)" !iters mindex 
					(Logo.output_program_pstr data.pro) 
					(Logo.output_program_pstr data2.pro));
				Vector.set db mindex data;
				Tensor.copy_ (Tensor.narrow dbf ~dim:0 ~start:mindex ~length:1) ~src:img;
				incr replace; 
			)
		); 
		if !iters mod 40 = 39 then 
			(* diminishing returns as this gets larger *)
			Caml.Gc.major (); 
		incr iters
	) done; 
	Logs.info(fun m -> m  "%d done; %d sampled; %d replacements" !i !iters !replace)
	
let save_database db = 
	let fil = open_out "db_prog.txt" in
	Printf.fprintf fil "%d\n" image_count; 
	Vector.iteri (fun i d -> 
		let pstr = Logo.output_program_pstr d.pro in
		Printf.fprintf fil "[%d] %s\n"  i pstr ) db ; 
	close_out fil; 
	Logs.app(fun m -> m  "saved %d to db_prog.txt" image_count); 
	
	(* verification .. human readable*)
	let fil = open_out "db_human_log.txt" in
	Printf.fprintf fil "%d\n" image_count; 
	Vector.iteri (fun i d -> 
		Printf.fprintf fil "\n[%d]\t" i ; 
		Logo.output_program_h d.pro fil) db ; 
	close_out fil; 
	Logs.debug(fun m -> m  "saved %d to db_human_log.txt" image_count)
	
let load_database device db dbf = 
	let lines = read_lines "db_prog.txt" in
	let a,progs = match lines with
		| h::r -> h,r
		| [] -> "0",[] in
	let ai = int_of_string a in
	if ai <> image_count then (
		Logs.err(fun m -> m "image_count mismatch, %d != %d" ai image_count); 
		false
	) else (
		List.iter (fun s -> 
			let sl = Pcre.split ~pat:"[\\[\\]]+" s in
			let _h,ids,ps = match sl with
				| h::m::t -> h,m,t
				| h::[] -> h,"",[]
				| [] -> "0","0",[] in
			let pid = int_of_string ids in
			let pr = if pid = 0 || (List.length ps) < 1
				then Some `Nop
				else parse_logo_string (List.hd ps) in
			(match pr with 
			| Some pro -> 
				let progenc = Logo.encode_program_str pro in
				let (_,_,segs) = Logo.eval (Logo.start_state ()) pro in
				let cost = segs_to_cost segs in
				let img,_ = Logo.segs_to_array_and_cost segs image_res in
				let data = {pid; pro; progenc; img; cost; segs} in
				Vector.push db data ; 
				let imgf = bigarray_img2tensor data.img device in
				Tensor.copy_ (Tensor.narrow dbf ~dim:0 ~start:pid ~length:1) ~src:imgf;
			| _ -> Logs.err(fun m -> m 
					"could not parse program %d %s" pid s ))
			) progs; 
		true
	)
	
(*
let () = 
	Unix.clear_nonblock stdin; 
	(*run_logo_file lg sout serr "semicircle.logo" ;*)
	Random.self_init (); 
	let lg = open_out "logo_Logs.txt" in
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
	let lg = open_out "logo_Logs.txt" in
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
	let lg = open_out "logo_Logs.txt" in
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

(* This should reach ~97% accuracy. *)
let hidden_nodes = 128
let epochs = 10000
let learning_rate = 1e-3

let test_torch () =
	Stdio.printf "cuda available: %b%!" (Cuda.is_available ());
	Stdio.printf "cudnn available: %b%!" (Cuda.cudnn_is_available ());
	let device = Torch.Device.cuda_if_available () in
	let mnist = Mnist_helper.read_files () in
	let { Dataset_helper.train_images; train_labels; _ } = mnist in
	let vs = Var_store.create ~name:"nn" ~device () in
	let linear1 =
		Layer.linear vs hidden_nodes ~activation:Relu ~input_dim:Mnist_helper.image_dim
	in
	let linear2 = Layer.linear vs Mnist_helper.label_count ~input_dim:hidden_nodes in
	let adam = Optimizer.adam vs ~learning_rate ~weight_decay:5e-5 in
	let model xs =
		Layer.forward linear1 xs
		|> Layer.forward linear2 in
	let img = train_images |> Torch.Tensor.to_device ~device in
	let lab = train_labels |> Torch.Tensor.to_device ~device in
	for index = 1 to epochs do
		(* Compute the cross-entropy loss. *)
		let loss =
			Tensor.cross_entropy_for_logits (model img) ~targets:lab
		in
		Optimizer.backward_step adam ~loss;
		if index mod 50 = 0
		then (
			(* Compute the validation error. *)
			let test_accuracy =
			Dataset_helper.batch_accuracy ~device mnist `test ~batch_size:1000 ~predict:model
			in
			Stdio.printf
			"%d %f %.2f%%\n%!"
			index
			(Tensor.float_value loss)
			(100. *. test_accuracy));
		Caml.Gc.full_major ()
	done
	
let progenc2progstr progenc = 
	progenc |> 
	Logo.string_to_intlist |> 
	Logo.decode_program 
	
let try_add_program state bi progenc = 
	let device,dba,dbf,fid = state in
	let progstr = progenc2progstr progenc in
	let g = parse_logo_string progstr in
	if bi mod 41 = 40 then Caml.Gc.major (); 
		(* clean up torch variables *)
	match g with
	| Some g2 -> (
		let pd = program_to_pdata g2 9999 image_res in
		match pd with
		| Some data -> (
			(*Logs.info (fun m -> m "Parsed! [%d]: %s \"%s\"" bi progenc progstr);*)
			let img = bigarray_img2tensor data.img device in
			let dist,mindex = dbf_dist dbf img in
			(* idea: if it's similar to a currently stored program, but has been hallucinated by the network, and is not too much more costly than what's there, 
			then replace the entry! *)
			if dist < 1.2 then (
				let data2 = dba.(mindex) in
				let c1 = progenc_cost data.progenc in
				let c2 = progenc_cost data2.progenc in
				if c1 < c2 then (
					let progstr2 = progenc2progstr data2.progenc in
					Logs.info (fun m -> m "#%d b:%d replacing equivalents [%d] %s with %s" !nreplace bi mindex progstr2 progstr);
					dba.(mindex) <- data; 
					Tensor.copy_ (Tensor.narrow dbf ~dim:0 ~start:mindex ~length:1) ~src:img; 
					Printf.fprintf fid "(%d) [%d] %s --> %s\n"
						!nreplace mindex progstr2 progstr; 
					flush fid; 
					Logo.segs_to_png data2.segs 64
					 (Printf.sprintf "/tmp/png/%05d_old.png" !nreplace);
					Logo.segs_to_png data.segs 64
					 (Printf.sprintf "/tmp/png/%05d_new.png" !nreplace);
					incr nreplace
					(* those two operations are in-place, so subsequent batches should contain the new program :-) *)
				)
			) )
			| _ -> () )
		| _ -> ()
	
let handle_message state bd msg =
	let _device,dba,_dbf,_fid = state in
	let l = String.length msg in
	let i = try String.index_from msg 0 ':' 
		with _ -> l in
	let cmd = String.sub msg 0 i in
	let data = if i >= l-1 then "" 
		else String.sub msg (i+1) (l-i-1) in
	match cmd with
	| "update_batch" -> (
		(* sent when python has a copy*)
		let bd = update_bea dba bd in
		bigfill_batchd dba bd; 
		Logs.debug(fun m -> m "new batch"); 
		bd,(Printf.sprintf "ok %d" !nreplace)
		)
	| "reset_batch" -> (
		let bea,fresh = reset_bea () in (* clear it *)
		let bd = {bd with bea;fresh} in
		let bd = update_bea dba bd in (* fill new entries *)
		bigfill_batchd dba bd; 
		bd,"batch has been reset."
		)
	| "edit_types" -> (
		(* these are ascii-encoded *)
		let typl = String.fold_left (fun a b -> 
			let typ = match b with
				| '0' -> "sub"
				| '1' -> "del"
				| '2' -> "ins"
				| _   -> "fin" in
				typ :: a) [] data in
		let typa = Array.of_list typl in
		let bea = Array.mapi (fun i be -> 
			let typ = typa.(i) in
			{be with edits=[(typ,0,'0')]} ) bd.bea in
		{bd with bea},"got edit types."
		)
	| "edit_pos" -> (
		let offs = Char.code '0' in
		let posl = String.fold_left (fun a b -> 
			let p = (Char.code b) - offs in
			p :: a) [] data in
		let posa = Array.of_list posl in
		let bea = Array.mapi (fun i be -> 
			let pos = posa.(i) in
			let typ,_,chr = List.hd be.edits in
			{be with edits=[(typ,pos,chr)]} ) bd.bea in
		{bd with bea},"got edit pos."
		)
	| "edit_chars" -> (
		let chrl = String.fold_left 
			(fun a b -> b :: a) [] data in
		let chra = Array.of_list chrl in
		let bea = Array.mapi (fun i be -> 
			let chr = chra.(i) in
			let typ,pos,_ = List.hd be.edits in
			{be with edits=[(typ,pos,chr)]} ) bd.bea in
		{bd with bea},"got edit chars."
		)
	| "apply_edits" -> (
		let bea = Array.mapi (fun i be -> 
			bd.fresh.(i) <- false; 
			apply_edits be ) bd.bea in
		let bd = {bd with bea} in
		bigfill_batchd dba bd; 
		(*Logs.info (fun m -> m "apply_edits");*) 
		bd,"applied the edits."
		)
	| "print_progenc" -> (
		Logs.info (fun m -> m "c_progenc[] "); (* FIXME debug *)
		Array.iteri (fun i be -> 
			try_add_program state i be.c_progenc ) bd.bea; 
		bd,"printed."
		)
	| _ -> bd,"Unknown command"

let rec handle_connection state bd fdlist ic oc () =
	(* init batch datastructures *)
	Lwt_io.read_line_opt ic >>=
	(fun msg ->
		match msg with
		| Some msg -> (
			let bd,reply = handle_message state bd msg in
			(*Logs.info (fun m -> m "%s" reply);*) 
			Lwt_io.write_line oc reply 
			>>= handle_connection state bd fdlist ic oc )
		| None -> (
			List.iter (fun fd -> Unix.close fd) fdlist; 
			Logs_lwt.info (fun m -> m "Connection closed")
			>>= return) )
	
let accept_connection state conn =
	let device,dba,dbf = state in
	let fd, _ = conn in
	let ic = Lwt_io.of_fd ~mode:Lwt_io.Input fd  in
	let oc = Lwt_io.of_fd ~mode:Lwt_io.Output fd in
	let fid = open_out "/tmp/png/replacements_log.txt" in
	let state2 = device,dba,dbf,fid in
	let bd,fdlist = init_batchd () in
	let bd = update_bea dba bd in
	bigfill_batchd dba bd ; 
	Lwt.on_failure (handle_connection state2 bd fdlist ic oc ()) 
		(fun e -> 
			Logs.err (fun m -> m "%s" (Printexc.to_string e) );
			Printexc.print_backtrace stdout; 
			flush stdout; 
			List.iter (fun fd -> Unix.close fd) fdlist;
			close_out fid);
	Logs_lwt.info (fun m -> m "New connection") 
	>>= return
	
let create_socket () =
	let open Lwt_unix in
	let sock = socket PF_INET SOCK_STREAM 0 in
	let adr = ADDR_INET(listen_address, port) in
	Lwt.async( fun () -> bind sock adr );
	listen sock backlog;
	sock

let create_server state sock =
	let rec serve () =
		Lwt_unix.accept sock >>= accept_connection state >>= serve
	in serve

let usage_msg = "program.exe -b <batch_size>"
let input_files = ref []
let output_file = ref ""
let anon_fun filename =
  input_files := filename :: !input_files
let speclist =
  [("-b", Arg.Set_int batch_size, "Training batch size");
   ("-o", Arg.Set_string output_file, "Set output file name")]

let () = 
	Arg.parse speclist anon_fun usage_msg;
	Random.self_init (); 
	let () = Logs.set_reporter (Logs.format_reporter ()) in
	let () = Logs.set_level (Some Logs.Info) in
	(* App, Error, Warning, Info, Debug *)

	Logs.info(fun m -> m "batch_size:%d" !batch_size);
	Logs.info(fun m -> m "cuda available: %b%!" 
				(Cuda.is_available ()));
	Logs.info(fun m -> m "cudnn available: %b%!"
				(Cuda.cudnn_is_available ()));
	let device = Torch.Device.cuda_if_available () in
	(*let device = Torch.Device.Cpu in*) (* slower *)
	let db = Vector.create ~dummy:nulpdata in
	let dbf = Tensor.( 
		( ones [image_count; image_res; image_res] ) * (f (-1.0))) 
		|> Tensor.to_device ~device in
	
	let g = if Sys.file_exists "db_prog.txt" 
		then load_database device db dbf 
		else false in
	if g then (
		Logs.app(fun m -> m "Loaded %d programs from db_prog.txt" image_count); 
	) else (
		Logs.app(fun m -> m "Generating %d programs" image_count);
		let start = Unix.gettimeofday () in
		init_database device db dbf;
		let stop = Unix.gettimeofday () in
		Logs.app(fun m -> m "Execution time: %fs\n%!" (stop -. start)); 
	
		Logs.info(fun m -> m "render_simplest: first 10 programs");
		for i = 0 to 9 do (
			let p = Vector.get db i in
			Logs.info(fun m -> m "%d: %s\n" i
					(Logo.output_program_pstr p.pro)); 
		) done; 
		save_database db ; 
	); 
	let dba = render_simplest db false in (* don't sort -- criteria is different! *)
	
	let sock = create_socket () in
	let state = device,dba,dbf in
	let serve = create_server state sock in
	Lwt_main.run @@ serve () 

(*
let image_dist dbf img = 
	let d = Tensor.( (dbf - img) ) in
	(* per-element square and sum *)
	let d = Tensor.einsum ~equation:"ijk,ijk -> i" [d;d] ~path:None in
	let mindex = Tensor.argmin d ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d mindex |> Tensor.float_value in
	dist,mindex

let () = 
	Unix.clear_nonblock stdin; 
	Printf.printf "cuda available: %b\n%!" (Cuda.is_available ());
	let device = Torch.Device.cuda_if_available () in
	(* dbf is a tensor of images to be compared (MSE) against *)
	let dbf = Tensor.( 
		( ones [image_count; image_res; image_res] ) * (f (-1.0))) 
		|> Tensor.to_device ~device in
	let start = Unix.gettimeofday () in
	for i = 0 to 100_000 do (
		(* generate a random image *)
		let img = Tensor.(randn [image_res; image_res] ) 
			|> Tensor.to_device ~device in
		ignore( image_dist dbf img ); 
		if i mod 30 = 29 then 
			Caml.Gc.full_major()
		(* in the actual program, we do something with dist,mindex *)
	) done; 
	let stop = Unix.gettimeofday () in
	Printf.printf "100k image_dist calc time: %fs\n%!" 
		(stop -. start);
*)
(*
let () = 
	Unix.clear_nonblock stdin; 
	Random.self_init (); 

	let lg = open_out "logo_Logs.txt" in
	let ic = in_channel_of_descr stdin in
	let sout = out_channel_of_descr stdout in
	let serr = out_channel_of_descr stderr in 
	let channels = (sout, serr, ic, lg) in
	Printf.fprintf lg "hello\n";
	
	
	let db = Vector.create ~dummy:nulpdata in
	let dba = Array.make 1 nulpdata in
	(*
	Printf.printf "cuda available: %b\n%!" (Cuda.is_available ());
	Printf.printf "cudnn available: %b\n%!" (Cuda.cudnn_is_available ());
	let device = Torch.Device.cuda_if_available () in
	(*let device = Torch.Device.Cpu in*) (* slower *)
	let dbf = Tensor.( 
		( ones [image_count; image_res; image_res] ) * (f (-1.0))) 
		|> Tensor.to_device ~device in
	init_database channels device db dbf;*) 
	loop_random channels 0 db dba;
	(*let _b = make_batch lg dba 1 in*)

	close_out sout; 
	close_out serr;
	close_out lg; 
*)
