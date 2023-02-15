(*open Core :-X core doesnt yet support byte output, necessary for protobufs*) (* but we don't use protobufs anymore *)
open Lexer
open Lexing
open Printf
(*open Unix*)
open Logo
open Torch

module Dtask = Domainslib.Task

type pequiv = 
	{ epro : Logo.prog
	; eprogenc : string
	; eimg : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t
	; escost : float
	; epcost : int
	; esegs : Logo.segment list
	}

type pdata = (* db is a Vector of pdata *)
	{ pid : int
	; pro  : Logo.prog
	; progenc : string
	; img  : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t
	; scost : float (* segment cost *)
	; pcost : int (* program cost *)
	; segs : Logo.segment list
	; equiv : pequiv list
	}
	
let nulpdata = 
	{ pid = -1
	; pro  = `Nop 
	; progenc = ""
	; img  = Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout 1 1
	; scost = -1.0 (* sequence cost *)
	; pcost = -1 (* program cost *)
	; segs = [] 
	; equiv = []
	}
	
let nulpequiv = 
	{ epro = `Nop
	; eprogenc = ""
	; eimg = Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout 1 1
	; escost = -1.0
	; epcost = -1
	; esegs = []
	}

type batche = (* batch edit structure *)
	{ a_pid : int 
	; b_pid : int
	; a_progenc : string (* from *)
	; b_progenc : string (* to; something if supervised; blank if dream *)
	; c_progenc : string (* program being edited *)
	; edits : (string*int*char) list (* supervised *)
	; edited : float array (* tell python which chars have been changed*)
	; count : int (* count & cap total # of edits *)
	; indx : int
	}
	
let nulbatche = 
	{ a_pid = 0
	; b_pid = 0
	; a_progenc = ""
	; b_progenc = ""
	; c_progenc = ""
	; edits = []
	; edited = [| 0.0 |]
	; count = 0 
	; indx = 0
	}
	
type batchd = (* batch data structure *)
	{ bpro : (float, Bigarray.float32_elt, Bigarray.c_layout)
				Bigarray.Array3.t (* btch , p_ctx , p_indim *)
	; bimg : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array3.t (* btch*3, image_res, image_res *)
	; bedts : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array2.t (* btch , e_indim *)
	; bedtd : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array2.t (* btch , e_indim *)
	; posenc : (float, Bigarray.float32_elt, Bigarray.c_layout) 
				Bigarray.Array2.t (* p_ctx , poslen*2 *)
	; posencn : Tensor.t (* normalized position encoding *)
	; bea : batche array
	; fresh : bool array
	}
	
type dreamcheck = 
	{ be : batche
	; mutable decode : string list
	; mutable correct_cnt : int
	}
	
let nuldream = 
	{ be = nulbatche
	; decode = []
	; correct_cnt = 0
	}
	
type tsteak = (* thread state *)
	{ device : Torch.Device.t
	; db : pdata Vector.t
	; dbf : Tensor.t
	; mnist : Tensor.t
	; dbf_enc : Tensor.t
	; mnist_enc : Tensor.t
	; vae : Vae.VAE.t
	; db_mutex : Mutex.t
	; superv : bool (* supervised or generative *)
	; sockno : int
	; fid : out_channel (* log for e.g. replacements *)
	; mutable batchno : int (* for e.g doing garbage collection *)
	; mutable pool : Domainslib.Task.pool (* needs to be replaced for dream *)
	; mutable dreamn : int
	; dreams : dreamcheck array option
	}
	
let pi = 3.1415926
let image_count = 6*2048*2
let image_res = 30
let batch_size = ref (256*3)
let toklen = 30
let poslen = 6
let p_indim = toklen + 1 + poslen*2 (* 31 + 12 = 43 *)
let e_indim = 5 + toklen + poslen*2
let p_ctx = 64
let nreplace = ref 0 (* number of repalcements during hallucinations *)
let glive = ref true 
let gparallel = ref false

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
	| SyntaxError _msg ->
		(*Logs.debug (fun m -> m "%s: %s" 
			(print_position lexbuf) msg);*)  (* these errors overwhelm while dreaming *)
		None
	| Parser.Error ->
		(*Logs.debug (fun m -> m "%s: syntax error" 
			(print_position lexbuf));*)
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
	
let progenc_cost s = 
	String.fold_left (fun a b -> a + (Char.code b)) 0 s
	
let program_to_pdata pro id res = 
	let progenc = Logo.encode_program_str pro in
	let (_,_,segs) = Logo.eval (Logo.start_state ()) pro in
	let scost = segs_to_cost segs in
	let pcost = progenc_cost progenc in
	let lx,hx,ly,hy = segs_bbx segs in
	let dx = hx-.lx in
	let dy = hy-.ly in
	let maxd = max dx dy in
	if maxd >= 2. && maxd <= 9. && scost >= 4. && scost <= 64. && List.length segs < 8 && String.length progenc < 24 then (
		let img, _ = Logo.segs_to_array_and_cost segs res in
		let equiv = [] in
		Some {pid=id; pro; progenc; img; 
				scost; pcost; segs; equiv}
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
	let scost = 0.0 in
	let pcost = 0 in
	let img, _ = Logo.segs_to_array_and_cost segs res in
	let equiv = [] in
	{pid=id; pro; progenc; img; 
	scost; pcost; segs; equiv}

let render_simplest db =
	(* render the shortest 4*1024 programs in the database.*)
	let dbl = Vector.length db in
	let res = 48 in
	let lg = open_out "/tmp/png/db_init.txt" in
	for id = 0 to min (4*1024-1) (dbl-1) do (
		let data = Vector.get db id in
		let (_,_,segs) = Logo.eval (Logo.start_state ()) data.pro in
		Logo.segs_to_png segs res (Printf.sprintf "/tmp/png/test%d.png" id);
		let bf = Buffer.create 30 in
		Logo.output_program_p bf data.pro;
		fprintf lg "%d %s\n" id (Buffer.contents bf);
	) done;
	close_out lg
	
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
	
(* both these are exclusively float32 *)
let bigarray2_of_tensor m = 
	let c = Tensor.to_bigarray ~kind:Bigarray.float32 m in
	Bigarray.array2_of_genarray c
	
let tensor_of_bigarray2 m device =
	let o = Tensor.of_bigarray (Bigarray.genarray_of_array2 m) in
	o |> Tensor.to_device ~device
	
(*let tensor_of_bigarray1img img device = 
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
		|> Torch.Tensor.to_device ~device*)
		
let db_get steak i = 
	Mutex.lock steak.db_mutex ; 
	let r = Vector.get steak.db i in
	Mutex.unlock steak.db_mutex ; 
	r
	
let db_set steak i d img = 
	Mutex.lock steak.db_mutex ; 
	Vector.set steak.db i d ;
	Tensor.copy_ (Tensor.narrow steak.dbf ~dim:0 ~start:i ~length:1) ~src:img; 
	Mutex.unlock steak.db_mutex

let db_push steak d imgf = 
	(*imgf is a tensor on same device as dbf*)
	let added = ref false in
	Mutex.lock steak.db_mutex; 
	let l = Vector.length steak.db in
	if l < image_count then (
		Vector.push steak.db d ;
		Tensor.copy_ (Tensor.narrow steak.dbf ~dim:0 ~start:l ~length:1) ~src:imgf;
		added := true
	); 
	Mutex.unlock steak.db_mutex; 
	!added,l
	
let db_len steak = 
	Mutex.lock steak.db_mutex ; 
	let r = Vector.length steak.db in
	Mutex.unlock steak.db_mutex ; 
	r

let dbf_dist steak img = 
	(* using Cosine Similarity *)
	let ir = image_res*image_res in
	let a = Tensor.view steak.dbf ~size:[-1;ir] in
	let b = Tensor.view img ~size:[1;-1] 
		|> Tensor.expand ~implicit:false ~size:[image_count; ir] in
	let d = Tensor.cosine_similarity ~x1:a ~x2:b ~dim:1 ~eps:1e-7 in
	let maxdex = Tensor.argmax d ~dim:0 ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d maxdex |> Tensor.float_value in
	abs_float (1.0-.dist),maxdex
	
let dbf_dist2 steak img = 
	(*Mutex.lock steak.db_mutex;*) 
	let d = Tensor.( (steak.dbf - img) ) in (* broadcast *)
	(* this is not good: allocates a ton of memory!! *)
	(* what we need to do is substract, square, sum. at the same time *)
	(*Mutex.unlock steak.db_mutex;*) (* we have a copy *)
	let d = Tensor.einsum ~equation:"ijk,ijk -> i" [d;d] ~path:None in
	let mindex = Tensor.argmin d ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d mindex |> Tensor.float_value in
	dist,mindex
	

let tensor_of_bigarray1img img device = 
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
	
let normalize_tensor m = 
	(* normalizes length along dimension 1 *)
	let len = Tensor.einsum ~equation:"ij,ij -> i" [m;m] ~path:None 
			|> Tensor.sqrt in
	let len = Tensor.(f 1. / len) in
	Tensor.einsum ~equation:"ij,i -> ij" [m;len] ~path:None 
	
let reset_bea () = 
	let bea = Array.init !batch_size (fun _i -> nulbatche) in
	let fresh = Array.init !batch_size (fun _i -> true) in
	bea,fresh
	
let init_batchd filnum =
	Logs.debug (fun m -> m "init_batchd"); 
	let mkfnam s = 
		Printf.sprintf "%s_%d.mmap" s filnum in
	let fd_bpro,bpro = mmap_bigarray3 (mkfnam "bpro") 
			!batch_size p_ctx p_indim in
	let fd_bimg,bimg = mmap_bigarray3 (mkfnam "bimg") 
			(!batch_size*3) image_res image_res in
			(* note: needs a reshape on python side!! *)
	(* supervised batch of edits: ocaml to python *)
	let fd_bedts,bedts = mmap_bigarray2 (mkfnam "bedts") 
			!batch_size e_indim in
	(* hallucinated batch of edits: python to ocaml *)
	let fd_bedtd,bedtd = mmap_bigarray2 (mkfnam "bedtd") 
			!batch_size e_indim in
	let fd_posenc,posenc = mmap_bigarray2 (mkfnam "posenc")
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
	let device = Torch.Device.Cpu in
	let posencn = tensor_of_bigarray2 posenc device |> normalize_tensor in
	Bigarray.Array3.fill bpro 0.0 ;
	Bigarray.Array3.fill bimg 0.0 ;
	Bigarray.Array2.fill bedts 0.0 ;
	Bigarray.Array2.fill bedtd 0.0 ; 
	(* return batchd struct & list of files to close later *)
	{bpro; bimg; bedts; bedtd; posenc; posencn; bea; fresh}, 
	[fd_bpro; fd_bimg; fd_bedts; fd_bedtd; fd_posenc]
	
let progenc_to_edits a b = 
	let dist, edits = Levenshtein.distance a.progenc b.progenc true in
	let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
	(* verify .. a bit of overhead *)
	(*let re = Levenshtein.apply_edits a.progenc edits in
	if re <> b.progenc then (
		Logs.err(fun m -> m  
			"error! %s edits should be %s was %s"
			a.progenc b.progenc re)
	);*)
	(* edits are applied in reverse *)
	(* & add a 'done' edit/indicator *)
	let edits = ("fin",0,'0') :: edits in
	dist, (List.rev edits)
	
let edit_criteria edits dosub = 
	let count_type typ = 
		List.fold_left 
		(fun a (t,_p,_c) -> if t = typ then a+1 else a) 0 edits 
	in
	let nsub = count_type "sub" in
	let ndel = count_type "del" in
	let nins = count_type "ins" in
	let r = ref false in
	if dosub then (
		if nsub = 1 && ndel = 0 && nins = 0 then r := true 
	) else (
		if nsub = 0 && ndel <= 6 && nins = 0 then r := true; 
		if nsub = 0 && ndel = 0 && nins <= 6 then r := true
	);
	!r
	
let rec new_batche steak bn dosub = 
	(* only supervised mode! *)
	let ndb = db_len steak in
	let nb = (Random.int (ndb-1)) + 1 in
	(*let na = if doedit then (Random.int (ndb-1)) + 1 else 0 in*)
	let na = (Random.int ndb) in
	(* small portion of the time na is the empty program *)
	let a = db_get steak na in
	let b = db_get steak nb in
	let a_ns = List.length a.segs in
	let b_ns = List.length b.segs in
	let a_np = String.length a.progenc in
	let b_np = String.length b.progenc in
	let lim = (p_ctx/2)-4 in 
	(* check this .. really should be -2 as in bigfill_batchd *)
	if a_ns <= 8 && b_ns <= 8 && a_np < lim && b_np < lim then (
		let dist,edits = progenc_to_edits a b in
		(*Logs.debug(fun m -> m  
			"trying [%d] [%d] for batch; dist %d" na nb dist);*)
		if edit_criteria edits dosub && dist > 0 then (
			(*Logs.debug(fun m -> m 
				"|%d adding [%d] %s [%d] %s to batch; dist:%d"
				bn na a.progenc nb b.progenc dist);*)
			let edited = Array.make p_ctx 0.0 in
			{a_pid=na; b_pid=nb; 
				a_progenc = a.progenc; 
				b_progenc = b.progenc; 
				c_progenc = a.progenc; 
				edits; edited; count=0; indx=na} (* FIXME indx *)
		) else new_batche steak bn dosub
	) else new_batche steak	bn dosub

(*let rec new_batche doedit db = 
	(* only supervised mode! *)
	let ndb = (Vector.length db) in
	let nb = (Random.int (ndb-1)) + 1 in
	let na = if doedit then (Random.int (ndb-1)) + 1 else 0 in
	let distthresh = if doedit then 5 else 15 in (* FIXME: 4,12 -> 5,15 *)
	(*let na = if (Random.int 10) = 0 then 0 else (Random.int nb) in*)
	(* small portion of the time na is the empty program *)
	let a = Vector.get db na in
	let b = Vector.get db nb in
	let a_ns = List.length a.segs in
	let b_ns = List.length b.segs in
	let a_np = String.length a.progenc in
	let b_np = String.length b.progenc in
	let lim = (p_ctx/2)-4 in 
	(* check this .. really should be -2 as in bigfill_batchd *)
	if a_ns <= 8 && b_ns <= 8 && a_np < lim && b_np < lim then (
		let dist,edits = Levenshtein.distance a.progenc b.progenc false in
		(*Logs.debug(fun m -> m  
			"trying [%d] [%d] for batch; dist %d" na nb dist);*)
		if dist > 0 && dist < distthresh then (
			let edits = progenc_to_edits a b in
			(*Logs.debug(fun m -> m 
				"adding [%d] %s [%d] %s to batch (unsorted pids: %d %d)"
				na a.progenc nb b.progenc a.pid b.pid);*)
			let edited = Array.make p_ctx 0.0 in
			{a_pid=na; b_pid=nb; 
				a_progenc = a.progenc; 
				b_progenc = b.progenc; 
				c_progenc = a.progenc; 
				edits; edited; count=0; indx=na} (* FIXME indx *)
		) else new_batche doedit db
	) else new_batche doedit db*)
	
let new_batche_sup steak bn = 
	(* supervised only -- use this *)
	(* generate mode lasts longer, hence probabilities need to be adjusted *)
	let dosub = (Random.int 10) < 5  in
	new_batche steak bn dosub

	
let dbf_to_png bigt i filename = 
	let dbfim = Tensor.narrow bigt ~dim:0 ~start:i ~length:1 in
	Torch_vision.Image.write_image Tensor.((f 1. - dbfim) * f 255.) ~filename

(* let inspect_counter = Atomic.make 0  *)

let new_batche_unsup steak = 
	(* for now, just set the target B to a sample from MNIST; ultimately will need to have longer interactions & intermediate starting points *)
	let mid = Random.int 60000 in
	(* TODO: select a starting point closer to the target, w/threshold.
		goal is conflated with longer interactions, guess ? *)
	let dbfn,cols = Tensor.shape2_exn steak.dbf_enc in
	let b = Tensor.narrow steak.mnist_enc ~dim:0 ~start:mid ~length:1 
			|> Tensor.expand ~implicit:false ~size:[dbfn;cols] in
	let d = Tensor.cosine_similarity ~x1:steak.dbf_enc ~x2:b ~dim:1 ~eps:1e-7 in
	(* add a bit of noise .. ?? *)
	let d = Tensor.( d + (f 0.05 * (randn [image_count;]))) in
	assert ((Tensor.shape1_exn d) = image_count) ; (* sanity.. *)
	let indx = Tensor.argmax d ~dim:0 ~keepdim:true |> Tensor.int_value in
	let a = db_get steak indx in
	let edited = Array.make p_ctx 0.0 in
	(* output these subs for inspection *)
	(*let filename = Printf.sprintf "/tmp/png/b%05d_target.png" (Atomic.get inspect_counter) in
	dbf_to_png steak.mnist mid filename; 
	Logo.segs_to_png a.segs 64
		(Printf.sprintf "/tmp/png/b%05d_closest.png" (Atomic.get inspect_counter));
	Atomic.incr inspect_counter ; *)
	{ a_pid = indx; b_pid = image_count + mid; 
		a_progenc = a.progenc; 
		b_progenc = ""; 
		c_progenc = a.progenc; 
		edits = []; edited; count=0; indx=mid}
	(* note: bd.fresh is set in the calling function *)
		
(*let new_batche_dream_x steak = (* use this *)
	if (Random.int 10) < 7 then (
		new_batche_sup steak
	) else (
		new_batche_unsup steak
	)*)
	
let new_batche_dream steak = (* use this *)
	(* not threaded !! *)
	match steak.dreams with
	| Some dreams -> (
		let i = steak.dreamn in
		let d = dreams.(i) in
		steak.dreamn <- (i+1) mod (Array.length dreams); 
		let edited = Array.make p_ctx 0.0 in (* memory thrash *)
		{d.be with edited} )
	| None -> ( nulbatche )
	
let update_edited be ed = 
	(* update the 'edited' array, which indicates what has changed in the program string *)
	let typ,pp,_chr = ed in 
	let la = String.length be.a_progenc in
	let lc = String.length be.c_progenc in
	(* in-place array modification *)
	(match typ with 
	| "sub" -> (
		let pp = if pp > lc-1 then lc-1 else pp in
		let pp = if pp < 0 then 0 else pp in
		be.edited.(la+pp) <- 0.5 )
	| "del" -> (
		(* lc is already one less at this point -- c has been edited *)
		let pp = if pp > lc-1 then lc-1 else pp in
		let pp = if pp < 0 then 0 else pp in
		(* shift left *)
		for i = la+pp to p_ctx-2 do (
			be.edited.(i) <- be.edited.(i+1)
		) done; 
		be.edited.(la+pp) <- ( -1.0 ) )
	| "ins" -> (
		if lc < p_ctx/2 && la+pp < p_ctx then (
			let pp = if pp > lc then lc else pp in
			let pp = if pp < 0 then 0 else pp in
			(* shift right one *)
			for i = p_ctx-2 downto la+pp do (
				be.edited.(i+1) <- be.edited.(i)
			) done; 
			be.edited.(la+pp) <- 1.0 ) )
	| _ -> () )

(* seems like this would be much faster if simply implemented directly in ocaml with bigarrays!! *)
(* also, simpler and more comprehensible ... and might as well use the parallel pool for this *)
(* eh .. this scales easily to very large batches *)
(* TODO: softmax + temperature decoding (via torch *)
	
let decode_edit bd ba_edit = 
	(* decode model output (from python) *)
	let sta = Unix.gettimeofday () in
	let device = Torch.Device.Cpu in
	let m = tensor_of_bigarray2 ba_edit device in
	(* typ = th.argmax(y[:,0:4], 1)  (0 is the batch dim) *)
	let typ = Tensor.argmax (Tensor.narrow m ~dim:1 ~start:0 ~length:4) 
			~dim:1 ~keepdim:false in
	let chr = (Tensor.argmax (Tensor.narrow m ~dim:1 ~start:4 ~length:toklen)
			~dim:1 ~keepdim:false) in
	(* now need to compute cosine distance.  normalize vectors first *)
	let pos = Tensor.narrow m ~dim:1 ~start:(5+toklen) ~length:(poslen*2)
			|> normalize_tensor in (* wait where does 5 come from? *)
	let pos = Tensor.expand pos ~size:[p_ctx;!batch_size;poslen*2] ~implicit:true in
	let posenc = Tensor.expand bd.posencn ~size:[!batch_size;p_ctx;poslen*2] ~implicit:true in
	let sim = Tensor.einsum ~equation:"cbp,bcp -> bc" [pos;posenc] ~path:None in
	let loc = Tensor.argmax sim ~dim:1 ~keepdim:false in
	(* location must be clipped to within the program. *)
	let edit_arr = Array.init !batch_size (fun i -> 
		let etyp = match Tensor.get_int1 typ i with
			| 0 -> "sub"
			| 1 -> "del" 
			| 2 -> "ins"
			| _ -> "fin" in
		let echr = (Tensor.get_int1 chr i) + Char.code('0') |> Char.chr in
		let eloc = Tensor.get_int1 loc i in
		(etyp,eloc,echr) ) in
	(* debug *)
	(*for i = 0 to (min 1 !batch_size) do (
		let typ,loc,chr = edit_arr.(i) in
		Logs.debug (fun m -> m "decode_edit %d: %s,%c,%d" i typ chr loc)
	) done;*)
	let fin = Unix.gettimeofday () in
	Logs.debug (fun m -> m "decode_edit time %f" (fin-.sta)); 
	edit_arr
	
let apply_edits be = 
	(* apply after (of course) sending to python. *)
	(* edit length must be > 0 *)
	(* updates batche ds; pops first edit *)
	let ed = List.hd be.edits in
	(* note: Levenshtein clips the edit positions *)
	let c = Levenshtein.apply_edits be.c_progenc [ed] in
	let be2 = { be with c_progenc=c; edits=(List.tl be.edits) } in
	update_edited be2 ed; 
	be2
	
let progenc2progstr progenc = 
	progenc |> 
	Logo.string_to_intlist |> 
	Logo.decode_program 
	
let pdata_to_edata p = 
	{ epro = p.pro
	; eprogenc = p.progenc
	; eimg = p.img
	; escost = p.scost
	; epcost = p.pcost
	; esegs = p.segs }
	
let better_counter = Atomic.make 0
	
let try_add_program steak progenc be = 
	let progstr = progenc2progstr progenc in
	let g = parse_logo_string progstr in
	match g with
	| Some g2 -> (
		let pd = program_to_pdata g2 99999 image_res in
		match pd with
		| Some data -> (
			(*Logs.info (fun m -> m "Parsed! [%d]: %s \"%s\"" bi progenc progstr);*)
			let imgf = tensor_of_bigarray2 data.img steak.device in
			let dist,mindex = dbf_dist steak imgf in
			steak.batchno <- steak.batchno + 1 ; 
			(* idea: if it's similar to a currently stored program, but has been hallucinated by the network, and is not too much more costly than what's there, 
			then replace the entry! *)
			if dist < 0.006 then (
				let data2 = db_get steak mindex in
				let c1 = data.pcost in
				let c2 = data2.pcost in
				if c1 < c2 then (
					let progstr2 = progenc2progstr data2.progenc in
					Logs.info (fun m -> m "#%d b:%d replacing equivalents [%d] %s with %s" !nreplace steak.batchno mindex progstr2 progstr);
					let ed = pdata_to_edata data2 in
					let data = {data with equiv = (ed :: data2.equiv)} in
					db_set steak mindex data imgf; 
					Printf.fprintf steak.fid 
						"(%d) [%d] d:%f %s --> %s | pcost %d -> %d | scost %f -> %f\n"
						!nreplace mindex dist progstr2 progstr c2 c1
						data2.scost data.scost; 
					flush steak.fid; 
					Logo.segs_to_png data2.segs 64
					 (Printf.sprintf "/tmp/png/%05d_old.png" !nreplace);
					Logo.segs_to_png data.segs 64
					 (Printf.sprintf "/tmp/png/%05d_new.png" !nreplace);
					(* get the dbf entry too, to check *)
					let filename = Printf.sprintf 
						"/tmp/png/%05d_dbf.png" !nreplace in
					dbf_to_png steak.dbf mindex filename; 
					incr nreplace
					(* those two operations are in-place, so subsequent batches should contain the new program :-) *)
				)
			) ; 
			if dist > 5.0 then (
				let added,l = db_push steak data imgf in
				if added then (
					Logs.info(fun m -> m 
						"try_add_program: adding new [%d] = %s" l 
						(Logo.output_program_pstr data.pro) )
				) else (
					Logs.debug(fun m -> m 
						"try_add_program: could not add new, db full. [%d]" l )
				)
			) ; 
			(* new: see if the edits move the image closer to the mnist target *)
			if be.b_pid >= image_count then (
				(*let cpu = Torch.Device.Cpu in*)
				let a = db_get steak be.a_pid in
				let mid = be.b_pid - image_count in
				let aimg = tensor_of_bigarray2 a.img steak.device in
				let bimg = Tensor.narrow steak.mnist ~dim:0 ~start:mid ~length:1 
					|> Tensor.to_device ~device:steak.device in
				let cimg = imgf in
				let encode v = 
					Tensor.view v ~size:[image_res*image_res;] 
					|> Vae.encode1_ext steak.vae in
				let aenc = encode aimg in
				let benc = encode bimg in
				let cenc = encode cimg in
				let cos_ab = Tensor.cosine_similarity ~x1:aenc ~x2:benc ~dim:0 ~eps:1e-7 |> Tensor.float_value in
				let cos_cb = Tensor.cosine_similarity ~x1:cenc ~x2:benc ~dim:0 ~eps:1e-7 |> Tensor.float_value in
				let ab = Tensor.( mean((aimg - bimg) * (aimg - bimg)) ) 
					|> Tensor.float_value in
				let cb = Tensor.( mean((cimg - bimg) * (cimg - bimg)) ) 
					|> Tensor.float_value in 
				if cos_cb > cos_ab then (
					let q = Atomic.get better_counter in
					Logs.info (fun m -> m "Made an improvement! see %d; cos: %f --> %f ; mse: %f --> %f" q cos_ab cos_cb ab cb); 
					let filename = Printf.sprintf "/tmp/png/b%05d_a_target.png" q in
					dbf_to_png steak.mnist mid filename; 
					Logo.segs_to_png a.segs 64
						(Printf.sprintf "/tmp/png/b%05d_b_old.png" q);
					Logo.segs_to_png data.segs 64
						(Printf.sprintf "/tmp/png/b%05d_c_new.png" q);
					Atomic.incr better_counter
				)
			); 
			(*if dist >= 0.6 && dist <= 5.0 then (
				let data2 = db_get steak mindex in
				Logs.info(fun m -> m 
						"try_add_program: new \n\t%s sim existing\n\t%s"  
						(Logo.output_program_pstr data.pro)
						(Logo.output_program_pstr data2.pro) )
			)*) )
			| _ -> () )
		| _ -> ()

let update_bea_sup steak bd = 
	let sta = Unix.gettimeofday () in
	(* need to run through twice, first to apply the edits, second to replace the finished elements. *)
	let innerloop i = 
		let be = bd.bea.(i) in
		let be2 = if List.length be.edits > 0 then (
			bd.fresh.(i) <- false; 
			apply_edits be 
			) else be in
		let be3 = if List.length be.edits = 0 then (
			bd.fresh.(i) <- true; (* update image flag *)
			new_batche_sup steak i
			) else be2 in
		bd.bea.(i) <- be3 in (* /innerloop *)
	if !gparallel then
		Dtask.parallel_for steak.pool ~start:0 ~finish:(!batch_size-1)
			~body:innerloop
	else
		for i=0 to (!batch_size-1) do
			innerloop i done;
	let fin = Unix.gettimeofday () in
	Logs.debug (fun m -> m "update_bea_sup time %f" (fin-.sta));
	(* array update was in-place, so just return bd. *)
	bd
	
let update_bea_dream steak bd = 
	let sta = Unix.gettimeofday () in
	Logs.debug (fun m -> m "entering update_bea_dream");
	(* this one only needs one run-through. *)
	let edit_arr = decode_edit bd bd.bedtd in
	let innerloop i = 
		let be = bd.bea.(i) in
		let cnt = be.count in
		let typ,loc,chr = edit_arr.(i) in
		(* edited is initialized in update_be_sup/new_batche_sup; same here *)
		let edited = if Array.length be.edited <> p_ctx 
			then Array.make p_ctx 0.0 else be.edited in
		let be2 = {be with edits=[(typ,loc,chr)];count=cnt+1;edited} in
		let be3 = apply_edits be2 in
		let be4 = if typ = "fin" || be2.count >= p_ctx/2 then (
			(* log it! *)
			(match steak.dreams with
			| Some dreams -> (
				let a = be.a_pid in
				let b = be.b_pid in
				let i = be.indx in
				let s = progenc2progstr be.c_progenc in
				if b < image_count then (
					if be.c_progenc = be.b_progenc then ( 
						dreams.(i).decode <- (s :: dreams.(i).decode);
						dreams.(i).correct_cnt <- dreams.(i).correct_cnt+1; 
						Logs.debug (fun m -> m "dream:%d [%d]->[%d] %s decoded correctly." i a b s)
					) else (
						(* if wrong, save one example decode *)
						if (List.length dreams.(i).decode) = 0 then
						dreams.(i).decode <- (s :: dreams.(i).decode);
					) 
				) else (
					let mid = b - image_count in
					if mid < 60000 then (
						dreams.(i).decode <- (s :: dreams.(i).decode);
					)
				) )
			| None -> () ); 
			(*try_add_program steak be3.c_progenc be3;*) (* FIXME *)
			bd.fresh.(i) <- true; 
			new_batche_dream steak
		) else (
			bd.fresh.(i) <- false; 
			be3 
		) in
		bd.bea.(i) <- be4; 
	in (* /innerloop *)
	if !gparallel then (* this might work again? *)
		Dtask.parallel_for steak.pool ~start:0 ~finish:(!batch_size-1) 
			~body:innerloop
	else 
		for i = 0 to (!batch_size-1) do (* non-parallel version *)
			innerloop i done; 
	let fin = Unix.gettimeofday () in
	Logs.debug (fun m -> m "update_bea_dream time %f" (fin-.sta));
	Mutex.lock steak.db_mutex; 
(* 	Caml.Gc.major (); (* clean up torch variables *) *)
	Mutex.unlock steak.db_mutex; 
	let fin2 = Unix.gettimeofday () in
	Logs.debug (fun m -> m "update_bea_dream: Caml.Gc.major time %f;" 
			(fin2 -. fin)); 
	(* cannot do this within a parallel for loop!!! *)
	(* in-place update of bea *)
	bd
	
let bigfill_batchd steak bd = 
	let sta = Unix.gettimeofday () in
	(* convert bea to the 3 mmaped bigarrays *)
	(* first clear them *)
	(*Bigarray.Array3.fill bd.bimg 0.0 ;*)
	(* fill one-hots *)
	Bigarray.Array3.fill bd.bpro 0.0 ;
	Bigarray.Array2.fill bd.bedts 0.0 ; 
	for u = 0 to !batch_size-1 do (
		bd.bedts.{u,0} <- -1. (* TEST (checked via min, python side) *)
	) done; 
	let innerloop u = 
		let be = bd.bea.(u) in
		(* - bpro - *)
		let offs = Char.code '0' in
		let llim = (p_ctx/2)-2 in
		let l = String.length be.a_progenc in
		if l > llim then 
			Logs.debug(fun m -> m  "too long(%d):%s" l be.a_progenc);
		String.iteri (fun i c -> 
			let j = (Char.code c) - offs in
			if i < llim then bd.bpro.{u,i,j} <- 1.0 ) be.a_progenc ; 
		let lc = String.length be.c_progenc in
		if lc > llim then 
			Logs.debug(fun m -> m  "too long(%d):%s" lc be.c_progenc);
		String.iteri (fun i c -> 
			let j = (Char.code c) - offs in
			if (i+l) < 2*llim then (
				bd.bpro.{u,i+l,j} <- 1.0; 
				(* indicate this is c, to be edited *)
				bd.bpro.{u,i+l,toklen-1} <- 1.0 ) ) be.c_progenc ;
		(* copy over the edited tags (set in apply_edits) *)
		assert (Array.length be.edited = p_ctx) ; 
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
			let pid_to_ba2 pid = 
				if pid < image_count then (
					let a = db_get steak pid in
					a.img
				) else (
					let mid = pid - image_count in
					Tensor.narrow steak.mnist ~dim:0 ~start:mid ~length:1 
					|> Tensor.squeeze
					|> bigarray2_of_tensor
				) in
			assert (be.a_pid < image_count); 
			assert (be.b_pid < (image_count + 60000)); 
			let aimg = pid_to_ba2 be.a_pid in
			let bimg = pid_to_ba2 be.b_pid in
			for i=0 to (image_res-1) do (
				for j=0 to (image_res-1) do (
					let aif = aimg.{i,j} in
					let bif = bimg.{i,j} in
					bd.bimg.{3*u+0,i,j} <- aif; 
					bd.bimg.{3*u+1,i,j} <- bif;
					bd.bimg.{3*u+2,i,j} <- aif -. bif;
				) done
			) done ; 
			bd.fresh.(u) <- false
		); 
	  (* - bedts - *)
		bd.bedts.{u,0} <- 0.0; 
		let (typ,pp,c) = if (List.length be.edits) > 0 
			then List.hd be.edits
			else ("fin",0,'0') in
		(* during dreaming, the edit list is drained dry: 
		we apply the edits (thereby emptying the 1-element list),
		update the program encodings, 
		and ask the model to generate a new edit. *)
		(match typ with
		| "sub" -> bd.bedts.{u,0} <- 1.0
		| "del" -> bd.bedts.{u,1} <- 1.0
		| "ins" -> bd.bedts.{u,2} <- 1.0
		| "fin" -> bd.bedts.{u,3} <- 1.0
		| _ -> () ); 
		let ci = (Char.code c) - offs in
		if ci >= 0 && ci < toklen then 
			bd.bedts.{u,4+ci} <- 1.0 ; 
		(* position encoding *)
		if pp >= 0 && pp < p_ctx then (
			for i = 0 to poslen*2-1 do (
				bd.bedts.{u,5+toklen+i} <- bd.posenc.{pp,i}
			) done
		) in (* /innerloop *)
	if !gparallel then
		Dtask.parallel_for steak.pool ~start:0 ~finish:(!batch_size-1)
			~body:innerloop
	else
		for i=0 to (!batch_size-1) do
			innerloop i done; 
	let fin = Unix.gettimeofday () in
	Logs.debug (fun m -> m "bigfill_batchd time %f" (fin-.sta))
	
let truncate_list n l = 
	let n = min n (List.length l) in
	let a = Array.of_list l in
	Array.sub a 0 n |> Array.to_list

let sort_database device db =
	(* sorts the database & updates Tensor dbf *)
	(* sorts both primary keys and equivalent lists *)
	let dba = Vector.to_array db in (* easier .. *)
	Array.sort (fun a b -> compare a.pcost b.pcost) dba; 
	Logs.debug (fun m -> m "sort_database sort dba done.");
	(* remove duplicate equivalents *)
	let rec removeDup ls =
		if List.length ls > 1 then (
			let b = List.hd ls in
			let c = List.tl ls in
			if List.exists (fun d -> d.eprogenc = b.eprogenc) c
			then removeDup c
			else b :: (removeDup c)
		) else ls
	in
	(* sort the equivalents *)
	Array.iteri (fun i data -> 
		let dedup = removeDup data.equiv in
		let sl = List.sort (fun a b -> compare a.epcost b.epcost)
			dedup in
		(* remove head duplicates *)
		let sl = List.filter (fun a -> a.eprogenc <> data.progenc) sl in
		dba.(i) <- {data with equiv=sl} ) dba; 
	Logs.debug (fun m -> m "sort_database sort equivalents done.");
	let sta = Unix.gettimeofday () in
	let dbal = Array.length dba in
	let dbf = if dbal > image_count then (
		Logs.err (fun m -> m "size of database %d > %d, can't create tensor" dbal image_count); 
		Tensor.ones [1]
	) else (
		(* new dbf; otherwise there might be orphan images *)
		let dbf = Tensor.( 
			( ones [image_count; image_res; image_res] ) * (f (-1.0))) in
		let cpu = Torch.Device.Cpu in
		Array.iteri (fun i data -> 
			let imgf = tensor_of_bigarray2 data.img cpu in
			Tensor.copy_ (Tensor.narrow dbf ~dim:0 ~start:i ~length:1) ~src:imgf ) dba ; 
		dbf |> Tensor.to_device ~device 
	) in
	let fin = Unix.gettimeofday () in
	Logs.debug (fun m -> m "sort_database tensor copy done, %f" (fin-.sta)); 
	(Vector.of_array ~dummy:nulpdata dba),dbf

let init_database steak count = 
	(* generate 'count' initial program & image pairs *)
	Logs.info(fun m -> m  "init_database %d" count); 
	let i = ref 0 in
	let iters = ref 0 in
	let replace = ref 0 in
	let equivalents = ref 0 in
	while !i < count do (
		let data = if !i = 0 then
			generate_empty_logo !i image_res else
			generate_random_logo !i image_res in
		let imgf = tensor_of_bigarray2 data.img steak.device in
		let dist,mindex = dbf_dist steak imgf in
		if dist > 5. then (
			Logs.debug(fun m -> m 
				"%d: adding [%d] = %s" !iters !i 
				(Logo.output_program_pstr data.pro) ); 
			ignore( db_push steak data imgf ); 
			incr i;
		) else (
			(* see if there's a replacement *)
			let data2 = db_get steak mindex in
			let c1 = data.pcost in (* progenc_cost  *)
			let c2 = data2.pcost in
			if c1 < c2 then (
				Logs.debug(fun m -> m 
					"%d: replacing [%d][%d] = %s ( was %s)" 
					!iters mindex (List.length data2.equiv)
					(Logo.output_program_pstr data.pro) 
					(Logo.output_program_pstr data2.pro));
				let data = if dist < 0.6 then (
					let ed = pdata_to_edata data2 in
					{data with equiv = (ed :: data2.equiv)} 
					) else data in
				db_set steak mindex data imgf; 
				incr replace
			) else (
				(* they are equivalent; cost >=; see if it's already there *)
				(* if the list is short, add it; otherwise only add to list if it's better (above)*) 
				if dist < 0.6 then (
					if List.length data2.equiv < 32 then (
						let pres = List.exists (fun a -> a.eprogenc = data.progenc) data2.equiv in
						if not pres then ( 
						(* ok, add it to the list *)
						(*Logs.debug(fun m -> m 
							"%d: equivalent [%d][%d] = %s (best %s;)" 
							!iters mindex (List.length data2.equiv) 
							(Logo.output_program_pstr data.pro) 
							(Logo.output_program_pstr data2.pro));*)
							let ed = pdata_to_edata data in
							let eqlist = ed :: data2.equiv (*|> 
								List.sort (fun a b -> compare a.epcost b.epcost)*) in 
							let data2 = {data2 with equiv = eqlist} in
							let imgf = tensor_of_bigarray2 data2.img Torch.Device.Cpu in
							db_set steak mindex data2 imgf; 
							(* do not change the image *)
							incr equivalents
						)
					) 
				)
			)
		); 
		if !iters mod 40 = 39 then 
			(* needed to clean up torch allocations *)
			Caml.Gc.major (); 
		incr iters
	) done; 
	let db,dbf = sort_database steak.device steak.db in
	Logs.info(fun m -> m  "%d done; %d sampled; %d replacements; %d equivalents" !i !iters !replace !equivalents); 
	db,dbf
	
let save_database db fname = 
	(* saves in the current state -- not sorted. *)
	let dba = Vector.to_array db in (* inefficient *)
	let fil = open_out fname in
	Printf.fprintf fil "%d\n" (Array.length dba); 
	Array.iteri (fun i d -> 
		let pstr = Logo.output_program_pstr d.pro in
		Printf.fprintf fil "[%d] %s "  i pstr; 
		List.iter (fun ep -> 
			let ps = Logo.output_program_pstr ep.epro in
			if String.length ps > 0 then 
				Printf.fprintf fil "| %s " ps) d.equiv; 
		Printf.fprintf fil "\n") dba ; 
	close_out fil; 
	Logs.info(fun m -> m  "saved %d to %s" (Array.length dba) fname); 
	
	(* verification .. human readable*)
	let fil = open_out "db_human_log.txt" in
	Printf.fprintf fil "%d\n" (Array.length dba); 
	Array.iteri (fun i d -> 
		Printf.fprintf fil "\n[%d]\t" i ; 
		Logo.output_program_h d.pro fil) dba ; 
	close_out fil; 
	Logs.debug(fun m -> m  "saved %d to db_human_log.txt" (Array.length dba))
	(* save the equivalents too? *)

let load_database_line steak s pid equivalent =
	let sl = Pcre.split ~pat:"[\\[\\]]+" s in
	let _h,ids,ps = match sl with
		| h::m::t -> h,m,t
		| h::[] -> h,"",[]
		| [] -> "0","0",[] in
	let pid2 = int_of_string ids in
	if pid <> pid2 then 
		Logs.err (fun m -> m "load pid %d != line %d" pid2 pid); 
	let pr,prl = if pid = 0 || (List.length ps) < 1
		then Some `Nop, []
		else (
			let a = String.split_on_char '|' (List.hd ps) in
			parse_logo_string (List.hd a), (List.tl a) ) in
	(match pr with 
	| Some pro -> 
		let progenc = Logo.encode_program_str pro in
		let (_,_,segs) = Logo.eval (Logo.start_state ()) pro in
		let scost = segs_to_cost segs in
		let pcost = progenc_cost progenc in
		let img,_ = Logo.segs_to_array_and_cost segs image_res in
		let equiv = List.map (fun s -> 
			let g = parse_logo_string s in
			match g with 
			| Some epro -> ( 
				let eprogenc = Logo.encode_program_str epro in
				let (_,_,esegs) = Logo.eval (Logo.start_state ()) epro in
				let escost = segs_to_cost esegs in
				let epcost = progenc_cost eprogenc in
				let eimg,_ = Logo.segs_to_array_and_cost esegs image_res in
				Atomic.incr equivalent; 
				{epro; eprogenc; eimg; escost; epcost; esegs}
				)
			| _ -> ( 
				Logs.err(fun m -> m  "could not parse equiv program %d %s" pid s); 
				nulpequiv ) ) prl in
		let data = {pid; pro; progenc; img; 
						scost; pcost; segs; equiv} in
		let imgf = tensor_of_bigarray2 img steak.device in
		ignore( db_push steak data imgf );  (* mutex protected *)
	
	| _ -> Logs.err(fun m -> m 
				"could not parse program %d %s" pid s ))
	
let load_database steak = 
	let equivalent = Atomic.make 0 in
	let lines = read_lines "db_prog.txt" in
	let a,progs = match lines with
		| h::r -> h,r
		| [] -> "0",[] in
	let ai = int_of_string a in
	if ai > image_count then (
		Logs.err(fun m -> m "image_count too large, %d > %d" ai image_count) 
	) else (
		(* db_prog format: [$id] program | eq. prog | eq. prog ... \n *)
		let progsa = Array.of_list progs in
		let pl = Array.length progsa in
		(* this needs to be called within Dtask.run to be parallel *)
		(* actually ... running in parallel seems to screw things up! *)
		if !gparallel && false then (
			Dtask.parallel_for steak.pool ~start:0 ~finish:(pl-1) 
				~body:( fun i -> load_database_line steak progsa.(i) i equivalent )
		) else (
			for i = 0 to (pl-1) do 
				load_database_line steak progsa.(i) i equivalent 
			done
		); 
		Logs.info (fun m -> m "Loaded %d programs (%d max) and %d equivalents" 
			(List.length progs) image_count (Atomic.get equivalent)); 
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

let hidden_nodes = 128
let epochs = 10000
let learning_rate = 1e-3

let test_torch () =
	(* This should reach ~97% accuracy. *)
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
	for index = 1 to epochs-1 do
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
	
let handle_message steak bd msg =
	(*Logs.debug (fun m -> m "handle_message %s" msg);*)
	let msgl = String.split_on_char ':' msg in
	let cmd = if List.length msgl = 1 then msg 
		else List.hd msgl in
	match cmd with
	| "update_batch" -> (
		(* supervised: sent when python has a working copy of data *)
		(* dreaming : sent when the model has produced edits (maybe old) *)
		let bd = if steak.superv 
			then update_bea_sup steak bd
			else update_bea_dream steak bd in (* applies the dreamed edits *)
		bigfill_batchd steak bd; 
		steak.batchno <- steak.batchno + 1; 
		Logs.debug(fun m -> m "new batch %d" steak.batchno); 
		bd,(Printf.sprintf "ok %d" !nreplace)
		)
	| "decode_edit" -> (
		(* python sets bd.bedtd -- the dreamed edits *)
		(* read this independently of updating bd.bedts; so as to determine acuracy. *)
		let decode_edit_accuracy edit_arr str = 
			let typright, chrright, posright = ref 0, ref 0, ref 0 in
			let print = ref true in
			Array.iteri (fun i (typ,pos,chr) -> 
				if List.length (bd.bea.(i).edits) > 0 then (
					let styp,spos,schr = List.hd (bd.bea.(i).edits) in (* supervised *)
					if typ = styp then incr typright; 
					if pos = spos then incr posright; 
					if chr = schr then incr chrright; 
					if !print then (
						Logs.info (fun m -> m "|%d true: %s [%d] %c ; est: %s [%d] %c"
							i styp spos schr typ pos chr ); 
						print := false
					)
				) ) edit_arr ; 
			let pctg v = (foi !v) /. (foi !batch_size) in
			Logs.info (fun m -> m (* TODO: debug mode *)
				"decode_edit %s: correct typ %0.3f chr %0.3f pos %0.3f " 
				str (pctg typright) (pctg chrright) (pctg posright) );
		in
		(*let edit_sup = decode_edit bd bd.bedts in*)
		let edit_dream = decode_edit bd bd.bedtd in
		(*decode_edit_accuracy edit_sup "superv"; *)
		decode_edit_accuracy edit_dream "dream"; 
		bd, "ok" )
	| _ -> bd,"Unknown command"

let read_socket client_sock = 
	let maxlen = 256 in
	let data_read = Bytes.create maxlen in
	let data_length =
		try
			Unix.recv client_sock data_read 0 maxlen []
		with Unix.Unix_error (e, _, _) ->
			Logs.err (fun m -> m "Sock can't receive: %s ; shutting down"
				(Unix.error_message e));
			( try Unix.shutdown client_sock SHUTDOWN_ALL
			with Unix.Unix_error (e, _, _) ->
				Logs.err (fun m -> m "Sock can't shutdown: %s"
					(Unix.error_message e)) ); 
			0 in
	if data_length > 0 then 
		Some (Bytes.sub data_read 0 data_length |> Bytes.to_string )
	else None
	
let dreams_save mnist dreams = 
	let fid = open_out "dreamcheck.txt" in
	Array.iteri (fun i dc -> 
		let d = if List.length dc.decode >= 1 then 
				List.hd dc.decode else "" in
		if dc.be.b_pid < image_count then (
			let a = progenc2progstr dc.be.a_progenc in
			let b = progenc2progstr dc.be.b_progenc in
			Printf.fprintf fid "[%d] ncorrect:%d is:%s->%s decode:%s\n"
				i dc.correct_cnt a b d; 
			if i < 4096 then (
				ignore(run_logo_string b 64 
					(Printf.sprintf "/tmp/png/d%05d_real.png" i)); 
				ignore(run_logo_string d 64 
					(Printf.sprintf "/tmp/png/d%05d_decode.png" i)) );
		) else (
			let mid = dc.be.b_pid - image_count in
			if mid < 60000 then (
				Printf.fprintf fid "[%d] decode:%s\n" i d;
				dbf_to_png mnist mid
					(Printf.sprintf "/tmp/png/d%05d_real.png" i); 
				ignore(run_logo_string d 64 
					(Printf.sprintf "/tmp/png/d%05d_decode.png" i)) 
			)
		) ) dreams;
	close_out fid; 
	Logs.debug (fun m -> m "Saved %d to dreamcheck.txt" (Array.length dreams))

let servthread steak () = (* steak = thread state *)
	(let open Unix in
	Logs.info (fun m -> m "Starting server on %d" steak.sockno); 
	let server_sock = socket PF_INET SOCK_STREAM 0 in
	let listen_address = inet_addr_loopback in
	setsockopt server_sock SO_REUSEADDR true ; (* for faster restarts *)
	bind server_sock (ADDR_INET (listen_address, steak.sockno)) ;
	listen server_sock 2 ; (* 2 max connections *)
	while !glive do (
		let bd,fdlist = init_batchd (steak.sockno-4340) in
		let (client_sock, _client_addr) = accept server_sock in
		Logs.info (fun m -> m "new connection on %d" steak.sockno); 
		(* make new mmap files & batchd for each connection *)
		let rec message_loop bd =
			let msg = read_socket client_sock in
			(match msg with 
			| Some msg -> (
				let sta = Unix.gettimeofday () in
				let bd,resp = handle_message steak bd msg in 
				let fin = Unix.gettimeofday () in
				Logs.debug (fun m -> m "handle_message %s time %f" msg (fin-.sta));
				ignore ( send client_sock (Bytes.of_string resp) 
					0 (String.length resp) [] ) ; 
				message_loop bd )
			| _ -> (
				save_database steak.db "db_prog_new.txt"; 
				(match steak.dreams with
				| Some dreams -> ( dreams_save steak.mnist dreams )
				| _ -> () ); 
			) ) in
		if !gparallel then (
			Dtask.run steak.pool (fun () -> message_loop bd )
		) else (
			message_loop bd ); 
		(* if client disconnects, close the files and reopen *)
		Logs.debug (fun m -> m "disconnect %d" steak.sockno); 
		List.iter (fun fd -> close fd) fdlist; 
	) done; 
	) (* )open Unix *)
	
let measure_torch_copy_speed device = 
	let start = Unix.gettimeofday () in
	let nimg = 6*2048*2 in
	let dbf = Tensor.( zeros [nimg; image_res; image_res] ) in
	for i = 0 to nimg/2-1 do (
		let k = Tensor.ones [image_res; image_res] in
		Tensor.copy_ (Tensor.narrow dbf ~dim:0 ~start:i ~length:1) ~src:k; 
	) done; 
	let y = Tensor.to_device dbf ~device in
	let z = Tensor.sum y in
	let stop = Unix.gettimeofday () in
	Printf.printf "%d image_copy time: %fs\n%!" (nimg/2) (stop -. start);
	Printf.printf "%f\n%!" (Tensor.float_value z) 
	(* this is working just as fast or faster than python.*)
	(* something else must be going on in the larger program *)
	
let make_dreams db = 
	(* make an array of dreams to test *)
	let dreams = Vector.create ~dummy:nuldream in
	let aa = Vector.get db 0 in
	(*let dba = Vector.to_array db in
	Array.iteri (fun i b -> 
		let edits = progenc_to_edits aa b in (* not needed, maybe useful *)
		let edited = Array.make p_ctx 0.0 in
		let be = {a_pid=0; 
					b_pid=i; 
					a_progenc=aa.progenc; (* null program *)
					b_progenc=b.progenc; 
					c_progenc=""; 
					edits; 
					edited; 
					count=0; 
					indx=i } in
		Vector.push dreams {be; decode=[]; correct_cnt=0}
		) dba ;*) 
	let ndenovo = Vector.length dreams in
	(*let dba2 = Array.sub dba 0 2048 in
	(* add in all 2-edits *)
	Array.iteri (fun i a -> 
		Array.iteri (fun j b -> 
			let dist,edits = Levenshtein.distance a.progenc b.progenc true in
			if dist > 0 && dist <= 2 then (
				let edited = Array.make p_ctx 0.0 in
				let be = {a_pid = i; 
							b_pid = j;
							a_progenc = a.progenc; 
							b_progenc = b.progenc; 
							c_progenc = a.progenc; (* starting point *)
							edits; 
							edited; 
							count = 0; 
							indx = (Vector.length dreams)} in
				Vector.push dreams {be; decode=[]; correct_cnt=0}
			)
		) dba2
	) dba2 ; *)
	let nedits = Vector.length dreams in
	for i=0 to 2000-1 do (
		let edited = Array.make p_ctx 0.0 in (* don't forget this has to be replaced later *)
		let be = {a_pid = 0; 
					b_pid = image_count + i; 
					a_progenc = aa.progenc; 
					b_progenc = ""; 
					c_progenc = ""; 
					edits = []; 
					edited; 
					count=0; indx = (Vector.length dreams)} in
		Vector.push dreams {be; decode=[]; correct_cnt=0}
	) done; 
	let nall = Vector.length dreams in
	Logs.debug (fun m -> m "Generated %d dreams (%d denovo, %d edits, %d mnist)"
		nall ndenovo (nedits-ndenovo) (nall-nedits)); 
	Vector.to_array dreams

let usage_msg = "program.exe -b <batch_size>"
let input_files = ref []
let output_file = ref ""
let g_debug = ref false 
let anon_fun filename = (* just incase we need later *)
  input_files := filename :: !input_files
let speclist =
  [("-b", Arg.Set_int batch_size, "Training batch size");
   ("-o", Arg.Set_string output_file, "Set output file name"); 
   ("-g", Arg.Set g_debug, "Turn on debug");
   ("-p", Arg.Set gparallel, "Turn on parallel");]

let () = 
	Arg.parse speclist anon_fun usage_msg;
	Random.self_init (); 
	let () = Logs.set_reporter (Logs.format_reporter ()) in
	let () = Logs.set_level 
		(if !g_debug then Some Logs.Debug else Some Logs.Info) in
	Logs.debug (fun m -> m "Debug logging enabled."); 
	if !gparallel then 
		Logs.info (fun m -> m "Parallel enabled.")
	else 
		Logs.info (fun m -> m "Parallel disabled.") ; 
	Logs_threaded.enable (); 
	(* Logs levels: App, Error, Warning, Info, Debug *)

	Logs.info(fun m -> m "batch_size:%d" !batch_size);
	Logs.info(fun m -> m "cuda available: %b%!" 
				(Cuda.is_available ()));
	Logs.info(fun m -> m "cudnn available: %b%!"
				(Cuda.cudnn_is_available ()));
	let device = Torch.Device.cuda_if_available () in
	(*let device = Torch.Device.Cpu in*) (* slower *)
	(*for _i = 0 to 4 do 
		measure_torch_copy_speed device
	done; *)
	
	let mnistd = Mnist_helper.read_files ~prefix:"../otorch-test/data" () in
	let mimg = Tensor.reshape mnistd.train_images 
		~shape:[60000; 28; 28] in
	(* need to pad to 30 x 30, one pixel on each side *)
	let mnist = Tensor.zeros [60000; 30; 30] in
	Tensor.copy_ (
		Tensor.narrow mnist ~dim:1 ~start:1 ~length:28 |> 
		Tensor.narrow ~dim:2 ~start:1 ~length:28) ~src:mimg ;
		(* Tensor.narrow returns a pointer/view; copy_ is in-place. *)
		(* keep on CPU? *)

	let db = Vector.create ~dummy:nulpdata in
	let dbf = Tensor.( 
		( ones [image_count; image_res; image_res] ) * (f (-1.0))) 
		|> Tensor.to_device ~device in
	
	let db_mutex = Mutex.create () in
	let pool = Dtask.setup_pool ~num_domains:8 () in 
		(* tune this -- 8-12 seems ok *)
	let supfid = open_out "/tmp/png/replacements_sup.txt" in
	let dreamfid = open_out "/tmp/png/replacements_dream.txt" in
	let dbf_enc = Tensor.zeros [2;2] in
	let mnist_enc = Tensor.zeros [2;2] in
	let vae = Vae.dummy_ext () in
	let supsteak = {device; db; dbf; mnist; dbf_enc; mnist_enc; vae; db_mutex;
			superv=true; sockno=4340; fid=supfid; batchno=0; pool; 
			dreamn=0; dreams=None} in
			
	let db,dbf = if Sys.file_exists "db_prog.txt" then ( 
		if !gparallel then 
			Dtask.run supsteak.pool (fun () -> load_database supsteak )
		else 
			load_database supsteak ; 
		let db,dbf = sort_database device db in
		(*render_simplest db;*) 
		db,dbf
	) else ( 
		Logs.app(fun m -> m "Generating %d programs" (image_count/2));
		let () = Logs.set_level (Some Logs.Debug) in
		let start = Unix.gettimeofday () in
		let db,dbf = init_database supsteak (image_count/2) in
		(* init also sorts. *)
		let stop = Unix.gettimeofday () in
		Logs.app(fun m -> m "Execution time: %fs\n%!" (stop -. start)); 
		Logs.info(fun m -> m "render_simplest: first 10 programs");
		for i = 0 to 9 do (
			let p = Vector.get db i in
			Logs.info(fun m -> m "%d: %s" i
					(Logo.output_program_pstr p.pro)); 
		) done; 
		save_database db "db_prog.txt"; 
		db,dbf
	) in
	
	(* try to train the vae? *)
	let vae,dbf_enc,mnist_enc = Vae.train_ext dbf mnist device !batch_size in
	
	(* dreams test structure *)
	let dreams = make_dreams db in
	
	(* update the thread state *)
	let supsteak2 = {supsteak with db; dbf; dbf_enc; mnist_enc; vae} in 
	let dreamsteak = {device; db; dbf; mnist; 
			dbf_enc; mnist_enc; vae; db_mutex;
			superv=false; sockno=4341; fid=dreamfid; batchno=0; pool; 
			dreamn=0; dreams=(Some dreams)} in
			
	(* extra bit of complexity!! if Cuda hangs in one of the domains, e.g. for an out-of-memory error, you won't see it on stdout -- it will just stop. 
	to properly debug, will need to strip down to one thread, no domainslib *)
	(*let d = Domain.spawn (fun _ -> 
		let pool2 = Dtask.setup_pool ~num_domains:6 () in
		dreamsteak.pool <- pool2; 
		servthread dreamsteak () ) in*)
	servthread supsteak2 () ; 
	(*servthread dreamsteak () ;*)
	(*Domain.join d;*)
	close_out supfid; 
	close_out dreamfid; 
	Dtask.teardown_pool supsteak2.pool ; 
	Dtask.teardown_pool dreamsteak.pool ;

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
