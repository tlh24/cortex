open Logo

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
	(* recurs lim, binop recursion limit, active variables *)
	if not return_val then (
		match Random.int 6 with 
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
			`Move(gen_ast true (r-1,br,av), gen_ast true (r-1,br,av), nulptag)
		| 7 -> (* FIXME 3 *)
			let i = select_writevar av in
			av.(i) <- true;
			let n = `Const(foi(Random.int 12),nulptag) in
			`Loop(i, n, gen_ast false (r-1,br,av),nulptag)
		| 4 -> `Pen(gen_ast true (r-1,br,av),nulptag)
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
			match Random.int 4 with
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
	| `Pen(a,_) -> 1 + count_ast a
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
	| `Pen(a,_) -> ( if !n = sel then 
		(incr n; `Pen((mark_ast a n sel),1) ) else 
		(incr n; `Pen((mark_ast a n sel),nulptag) ) )
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
	| `Pen(a,w) -> ( if w > 0 then
		( gen_ast false st ) else 
		( `Pen(chng_ast a st, nulptag) ) )
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
				| a3,_ when a3 > neps && a3 < eps -> (* 0 * x *)
					(change := true;`Const(aa,nulptag))
				| _,b3 when b3 > neps && b3 < eps -> (* x * 0 *)
					(change := true;`Const(bb,nulptag))
				| a3,_ when a3 > 1.0-.eps && a3 < 1.0+.eps -> (* 1 * x *)
					(change := true;`Const(bb,nulptag))
				| _,b3 when b3 > 1.0-.eps && b3 < 1.0+.eps -> (* x * 1 *)
					(change := true;`Const(aa,nulptag))
				| a3,b3 when b3 > a3 -> 
					(change := true; `Binop(`Const(b3,bw),s,f,`Const(a3,aw),w))
				| _ -> `Binop(compress a change, s,f, compress b change,w) )
			| "/" -> (
				match aa,bb with
				| a3,_ when a3 > neps && a3 < eps ->  (* 0 / x *)
					(change := true;`Const(aa,nulptag))
				| _,b3 when b3 > neps && b3 < eps -> (* x / 0 *)
					(change := true;`Nop)
				| _,b3 when b3 > 1.0-.eps && b3 <1.0+.eps -> (* x / 1 *)
					(change := true;`Const(aa,nulptag))
				| a3,b3 when a3-.b3 > neps && a3-.b3 < eps -> (* x / x*)
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
	| `Def(i,a,w) -> `Def(i, compress a change, w)
	| `Pen(a,w) -> (
		match a with 
		| `Nop -> ( change := true; `Nop )
		| _ -> `Pen(compress a change, w)
		)
	| `Nop -> `Nop

let rec ast_has matchf ast = 
	let f = ast_has matchf in
	if matchf ast then true else (
	match ast with
	| `Save(_,a,_) -> f a
	| `Move(a,b,_) -> (f a || f b)
	| `Binop(a,_,_,b,_) -> (f a || f b)
	| `Seq(l,_) -> List.exists f l
	| `Loop(_,a,b,_) -> ( f a || f b )
	| `Call(_,l,_) -> List.exists f l
	| `Pen(a,_) -> f a
	| _ -> false )
	
let has_move ast = 
	let f a = match a with
		| `Move(_,_,_) -> true
		| _ -> false in
	ast_has f ast
	
let has_pen_nop ast = 
	let f a = match a with
		| `Pen(b,_) -> (
			match b with 
			| `Nop -> true
			| _ -> false )
		| _ -> false in
	ast_has f ast 
	
let rec compress_ast ast = 
	let change = ref false in
	let ast2 = compress ast change in
	let ast3 = if !change then compress_ast ast2 else ast2 in
	if has_move ast3 then ast3 else `Nop

let segs_bbx segs = 
	let minx = List.fold_left 
		(fun c (x0,_,x1,_,_) -> min (min x0 x1) c) 0.0 segs in
	let maxx = List.fold_left 
		(fun c (x0,_,x1,_,_) -> max (max x0 x1) c) 0.0 segs in
	let miny = List.fold_left 
		(fun c (_,y0,_,y1,_) -> min (min y0 y1) c) 0.0 segs in
	let maxy = List.fold_left 
		(fun c (_,y0,_,y1,_) -> max (max y0 y1) c) 0.0 segs in
	(minx,maxx,miny,maxy)
	
let segs_to_cost segs = 
	let seglen (x0,y0,x1,y1,_) = 
		Float.hypot (x0-.x1) (y0-.y1) in
	let cost = List.fold_left 
		(fun c s -> c +. (seglen s)) 0.0 segs in
	cost
	
