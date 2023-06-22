open Vgwrapper

(* variables will be referenced to the stack, de Brujin indexes *)
(* names are more interpretable by humans ... but that's a big space *)
let nulptag = 0
let nulimg = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout 1

type prog = [
	| `Var of int * int
	| `Save of int * prog * int
	| `Move of prog * prog * int(* angle & distance *)
	| `Binop of prog * string * (float -> float -> float) * prog * int
	| `Const of float * int
	| `Seq of prog list * int
	| `Loop of int * prog * prog * int(* iterations & body *)
	| `Call of int * prog list * int (* list of arguments *)
	| `Def of int * prog * int(* same sig as `Save *)
(* 	| Cmp of prog * (float -> float -> bool) * prog *)
	| `Pen of prog * int (* pen weight *)
	| `Nop
]

let fos = float_of_string
let ios = int_of_string
let foi = float_of_int
let iof = int_of_float

let pmark lg w = 
	if w>0 then Printf.fprintf lg "<marked> "
	
let enc_char1 c = 
	match c with
	| "(" -> 1
	| ")" -> 2
	| "," -> 3
	| ";" -> 4
	| "+" -> 5
	| "-" -> 6
	| "*" -> 7
	| "/" -> 8
	| "var" -> 9
	| "=" -> 10
	| "move" -> 11
	| "ua" -> 12
	| "ul" -> 13
	| "loop" -> 14
	| "def" -> 15
	| ":" -> 16
	| "call" -> 17
	| "pen" -> 18
	| _ -> 19
	(* 0 is not used atm *)
	
	
let dec_item1 i = 
	match i with 
	| 1 -> "( "
	| 2 -> ") "
	| 3 -> ", "
	| 4 -> "; " (* if we want a \n, add later *)
	| 5 -> "+ "
	| 6 -> "- "
	| 7 -> "* "
	| 8 -> "/ "
	| 9 -> "v"
	| 10 -> "= "
	| 11 -> "move "
	| 12 -> "ua "
	| 13 -> "ul "
	| 14 -> "loop "
	| 15 -> "d"
	| 16 -> ": "
	| 17 -> "c"
	| 18 -> "pen "
	| _ -> " "
	
let dec_item i =
	if i < 0 then 
		(string_of_int (i + 10))^" "
	else
		dec_item1 i
	
let rec enc_prog g s = 
	(* convert a program to a list of integers, s *)
	(* and a list of positions, pl *)
	(* output list is in REVERSE order *)
	let enc_char c s = 
		s := (enc_char1 c) :: !s in
		
	let enc_int i s = 
		let j = if i > 9 then 9 
			else (if i < 0 then 0 else i) in
		s := (j - 10) :: !s in
	
	match g with 
	| `Var(i,_) -> ( 
		enc_char "var" s; 
		enc_int i s)
	| `Save(i,a,_) -> ( 
		enc_char "var" s; 
		enc_int i s; 
		enc_char "=" s; 
		enc_prog a s )
	| `Move(a,b,_) -> (
		enc_char "move" s; 
		enc_prog a s; 
		enc_char "," s; 
		enc_prog b s ) 
	| `Binop(a,sop,_,b,_) -> (
		enc_prog a s; 
		enc_char sop s;
		enc_prog b s )
	| `Const(i,_) -> (
		match i with
		| x when x > 0.99 && x < 1.01 -> enc_char "ul" s; 
		| x when x > 6.28 && x < 6.29 -> enc_char "ua" s; 
		| x -> enc_int (iof x) s; 
		)
	| `Seq(l,_) -> (
		enc_char "(" s; 
		List.iteri (fun i a ->
			if i > 0 then (enc_char ";" s);
			enc_prog a s) l ; 
		enc_char ")" s )
	| `Loop(i,a,b,_) -> (
		enc_char "loop" s; 
		enc_int i s; 
		enc_char "," s; 
		enc_prog a s; 
		enc_char "," s; 
		enc_prog b s )
	| `Call(i,l,_) -> (
		enc_char "call" s; 
		enc_int i s; 
		List.iteri (fun i a ->
			if i > 0 then (enc_char "," s);
			enc_prog a s) l )
	| `Def(i,a,_) -> (
		enc_char "def" s; 
		enc_int i s; 
		enc_char ":" s;
		enc_prog a s )
	| `Pen(a,_) -> (
		enc_char "pen" s;
		enc_prog a s )
	| `Nop -> ()
	
let encode_program g = 
	let s = ref [] in
	enc_prog g s; 
	List.rev !s
	
let rec enc_ast g s q r = 
	(* convert a program to int list + int list list addresses. *)
	(* output list is in REVERSE order *)
	(* "q" is the list (int list) of addresses *)
	(* "r" is the address of the parent *)
	
	let enc_char c = 
		s := (enc_char1 c) :: !s;
		q := r :: !q in
		
	let clip i = if i > 9 then 9 
			else (if i < 0 then 0 else i) in
	
	let enc_int i p = 
		s := ((clip i) - 10) :: !s; 
		q := (p::r) :: !q in
		
	let enc_int_leaf i = 
		s := ((clip i) - 10) :: !s; 
		q := r :: !q in
	
	match g with 
	| `Var(i,_) -> ( 
		enc_char "var"; 
		enc_int i 0 )
	| `Save(i,a,_) -> ( 
		enc_char "="; 
		enc_int i 0; 
		enc_ast a s q (1::r))
	| `Move(a,b,_) -> (
		enc_char "move"; 
		enc_ast a s q (0::r);
		enc_ast b s q (1::r)) 
	| `Binop(a,sop,_,b,_) -> (
		enc_char sop; 
		enc_ast a s q (0::r);
		enc_ast b s q (1::r))
	| `Const(i,_) -> (
		match i with
		| x when x > 0.99 && x < 1.01 -> enc_char "ul"; 
		| x when x > 6.28 && x < 6.29 -> enc_char "ua"; 
		| x -> enc_int_leaf (iof x); 
		)
	| `Seq(l,_) -> (
		enc_char "("; (* meh *)
		List.iteri (fun i a -> enc_ast a s q (i::r)) l )
	| `Loop(i,a,b,_) -> (
		enc_char "loop"; 
		enc_int i 0; 
		enc_ast a s q (1::r); 
		enc_ast b s q (2::r))
	| `Call(i,l,_) -> (
		enc_char "call"; 
		enc_int i 0; 
		List.iteri (fun i a -> enc_ast a s q ((i+1)::r)) l )
	| `Def(i,a,_) -> (
		enc_char "def"; 
		enc_int i 0; 
		enc_ast a s q (1::r))
	| `Pen(a,_) -> (
		enc_char "pen";
		enc_ast a s q (1::r) )
	| `Nop -> ()
	;;
	
let encode_ast g = 
	let s = ref [] in
	let q = ref [] in
	enc_ast g s q []; 
	let ss = List.rev !s in
	let qq = List.map (fun a -> List.rev a) (List.rev !q) in
	ss,qq
	
let print_encoded_ast ss qq = 
	let indent l = String.init (List.length l) (fun _ -> ' ') in
	Printf.printf "element : address\n" ; 
	List.iter2 (fun a b -> 
		Printf.printf "%s%s:" (indent b) (dec_item a); 
		List.iter (fun c -> 
			Printf.printf "%d," c) b; 
		Printf.printf "\n") ss qq; 
	Printf.printf "\n"
	
	
(* for decoding, seems we need to extract the structural information
to get e.g. children when the # if kids is unknown *)
type node = 
	Node of int * node list
	| Null
	
let decode_ast_struct ss qq = 
	(* convert qq, type int list list, to string list *)
	let sq = List.map2 (fun s q -> s,q) ss qq in
	let get_kids prefix = 
		let pl = List.length prefix in
		(* first pass: address length is this + 1 *)
		let fl = List.filter (fun (_s,q) -> List.length q = (pl+1) ) sq in
		(* filter all that have the same prefix *)
		List.filter (fun (_s,q) -> 
			let rq = List.rev q in
			let q' = match rq with 
				| _::b -> List.rev b
				| _ -> List.rev prefix in
			(List.compare compare prefix q') = 0) fl
	in
		
	let rec build (root,prefix) =
		`Node(root, (get_kids prefix |> List.map build)) in
		
	let tree = build (List.hd sq) in
	
	let conv_i h = 
		match h with
		| `Node(i,_) -> i
		| _ -> 0 in
	
	let rec conv h = 
		match h with
		| `Node(parent,kids) -> (
			let a = Array.of_list kids in
			let n = nulptag in
			match (dec_item1 parent) with 
			| "v"-> `Var(conv_i a.(0), n)
			| "= " -> `Save(conv_i a.(0), conv a.(1), n)
			| "move " -> `Move(conv a.(0), conv a.(1), n)
			| "+ " -> `Binop(conv a.(0), "+", (fun d e -> d +. e), conv a.(1), n)
			| "- " -> `Binop(conv a.(0), "-", (fun d e -> d -. e), conv a.(1), n)
			| "* " -> `Binop(conv a.(0), "*", (fun d e -> d *. e), conv a.(1), n)
			| "/ " -> `Binop(conv a.(0), "/", (fun d e -> d /. e), conv a.(1), n)
			| "( " -> `Seq(List.map conv kids, n)
			| "loop " -> `Loop(conv_i a.(0), conv a.(1), conv a.(2), n)
			| "c" -> `Call(conv_i a.(0), 
							(match kids with
							| _::tl -> List.map conv tl
							| _ -> [`Nop] ), n)
			| "d" -> `Def( conv_i a.(0), conv a.(1), n)
			| "pen "-> `Pen(conv a.(0), n)
			| "ua "-> `Const(6.283185307179586, n)
			| "ul "-> `Const(1.0, n)
			| _ -> `Const(foi (parent+10), n) )
		| _ -> `Nop
	in
	conv tree

let decode_program il = 
	(* convert a program encoded as a list of ints to a string, 
	and then to the ast *)
	let sl = List.map dec_item il in
	List.fold_left (fun a b -> a^b) "" sl 
	(* parse in calling fun, via lexer and parser *)
	
let intlist_to_string e =
	(* convenience encoding -- see asciitable for what it turns to *)
	let offs = 10 + (Char.code '0') in 
	(* integers are mapped 1:1, 0 -> [-10] -> 0 -> '0' *)
	let cof_int i = Char.chr (i + offs) in
	let bf = Buffer.create 32 in
	List.iter (fun a -> Buffer.add_char bf (cof_int a)) e;
	Buffer.contents bf
	
let string_to_intlist e = 
	(* outputs a list, range [-10 17] *)
	let offs = 10 + (Char.code '0') in
	String.fold_left (fun a c -> 
		let i = (Char.code c) - offs in
		i :: a) [] e
	|> List.rev 
	
let output_program_p bf g = (* p is for parseable *)
	let gl = encode_program g in (* compressed encoding *)
	let s = decode_program gl in (* to minimize duplication *)
	Printf.bprintf bf "%s" s
	
let output_program_pstr g = 
	let bf = Buffer.create 64 in
	output_program_p bf g; 
	(Buffer.contents bf)

let output_program_plg lg g = 
	Printf.fprintf lg "%s" (output_program_pstr g;)
	
let encode_program_str g = 
	encode_program g |> intlist_to_string
	
let rec output_program_h g lg =
	match g with
	| `Var(i,w) -> Printf.fprintf lg "Var %d " i; pmark lg w
	| `Save(i,a,w) -> Printf.fprintf lg "Save %d " i; pmark lg w; 
			output_program_h a lg
	| `Move(a,b,w) -> Printf.fprintf lg "Move "; pmark lg w;
			output_program_h a lg; 
			Printf.fprintf lg ", " ; 
			output_program_h b lg
	| `Binop(a,s,_,b,w) -> Printf.fprintf lg "Binop "; pmark lg w;
			output_program_h a lg; 
			Printf.fprintf lg " %s " s; 
			output_program_h b lg
	| `Const(i,w) -> Printf.fprintf lg "Const %f " i; pmark lg w
	| `Seq(l,w) -> (
		Printf.fprintf lg "Seq "; pmark lg w; 
		output_list_h lg l "; " )
	| `Loop(i,a,b,w) -> Printf.fprintf lg "Loop [%d] " i; pmark lg w;
			output_program_h a lg; 
			Printf.fprintf lg ", " ; 
			output_program_h b lg
	| `Call(i, l,w) -> Printf.fprintf lg "Call %d " i; pmark lg w;
		output_list_h lg l ", "
	| `Def(i,a,w) -> Printf.fprintf lg "Def %d " i; pmark lg w;
		output_program_h a lg
	| `Pen(a,w) -> Printf.fprintf lg "Pen "; pmark lg w;
		output_program_h a lg;
	| `Nop -> Printf.fprintf lg "Nop "
	
and output_list_h lg l sep =
	Printf.fprintf lg "("; 
	List.iteri (fun i v -> 
		if i > 0 then Printf.fprintf lg "%s" sep ;
		output_program_h v lg) l ; 
	Printf.fprintf lg ")"
	
type ienc = [
	| `Leaf of int * int (* enc, pos *)
	| `Node of ienc * ienc list
	| `Nenc
]

let rec tag_pos g h = 
	(* convert the program g into an equivalent 
		ienc tree, where in each `Leaf(a, b)
		a = integer encoding, per enc_prog above
		b = position in encoded string, including separators *)
		
	let enc_char c = 
		let i = !h in incr h; 
		`Leaf(enc_char1 c, i)
	in
		
	let enc_int i = 
		let j = if i > 9 then 9 
		  else (if i < 0 then 0 else i) in
		let k = j-10 in
		let i = !h in incr h; 
		`Leaf(k, i)
	in
	
	(* note: the ocaml runtime evaluates arguments to constructors in reverse order. Which makes sense for currying and tail-recursion reasons ... 
	it just makes imperative execution a bit complicated. *)
	match g with 
	| `Var(i,_) -> ( (* v$i *)
		let j = enc_char "var"  in
		let k = enc_int i in
		`Node(j, [k] ) )
	| `Save(i,a,_) -> ( (* = v$i $a *)
		let j = enc_char "=" in
		let k = enc_char "var" in
		let l = enc_int i in
		let m = tag_pos a h in
		`Node(j, [k; l; m] ) )
	| `Move(a,b,_) -> ( (* move $a,$b *)
		let j = enc_char "move" in
		let aa = tag_pos a h in
		let c = enc_char "," in
		let bb = tag_pos b h in
		`Node( j, [aa; c; bb]) )
	| `Binop(a,sop,_,b,_) -> ( (* binop $a $b *)
		let j = enc_char sop in
		let aa = tag_pos a h in
		let bb = tag_pos b h in
		`Node(j, [aa; bb]) )
	| `Const(i,_) -> (
		match i with
		| x when x > 0.99 && x < 1.01 -> enc_char "ul"
		| x when x > 6.28 && x < 6.29 -> enc_char "ua"
		| x -> enc_int (iof x) )
	| `Seq(a,_) -> (
		let j = enc_char "(" in
		let l = ref [] in
		List.iter (fun a -> 
			l := (tag_pos a h) :: !l; 
			l := (enc_char ";") :: !l) a; 
		(* has one extra semicolon *)
		h := !h - 1; 
		let ll = (enc_char ")") :: (List.tl !l) |> List.rev in
		`Node(j,ll) )
	| `Loop(i,a,b,_) -> (
		let j = enc_char "loop" in
		let k = enc_int i in
		let aa = tag_pos a h in
		let bb = tag_pos b h in
		`Node( j, [k; aa; bb]) )
	| `Call(i,l,_) -> ( (* call, list of arguments (no sep) *)
		let j = enc_char "call" in
		let k = enc_int i in
		`Node(j, k :: (List.map (fun a -> tag_pos a h) l) ) )
	| `Def(i,a,_) -> (
		let j = enc_char "def" in
		let k = enc_int i in
		let aa = tag_pos a h in
		`Node(j, [k; aa]) )
	| `Pen(a,_) -> (
		let j = enc_char "pen" in
		let aa = tag_pos a h in
		`Node(j, [aa]) )
	| `Nop -> `Nenc
	
let rec untag_pos e = 
	match e with 
	| `Node(`Leaf(t,_), l) -> (
		let ut = dec_item t in
		Printf.printf "dec node %d %s\n%!" t ut; 
		match ut with 
		| "v" -> (
			let a = List.hd l in
			match a with
			| `Leaf(i,_) -> `Var(i+10, 0)
			| _ -> `Nop )
		| "= " -> (
			let a,b = List.nth l 1, List.nth l 2 in
			match a,b with 
			| `Leaf(i,_),_ -> `Save(i+10, untag_pos b, 0)
			| _ -> `Nop )
		| "move " -> (
			let a,b = List.hd l, List.nth l 2 in
			`Move(untag_pos a, untag_pos b, 0) )
		| "+ " -> (
			let a,b = List.hd l, List.nth l 1 in
			`Binop(untag_pos a, "+", ( +. ), untag_pos b, 0) )
		| "- " -> (
			let a,b = List.hd l, List.nth l 1 in
			`Binop(untag_pos a, "-", ( -. ), untag_pos b, 0) )
		| "* " -> (
			let a,b = List.hd l, List.nth l 1 in
			`Binop(untag_pos a, "*", ( *. ), untag_pos b, 0) )
		| "/ " -> (
			let a,b = List.hd l, List.nth l 1 in
			`Binop(untag_pos a, "/", ( /. ), untag_pos b, 0) )
		| "( " -> (
			let ar = Array.of_list l in
			let ll = List.init ((Array.length ar)/2) (fun i -> ar.(i*2)) 
				|> List.map untag_pos in
			`Seq(ll, 0) )
		| "loop " -> (
			let i,a,b = List.hd l, List.nth l 1, List.nth l 2 in
			match i with
			| `Leaf(ii,_) -> `Loop(ii+10, untag_pos a, untag_pos b, 0)
			| _ -> `Nop )
		| "c" -> (
			let i = List.hd l in
			match i with 
			| `Leaf(ii,_)-> `Call(ii+10, List.tl l |> List.map untag_pos, 0)
			| _ -> `Nop )
		| "d" -> (
			let i,a = List.hd l, List.nth l 1 in
			match i with 
			| `Leaf(ii,_) -> `Def(ii+10, untag_pos a, 0)
			| _ -> `Nop )
		| "pen " -> (
			let a = List.hd l in
			`Pen(untag_pos a, 0) )
		| "ua "-> `Const(6.283185307179586, 0)
		| "ul "-> `Const(1.0, 0)
		| _ -> ( (* int / const *)
			`Const((foi t) +. 10.0, 0) )
		)
	| `Leaf(t,_) -> ( 
		let ut = dec_item t in
		Printf.printf "dec leaf %d %s\n%!" t ut; 
		match ut with 
		| "ua "-> `Const(6.283185307179586, 0)
		| "ul "-> `Const(1.0, 0)
		| _ -> ( (* int / const *)
			`Const((foi t) +. 10.0, 0) ) )
	| _ -> `Nop
	;;
	
let test_tag_pos g = 
	(* simple test to see if the tagging makes sense .. *)
	let rec print_ienc e indent = 
		match e with
		| `Leaf(a,b) -> Printf.printf "%s[%d] %s\n" indent b (dec_item a)
		| `Node(a,b) -> 
			print_ienc a indent; 
			List.iter (fun c -> print_ienc c (indent^"  ")) b
		| _ -> ()
	in

	let h = ref 0 in
	let e = tag_pos g h in
	Printf.printf "total nodes: %d\n%!" !h; (* need this *)
	print_ienc e "" ; 
	Printf.printf "recon:\n%!"; 
	let gg = untag_pos e in
	Printf.printf "%s\n%!" (output_program_pstr gg)
	;;
	
type jenc = 
	(* flat array for sending to python *)
	{ enc : int (* int encoding of the prog symbol *)
	; pos : int
	; parent : int (* these could also be type int ref aka pointers *)
	; gparent : int
	; kid0 : int
	; kid1 : int
	; kidn : int (* last kid *)
	; l_sibling : int
	; r_sibling : int
}

let nuljenc = {
	enc = 0; 
	pos = -1; (* redundant, but w/e *)
	parent = -1; 
	gparent = -1; 
	kid0 = -1; 
	kid1 = -1; 
	kidn = -1; 
	l_sibling = -1; 
	r_sibling = -1; 
}
	
let tag_flatten_e e1 h = 
	let ar = Array.make h nuljenc in
	
	let unleaf k = 
		(* get the position of `ienc k *)
		match k with 
		| `Node(`Leaf(_,b),_) -> b
		| `Leaf(_,b) -> b
		| _ -> -1 in
		
	let is_delimiter a = 
		match a with 
		| `Leaf(b,_) -> (
			match b with 
			| 2 -> true (* ) *)
			| 3 -> true (* , *)
			| 4 -> true (* ; *)
			| _ -> false )
		| _ -> false in
		
	let remove_delimiters l = 
		List.filter (fun a -> not (is_delimiter a)) l in

	let rec flatten e ej left right = 
		match e with
		| `Node(`Leaf(a,b), l) -> (
				let al = remove_delimiters l |> Array.of_list in
				let n = Array.length al in
				Printf.printf "[%d] flatten, list len %d\n%!" b n ; 
				let kid0 = if n > 0 then unleaf al.(0) else -1 in
				let kid1 = if n > 1 then unleaf al.(1) else -1 in
				let kidn = if n > 0 then unleaf al.(n-1) else -1 in
				let f = { enc = a; pos = b; 
					parent = ej.pos; 
					gparent = ej.parent; 
					kid0; kid1; kidn; 
					l_sibling = left; 
					r_sibling = right; } in
				ar.(b) <- f; 
				(* iterate over the non-delimeters so they have "correct" *)
				(* redefine: iterate over whole list *)
				let al = Array.of_list l in
				let n = Array.length al in
				Array.iteri (fun i c -> 
					let ll = if i > 0 then al.(i-1) else `Nenc in
					let rr = if i < n-1 then al.(i+1) else `Nenc in
					let l2 = if is_delimiter ll && i > 1 
						then unleaf al.(i-2) else unleaf ll in
					let r2 = if is_delimiter rr && i < n-2 
						then unleaf al.(i+2) else unleaf rr in
						flatten c f l2 r2) al )
		| `Leaf(a,b) -> (
				let f = { enc = a; pos = b; 
					parent = ej.pos; 
					gparent = ej.parent; 
					kid0 = -1; kid1 = -1; kidn = -1; 
					l_sibling = left; 
					r_sibling = right; } in
				ar.(b) <- f; )
		| _ -> ()
	in
		
	flatten e1 nuljenc (-1) (-1); 
	ar
	;;
	
let tag_flatten g = 
	let h = ref 0 in
	let e = tag_pos g h in
	tag_flatten_e e !h 
	;;
	
let print_jenc e = 
	Printf.printf "[%d] \027[34m%s\027[0m \n   parent:%d gparent:%d kid0:%d kid1:%d kidn:%d l_sibling:%d r_sibling:%d\n%!"
	e.pos (dec_item e.enc) e.parent e.gparent e.kid0 e.kid1 e.kidn e.l_sibling e.r_sibling 
	;;
	
let test_tag_flatten g = 
	(* simple test to see if the tagging makes sense .. *)
	let ar = tag_flatten g in
	Array.iter print_jenc ar 
	;;
	
let move_jenc move c = {
		enc = c.enc; 
		pos = move c.pos; 
		parent = move c.parent; 
		gparent = move c.gparent; 
		kid0 = move c.kid0; 
		kid1 = move c.kid1; 
		kidn = move c.kidn; 
		l_sibling = move c.l_sibling; 
		r_sibling = move c.r_sibling }
	;;
	
let tag_insert ar pos chr = 
	let n = Array.length ar in
	assert (pos >= 0); 
	assert (pos <= n); 
	let move a = if a >= pos then a+1 else a in
	
	Array.init (n + 1)
		(fun i -> 
			if i < pos then move_jenc move ar.(i) 
			else (
				if i = pos then {nuljenc with enc = chr; pos}
				else ( (* i > pos *)
					move_jenc move ar.(i-1)
				) 
			)
		) 
	;;
	
let tag_delete ar pos = 
	let n = Array.length ar in
	assert (pos >= 0); 
	assert (pos < n);
	let move a = if a >= pos then a-1 else a in
	Array.init (n-1)
		(fun i -> 
			if i < pos then move_jenc move ar.(i)
			else move_jenc move ar.(i+1)
		)
	;;
	
let tag_substitute ar pos chr = 
	let a = ar.(pos) in
	ar.(pos) <- {a with enc = chr }; (* doesn't change links *)
	ar
	;;
	
let test_tag_edit g = 
	let print_sorted ar = Array.iter print_jenc ar in
		
	let ar = tag_flatten g in
	
	Printf.printf "---\ninserting to position 11 \"pen\"\n"; 
	let ar = tag_insert ar 11 (enc_char1 "pen") in
	print_sorted ar; 
	
	Printf.printf "---\ninserting to position 12 \"5\"\n"; 
	let ar = tag_insert ar 12 (5-10) in
	print_sorted ar;
	
	Printf.printf "---\ninserting to position 13 \";\"\n"; 
	let ar = tag_insert ar 13 (enc_char1 ";") in
	print_sorted ar;
	
	Printf.printf "---\ndeleting position 1 \n"; 
	let ar = tag_delete ar 1 in
	print_sorted ar;
	
	Printf.printf "---\ndeleting position 1 \n"; 
	let ar = tag_delete ar 1 in
	print_sorted ar;
	
	Printf.printf "---\ndeleting position 1 \n"; 
	let ar = tag_delete ar 1 in
	print_sorted ar;
	
	Printf.printf "---\nsubstituting pos 14 \"ua\"\n"; 
	let ar = tag_substitute ar 14 (enc_char1 "ua")in
	print_sorted ar;
	
	Printf.printf "---\nsubstituting pos 18 \"3\"\n"; 
	let ar = tag_substitute ar 14 (3-10)in
	print_sorted ar;
	;;
	
let progenc_cost s = 
	(* to break ties, add a slight bias for lower codes first *)
	let cost,_ = String.fold_left (fun (a,f) b -> 
		a +. ((foi (Char.code b)) *. f), 
		f *. 1.00058768275
		) (0.0,1.0) s in
	cost

	
and output_list_p bf l sep =
	Printf.bprintf bf "("; 
	List.iteri (fun i v -> 
		if i > 0 then Printf.bprintf bf "%s" sep ;
		output_program_p bf v) l ; 
	Printf.bprintf bf ")"


type state =
  { x : float
  ; y : float
  ; t : float (* theta *)
  ; p : float (* 1 = black ; 0 = clear ; -1 = white*)
  ; r : int (* execution count *)
  ; stk : float array
  }

let defs = Array.make 10 `Nop

type segment = float*float*float*float*float

let output_segments bf seglist = 
	List.iteri (fun i (x1,y1,x2,y2,a) -> 
		Printf.bprintf bf 
			"%d %f,%f %f,%f:%f\n" 
			i x1 y1 x2 y2 a) seglist
			
let output_segments_str seglist = 
	let bf = Buffer.create 64 in
	output_segments bf seglist; 
	(Buffer.contents bf)

let start_state () = 
	{x=d_from_origin; y=d_from_origin; t=0.0; p=1.0; r=0;
		stk=Array.make 10 (-1.0e9)}
 
(* eval needs to take a state & program
and return new state & (bool * float) result & segment list *)
let rec eval (st0:state) (pr:prog) = 
	let reclim = 512 in
	let overlim = reclim+10 in
	let nullret () = 
		(*Printf.printf "nulret!\n"; *)
		({st0 with r=overlim}, (false, (-1e9)), []) in
	if st0.r < reclim then (
	let st = {st0 with r=st0.r+1} in
	match pr with 
	| `Var(i,_) -> 
		if i >= 0 && i < 10 then (
			if st.stk.(i) < -1e6 then (
				nullret()
			) else (st, (true, st.stk.(i)), [])
		) else (
			nullret()
		)
	| `Save(i,a,_) -> 
		if i >= 0 && i < 10 then (
			let (sta, resa, seg) = eval st a in
			sta.stk.(i) <- (snd resa) ;
			(sta, resa, seg)
		)  else (
			nullret()
		)
	| `Move(a,b,_) -> (* distance, angle -- applied in that order *)
		let (sta, resa, _) = eval st a in
		let (stb, resb, _) = eval sta b in
		if stb.r < reclim then (
			let dist = snd resa in
			(* let dist = if dist < 0. then 0. else dist in (* not necc...*) *)
			let ang = snd resb in
			let t' = stb.t +. ang in
			let x' = stb.x +. (dist *. Float.cos(t')) in 
			let y' = stb.y +. (dist *. Float.sin(t')) in 
			let st2 = {stb with x = x'; y = y'; t = t' } in
			let seg = st.x, st.y, st2.x, st2.y, st2.p in
			(*Printf.printf "emitting seg (%d)\n" st2.r; *)
			(*Out_channel.flush stdout;*)
			(st2, (true, dist), [seg])
		) else nullret()
	| `Binop(a, s, f, b,_) -> 
		let (sta, resa, _) = eval st a in
		let (stb, resb, _) = eval sta b in
		if stb.r < reclim then (
			let ra = snd resa in
			let rb = snd resb in
			let r = match s with
			| "/" -> (
				if rb < 0.001 && rb > -0.001 
				then ra (* don't divide by zero *)
				else f ra rb )
			| _ -> f ra rb in
			(stb, (true, r), [])
		) else nullret()
	| `Const(f,_) -> 
		(st, (true, f), [])
	| `Seq(program_list,_) -> 
			List.fold_left (fun (st2,res2,segments) sub_prog -> 
				if st2.r < reclim then (
					let st3, res3, seg = eval st2 sub_prog in
					(st3, res3, (List.append seg segments) ) 
				) else (st2,res2,segments) )
				(st, (true,0.0), []) program_list
	| `Loop(indx, niter, body,_) -> 
		if indx >= 0 && indx < 10 then (
			let (sta, resa, _) = eval st niter in
			let n = iof (snd resa) in
			let n = if n <= 0 then 1 else n in (* not sure how it gets there.. *)
			let cntlist = List.init n (fun i -> i) in
			List.fold_left (fun (st2,res2,segments) i -> 
				if st2.r < reclim then (
					st2.stk.(indx) <- foi i;
					let st3, res3, seg = eval st2 body in
					(st3, res3, (List.append seg segments) ) 
				) else (st2,res2,segments) )
				(sta, (true,0.0),[]) cntlist
		) else nullret()
	| `Call(indx, program_list,_) ->
		(* this needs to be updated for the recursion limit *)
		if defs.(indx) <> `Nop then (
			if List.length program_list < 5 then (
				(* make a new stack and populate it *)
				(* arguments are not allowed to have side-effects *)
				let res = List.map (fun subprog ->
					let _st, res2, _seg = eval st subprog in
					res2 ) program_list in
				let st3 = {x=st.x;y=st.y;t=st.t;p=st.p;r=st.r; stk=(Array.make 10 (-1e9)) } in
				List.iteri (fun i v -> st3.stk.(i) <- snd v) res ;
				let _st4, res4, seg4 = eval st3 defs.(indx) in
				(* call does not affect state *)
				(st, res4, seg4)
			) else nullret()
		) else nullret()
	| `Def(indx, body, _) ->
		if (indx >= 0 && indx < 10) then (
			defs.(indx) <- body ) ;
		(st, (false, 0.0), [])
	| `Pen(a,_) -> 
		let (sta, (_,resa), _) = eval st a in
		(* you can scale the alpha with fractions .. *)
		let p = Float.max (Float.min resa 5.0) (-5.0) in
		({sta with p},(true, resa), [])
	| `Nop -> (st, (false, 0.0), [])
	) else (st0, (false, 0.0), [])

let center_segs l =
	let rec minimum = function
		| [] -> assert (false)
		| [x] -> x
		| h :: t -> min h (minimum t)
	and maximum = function
		| [] -> assert (false)
		| [x] -> x
		| h :: t -> max h (maximum t)
	in 
	match l with
	| [] -> []
	| _ ->
	(* build lists of all the x and y coordinates *)
		let xs = l |> List.map
		(function
			| (x,_,x',_,_) ->
				[x-.d_from_origin;
				x'-.d_from_origin;]) |> List.concat in
		let ys = l |> List.map
		(function
			| (_,y,_,y',_) ->
				[y-.d_from_origin;
				y'-.d_from_origin;]) |> List.concat in
		let x0 = xs |> minimum in
		let x1 = xs |> maximum in
		let y0 = ys |> minimum in
		let y1 = ys |> maximum in
		(* find center of min and max, x and y *)
		let dx = (x1-.x0)/.2.+.x0 in
		let dy = (y1-.y0)/.2.+.y0 in
		let d_from_origin = 0. in
		(* translate all the coordinates *)
		l |> List.map (fun(x,y,x',y',a) ->
			(x-.dx+.d_from_origin, y-.dy+.d_from_origin,
						x'-.dx+.d_from_origin, y'-.dy+.d_from_origin,a))
		
		
let segs_to_canvas segs =
  (*let segs = center_segs segs in*)
  let c = ref (new_canvas ()) in
  let lineto x y a = (c := (lineto !c x y a)) 
  and moveto x y = (c := (moveto !c x y)) in
  (* lineto and moveto are defined in vgwrapper.ml *)
  let total_cost = ref 0. in
  let eval_instruction (x1,y1,x2,y2,a) =
      total_cost := !total_cost +. 
			(sqrt ((x1-.x2)*.(x1-.x2) +. (y1-.y2)*.(y1-.y2))); 
			(* length of the line *)
		moveto x1 y1;
		lineto x2 y2 a; ()
  in
  List.iter eval_instruction segs ;
  !c,!total_cost

let segs_to_png segs resolution filename =
  let canvas,_ = segs_to_canvas segs in
  output_canvas_png canvas resolution filename

let segs_to_array_and_cost segs res =
	(* outputs a *float* image and cost *)
	(* float so we don't have to convert all the time.  more mem b/w tho*)
	let canvas,cost = segs_to_canvas segs in
	(* don't render if the resolution is zero, obvi *)
	if res > 0 then (
		let img = canvas_to_1Darray canvas res in
		(* these sometimes have stride > resolution; pack it *)
		let stride = (Bigarray.Array1.dim img) / res in
		let len = Bigarray.Array1.dim img in
		assert (len >= res * res);
		let o = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout (res*res) in
		for i = 0 to res-1 do (
			for j = 0 to res-1 do (
				let c = Bigarray.Array1.get img ((i*stride)+j) in
				o.{i*res+j} <- c ; 
			) done; 
		) done;
		(o, cost)
	) else ( nulimg, cost)

