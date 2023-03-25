open Lexer
open Lexing
(*open Logo*)
(*open Levenshtein*)

(* the Vector stores the actual data,
while the edges are stored as integers (pointers)
edges are doubly linked: if that node is updated, it changes
all nodes connected to it - and changes those nodes links too 
given that during running, we will not insert into the middle of the vector, 
only add to the end, or replace (not delete) in the middle, the links are stored as integer indexes within the vector.  
This makes access fast and simple, but requires care to maintain the mapping

Changing the representation of equivalents: they are still members of the same graph, which allows them to be nodes in a path (e.g. on the transition graph for refactors and such), only they don't possess image data.  
The original implementation kept these separate. 
*)
let soi = string_of_int
let ios = int_of_string

type progtyp = [
	| `Uniq  (* replace this with Bigarray image + imagef *)
	| `Equiv (* int = pointer to simplest *)
	| `Np 
]

type gdata = 
	{ progt : progtyp
	; pro : Logo.prog
	; progenc : string
	; progaddr : int list list (* TODO *)
	; scost : float
	; pcost : int
	; segs : Logo.segment list
	; img : int (* index to torch tensor(s), including encoding *)
	; outgoing : int list
	; equivalents : int list 
	; good : int (* count of correct decoding *)
	(* equivalents is like union-find: Uniq nodes points to all equivalents, 
		Equiv node points to the minimum cost.  
		Network of edit-distance similarity unaffected *)
	}
	
type tpimg = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t

let nulgdata = 
	{ progt = `Np
	; pro = `Nop
	; progenc = ""
	; progaddr = []
	; scost = 0.0
	; pcost = 0
	; segs = []
	; img = 0
	; outgoing = []
	; equivalents = []
	; good = 0
	}
	
type gstat = 
	{ g : gdata Vector.t
	; image_alloc : int
	; img_inv : int array (* points back to g, g[img_inv.(i)].img = i *)
	; mutable num_uniq : int
	; mutable num_equiv : int
	}

let create image_alloc =
	let g = Vector.create ~dummy:nulgdata in
	let img_inv = Array.make image_alloc 0 in
	{g; image_alloc; img_inv; num_uniq = 0; num_equiv = 0 }
	
let edit_criteria edits = 
	let count_type typ = 
		List.fold_left 
		(fun a (t,_p,_c) -> if t = typ then a+1 else a) 0 edits 
	in
	if List.length edits > 0 then (
		let nsub = count_type "sub" in
		let ndel = count_type "del" in
		let nins = count_type "ins" in
		let r = ref false in
		if nsub <= 3 && ndel = 0 && nins = 0 then r := true;
		if nsub = 0 && ndel <= 6 && nins = 0 then r := true; 
		if nsub = 0 && ndel = 0 && nins <= 8 then r := true;
		if nsub <= 1 && ndel = 0 && nins <= 3 then r := true;
		!r
	) else false

let get_edits a_progenc b_progenc = 
	let dist,edits = Levenshtein.distance a_progenc b_progenc true in
	let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
	dist,edits
	
let connect_uniq g indx = 
	let d = Vector.get g indx in
	(* need to connect to the rest of the graph *)
	let pe = d.progenc in
	let nearby = ref [] in
	Vector.iteri (fun i a -> 
		if i <> indx then (
		match a.progt with 
		| `Uniq -> (
			let dist,edits = get_edits pe a.progenc in
			let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
			if edit_criteria edits then (
				Printf.printf "connecting outgoing [%d] to [%d] dist %d\n" i indx dist; 
				nearby := i :: !nearby 
			) else ( 
				Printf.printf "not connecting outgoing [%d] to [%d] dist %d \n" i indx dist
			) )
		| _ -> () ) ) g; 
	List.iter (fun i -> 
		let d2 = Vector.get g i in
		let d2o = List.filter (fun a -> a <> indx) d2.outgoing in
		let d2o' = indx :: d2o in
		Vector.set g i {d2 with outgoing=d2o'} ) (!nearby) ; 
	(* update current node *)
	Vector.set g indx {d with outgoing=(!nearby)}
	
let add_uniq gs d = 
	(* add a unique node to the graph structure *)
	(* returns where it's stored; for now = imgf index *)
	let l = gs.num_uniq in
	if l < gs.image_alloc then (
		let d' = {d with img = l } in
		Vector.push gs.g d'; 
		connect_uniq gs.g l; 
		gs.num_uniq <- gs.num_uniq + 1; 
		l
	) else 0
	(* at some point, will need to replace old / less-useful nodes.
		TBD there! *)
	
let replace_equiv gs indx d2 =
	(* d2 is equivalent to gs.g[indx], but lower cost *)
	(* add d2 the end, and update d1 = g[indx]. *)
	let d1 = Vector.get gs.g indx in
	(* d1 gets added to it's own equivalents list *)
	let d2' = {d2 with equivalents= indx::d1.equivalents; 
							img = d1.img} in
	Vector.push gs.g d2';
	gs.num_equiv <- gs.num_equiv + 1; 
	let ni = (Vector.length gs.g) - 1 in (* new index *)
	gs.img_inv.(d2'.img) <- ni ; (* back pointer *)
	(* update the incoming equivalent pointers *)
	List.iter (fun i -> 
		let e = Vector.get gs.g i in
		let e' = {e with equivalents = [ni]} in
		Vector.set gs.g i e') d2'.equivalents; 
	(* update d1 *)
	let d1' = {d1 with progt = `Equiv } in
	Vector.set gs.g indx d1'; 
	(* finally, update d2's edit connections *)
	connect_uniq gs.g ni ; 
	ni (* return the location of the new node, same as add_uniq *)
	
let add_equiv gs indx d2 =
	(* d2 is equivalent to gs.g[indx], but higher cost *)
	(* yeah, this might not be so useful .. eh *)
	let d1 = Vector.get gs.g indx in
	let d2' = {d2 with equivalents = [indx]; img = d1.img} in
	Vector.push gs.g d2'; 
	gs.num_equiv <- gs.num_equiv + 1;
	let ni = (Vector.length gs.g) - 1 in (* new index *)
	let d1' = {d1 with equivalents = ni::d1.equivalents} in
	Vector.set gs.g indx d1'; 
	connect_uniq gs.g ni ; 
	ni (* return the location of the new node, same as add_uniq *)
	
let incr_good g i = 
	(* hopefully the compiler replaces this with an in-place op *)
	let d = Vector.get g i in
	let d' = {d with good = (d.good + 1)} in
	Vector.set g i d' 
	
let sort_graph g = 
	(* sort the array by scost ascending; return new vector *)
	let ar = Vector.to_array g |>
		Array.mapi (fun i a -> (i,a)) in
	Array.sort (fun (_,a) (_,b) -> compare a.scost b.scost) ar ; 
	let indxs,sorted = Array.split ar in
	(* indxs[i] = original pointer in g *)
	(* invert the indexes so that mappin[i] points to the sorted array *)
	let mappin = Array.make (Array.length indxs) 0 in
	Array.iteri (fun i a -> mappin.(a) <- i; 
		Printf.printf "old %d -> new %d\n" a i) indxs; 
	Array.map (fun a -> 
		let outgoing = List.map (fun b -> mappin.(b)) a.outgoing in
		let equivalents = List.map (fun b -> mappin.(b)) a.equivalents in
		{a with outgoing; equivalents} ) sorted |>
		Vector.of_array ~dummy:nulgdata
	(* does not update the img indexes *)

let progt_to_str = function
	| `Uniq -> "uniq" | `Equiv -> "equiv" | `Np -> "np"
	
let str_to_progt = function
	| "uniq" -> `Uniq | "equiv" -> `Equiv | "np" -> `Np | _ -> `Np
	
let print_graph g = 
	let print_node i d = 
		let c = progt_to_str d.progt in
		Printf.printf "node [%d] = %s '%s'\n\tcost:%f,%d img:%d\n"
			i c (Logo.output_program_pstr d.pro) d.scost d.pcost d.img; 
		Printf.printf "\toutgoing: "; 
		List.iter (fun j -> Printf.printf "%d," j) d.outgoing;
		Printf.printf "\tequivalents: "; 
		List.iter (fun j -> Printf.printf "%d," j) d.equivalents; 
		Printf.printf "\n"
	in
	Vector.iteri print_node g 
	
(* these functions from program.ml *)
let parse_with_error lexbuf =
	let prog = try Some (Parser.parse_prog Lexer.read lexbuf) with
	| SyntaxError msg -> Printf.printf "SyntaxError %s\n" msg; None
	| Parser.Error -> None in
	prog 

let parse_logo_string s = 
	let lexbuf = Lexing.from_string s in
	lexbuf.lex_curr_p <- { lexbuf.lex_curr_p with pos_fname = "from string" };
	parse_with_error lexbuf 
	
let pro_to_gdata pro = 
	let (_,_,segs) = Logo.eval (Logo.start_state ()) pro in
	let _img,scost = Logo.segs_to_array_and_cost segs 30 in
	let proglst,progaddr = Logo.encode_ast pro in
	let progenc = Logo.intlist_to_string proglst in
	let pcost = Logo.progenc_cost progenc in
	{nulgdata with pro; progenc; progaddr; scost; pcost; segs }
	(* caller needs to change progt *)
	
let save fname g =
	let open Sexplib in
	(* output directly, preserving order -- indexes should be preserved *)
	let intlist_to_sexp o = 
		Sexp.List (List.map (fun a -> Sexp.Atom (string_of_int a) ) o) 
	in
	let sexp = Sexp.List ( List.map (fun d -> 
		Sexp.List 
			[ Sexp.Atom (progt_to_str d.progt) 
			; Sexp.Atom (Logo.output_program_pstr d.pro)
			; (intlist_to_sexp d.outgoing)
			; (intlist_to_sexp d.equivalents)
			; Sexp.Atom (string_of_int d.good) ] )
			(Vector.to_list g) )
	in
	let oc = open_out fname in
	Sexplib.Sexp.output_hum oc sexp;
	close_out oc

let load fname = 
	let open Sexplib in
	let ic = open_in fname in
	let sexp = Sexp.input_sexps ic |> List.hd in
	close_in ic;
	(* extra layer of 'list'; not sure why *)
	let sexp' = Conv.list_of_sexp (function 
		|  Sexp.List(a) -> a
		| _ ->  []) sexp in
	let strip_int k =
		Conv.list_of_sexp (function
			| Sexp.Atom s -> ios s
			| _ -> (-1) ) k in
	let gl = List.map (function 
		| [Sexp.Atom pt; Sexp.Atom pstr; out; equiv; Sexp.Atom gd] -> (
			let prog = parse_logo_string pstr in
			(match prog with
			| Some(pro) -> 
				let d = pro_to_gdata pro in
				let d = {d with 
					progt = (str_to_progt pt); 
					outgoing = (strip_int out); 
					equivalents = (strip_int equiv); 
					good = (ios gd)} in
				d 
			| _ -> assert(0 <> 0); nulgdata) )
		| _ -> failwith "Invalid S-expression format") sexp' in
	Vector.of_list ~dummy:nulgdata gl
	
(* test code *)
let () = 
	let gs = create 5 in
	
	let add_str equiv s = 
		let prog = parse_logo_string s in
		let l = match prog with
		| Some(pro) -> (
			let d = pro_to_gdata pro in
			if equiv >= 0 then (
				let d = {d with progt = `Uniq} in
				replace_equiv gs equiv d ; (* fills out connectivity *)
			) else (
				let d = {d with progt = `Uniq} in
				add_uniq gs d ; (* fills out connectivity *)
			) )
		| _ -> 0 in
		Printf.printf "added [%d] = '%s' \n" l s
	in
	
	add_str (-1) "( move ua , ua / 4 ; move ua , 3 / 3 )"; 
	add_str (-1) "( move ua , ua / 4 ; move ua , ua / 3 )";
	add_str (-1) "( move ua , 0 ; move ua , ua / 3 )"; 
	add_str 0    "( move ua , ua / 4 ; move ua , 1 )"; 
	
	print_graph gs.g; 
	let sorted = sort_graph gs.g in
	Printf.printf "--- sorted ---\n"; 
	print_graph sorted ; 
	Printf.printf "--- saving & reloaded ---\n"; 
	let fname = "db_prog.S" in
	save fname sorted; 
	print_graph (load fname)
	
(* Todo: move imgf here & manage it when loading / reloading *)
