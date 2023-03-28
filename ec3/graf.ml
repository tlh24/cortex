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

module SI = Set.Make(Int);;

let nulimg = Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout 1 1

type progtyp = [
	| `Uniq  (* replace this with Bigarray image + imagef *)
	| `Equiv (* int = pointer to simplest *)
	| `Np 
]

type edata = 
	{ pro : Logo.prog
	; progenc : string
	; progaddr : int list list 
	; scost : float
	; pcost : int
	; segs : Logo.segment list
	}

type gdata = 
	{ progt : progtyp
	; ed : edata
	; imgi : int (* index to torch tensors *)
	; outgoing : SI.t (* within edit distance *)
	; equivroot : int 
	; equivalents : SI.t
	; good : int (* count of correct decoding *)
	(* equivalents is like union-find: Uniq nodes points to all equivalents, 
		Equiv node points to the minimum cost.  
		Network of edit-distance similarity unaffected *)
	}
	
type tpimg = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t

let nuledata = 
	{ pro = `Nop
	; progenc = ""
	; progaddr = []
	; scost = 0.0
	; pcost = 0
	; segs = []
	}

let nulgdata = 
	{ progt = `Np
	; ed = nuledata
	; imgi = (-1)
	; outgoing = SI.empty
	; equivroot = (-1)
	; equivalents = SI.empty
	; good = 0
	}
	
type gstat = 
	{ g : gdata Vector.t
	; image_alloc : int
	; img_inv : int array (* points back to g, g[img_inv.(i)].imgi = i *)
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
	let pe = d.ed.progenc in
	let nearby = ref [] in
	Vector.iteri (fun i a -> 
		if i <> indx then (
		match a.progt with 
		| `Uniq -> (
			let dist,edits = get_edits pe a.ed.progenc in
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
		let d2o = SI.add indx d2.outgoing in
		Vector.set g i {d2 with outgoing=d2o} ) (!nearby) ; 
	(* update current node *)
	Vector.set g indx {d with outgoing=(SI.of_list (!nearby)) }
	
let add_uniq gs d = 
	(* add a unique node to the graph structure *)
	(* returns where it's stored; for now = imgf index *)
	let l = gs.num_uniq in
	if l < gs.image_alloc then (
		let d' = {d with imgi = l } in
		Vector.push gs.g d'; 
		let ni = (Vector.length gs.g) - 1 in
		connect_uniq gs.g ni; 
		gs.num_uniq <- gs.num_uniq + 1; 
		ni,l (* index to gs.g and dbf respectively *)
	) else (-1),(-1)
	
let replace_equiv gs indx d2 =
	(* d2 is equivalent to gs.g[indx], but lower cost *)
	(* add d2 the end, and update d1 = g[indx]. *)
	let d1 = Vector.get gs.g indx in
	let d2' = {d2 with equivalents = (SI.add indx d1.equivalents); 
							imgi = d1.imgi} in
	Vector.push gs.g d2';
	gs.num_equiv <- gs.num_equiv + 1; 
	let ni = (Vector.length gs.g) - 1 in (* new index *)
	gs.img_inv.(d2'.imgi) <- ni ; (* back pointer *)
	(* update the incoming equivalent pointers; includes d1 !  *)
	SI.iter (fun i -> 
		let e = Vector.get gs.g i in
		let e' = {e with progt = `Equiv; equivroot = ni; equivalents = SI.empty} in
		Vector.set gs.g i e' ) d2'.equivalents; 
	(* finally, update d2's edit connections *)
	connect_uniq gs.g ni ; 
	ni (* return the location of the new node, same as add_uniq *)
	
let add_equiv gs indx d2 =
	(* d2 is equivalent to gs.g[indx], but higher (or equivalent) cost *)
	(* union-find the root equivalent *)
	let rec find_root j = 
		let d = Vector.get gs.g j in
		if d.equivroot >= 0 then 
			find_root d.equivroot
		else
			j in
	let equivroot = find_root indx in
	let d1 = Vector.get gs.g equivroot in
	(* need to verify that this is not in the graph.*)
	let has = SI.fold (fun j a -> 
		let d = Vector.get gs.g j in
		if d.ed.progenc = d2.ed.progenc then true else a) 
		d1.equivalents false in
	if not has then (
		let d2' = {d2 with equivroot; imgi = d1.imgi} in
		Vector.push gs.g d2'; 
		gs.num_equiv <- gs.num_equiv + 1;
		let ni = (Vector.length gs.g) - 1 in (* new index *)
		let d1' = {d1 with equivalents = SI.add ni d1.equivalents} in
		Vector.set gs.g equivroot d1'; 
		connect_uniq gs.g ni ; 
		ni (* return the location of the new node, same as add_uniq *)
	) else (-1)
	
let incr_good g i = 
	(* hopefully the compiler replaces this with an in-place op *)
	let d = Vector.get g i in
	let d' = {d with good = (d.good + 1)} in
	Vector.set g i d' 
	
let sort_graph g = 
	(* sort the array by scost ascending; return new vector *)
	let ar = Vector.to_array g |>
		Array.mapi (fun i a -> (i,a)) in
	Array.sort (fun (_,a) (_,b) -> compare a.ed.pcost b.ed.pcost) ar ; 
	let indxs,sorted = Array.split ar in
	(* indxs[i] = original pointer in g *)
	(* invert the indexes so that mappin[i] points to the sorted array *)
	let mappin = Array.make (Array.length indxs) 0 in
	Array.iteri (fun i a -> mappin.(a) <- i; 
		Printf.printf "old %d -> new %d\n" a i) indxs; 
	Array.map (fun a -> 
		let outgoing = SI.map (fun b -> mappin.(b)) a.outgoing in
		let equivalents = SI.map (fun b -> mappin.(b)) a.equivalents in
		let equivroot = if a.equivroot >= 0 then
			mappin.(a.equivroot) else (-1) in
		{a with outgoing; equivalents; equivroot} ) sorted |>
		Vector.of_array ~dummy:nulgdata
	(* does not update the img indexes *)

let progt_to_str = function
	| `Uniq -> "uniq" | `Equiv -> "equiv" | `Np -> "np"
	
let str_to_progt = function
	| "uniq" -> `Uniq | "equiv" -> `Equiv | "np" -> `Np | _ -> `Np
	
let print_graph g = 
	let print_node i d = 
		let c = progt_to_str d.progt in
		Printf.printf "node [%d] = %s '%s'\n\tcost:%.2f, %d imgi:%d\n"
			i c (Logo.output_program_pstr d.ed.pro) d.ed.scost d.ed.pcost d.imgi; 
		Printf.printf "\toutgoing: "; 
		SI.iter (fun j -> Printf.printf "%d," j) d.outgoing;
		Printf.printf "\tequivalents: "; 
		SI.iter (fun j -> Printf.printf "%d," j) d.equivalents; 
		Printf.printf "\tequivroot: \027[31m %d\027[0m\n" d.equivroot
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
	
let pro_to_edata pro res = 
	let (_,_,segs) = Logo.eval (Logo.start_state ()) pro in
	let img,scost = Logo.segs_to_array_and_cost segs res in
	let proglst,progaddr = Logo.encode_ast pro in
	let progenc = Logo.intlist_to_string proglst in
	let pcost = Logo.progenc_cost progenc in
	{pro; progenc; progaddr; scost; pcost; segs },img
	
let pro_to_edata_opt pro res = 
	(* returns `None if the prorgam doesn't meet segment criteria *)
	let ed,img = pro_to_edata pro res in
	let lx,hx,ly,hy = Ast.segs_bbx ed.segs in
	let dx = hx-.lx in
	let dy = hy-.ly in
	let maxd = max dx dy in
	if maxd >= 2. && maxd <= 9. && ed.scost >= 4. && ed.scost <= 64. 
	&& List.length ed.segs < 8 && String.length ed.progenc < 24 then (
		Some (ed, img)
	) else None 
	
	
let save fname g =
	let open Sexplib in
	(* output directly, preserving order -- indexes should be preserved *)
	let intset_to_sexp o = 
		Sexp.List (List.map (fun a -> Sexp.Atom (string_of_int a) )
			(SI.elements o) ) 
	in
	let sexp = Sexp.List ( List.map (fun d -> 
		Sexp.List 
			[ Sexp.Atom (progt_to_str d.progt) 
			; Sexp.Atom (Logo.output_program_pstr d.ed.pro)
			; (intset_to_sexp d.outgoing)
			; (intset_to_sexp d.equivalents)
			; Sexp.Atom (string_of_int d.equivroot)
			; Sexp.Atom (string_of_int d.good) ] )
			(Vector.to_list g) )
	in
	let oc = open_out fname in
	Sexplib.Sexp.output_hum oc sexp;
	close_out oc

let load fname res = 
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
			| _ -> (-1) ) k |> SI.of_list in
	let num_uniq = ref 0 in
	let gl = List.map (function 
		| [Sexp.Atom pt; Sexp.Atom pstr; out; equiv; 
							Sexp.Atom eqrt; Sexp.Atom gd] -> (
			let prog = parse_logo_string pstr in
			(match prog with
			| Some(pro) -> 
				let progt = (str_to_progt pt) in
				let outgoing = (strip_int out) in
				let equivalents = (strip_int equiv) in
				let equivroot = (ios eqrt) in
				let good = (ios gd) in
				let imgi,res2 = match progt with 
					| `Uniq -> incr num_uniq; (!num_uniq - 1),res
					| _ -> (-1),4 in
				let ed,img = pro_to_edata pro res2 in
				let d = {progt; ed; imgi ; outgoing; equivalents; equivroot; good} in
				d,img
			| _ -> assert(0 <> 0); nulgdata,nulimg) )
		| _ -> failwith "Invalid S-expression format") sexp' in
	let gll,imgs = List.split gl in
	let g = Vector.of_list ~dummy:nulgdata gll in
	Printf.printf "check!!!\n"; 
	print_graph g; 
	(* need to go back and fix the imgi indexes *)
	Vector.iteri (fun i d -> 
		match d.progt with 
		| `Equiv -> (
			let e = Vector.get g d.equivroot in
			Vector.set g i {d with imgi = e.imgi} )
		| `Uniq -> ()
		| _ -> Printf.printf "error at %d %d\n" i d.ed.pcost; assert (0 <> 0);
		) g; 
	g,imgs
	
(* test code *)
let () = 
	let gs = create 5 in
	
	let add_str mode equiv s = 
		let prog = parse_logo_string s in
		let l = match prog with
		| Some(pro) -> (
			let ed,_img = pro_to_edata pro 4 in
			if mode = `Uniq then (
				if equiv >= 0 then (
					let d = {nulgdata with ed; progt = mode} in
					replace_equiv gs equiv d ; (* fills out connectivity *)
				) else (
					let d = {nulgdata with ed; progt = mode} in
					let a,_ = add_uniq gs d in (* fills out connectivity *)
					a
				) 
			) else (
				let d = {nulgdata with ed; progt = mode} in
					add_equiv gs equiv d ;
			) )
		| _ -> 0 in
		if l >= 0 then 
			Printf.printf "added [%d] = '%s' \n" l s
		else 
			Printf.printf "did not add [%d] = '%s' \n" l s
	in
	
	add_str `Uniq (-1) "( move ua , ua / 4 ; move ua , 3 / 3 )"; 
	add_str `Uniq (-1) "( move ua , ua / 4 ; move ua , ua / 3 )";
	add_str `Uniq (-1) "( move ua , 0 ; move ua , ua / 3 )"; 
	add_str `Uniq  0   "( move ua , ua / 4 ; move ua , 1 )";
	add_str `Equiv 0   "( move ua , ua / 4 ; move ua , ul * ul )";
	add_str `Equiv 0   "( move ua , ua / 4 ; move ua , ul * ul )";
	incr_good gs.g 0; 
	
	print_graph gs.g; 
	let sorted = sort_graph gs.g in
	Printf.printf "--- sorted ---\n"; 
	print_graph sorted ; 
	Printf.printf "--- saving & reloaded ---\n"; 
	let fname = "db_prog.S" in
	save fname sorted; 
	let gp,_ = load fname 30 in
	print_graph gp
	
