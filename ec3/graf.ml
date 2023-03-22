open Logo
open Levenshtein

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

type progtyp = [
	| `Uniq  (* replace this with Bigarray image + imagef *)
	| `Equiv (* int = pointer to simplest *)
	| `Np 
]

type gdata = 
	{ progt : progtyp
	; pro : Logo.prog
	; progenc : string
	; proaddr : int list list (* TODO *)
	; scost : float
	; pcost : int
	; img : int (* index to torch tensor(s), including encoding *)
	; outgoing : int list
	; equivalents : int list 
	(* equivalents is like union-find: Uniq nodes points to all equivalents, 
		Equiv node points to the minimum cost.  
		Network of edit-distance similarity unaffected *)
	}
	
type tpimg = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t

let nulgdata = 
	{ progt = `Np
	; pro = `Nop
	; progenc = ""
	; proaddr = []
	; scost = 0.0
	; pcost = 0
	; img = 0
	; outgoing = []
	; equivalents = []
	}

let create () =
	Vector.create ~dummy:nulgdata
	
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
		!r
	) else false

let get_edits a_progenc b_progenc = 
	let dist,edits = Levenshtein.distance pe pd.progenc true in
	let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
	dist,edits
	
let connect_node_uniq g indx = 
	let d = Vector.get g indx in
	(* need to connect to the rest of the graph *)
	let pe = d.progenc in
	let nearby = ref [] in
	Vector.iteri g (fun i a -> 
		match a.progt with 
		| `Uniq -> (
			let dist,edits = get_edits pe a.progenc true in
			let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
			if edit_criteria edits then (
					nearby := [i :: !nearby] 
			) )
		| _ -> () ); 
	List.iter (fun i -> 
		let d2 = Vector.get g i in
		let d2o = List.filter (fun a -> a <> indx) d2.outgoing in
		let d2o' = [indx :: d2o] in
		Vector.set g i {d2 with outgoing=d2o'} ) (!nearby) ; 
	Vector.set g i {d with outgoing=(!nearby)}
	
let add_uniq g d = 
	(* add a unique node to the graph structure *)
	Vector.push g d; 
	let indx = Vector.length g in
	connect_node_uniq g indx; 
	indx
	
let replace_equiv g indx d2 =
	(* d2 is equivalent to g[indx], but lower cost *)
	(* add d2 the end, and update d1 = g[indx]. *)
	let d1 = Vector.get g indx in
	(* d1 gets added to it's own equivalents list *)
	let d2' = {d2 with equivalents=[indx::d1.equivalents]} in
	Vector.push g d2'; 
	let ni = Vector.length g in
	(* update the incoming equivalent pointers *)
	List.iter (fun i -> 
		let e = Vector.get g i in
		let e' = {e with equivalents = [ni]} in
		Vector.set g i e') d2'.equivalents; 
	(* update d1 *)
	let d1' = {d1 with progt = `Equiv } in
	Vector.set g indx d1'; 
	(* update d2's edit connections *)
	(* ideally, we'd use the existing structure ... but will be slow here *)
	connect_node_uniq g ni
	
let sort g = 
	(* sort the array by scost; return indexes *)
	let indxs,sorted = 
		Vector.to_array g |>
		Array.mapi (fun i a -> (i,a))  |>
		Array.sort (fun (_,a) (_,b) -> compare a.scost b.scost) |> 
		Array.split in
	(* indxs[i] = original pointer in g *)
	(* invert the indexes so that mappin[i] points to the sorted array *)
	let mappin = Array.make (Array.length indxs) 0 in
	Array.iteri (fun i a -> mappin.(a) = i) indxs; 
	Array.map (fun a -> 
		let outgoing = List.map (fun b -> mappin.(b)) a.outgoing in
		let equivalents = List.map (fun b -> mappin.(b)) a.equivalents in
		{a with outgoing; equivalents} ) sorted |>
		Vector.from_array
