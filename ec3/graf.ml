open Lexer
open Lexing
open Domainslib

(* the Array stores the actual data,
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
let foi = float_of_int
let iof = int_of_float

module SI = Set.Make( 
	(* set for outgoing connections *)
	(* node#, edit type, edit count, cost *)
	(* type and count are for sorting / selecting *)
	struct
		let compare (a,_,_,_) (b,_,_,_) = compare a b
		type t = int * string * int * float
	end ) ;;

type progtyp = [
	| `Uniq  (* replace this with Bigarray image + imagef *)
	| `Equiv (* int = pointer to simplest *)
	| `Np 
]

let progt_to_str = function
	| `Uniq -> "uniq" | `Equiv -> "equiv" | `Np -> "np"
	
let str_to_progt = function
	| "uniq" -> `Uniq | "equiv" -> `Equiv | _ -> `Np

type edata = 
	{ pro : Logo.prog
	; progenc : string
	; progaddr : int list list 
	; scost : float
	; pcost : float
	; segs : Logo.segment list
	}

type gdata = 
	{ progt : progtyp
	; ed : edata
	; imgi : int (* index to torch tensors *)
	; outgoing : SI.t (* within edit distance *)
	; equivroot : int 
	; equivalents : SI.t
	; good : int (* count of correct decoding / used *)
	(* equivalents is like union-find: Uniq nodes points to all equivalents, 
		Equiv node points to the minimum cost.  
		Network of edit-distance similarity unaffected *)
	}
	
type tpimg = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t

let nuledata = 
	{ pro = `Nop
	; progenc = ""
	; progaddr = []
	; scost = 999999.0
	; pcost = 999999.0
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
	{ g : gdata array
	; image_alloc : int (* number of uniq allocated in torch mem*)
	; all_alloc : int	  (* number of uniq + equiv = Array.len g *)
	; img_inv : int array (* points back to g, g[img_inv.(i)].imgi = i *)
	; mutable free_slots : int list (* in g *)
	; mutable free_img : int list (* in g *)
	; mutable num_uniq : int
	; mutable num_equiv : int
	}

let create all_alloc image_alloc =
	let g = Array.make all_alloc nulgdata in
	let img_inv = Array.make image_alloc 0 in
	let free_slots = List.init all_alloc (fun i -> i) in
	let free_img = List.init image_alloc (fun i -> i) in
	{g; image_alloc; all_alloc; img_inv; free_slots; free_img; num_uniq = 0; num_equiv = 0 }

let count_edit_types edits = 
	let count_type typ = 
		List.fold_left 
		(fun a (t,_p,_c) -> if t = typ then a+1 else a) 0 edits 
	in
	let nsub = count_type "sub" in
	let ndel = count_type "del" in
	let nins = count_type "ins" in
	nsub,ndel,nins

let edit_cost edits maxlen =
	(* find the first edit *)
	let firsted = List.fold_left
			(fun a (_t,p,_c) ->
				if p < a then p else a) 99999 edits in
	(* number of edits minus position of first edit *)
	(* might require tweaking!! TODO *)
	let edlen = foi (List.length edits) *. 0.8 in
	let edpos = foi (maxlen - firsted) *. 0.5 in
	let cost = edlen +. edpos *. edpos in
	(*Logs.debug (fun m->m "edit cost %f (%f %f)" cost edlen edpos);*)
	cost
	;;
	
let edit_criteria_b edits = 
	if List.length edits > 0 && List.length edits <= 6 then (
		let nsub,ndel,nins = count_edit_types edits in
		let typ = ref "nul" in
		(* note! these rules must be symmetric for the graph to make sense *)
		if nsub > ndel && nsub > nins then typ := "sub";
		if ndel > nsub && ndel > nins then typ := "del"; 
		if nins > nsub && nins > ndel then typ := "ins"; 
		true,!typ
	) else false,"nul"
	
let edit_criteria edits maxlen = 
	if List.length edits > 0 then (
		let nsub,ndel,nins = count_edit_types edits in
		let r = ref false in
		let typ = ref "nul" in
		let cost = edit_cost edits maxlen in
		(* note! these rules must be symmetric for the graph to make sense *)
		if nsub <= 3 && ndel = 0 && nins = 0 then (
			r := true; typ := "sub" );
		if nsub = 0 && ndel <= 6 && nins = 0 then (
			r := true; typ := "del" ); 
		if nsub = 0 && ndel = 0 && nins <= 6 then (
			r := true; typ := "ins" );
		(*if nsub <= 1 && ndel = 0 && nins <= 3 then r := true;
		if nsub <= 1 && ndel <= 3 && nins <= 0 then r := true;*)
		!r,!typ,cost
	) else false,"",-1.0

let get_edits a_progenc b_progenc = 
	(* eliminate out-of-criteria distances to save time *)
	let d = abs ((String.length a_progenc) - (String.length b_progenc)) in
	if d <= 6 then (
		let dist,edits = Levenshtein.distance a_progenc b_progenc true in
		let edits = List.filter (fun (s,_p,_c) -> s <> "con") edits in
		let maxlen = max (String.length a_progenc) (String.length b_progenc) in
		(* verify .. a bit of overhead // NOTE: seems very safe! *)
		(*let re = Levenshtein.apply_edits a_progenc edits in
		if re <> b_progenc then (
			Logs.err(fun m -> m  
				"error! %s edits should be %s was %s"
				a_progenc b_progenc re)
		);*)
		(* edits are applied in reverse order *)
		(* & add a 'done' edit/indicator *)
		let edits = ("fin",0,'0') :: edits in
		dist, (List.rev edits), maxlen
	) else (
		d, [], 256
	)
	
let connect mtx pool g indx = 
	let d = g.(indx) in
	(* connect new node to the rest of the graph *)
	let pe = d.ed.progenc in
	let pel = String.length pe in
	(* levenshtein is very expensive, parallelize it! *)
	let body i = 
		let a = g.(i) in
		let ae = a.ed.progenc in
		let ael = String.length ae in
		let dist = abs (pel - ael) in
		if a.progt <> `Np && i <> indx && dist <= 6 then (
			let cnt,edits,maxlen = get_edits pe a.ed.progenc in
			let b,typ,cost = edit_criteria edits maxlen in
			if b then [ (i,typ,cnt,cost) ]
			else [ (0,"nul",0,-1.0) ]
		) else (
			[ (0,"nul",0,-1.0) ]
		)
	in
	let reduce a b = List.rev_append a b in
	let nearby = Task.run pool (fun () -> 
		Task.parallel_for_reduce 
			~chunk_size:32 ~start:0 ~finish:((Array.length g)-1) 
			~body pool reduce [] )
		|> List.filter (fun (_,_,_,b) -> b >= 0.0) in
	(*let nearby = ref [] in
	Array.iteri (fun i a -> 
		if a.progt <> `Np && i <> indx then (
			let cnt,edits = get_edits pe a.ed.progenc in
			let b,typ = edit_criteria edits in
			if b then (
				nearby := (i,typ,cnt,0) :: !nearby; 
				(*Printf.printf "connect: [%d] conn [%d] : %d \n" indx i dist
			) else (
				Printf.printf "connect: [%d] nocon [%d] : %d\n" indx i dist*)
			)
		 ) ) g; *)
	(* editing the graph; lock the mutex. *)
	Mutex.lock mtx; 
	List.iter (fun (i,typ,cnt,cost) ->
		let invtyp = match typ with
			| "del" -> "ins"
			| "ins" -> "del"
			| _ -> "sub" in
		(* note: cost does not depend on edit type.
			position of first edit should also not change. *)
		let d2 = g.(i) in
		let d2o = SI.add (indx,invtyp,cnt,cost) d2.outgoing in
		g.(i) <- {d2 with outgoing=d2o} ) nearby ; 
	(* update current node *)
	g.(indx) <- {d with outgoing=(SI.of_list nearby) }; 
	Mutex.unlock mtx; 
	;;
	
let get_slot gs = 
	let nfree = List.length gs.free_slots in
	if nfree > 0 then (
		let ni = List.hd gs.free_slots in
		gs.free_slots <- List.tl gs.free_slots; 
		ni
	) else ( -1 )
	;;
	
let get_slot_img gs = 
	let nfree = List.length gs.free_img in
	if nfree > 0 then (
		let ni = List.hd gs.free_img in
		gs.free_img <- List.tl gs.free_img; 
		ni
	) else ( -1 )
	;;
	
let add_uniq mtx pool gs ed = 
	(* add a unique node to the graph structure *)
	(* returns where it's stored; for now = imgf index *)
	 Mutex.lock mtx; 
	let ni = get_slot gs in
	let imgi = get_slot_img gs in
	 Mutex.unlock mtx; 
	
	if imgi >= 0 && ni >= 0 then (
		(*if imgi = 709 then 
			Logs.debug (fun m -> m "add_uniq ni:%d imgi:%d" ni imgi);*)
		let d = {nulgdata with ed; progt = `Uniq; imgi} in
		 Mutex.lock mtx;
		gs.g.(ni) <- d; 
		gs.img_inv.(d.imgi) <- ni ; (* back pointer *)
		gs.num_uniq <- gs.num_uniq + 1; 
		 Mutex.unlock mtx;
		connect mtx pool gs.g ni; 
		ni,imgi (* index to gs.g and dbf respectively *)
	) else (-1),(-1)
	;;
	
let replace_equiv mtx pool gs indx ed =
	(* ed is equivalent to gs.g[indx], but lower cost *)
	(* add ed the end, and update d1 = g[indx]. *)
	(* outgoing connections are not changed *)
	let d1 = gs.g.(indx) in
	if SI.cardinal d1.equivalents < 16 then (
		 Mutex.lock mtx; 
		let ni = get_slot gs in
		 Mutex.unlock mtx; 
		if ni >= 0 then (
			if d1.ed.progenc <> ed.progenc then (
				let eq2 = SI.add (indx,"",0,-1.0) d1.equivalents in (* fixed below *)
				let equivalents = SI.map (fun (i,_,_,_) -> 
					let e = gs.g.(i) in
					let cnt,edits,maxlen = get_edits ed.progenc e.ed.progenc in
					let _,typ,cost = edit_criteria edits maxlen in (* not constrained! *)
					i,typ,cnt,cost) eq2 in
				let imgi = d1.imgi in
				assert( imgi >= 0 ); 
				let d2 = {nulgdata with ed; 
							progt = `Uniq; 
							equivalents; 
							imgi } in
				 Mutex.lock mtx;
				gs.g.(ni) <- d2;
				gs.num_equiv <- gs.num_equiv + 1; 
				gs.img_inv.(imgi) <- ni ; (* back pointer *)
				(* update the incoming equivalent pointers; includes d1 *)
				SI.iter (fun (i,_,_,_) -> 
					let e = gs.g.(i) in
					let e' = {e with progt = `Equiv; equivroot = ni; equivalents = SI.empty} in
					gs.g.(i) <- e' ) d2.equivalents; 
				 Mutex.unlock mtx;
				Logs.debug (fun m -> m "replace_equiv newi:%d imgi:%d old:%d" 
					ni d2.imgi indx);
				(* finally, update d2's edit connections *)
				connect mtx pool gs.g ni ; 
				ni,imgi (* return the location of the new node, same as add_uniq *)
			) else (-1),(-1)
		) else (-1),(-1)
	) else (-1),(-1)
	;;
	
let add_equiv mtx pool gs indx ed =
	(* ed is equivalent to gs.g[indx], but higher (or equivalent) cost *)
	(* union-find the root equivalent *)
	let rec find_root j = 
		let d = gs.g.(j) in
		if d.equivroot >= 0 && d.equivroot <> j then 
			find_root d.equivroot
		else
			j 
	in
	 Mutex.lock mtx;
	let ni = get_slot gs in
	 Mutex.unlock mtx;
	if ni >= 0 then (
		let equivroot = find_root indx in
		let d1 = gs.g.(equivroot) in
		if d1.ed.progenc <> ed.progenc then (
			(* need to verify that this is not in the graph.*)
			let has = SI.fold (fun (j,_,_,_) a -> 
				let d = gs.g.(j) in
				if d.ed.progenc = ed.progenc then true else a) 
				d1.equivalents false in
			if not has then (
				let d2 = {nulgdata with ed; progt = `Equiv; equivroot; imgi = d1.imgi} in
				let cnt,edits,maxlen = get_edits d1.ed.progenc ed.progenc in
				let _b,typ,cost = edit_criteria edits maxlen in (* NOTE not constrained! *)
				let d1' = {d1 with equivalents = 
									SI.add (ni,typ,cnt,cost) d1.equivalents} in
				 Mutex.lock mtx;
				gs.g.(ni) <- d2; 
				gs.num_equiv <- gs.num_equiv + 1;
				gs.g.(equivroot) <- d1'; 
				 Mutex.unlock mtx;
				connect mtx pool gs.g ni ; 
				Logs.debug (fun m -> m "add_equiv newi:%d imgi:%d old:%d" 
					ni d2.imgi indx);
				ni (* return the location of the new node, same as add_uniq *)
			) else (-1)
		) else (-1)
	) else (-1)
	;;
	
let remove mtx gs indx = 
	(* remove a node from the graph, 
		e.g. if it's never used, or not well connected
		free up space for new nodes! *)
	(* our pointer from outgoing *)
	let d = gs.g.(indx) in
	
	(* this is a complex function, so just lock the whole thing *)
	 Mutex.lock mtx;
	
	SI.iter (fun (i,_,_,_) -> 
		let e = gs.g.(i) in
		let outgoing = SI.filter (fun (a,_,_,_) -> a <> indx) e.outgoing in
		let e' = {e with outgoing} in
		gs.g.(i) <- e' ) d.outgoing ; 
	
	if d.progt = `Uniq then (
		(* complexity: root equivalent nodes need to be remapped *)
		if SI.cardinal d.equivalents > 0 then (
			(* need to find a new equivroot *)
			let equivroot,_ = 
				SI.elements d.equivalents
				|> List.map (fun (a,_,_,_) -> 
					let e = gs.g.(a) in
					let f = Logo.progenc_cost e.ed.progenc in
					a,f )
				|> List.sort (fun (_,a) (_,b) -> compare a b) 
				|> List.hd in
			let equivalents = SI.filter (
				fun (a,_,_,_) -> 
					a <> equivroot ) d.equivalents in
			let e = gs.g.(equivroot) in
			let imgi = d.imgi in
			let progt = `Uniq in (* convert type *)
			let e' = {e with progt; imgi; equivroot; equivalents} in
			gs.g.(equivroot) <- e'; 
			SI.iter (fun (a,_,_,_) ->
				let h = gs.g.(a) in
				gs.g.(a) <- {h with equivroot} ) equivalents; 
			(* update img_inv, too *)
			gs.img_inv.(imgi) <- equivroot; 
			(* if we have elected a new root, imgi needs to stay around! *)
			gs.num_equiv <- gs.num_equiv - 1; 
		) else (
			(* otherwise, can free the imgf alloc *)
			gs.free_img <- d.imgi :: gs.free_img; 
			gs.num_uniq <- gs.num_uniq - 1; 
		); 
	) ;
	
	if d.progt = `Equiv then (
		(* remove us from our equivroot *)
		let e = gs.g.(d.equivroot) in
		let equivalents = SI.filter (fun (a,_,_,_) -> a <> indx) e.equivalents in
		gs.g.(d.equivroot) <- {e with equivalents}; 
		(* add to the free lists *)
		gs.num_equiv <- gs.num_equiv - 1; 
	); 
	
	gs.free_slots <- indx :: gs.free_slots;
	gs.g.(indx) <- nulgdata; 
	
	 Mutex.unlock mtx;
	;;
	
let incr_good gs i = 
	(* hopefully the compiler replaces this with an in-place op *)
	let d = gs.g.(i) in
	gs.g.(i) <- {d with good = (d.good + 1)} 
	;;
	
let sort_graph g = 
	(* sort the array by pcost ascending *)
	let ar = Array.mapi (fun i a -> (i,a)) g in
	Array.sort (fun (_,a) (_,b) -> compare a.ed.pcost b.ed.pcost) ar ; 
	let indxs,sorted = Array.split ar in
	(* indxs[i] = original pointer in g *)
	(* invert the indexes so that mappin[i] points to the sorted array *)
	let mappin = Array.make (Array.length indxs) 0 in
	Array.iteri (fun i a -> mappin.(a) <- i; 
		(*Logs.debug (fun m -> m "old %d -> new %d\n" a i)*)) indxs; 
	Array.map (fun a -> 
		let outgoing = SI.map (fun (b,typ,cnt,cost) ->
			mappin.(b),typ,cnt,cost
				) a.outgoing in
		let equivalents = SI.map (fun (b,typ,cnt,cost) ->
		mappin.(b),typ,cnt,cost
				) a.equivalents in
		let equivroot = if a.equivroot >= 0 then
			mappin.(a.equivroot) else (-1) in
		{a with outgoing; equivalents; equivroot} ) sorted 
	(* does not update the img indexes *)
	
let print_graph g = 
	let print_node i d = 
		let c = progt_to_str d.progt in
		Printf.printf "node [%d] = %s '%s'\n\tcost:%.2f, %.2f imgi:%d\n"
			i c (Logo.output_program_pstr d.ed.pro) d.ed.scost d.ed.pcost d.imgi; 
		Printf.printf "\toutgoing: "; 
		SI.iter (fun (j,typ,cnt,cost) -> Printf.printf "%d(%s %d %f)," j typ cnt cost) d.outgoing;
		Printf.printf "\tequivalents: "; 
		SI.iter (fun (j,typ,cnt,cost) -> Printf.printf "%d(%s %d %f," j typ cnt cost) d.equivalents;
		Printf.printf "\tequivroot: \027[31m %d\027[0m\n" d.equivroot
	in
	Array.iteri print_node g 
	
let get_stats gs = 
	Printf.sprintf "uniq:%d equiv:%d all_alloc:%d image_alloc:%d free_slots:%d free_img:%d" 
		gs.num_uniq gs.num_equiv gs.all_alloc gs.image_alloc 
		(List.length gs.free_slots) (List.length gs.free_img)
	
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
	let _proglst,progaddr = Logo.encode_ast pro in (*!! danger !!*)
	let progenc = Logo.encode_program pro |> Logo.intlist_to_string in
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
	(* output directly, keeping order -- indexes should be preserved *)
	let intset_to_sexp o = 
		Sexp.List (List.map (fun (i,typ,cnt,_cost) -> Sexp.List
			[ Sexp.Atom (soi i)
			; Sexp.Atom typ
			; Sexp.Atom (soi cnt) ])
			(SI.elements o) ) 
	in
	let sexp = Sexp.List ( List.mapi (fun i d -> 
		Sexp.List 
			[ Sexp.Atom (soi i)
			; Sexp.Atom (progt_to_str d.progt) 
			; Sexp.Atom (Logo.output_program_pstr d.ed.pro)
			; (intset_to_sexp d.outgoing)
			; (intset_to_sexp d.equivalents)
			; Sexp.Atom (soi d.equivroot)
			; Sexp.Atom (soi d.good) ] )
			(Array.to_list g) )
	in
	let oc = open_out fname in
	Sexplib.Sexp.output_hum oc sexp;
	close_out oc

let load gs fname = 
	(* does *not* render the programs *)
	Printf.printf "Graf.load %s%!" fname; flush stdout; 
	let open Sexplib in
	let ic = open_in fname in
	Printf.printf "Parsing Sexps...\n%!"; flush stdout; 
	let sexp = Sexp.input_sexps ic |> List.hd in
	close_in ic;
	(* extra layer of 'list'; not sure why *)
	let sexp' = Conv.list_of_sexp (function 
		|  Sexp.List(a) -> a
		| _ ->  []) sexp in
	let sexp_to_intset k =
		Conv.list_of_sexp (function
			| Sexp.List s -> (
				match s with
				| [Sexp.Atom i; Sexp.Atom typ; Sexp.Atom cnt] -> 
					(ios i),typ,(ios cnt),-1.0
				| _ -> (-1,"",0,-1.0) )
			| _ -> (-1,"",0,-1.0) ) k |> SI.of_list in
	gs.num_uniq <- 0; 
	gs.num_equiv <- 0; 
	List.iter (function 
		| [Sexp.Atom i; Sexp.Atom pt; Sexp.Atom pstr; out; equiv; 
							Sexp.Atom eqrt; Sexp.Atom gd] -> (
			let prog = match pstr with
				| "" -> Some(`Nop)
				| _ -> parse_logo_string pstr in
			(match prog with
			| Some(pro) -> (
				let progt = (str_to_progt pt) in
				if gs.num_uniq < gs.image_alloc 
						&& gs.num_uniq + gs.num_equiv < gs.all_alloc then (
					let outgoing = (sexp_to_intset out) in
					let equivalents = (sexp_to_intset equiv) in
					let equivroot = (ios eqrt) in
					let good = (ios gd) in
					let imgi = match progt with 
						| `Uniq -> gs.num_uniq <- gs.num_uniq + 1; (gs.num_uniq - 1)
						| `Equiv -> gs.num_equiv <- gs.num_equiv + 1; (-1)
						| _ -> (-1) in
					let ed,_ = pro_to_edata pro 0 in
					if progt <> `Np then (
						let d = {progt; ed; imgi ; outgoing; equivalents; equivroot; good} in
						let ii = ios i in
						if imgi >= gs.image_alloc then 
							assert (0 <> 0) ; 
						if ii >= gs.all_alloc then 
							assert (0 <> 0) ; 
						if imgi >= 0 then gs.img_inv.(imgi) <- ii; 
						gs.free_img <- List.filter (fun a -> a <> imgi) 
										gs.free_img;
						gs.free_slots <- List.filter (fun a -> a <> ii) 
										gs.free_slots; 
						gs.g.(ii) <- d; 
						Printf.printf "\rloaded %d %!" ii ; 
						flush stdout; 
						flush stderr; 
					)
				) else (
					if progt = `Uniq || progt = `Equiv then ( 
					Logs.err (fun m->m "%s too big; num_uniq %d num_equiv %d image_alloc %d all_alloc %d (used %d) adding %s"
						fname
						gs.num_uniq gs.num_equiv gs.image_alloc gs.all_alloc
						(gs.num_uniq + gs.num_equiv) 
						(match progt with |`Uniq -> "uniq" |`Equiv -> "equiv" | _ -> "nul")
						) )
				) )
			| _ -> (
				Logs.err (fun m -> m "Failed to parse pstr \"%s\"" pstr); 
				assert(0 <> 0); 
				) 
			) )
		| _ -> failwith "Invalid S-expression format"
		) sexp' ; 
	(*Printf.printf "check!!!\n"; 
	print_graph g; *)
	(* need to go back and fix the imgi indexes *)
	Array.iteri (fun i d -> 
		match d.progt with 
		| `Equiv -> (
			let e = gs.g.(d.equivroot) in
			gs.g.(i) <- {d with imgi = e.imgi} )
		| `Uniq -> ()
		| _ -> () (* null entry *)
		) gs.g; 
	(* need to go back and fix the edit costs *)
	(* probably should save the costs to file! *)
	Array.iteri (fun i d ->
		match d.progt with
		| `Equiv
		| `Uniq -> (
				let domap st =
					SI.map (fun (i,_,_,_) ->
					let e = gs.g.(i) in
					let cnt,edits,maxlen = get_edits d.ed.progenc e.ed.progenc in
					let _,typ,cost = edit_criteria edits maxlen in
					(i,typ,cnt,cost)
					) st
				in
				let outgoing = domap d.outgoing in
				let equivalents = domap d.equivalents in
				gs.g.(i) <- {d with outgoing; equivalents}
			)
		| _ -> () (* null entry *)
		) gs.g;
	gs
	;;


(* use a priority-search-queue to hold the list of nodes *)
module QI = struct type t = int let compare (a: int) b = compare a b end
module QF = struct type t = float let compare (a: float) b = compare a b end
module Q = Psq.Make (QI) (QF) ;; (* make (key) (priority) *)

let dijkstra gs start dbg = 
	(* starting from node gs.g.(start), 
		find the distances to all other nodes
		bias to postfix as opposed to prefix editing *)
	let d = gs.g.(start) in
	let n = Array.length gs.g in
	let dist = Array.make n (-1.0) in
	let prev = Array.make n (-1) in
	let visited = Array.make n false in 
	dist.(start) <- 0.0;
	visited.(start) <- true; 
	let q = SI.fold (fun (i,_,_cnt,cost) a ->
		dist.(i) <- cost;
		prev.(i) <- start; 
		Q.add i cost a
		) d.outgoing Q.empty in
	if dbg then Logs.debug (fun m->m "psq size %d" (Q.size q));
	if dbg then Logs.debug (fun m->m "outgoing size %d" (SI.cardinal d.outgoing));
	
	let rec dijk qq = 
		if Q.is_empty qq then ()
		else (
			match Q.pop qq with
			| Some ((i,p),q) -> (
				let e = gs.g.(i) in
				(* visited nodes are never added to the queue *)
				assert(visited.(i) = false);
				if dbg then Logs.debug (fun m -> m "dijk visit [%d] %f %s"
					i p (Logo.output_program_pstr e.ed.pro));
				if dbg then Logs.debug (fun m->m "psq size %d" (Q.size q));
				if dbg then Logs.debug (fun m->m "outgoing size %d" (SI.cardinal e.outgoing));
				if dbg then Logs.debug (fun m->m "equivalents size %d" (SI.cardinal e.equivalents));
				visited.(i) <- true; 
				let de = dist.(i) in
				assert( p = de ); 
				let foldadd set sq = 
					SI.fold (fun (j,typ,_cnt,cost) a ->
					if not visited.(j) && typ <> "nul" then (
						let nd = de +. cost in
						if dist.(j) < 0. then (
							(* new route *)
							if dbg then Logs.debug (fun m -> m "new route to %d cost %f" j nd);
							dist.(j) <- nd; 
							prev.(j) <- i; 
							Q.add j nd a
						) else (
							if nd < dist.(j) then (
								(* new route is shorter *)
								if dbg then Logs.debug (fun m -> m "shorter route to %d cost %f" j nd);
								dist.(j) <- nd; 
								prev.(j) <- i; 
								Q.adjust j (fun _ -> nd) a
							) else ( a ) )
					) else ( a ) ) set sq 
				in
				q 
				|> foldadd e.outgoing 
				|> foldadd e.equivalents 
				|> dijk )
			| _ -> () 
		) 
	in
	
	dijk q; 
	
	let root = "/tmp/ec3/dijkstra" in
	let fid = open_out (Printf.sprintf "%s/paths.txt" root) in
	
	let rec print_path final step present = 
		let d = gs.g.(present) in
		Printf.fprintf fid "final:%d step:%d present:%d %s \n" 
			final step present (Logo.output_program_pstr d.ed.pro); 
		let fname = Printf.sprintf "%s/%d_%d.png" root final step in
		ignore(Logoext.run_prog (Some d.ed.pro) 64 fname false); 
		let j = prev.(present) in
		if j >= 0 then print_path final (step+1) j else ()
	in

	for i = 0 to 19 do (
		Printf.fprintf fid "--%d--" i; 
		let k = Random.int n in
		print_path k 0 k
	) done; 

	close_out fid; 
	
	(dist, prev)
	
let edge_use gs l =
	(* reset the good counts *)
	Array.iteri (fun i a -> gs.g.(i) <- {a with good=0} ) gs.g;
	(* increment the counts of the used edges from the list l *)
	List.iter (fun (pre,post) -> 
		let a = gs.g.(pre) in
		let outgoing = SI.map (fun (b,typ,cnt,cst) ->
			let cost = if b = post then cst +. 1.0 else cst in
			(b,typ,cnt,cost) ) a.outgoing in
		(* note: 'nul' is filtered in dijkstra above *)
		(* (equivalents can have large edit distance) *)
		let equivalents = SI.map (fun (b,typ,cnt,cst) ->
			let cost = if b = post then cst +. 1.0 else cst in
			(b,typ,cnt,cost) ) a.equivalents in
		let good1 = a.good + 1 in
		gs.g.(pre) <- {a with outgoing; equivalents; good=good1}; 
		(* update post too *) 
		let c = gs.g.(post) in
		let good2 = c.good + 1 in
		gs.g.(post) <- {c with good=good2} ) l; 
	let unused = ref 0 in
	Array.iter (fun a -> 
		if a.progt <> `Np && a.good <= 0 then incr unused) gs.g; 
	Logs.info (fun m->m "Graf.edge_use: unused database nodes: %d" !unused)
	;;
	
let dist_to_good gs dist = 
	(* set the 'good' field to distance *)
	Array.iteri (fun i e -> 
		let e' = {e with good = dist.(i) } in
		gs.g.(i) <- e' 
		) gs.g 
	;;
	
let remove_unused mtx gs = 
	Array.iteri (fun i a -> 
		if a.progt <> `Np && a.good < 0 then remove mtx gs i) gs.g
	;;
	
let remove_unreachable mtx gs = 
	let dist,prev = dijkstra gs 0 false in
	
	let unreachable,_ = Array.fold_left (fun (a,i) b -> 
		if gs.g.(i).progt <> `Np && b < 0.0
			then (a+1,i+1) 
			else (a,i+1)
			) (0,0) dist in
			
	(* need to remove `Equiv that are not stepping-points to `Uniq 
		these will not be used in training. *)
	let n = Array.length gs.g in
	let terminal = Array.make n true in
	Array.iter (fun p -> if p >= 0 then terminal.(p) <- false) prev; 
	let term_equiv = ref 0 in
	Array.iteri (fun i term -> 
		if term && gs.g.(i).progt = `Equiv then (
			incr term_equiv; 
			dist.(i) <- (-1.0 ) ) ) terminal;
			
	(* remove programs that are too long *)
	let too_long = ref 0 in
	Array.iteri (fun i d -> 
		let p_ctx = 96 in
		if String.length d.ed.progenc > p_ctx/2-2 then (
			dist.(i) <- (-1.0) ;
			incr too_long
		) ) gs.g; 
		
	Logs.debug (fun m->m "remove_unreachable: \n\t%d of %d unreachable, \n\t%d terminal `Equiv; \n\t%d too long"
		unreachable (gs.num_uniq + gs.num_equiv) !term_equiv !too_long); 
		
	Logs.debug (fun m->m "old graph (db_preremove.S), %s" (get_stats gs));
	dist_to_good gs (Array.map iof dist);
	save "db_preremove.S" gs.g;
	remove_unused mtx gs ; 
	Logs.debug (fun m->m "new graph (db_removed.S), %s" (get_stats gs));
	save "db_removed.S" gs.g;
	Logs.debug (fun m->m "remove_unreachable done");
	(* caller must render the database!! *)
	(* render_database steak *)
	;;
		
let gexf_out gs = 
	(* save GEXF file for gephi visualization *)
	let fid = open_out "../prog-gephi-viz/db.gexf" in
	Printf.fprintf fid "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"; 
	Printf.fprintf fid "<gexf xmlns=\"http://gexf.net/1.3\" version=\"1.3\">\n"; 
	Printf.fprintf fid "<graph mode=\"static\" defaultedgetype=\"directed\">\n"; 
	Printf.fprintf fid "<attributes class=\"node\">\n"; 
	Printf.fprintf fid "<attribute id=\"0\" title=\"progt\" type=\"string\"/>\n"; 
	Printf.fprintf fid "</attributes>\n";
	Printf.fprintf fid "<attributes class=\"edge\">\n"; 
	Printf.fprintf fid "<attribute id=\"0\" title=\"typ\" type=\"string\"/>\n"; 
	Printf.fprintf fid "<attribute id=\"1\" title=\"cost\" type=\"int\"/>\n";
	Printf.fprintf fid "</attributes>\n"; 
	Printf.fprintf fid "<nodes>\n"; 
	Array.iteri (fun i d -> 
		Printf.fprintf fid "<node id=\"%d\" label=\"%s\" >\n" i
			(Logo.output_program_pstr d.ed.pro); 
		let pts = match d.progt with
			| `Uniq -> "uniq"
			| `Equiv -> "equiv"
			| _ -> "nul" in
		Printf.fprintf fid 
		"<attvalues><attvalue for=\"0\" value=\"%s\"/></attvalues>\n" pts; 
		Printf.fprintf fid "</node>\n"
		) gs.g ; 
	Printf.fprintf fid "</nodes>\n";
	
	Printf.fprintf fid "<edges>\n"; 
	Array.iteri (fun i d -> 
		SI.iter (fun (j,typ,_cnt,cost) ->
			if cost <> 0. then (
				Printf.fprintf fid "<edge source=\"%d\" target=\"%d\">\n" i j;
				Printf.fprintf fid "<attvalues>\n";
				Printf.fprintf fid "<attvalue for=\"0\" value=\"%s\"/>\n" typ;
				Printf.fprintf fid "<attvalue for=\"1\" value=\"%f\"/>\n" cost;
				Printf.fprintf fid "</attvalues>\n";
				Printf.fprintf fid "</edge>\n"
			)
		) d.outgoing
		) gs.g ; 
	Printf.fprintf fid "</edges>\n";
	Printf.fprintf fid "</graph>\n";
	Printf.fprintf fid "</gexf>\n";
	close_out fid; 
	Printf.printf "saved ../prog-gephi-viz/db.gexf\n";; 

(*let () = 
	let gs = create 5 in
	
	let add_str mode equiv s = 
		let prog = parse_logo_string s in
		let l = match prog with
		| Some(pro) -> (
			let ed,_ = pro_to_edata pro 0 in
			if mode = `Uniq then (
				if equiv >= 0 then (
					let a,_ = replace_equiv gs equiv ed in (* fills out connectivity *)
					a
				) else (
					let a,_ = add_uniq gs ed in (* fills out connectivity *)
					a
				) 
			) else (
				add_equiv gs equiv ed ;
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
	incr_good gs 0; 
	
	print_graph gs.g; 
	let sorted = sort_graph gs.g in
	Printf.printf "--- sorted ---\n"; 
	print_graph sorted ; 
	Printf.printf "--- saving & reloaded ---\n"; 
	let fname = "test_prog.S" in
	save fname sorted; 
	let gp = load gs fname in
	print_graph gp.g
	*)
