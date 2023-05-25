(* module for implementing an iteratively-broadening edit-tree *)

type t = 
	{ progenc : string
	; edited : float array (* tell python which chars have been changed*)
	; mutable edits : (string*int*char) list (* edits to gen kids *)
	; mutable probs : float list
	; mutable kids : t option array
	}
	
let nuledtree = 
	{ progenc = ""
	; edited = [| 0.0 |]
	; edits = []
	; probs = []
	; kids = Array.make 1 None
	}
	
let p_ctx = 96
let soi = string_of_int
let iof = int_of_float
let adr2str adr = 
	List.fold_left (fun a b -> a^(soi b)^",") "" adr 
	
let update_edited ?(inplace=false) origed ed lc = 
	(* update the 'edited' array, which indicates what has changed in the program string *)
	(* lc is the length of the string being edited *)
	let typ,pp,_chr = ed in 
	let edited = if inplace then origed else Array.copy origed in
	(match typ with 
	| "sub" -> (
		let pp = if pp > lc-1 then lc-1 else pp in
		let pp = if pp < 0 then 0 else pp in
		edited.(pp) <- 2.0 )
	| "del" -> (
		(* lc is already one less at this point -- c has been edited *)
		let pp = if pp > lc-1 then lc-1 else pp in
		let pp = if pp < 0 then 0 else pp in
		(* shift left *)
		for i = pp to (p_ctx/2)-2 do (
			edited.(i) <- edited.(i+1)
		) done; 
		assert (pp < (p_ctx/2)); 
		edited.(pp) <- ( -1.0 ) ) 
	| "ins" -> (
		if lc < p_ctx/2 && pp < p_ctx/2 then (
			let pp = if pp > lc then lc else pp in
			let pp = if pp < 0 then 0 else pp in
			(* shift right one *)
			for i = (p_ctx/2)-2 downto pp do (
				edited.(i+1) <- edited.(i)
			) done; 
			edited.(pp) <- 1.0 ) )
	| _ -> () ); 
	edited
		
let rec flatten node adr pr = 
	(* flatten the node to a list of 
		*leaf* addresses and probabilities *)
	(* addresses are assembled backwards *)
	(* out type is ((int list) * float) list *)
	List.combine (List.combine node.probs node.edits)
		(Array.to_list node.kids)
	|> List.fold_left (fun (i,a) ((p,ed),k) -> 
		match k,ed with 
		| None,("don",_,_) -> (i+1), a (* don't add 'don' nodes *)
		| None,_ -> (i+1), (i::adr, (pr +. p))::a
		| Some k,_ -> ( 
			if String.length k.progenc < p_ctx/2-1 then (
				let _,b = flatten k (i::adr) (pr +. p) in
				(i+1), (List.rev_append b a) 
			) else (i+1,a) )
		) (0,[])
	
let rec index node targadr curadr out = 
	(* given an address, out := edit,kid *)
	(* usually: generate a new node *)
	List.combine (Array.to_list node.kids) node.edits
	|> List.iteri (fun i (d,ed) -> 
		let adr = i :: curadr in
		match d with 
		| Some kid -> (
			if adr = targadr then out := ed,kid
			else index kid targadr adr out (* recurse *)
			)
		| None -> (
			if adr = targadr then (
				let progenc = Levenshtein.apply_edits node.progenc [ed] in
				let edited = update_edited node.edited ed (String.length progenc) in
				let edits = [] in (* from model! *)
				let probs = [] in (* must be evaled! *)
				let kids = [| |] in
				let kid = { progenc; edited; edits; probs; kids } in
				node.kids.(i) <- Some kid; 
				out := ed,kid
			) )
		) 
	
let select node = 
	(* select the leaf node on the parse data tree 
		with the highest log probability *)
	let out = ref (("con",0,'0'),nuledtree) in
	let _,flat = flatten node [] 0.0 in
	if (List.length flat) > 0 then (
		let adr,_prob = 
			List.sort (fun (_,a) (_,b) -> compare b a) flat 
			|> List.hd in
		(*Printf.printf "selected %s %f\n" (adr2str adr) prob;*) 
		index node adr [] out; 
		let edit,kid = !out in
		adr,edit,kid
	) else ([], ("con",0,'0'), nuledtree)
	
let rec update node targadr curadr eds pr = 
	(* given an targe address, update the probs and edits *)
	if targadr = curadr then (
		node.edits <- eds ; 
		node.probs <- pr ; 
		node.kids <- Array.make (List.length eds) None ; 
	) else (
		Array.iteri (fun i d -> 
			match d with
			| Some kid -> update kid targadr (i::curadr) eds pr
			| _ -> ()
			) node.kids
	)
	
(* -- external interface -- *)

let make_root progenc = 
	{ progenc; 
	 edited = Array.make (p_ctx / 2) 0.0; 
	 edits = []; 
	 probs = []; 
	 kids = Array.make 10 None }

let model_update root adr eds pr =
	(* update the edit tree based on the model's output *)
	update root adr [] eds pr
	
let model_done root adr = 
	update root adr [] [("don",0,'0')] [log 1.0]
	
let model_select root = 
	let adr,edit,kid = select root in
	adr,edit,kid.progenc,kid.edited
	
	 
(* -- testing -- *)
	 
let rec print node adr prefix prob =
	let adrs = adr2str adr in
	let editeds = Array.fold_left (fun a b -> a^(b |> iof |> soi)) "" node.edited in
	Printf.printf "%snode @ %s \"%s\" edited:%s \n" prefix adrs node.progenc editeds ; 
	let prefixx = "  "^prefix in
	List.combine node.probs node.edits 
	|> List.iteri (fun i (p,(typ,pos,chr)) -> 
		let pp = p +. prob in
		Printf.printf "%sedit[%d] logp %0.3f (%0.3f) : %s %d %c\n" 
			prefixx i pp p typ pos chr ; 
		let adrr = i :: adr in
		match node.kids.(i) with
		| Some k -> print k adrr prefixx pp
		| _ -> () ); 
	flush stdout

let getprogenc root = root.progenc
