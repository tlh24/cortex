(* module for implementing an iteratively-broaening parse-tree *)

type edtree = 
	{ progenc : string
	; edited : float array (* tell python which chars have been changed*)
	; mutable edits : (string*int*char) list
	; mutable probs : float list
	; mutable kids : edtree option array
	}
	
let nuledtree = 
	{ progenc = ""
	; edited = [| 0.0 |]
	; edits = []
	; probs = []
	; kids = Array.make 10 None
	}
	
let p_ctx = 12
let soi = string_of_int
let iof = int_of_float
	
let update_edited origed ed lc = 
	(* update the 'edited' array, which indicates what has changed in the program string *)
	(* lc is the length of the string being edited *)
	let typ,pp,_chr = ed in 
	let edited = Array.copy origed in
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
	(* sig is ((int list) * float) list *)
	List.combine (List.combine node.probs node.edits)
		(Array.to_list node.kids)
	|> List.fold_left (fun (i,a) ((p,ed),k) -> 
		match k,ed with 
		| None,("fin",_,_) -> (i+1), a (* don't add 'fin' nodes *)
		| None,_ -> (i+1), (i::adr, (pr +. p))::a
		| Some k,_ -> ( let _,b = flatten k (i::adr) (pr +. p) in
			(i+1), (List.rev_append b a) )
		) (0,[])
	
let rec index node targadr curadr out = 
	(* given an address, out := kid *)
	(* generate a new one if needed *)
	List.combine (Array.to_list node.kids) node.edits
	|> List.iteri (fun i (d,ed) -> 
		let adr = i :: curadr in
		match d with 
		| Some kid -> (
			if adr = targadr then out := kid
			else index kid targadr adr out
			)
		| None -> (
			if adr = targadr then (
				let progenc = Levenshtein.apply_edits node.progenc [ed] in
				let edited = update_edited node.edited ed (String.length progenc) in
				let edits = [] in (* from model! *)
				let probs = [] in (* must be evaled ! *)
				let kids = [| |] in
				let kid = { progenc; edited; edits; probs; kids } in
				node.kids.(i) <- Some kid; 
				out := kid
			) )
		) 
	
let adr2str adr = 
	List.fold_left (fun a b -> a^(soi b)^",") "" adr 
	
let select node = 
	(* select the leaf node on the parse data tree 
		with the highest log probability *)
	let _,flat = flatten node [] 0.0 in
	let adr,prob = 
		List.sort (fun (_,a) (_,b) -> compare b a) flat 
		|> List.hd in
	Printf.printf "selected %s %f\n" (adr2str adr) prob; 
	let out = ref nuledtree in
	index node adr [] out; 
	adr, !out
	
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
	
let model_update node adr eds pr =
	(* update the edit tree based on the model's output *)
	update node adr [] eds pr
	
let make_root progenc = 
	{ progenc; 
	 edited = Array.make (p_ctx / 2) 0.0; 
	 edits = []; 
	 probs = []; 
	 kids = Array.make 10 None }
	 
let rec print node adr prefix =
	let adrs = adr2str adr in
	let editeds = Array.fold_left (fun a b -> a^(b |> iof |> soi)) "" node.edited in
	Printf.printf "%snode @ %s \"%s\" edited:%s \n" prefix adrs node.progenc editeds ; 
	let prefixx = "  "^prefix in
	List.combine node.probs node.edits 
	|> List.iteri (fun i (p,(typ,pos,chr)) -> 
		Printf.printf "%sedit[%d] logp %f : %s %d %c\n" 
			prefixx i p typ pos chr ; 
		let adrr = i :: adr in
		match node.kids.(i) with
		| Some k -> print k adrr prefixx
		| _ -> () )

let () = 
	Printf.printf "address test %s\n" (adr2str [1;2;3;4]); 
	let r = make_root "test" in
	let edits = [("sub",0,'h');("ins",1,'o');("del",3,'t')] in
	let probs = [0.2; 0.3; 0.4] |> List.map log in
	model_update r [] edits probs; 
	print r [] "" ; Printf.printf "\n"; 
	
	let adr,kid = select r in
	print kid adr "1-" ; 
	let edits2 = [("sub",1,'x');("fin",0,'z');("ins",3,'t')] in
	let probs2 = [0.5; 0.15; 0.05] |> List.map log in
	model_update r adr edits2 probs2; 
	print r [] "" ; Printf.printf "\n"; 
	
	let adr,kid = select r in
	print kid adr "2-" ; 
	let edits3 = [("fin",1,'w');("fin",0,'a');("fin",2,'v')] in
	let probs3 = [0.1; 0.15; 0.05] |> List.map log in
	model_update r adr edits3 probs3; 
	print r [] "" ; Printf.printf "\n"; 
	
	let adr,kid = select r in
	print kid adr "3-" ; 
	let edits3 = [("fin",1,'y');("fin",0,'o');("fin",2,'u')] in
	let probs3 = [0.1; 0.15; 0.05] |> List.map log in
	model_update r adr edits3 probs3; 
	print r [] "" ; Printf.printf "\n"; 
	
	let adr,kid = select r in
	print kid adr "3-" ; 
	let edits3 = [("fin",1,'b');("fin",0,'n');("fin",2,'m')] in
	let probs3 = [0.1; 0.15; 0.05] |> List.map log in
	model_update r adr edits3 probs3; 
	print r [] "" ; Printf.printf "\n"; 
	
	let adr,kid = select r in
	print kid adr "4-" ; 
	let edits3 = [("fin",1,'g');("fin",0,'h');("fin",2,'j')] in
	let probs3 = [0.1; 0.15; 0.05] |> List.map log in
	model_update r adr edits3 probs3; 
	print r [] "" ; Printf.printf "\n";
