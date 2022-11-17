(* compute the levenshtein edits in ocaml using 
https://www.codeproject.com/Articles/13525/Fast-memory-efficient-Levenshtein-algorithm-2 *)

let distance srow scol getedits = 
	let rlen = String.length srow in
	let clen = String.length scol in
	let mat = if getedits then Array.make_matrix (rlen+1) clen 0 
		else Array.make_matrix 2 2 0 in
	let edits = ref [] in
	if rlen = 0 then (
		clen, []
	) else (
		if clen = 0 then (
			rlen, []
		) else (
			let v0 = Array.init (rlen+1) (fun i -> i) in
			let v1 = Array.init (rlen+1) (fun i -> i) in
			(* loop over all columns *)
			String.iteri (fun col cc -> 
				let coli = col+1 in
				(* set the 0th element to column number *)
				v1.(0) <- coli; 
				String.iteri (fun row cr -> 
					let rowi = row+1 in
					let cost = if cc = cr then 0 else 1 in
					let a = v0.(rowi) + 1 in
					let b = v1.(rowi-1) + 1 in
					let c = v0.(rowi-1) + cost in
					let d = if a<b then (if a<c then a else c)
						else ( if b<c then b else c) in
					v1.(rowi) <- d; 
					) srow; 
				(* check the contents *)
				(*Printf.printf "v0\tv1\n"; 
				Array.iteri (fun i _ -> 
					Printf.printf "%d\t%d\n" v0.(i) v1.(i)
				) v0; 
				Printf.printf "\n";*) 
				(* swap -- faster later *)
				Array.iteri (fun i _ -> 
					let tmp = v0.(i) in
					v0.(i) <- v1.(i); 
					if getedits then mat.(i).(col) <- v1.(i); 
					v1.(i) <- tmp) v0; 
			) scol; 
			(* print the matrix
			for r = 0 to rlen+1 do (
				let s = if r >= 2 then String.get srow (r-2) else ' ' in
				Printf.printf "%c " s; 
				if r = 0 then (
					for c = 0 to clen-1 do (
						Printf.printf "%c " (String.get scol c)
					) done
				) else (
					for c = 0 to clen-1 do (
						Printf.printf "%d " mat.(r-1).(c)
					) done
				); 
				Printf.printf "\n"
			) done; *)
			(* to get the edits we have to work backwards
			through the matrix, as in the smith-waterman algorithm *)
			if getedits then (
				let r = ref rlen in
				let c = ref (clen-1) in
				while !r > 1 && !c >= 0 do (
					let rw,cw = (String.get srow (!r-1)),
								(String.get scol !c) in
					(*Printf.printf "row %d column %d srow %c scol %c\n"
								!r !c rw cw;*)
					let ak,ae = mat.(!r-1).(!c), 1 in
					let bk,be = if !c>0 then mat.(!r).(!c-1), 2
									else 1000000, 2 in
					let ck,ce = if !c>0 then mat.(!r-1).(!c-1), 3
									else (1000000, 3) in
					let _dk,de = if ak<bk then (if ak<ck then ak,ae else ck,ce)
						else ( if bk<ck then bk,be else ck,ce) in
					let ed,rr,cc = match de with
						| 1 -> ("del",!r-1, rw), !r-1, !c
						| 2 -> ("ins",!r, cw), !r, !c-1
						| _ -> if cw = rw then ("con",!r-1, cw), !r-1, !c-1
								else ( "sub", !r-1, cw), !r-1, !c-1 in
					edits := ed :: !edits ;
					r := rr;
					c := cc;
				) done; 
				(* add in the last edit(s) -- potentially tricky*)
				(*Printf.printf "final row %d column %d\n" !r !c;*)
				let rw,cw = (String.get srow (!r-1)),
								(String.get scol !c) in
				let ed = match !r,!c with
					| 2,0 -> ("del",(!r-2), String.get srow (!r-2))
					| 1,0 -> if rw = cw then ("con",(!r-1), rw)
								else ("sub",(!r-1),cw)
					| 1,rr -> (
						let f = if rw=cw then ("con",(!r-1), rw)
						else ("sub",(!r-1),cw) in
						edits := f :: !edits ;
						(* manage the prefix *)
						(* in the case of a single fixed prefix, this adds an unecessary sub then insert 0 instead of insert 1 *)
						for j = rr-1 downto 0 do (
							let cw = String.get scol j in
							edits := ("ins",0,cw) :: !edits 
						) done; 
						("nul",0,'0') )
					| _ -> ("nul",0,'0') in
				edits := ed :: !edits ;
			); 
			v0.(rlen), !edits
		)
	)
	
let print_edits edits = 
	Printf.printf "edits:\n"; 
	List.iteri (fun i (s,ri,c) ->
		Printf.printf "%d: %s %d %c \n" i s ri c
		) edits
	
let apply_edits s1 edits = 
	List.fold_left (fun ss (s,p,c) ->
		let len = String.length ss in
		let cc = String.make 1 c in
		(*Printf.printf "ss %s; %s %d %c\n" ss s p c;*) 
		match s with
		| "sub" -> (
			let a = if p > 0 then String.sub ss 0 p else "" in
			let b = if p < len-1 then
					String.sub ss (p+1) (len-p-1) else "" in
			(*Printf.printf "a %s\n" a; 
			Printf.printf "b %s\n" b;*) 
			a ^ cc ^ b )
		| "del" -> (
			let a = if p > 0 then String.sub ss 0 p else "" in
			let b = if p < len-1 then
					String.sub ss (p+1) (len-p-1) else "" in
			a ^ b )
		| "ins" -> ( (* insert before p; can be at end of string*)
			let a = if p > 0 then String.sub ss 0 p else "" in
			let b = if p < len then String.sub ss p (len-p) else "" in
			(*Printf.printf "a %s\n" a; 
			Printf.printf "b %s\n" b;*) 
			a ^ cc ^ b )
		| _ -> ss ) s1 (List.rev edits)
		
let () = 
	let s1 = "apples123" in
	let s2 = "apsxsples" in
	let dist, edits = distance s1 s2 true in
	Printf.printf "distance = %d done\n" dist; 
	(* verify this *)
	let s3 = apply_edits s1 edits in
	Printf.printf "%s to %s, verify %s\n" s1 s2 s3
