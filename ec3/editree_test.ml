open Editree

let () = 

	Printf.printf "address test %s\n" (adr2str [1;2;3;4]); 
	let r = make_root "test" in
	let edits = [("sub",0,'h');("ins",1,'o');("del",3,'t')] in
	let probs = [0.2; 0.3; 0.4] |> List.map log in
	model_update r [] edits probs; 
	print r [] "" 0.0; Printf.printf "\n"; 
	
	let adr,_,kid = select r in
	print kid adr "1-" 0.0; 
	let edits2 = [("sub",1,'x');("fin",0,'z');("ins",3,'t')] in
	let probs2 = [0.5; 0.15; 0.05] |> List.map log in
	model_update r adr edits2 probs2; 
	print r [] "" 0.0; Printf.printf "\n"; 
	
	let adr,_,kid = select r in
	print kid adr "2-" 0.0; 
	let edits3 = [("fin",1,'w');("fin",0,'a');("fin",2,'v')] in
	let probs3 = [0.1; 0.15; 0.05] |> List.map log in
	model_update r adr edits3 probs3; 
	print r [] "" 0.0; Printf.printf "\n"; 
	
	let adr,_,kid = select r in
	print kid adr "3-" 0.0; 
	let edits3 = [("fin",1,'y');("fin",0,'o');("fin",2,'u');("ins",2,'1')] in
	let probs3 = [0.1; 0.15; 0.05; 0.01] |> List.map log in
	model_update r adr edits3 probs3; 
	print r [] "" 0.0; Printf.printf "\n"; 
	
	let adr,_,kid = select r in
	print kid adr "3-" 0.0; 
	let edits3 = [("fin",1,'b');("fin",0,'n')] in
	let probs3 = [0.1; 0.15] |> List.map log in
	model_update r adr edits3 probs3; 
	print r [] "" 0.0; Printf.printf "\n"; 
	
	let adr,_,kid = select r in
	print kid adr "4-" 0.0; 
	let edits3 = [("fin",1,'g');("fin",0,'h')] in
	let probs3 = [0.1; 0.15] |> List.map log in
	model_update r adr edits3 probs3; 
	print r [] "" 0.0; Printf.printf "\n";
