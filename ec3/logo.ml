open Vgwrapper

(* variables will be referenced to the stack, de Brujin indexes *)
(* names are more interpretable by humans ... but that's a big space *)
type ptag = int (* future-proof *)
let nulptag = 0

type prog = [
	| `Var of int * ptag
	| `Save of int * prog * ptag
	| `Move of prog * prog * ptag(* angle & distance *)
	| `Binop of prog * string * (float -> float -> float) * prog * ptag
	| `Const of float * ptag
	| `Seq of prog list * ptag
	| `Loop of int * prog * prog * ptag(* iterations & body *)
	| `Call of int * prog list * ptag (* list of arguments *)
	| `Def of int * prog * ptag(* same sig as `Save *)
(* 	| Cmp of prog * (float -> float -> bool) * prog *)
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
	| _ -> 18
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
	| _ -> " "
	
let enc_char c s = 
	s := (enc_char1 c) :: !s
	
let enc_int i s = 
	let j = if i > 9 then 9 
		else (if i < 0 then 0 else i) in
	s := (j - 10) :: !s
	
let dec_item i =
	if i < 0 then 
		(string_of_int (i + 10))^" "
	else
		dec_item1 i
	
let rec enc_prog g s = 
	(* convert a program to integers.  For the transformer. *)
	(* output list is in REVERSE order *)
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
	| `Nop -> ()
	
let encode_program g = 
	let s = ref [] in
	enc_prog g s; 
	List.rev !s
	
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
	| `Nop -> Printf.fprintf lg "Nop "
	
and output_list_h lg l sep =
	Printf.fprintf lg "("; 
	List.iteri (fun i v -> 
		if i > 0 then Printf.fprintf lg "%s" sep ;
		output_program_h v lg) l ; 
	Printf.fprintf lg ")"
	
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
  ; p : bool (* pen down = true *)
  ; r : int (* execution count *)
  ; stk : float array
  }

let defs = Array.make 10 `Nop

type segment = float*float*float*float

let output_segments bf seglist = 
	List.iteri (fun i (x1,y1,x2,y2) -> 
		Printf.bprintf bf 
			"%d %f,%f %f,%f\n" 
			i x1 y1 x2 y2) seglist
			
let output_segments_str seglist = 
	let bf = Buffer.create 64 in
	output_segments bf seglist; 
	(Buffer.contents bf)

let start_state () = 
	{x=0.0; y=0.0; t=0.0; p=true; r=0;
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
			let seg = st.x, st.y, st2.x, st2.y in
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
			(*Printf.printf "loop of %d using v%d (%d)\n" n indx (st.r);*) 
			(*Out_channel.flush stdout;*)
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
			| (x,_,x',_) ->
				[x-.d_from_origin;
				x'-.d_from_origin;]) |> List.concat in
		let ys = l |> List.map
		(function
			| (_,y,_,y') ->
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
		l |> List.map (fun(x,y,x',y') ->
			(x-.dx+.d_from_origin, y-.dy+.d_from_origin,
						x'-.dx+.d_from_origin, y'-.dy+.d_from_origin))
		
		
let segs_to_canvas segs =
  let segs = center_segs segs in
  let c = ref (new_canvas ()) in
  let lineto x y = (c := (lineto !c x y)) 
  and moveto x y = (c := (moveto !c x y)) in
  (* lineto and moveto are defined in VGWrapper.ml *)
  let total_cost = ref 0. in
  let eval_instruction (x1,y1,x2,y2) =
      total_cost := !total_cost +. 
			(sqrt ((x1-.x2)*.(x1-.x2) +. (y1-.y2)*.(y1-.y2))); 
			(* length of the line *)
		moveto x1 y1;
		lineto x2 y2; ()
  in
  List.iter eval_instruction segs ;
  !c,!total_cost

let segs_to_png segs resolution filename =
  let canvas,_ = segs_to_canvas segs in
  output_canvas_png canvas resolution filename

let segs_to_array_and_cost segs resolution =
	(* outputs a *float* image and cost *)
	(* float so we don't have to convert all the time.  more mem b/w tho*)
	let canvas,cost = segs_to_canvas segs in
	let img = canvas_to_1Darray canvas resolution in
	(* these sometimes have stride > resolution; pack it *)
	let stride = (Bigarray.Array1.dim img) / resolution in
	let len = Bigarray.Array1.dim img in
	assert (len >= resolution * resolution);
	let o = Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout resolution resolution in
	for i = 0 to resolution-1 do (
		for j = 0 to resolution-1 do (
			let c = Bigarray.Array1.get img ((i*stride)+j) |> foi in
			o.{i,j} <- c /. 255.0; 
		) done; 
	) done;
	(o, cost)
