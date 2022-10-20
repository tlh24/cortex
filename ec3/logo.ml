open Core
open Base.Poly
open Vgwrapper

(* variables will be referenced to the stack, de Brujin indexes *)
(* names are more interpretable by humans ... but that's a big space *)
type prog = [
	| `Var of int 
	| `Save of int * prog
	| `Move of prog * prog (* angle and distance *)
	| `Binop of prog * string * (float -> float -> float) * prog
	| `Const of float
	| `Seq of prog list
	| `Loop of int * prog * prog (* iterations and body *)
	| `Call of int * prog list (* list of arguments *)
	| `Def of int * prog (* same sig as `Save *)
(* 	| Cmp of prog * (float -> float -> bool) * prog *)
	| `Nop
]

let fos = float_of_string
let ios = int_of_string
let foi = float_of_int
let iof = int_of_float

let rec output_program_h lg g =
	match g with
	| `Var i -> Printf.fprintf lg "Var %d " i
	| `Save(i,a) -> Printf.fprintf lg "Save %d " i; 
			output_program_h lg a
	| `Move(a,b) -> Printf.fprintf lg "Move "; 
			output_program_h lg a; 
			Printf.fprintf lg ", " ; 
			output_program_h lg b
	| `Binop(a,s,_,b) -> Printf.fprintf lg "Binop "; 
			output_program_h lg a; 
			Printf.fprintf lg " %s " s; 
			output_program_h lg b
	| `Const(i) -> Printf.fprintf lg "Const %f " i
	| `Seq l -> output_list lg l true "; "
	| `Loop(i,a,b) -> Printf.fprintf lg "Loop [%d] " i; 
			output_program_h lg a; 
			Printf.fprintf lg ", " ; 
			output_program_h lg b
	| `Call(i, l) -> Printf.fprintf lg "Call %d " i;
		output_list lg l true ", "
	| `Def(i, a) -> Printf.fprintf lg "Def %d " i;
		output_program_h lg a
	| `Nop -> Printf.fprintf lg "Nop "
			
and output_program_p lg g = (* p is for parseable *)
	match g with
	| `Var i -> Printf.fprintf lg "v%d " i
	| `Save(i,a) -> Printf.fprintf lg "v%d = ( " i; 
			output_program_p lg a; 
			Printf.fprintf lg ") "; 
	| `Move(a,b) -> Printf.fprintf lg "move "; 
			output_program_p lg a; 
			Printf.fprintf lg ", " ; 
			output_program_p lg b
	| `Binop(a,s,_,b) -> 
			Printf.fprintf lg "( "; 
			output_program_p lg a; 
			Printf.fprintf lg " %s " s; 
			output_program_p lg b; 
			Printf.fprintf lg ") "
	| `Const(i) -> (
		match i with
		| x when x > 0.99 && x < 1.01 -> Printf.fprintf lg "ul "; 
		| x when x > 6.28 && x < 6.29 -> Printf.fprintf lg "ua "; 
		| x -> Printf.fprintf lg "%d " (int_of_float x); 
		)
	| `Seq l -> output_list lg l false "; "
	| `Loop(i,a,b) -> Printf.fprintf lg "loop %d , " i; 
			output_program_p lg a; 
			Printf.fprintf lg ", (" ; 
			output_program_p lg b; 
			Printf.fprintf lg ") " 
	| `Call(i, l) -> Printf.fprintf lg "c%d " i;
		output_list lg l false ", "
	| `Def(i, a) -> Printf.fprintf lg "d%d " i;
		output_program_p lg a
	| `Nop -> ()
	
and output_list lg l h sep =
	Printf.fprintf lg "("; 
	List.iteri ~f:(fun i v -> 
		if i > 0 then Printf.fprintf lg "%s" sep ;
		if h then output_program_h lg v
		else output_program_p lg v) l ; 
	Printf.fprintf lg ")"
	

type state =
  { x : float
  ; y : float
  ; t : float (* theta *)
  ; p : bool (* pen down = true *)
  ; r : int (* execution count *)
  ; stk : float array
  }

let defs = Array.create ~len:10 `Nop

type segment = float*float*float*float

let output_segments lg seglist = 
	List.iteri ~f:(fun i (x1,y1,x2,y2) -> 
		Printf.fprintf lg 
			"%d %f,%f %f,%f\n" 
			i x1 y1 x2 y2) seglist

let start_state = {x=0.0; y=0.0; t=0.0; p=true; r=0;
		stk=Array.create ~len:10 0.0}
 
(* eval needs to take a state & program
and return new state & (bool * float) result & segment list *)
let rec eval (st:state) (pr:prog) = 
	match pr with 
	| `Var(i) -> 
		if i >= 0 && i < 10 then (
			(st, (true, st.stk.(i)), [])
		) else (
			(st, (false, 0.0), [])
		)
	| `Save(i, a) -> 
		if i >= 0 && i < 10 then (
			let (sta, resa, seg) = eval st a in
			sta.stk.(i) <- (snd resa) ;
			(sta, resa, seg)
		)  else (
			(st, (false, 0.0), [])
		)
	| `Move(a, b) -> (* distance, angle -- applied in that order *)
		let (sta, resa, _) = eval st a in
		let (stb, resb, _) = eval sta b in
		let dist = snd resa in
		let ang = snd resb in
		let x' = stb.x +. (dist *. Float.cos(stb.t)) in 
		let y' = stb.y +. (dist *. Float.sin(stb.t)) in 
		let t' = stb.t +. ang in
		let st2 = {stb with x = x'; y = y'; t = t' } in
		let seg = st.x, st.y, st2.x, st2.y in
		(st2, (true, dist), [seg])
	| `Binop(a, s, f, b) -> 
		let (sta, resa, _) = eval st a in
		let (stb, resb, _) = eval sta b in
		let ra = snd resa in
		let rb = snd resb in
		let r = match s with
		| "/" -> (
			if rb < 0.001 && rb > -0.001 
			then ra (* don't divide by zero *)
			else f ra rb )
		| _ -> f ra rb in
		(stb, (true, r), [])
	| `Const(f) -> 
		(st, (true, f), [])
	| `Seq(program_list) -> 
			List.fold_left ~f:(fun (st2,_,segments) sub_prog -> 
				let st3, res3, seg = eval st2 sub_prog in
				(st3, res3, (List.append seg segments) ) )
				~init:(st, (true,0.0), []) program_list
	| `Loop(indx, niter, body) -> 
		if indx >= 0 && indx < 10 then (
			let (sta, resa, _) = eval st niter in
			let n = iof (snd resa) in
			let cntlist = List.init n ~f:(fun i -> i) in
			List.fold_left ~f:( fun(st2,_,segments) i -> 
				st2.stk.(indx) <- foi i;
				let st3, res3, seg = eval st2 body in
				(st3, res3, (List.append seg segments) ) )
				~init:(sta, (true,0.0), []) cntlist
		) else (
			(st, (false, 0.0), [])
		)
	| `Call(indx, program_list) ->
		if defs.(indx) <> `Nop then (
			if List.length program_list < 5 then (
				(* make a new stack and populate it *)
				(* arguments are not allowed to have side-effects *)
				let res = List.map ~f:(fun subprog ->
					let _st, res2, _seg = eval st subprog in
					res2 ) program_list in
				let st3 = {x=st.x; y=st.y; t=st.t; p=st.p; r=st.r;
					stk=Array.create ~len:10 0.0 } in
				List.iteri ~f:(fun i v -> st3.stk.(i) <- snd v) res ;
				let _st4, res4, seg4 = eval st3 defs.(indx) in
				(* call does not affect state *)
				(st, res4, seg4)
			) else (st, (false, 0.0), [])
		) else (st, (false, 0.0), [])
	| `Def(indx, body) ->
		if (indx >= 0 && indx < 10) then (
			defs.(indx) <- body ) ;
		(st, (false, 0.0), [])
	| `Nop -> (st, (false, 0.0), [])

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
		~f:(function
			| (x,_,x',_) ->
				[x-.d_from_origin;
				x'-.d_from_origin;]) |> List.concat in
		let ys = l |> List.map
		~f:(function
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
		l |> List.map ~f:(fun(x,y,x',y') ->
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
  List.iter ~f:eval_instruction segs ;
  !c,!total_cost

let segs_to_png segs resolution filename =
  let canvas,_ = segs_to_canvas segs in
  output_canvas_png canvas resolution filename

let segs_to_array_and_cost segs resolution =
  let canvas,cost = segs_to_canvas segs in
  (canvas_to_1Darray canvas resolution, cost)
