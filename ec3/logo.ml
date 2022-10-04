open Core
open Base.Poly
open Out_channel
open Vgwrapper

(* variables will be referenced to the stack, de Brujin indexes *)
(* names are more interpretable by humans ... but that's a big space *)
type prog = [
	| `Var of int 
	| `Save of int * prog
	| `Move of prog * prog (* angle and distance *)
	| `Binop of prog * (float -> float -> float) * prog
	| `Const of float
	| `Seq of prog list
	| `Loop of int* prog * prog (* iterations and body *)
(* 	| Call of int *)
(* 	| Cmp of prog * (float -> float -> bool) * prog *)
(* no 'let' here yet *)
]

let fos = float_of_string
let ios = int_of_string
let foi = float_of_int
let iof = int_of_float

let rec output_program = function
	| `Var i -> printf "Var %d " i
	| `Save(i,a) -> printf "Save %d " i; 
			output_program a
	| `Move(a,b) -> printf "Move "; 
			output_program a; 
			printf ", " ; 
			output_program b
	| `Binop(a,_,b) -> printf "Binop "; 
			output_program a; 
			output_program b
	| `Const(i) -> printf "Const %f " i
	| `Seq l -> output_list l
	| `Loop(i,a,b) -> printf "Loop [%d] " i; 
			output_program a; 
			printf ", " ; 
			output_program b
	
and output_list l = 
	printf "("; 
	List.iteri ~f:(fun i v -> 
		if i > 0 then printf "; " ; 
		output_program v) l ; 
	printf ")"
	

type state =
  { x : float
  ; y : float
  ; t : float (* theta *)
  ; p : bool (* pen down = true *)
  ; r : int (* execution count *)
  }
  
let stack = Array.create ~len:10 0.0
(* somehow state is also going to need a stack .. ?? *)
(* is state mutable?  within a sequence, yes, it needs to be continuous. 
when calling a function -- no, I think not. *)

type segment = float*float*float*float

let output_segments seglist = 
	List.iteri ~f:(fun i (x1,y1,x2,y2) -> 
		printf "%d %f,%f %f,%f\n" i x1 y1 x2 y2) seglist

let start_state = {x=0.0; y=0.0; t=0.0; p=true; r=0}
 
(* eval needs to take a state & program
and return new state & (bool * float) result & segment list *)
let rec eval (st:state) (pr:prog) = 
	match pr with 
	| `Var(i) -> 
		if i >= 0 && i < 10 then (
			(st, (true, stack.(i)), []) 
		) else (
			(st, (false, 0.0), [])
		)
	| `Save(i, a) -> 
		if i >= 0 && i < 10 then (
			let (sta, resa, seg) = eval st a in
			stack.(i) <- (snd resa) ;
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
	| `Binop(a, f, b) -> 
		let (sta, resa, _) = eval st a in
		let (stb, resb, _) = eval sta b in
		let r = f (snd resa) (snd resb) in
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
				stack.(indx) <- foi i; 
				let st3, res3, seg = eval st2 body in
				(st3, res3, (List.append seg segments) ) )
				~init:(sta, (true,0.0), []) cntlist
		) else (
			(st, (false, 0.0), [])
		)

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
