open Core
open Base.Poly
open Out_channel

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
