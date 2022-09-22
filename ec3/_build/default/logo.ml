open Core
open Base.Poly
open Out_channel

(* variables will be referenced to the stack, de Brujin indexes *)
(* names are more interpretable by humans ... but that's a big space *)
type prog = [
	| `Var of int 
	| `Move of prog * prog (* angle and distance *)
	| `Binop of prog * (float -> float -> float) * prog
	| `Save of int * prog
	| `Const of float
	| `Seq of prog list
	| `Loop of prog * prog (* iterations and body *)
(* 	| Call of int *)
(* 	| Cmp of prog * (float -> float -> bool) * prog *)
(* no 'let' here yet *)
]

let fos = float_of_string
let ios = int_of_string

let rec output_program = function
	| `Var i -> printf "Var %d " i
	| `Move(a,b) -> printf "Move "; 
			output_program a; 
			printf ", " ; 
			output_program b
	| `Binop(a,_,b) -> printf "Binop "; 
			output_program a; 
			output_program b
	| `Save(i,a) -> printf "Save %d " i; 
			output_program a
	| `Const(i) -> printf "Const %f " i
	| `Seq l -> output_list l
	| `Loop(a,b) -> printf "Loop  "; 
			output_program a; 
			printf ", " ; 
			output_program b
	
and output_list l = 
	printf "("; 
	List.iteri ~f:(fun i v -> 
		if i > 0 then printf "; " ; 
		output_program v) l ; 
	printf ")"
	

(* realisticly .. the interpreter can act on the AST, and doesn't need to act on the string representation of the program *)
(* maybe initially just do the string representation .. ? *)
(* want a DSL that can represent: 
(
	(move 0d (/a 1a 6))
	(loop i 7 
		(move ( *l 1l i) (/a 1a 4))
	)
)
*)
