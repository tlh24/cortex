open Logo

(* q: how does ocaml do reference counting? 
if a data structure is wholly immutable, ok to just store it? 
or should we keep a ref..*)

type gdata = 
	{ progenc : string
	; outgoing : gdata ref Vector.t
	; incoming : gdata ref Vector.t
	}

let nulgdata = 
