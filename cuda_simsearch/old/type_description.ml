(* open Ctypes *)

module Types (F : Ctypes.TYPE) = struct
	open F

	let db_size = constant "DB_SIZE" int
end
