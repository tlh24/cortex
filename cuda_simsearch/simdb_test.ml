let dbSize = (400 * 256)
let dbDim = 900

let foi = float_of_int

open Ctypes

type imgdb = unit ptr
let imgdb : imgdb typ = ptr void

open Foreign

let libss = Dl.dlopen ~flags:[Dl.RTLD_NOW] ~filename:"./libsimsearch.so" 

let simdb_allocate = foreign "simdb_allocate" 
			(int @-> returning imgdb) ~from:libss;;
let simdb_free = foreign "simdb_free" 
			(imgdb @-> returning void) ~from:libss;;
let simdb_setU = foreign "simdb_set" 
			(imgdb @-> int @-> ptr void @-> returning void) ~from:libss;;
let simdb_queryU = foreign "simdb_query" 
			(imgdb @-> ptr void @-> ptr void @-> ptr void @-> returning void) ~from:libss;;
			
(* Note: 
	https://github.com/dbuenzli/tsdl/blob/master/src/tsdl.ml
	is a good reference for Ctypes FFI, 
	including Bigarray accesses *)

let simdb_set sdb i ba = 
	let len = Bigarray.Array1.dim ba in
	if len <> dbDim then 
		invalid_arg (Printf.sprintf "array length %d not %d" len dbDim)
	else 
		let ps = to_voidp (bigarray_start array1 ba) in
		simdb_setU sdb i ps
	;;
		
let simdb_query sdb ba = 
	let len = Bigarray.Array1.dim ba in
	if len <> dbDim then (
		invalid_arg (Printf.sprintf "array length %d not %d" len dbDim)
	) else (
		let ps = to_voidp (bigarray_start array1 ba) in
		let dist = allocate float (-1.0) in
		let indx = allocate int (-1) in
		simdb_queryU sdb ps (to_voidp dist) (to_voidp indx); 
		!@dist, !@indx
	)

let () =
	let sdb = simdb_allocate dbSize in
	let ba = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout dbDim in
	for k = 0 to dbSize-1 do (
		for j = 0 to dbDim-1 do (
			ba.{j} <- Random.int 256
		) done; 
		simdb_set sdb k ba
	) done; 
	(* make a new random query *)
	let rand_indx = Random.int dbSize in
	for j = 0 to dbDim-1 do (
		ba.{j} <- Random.int 256
	) done; 
	simdb_set sdb rand_indx ba; 
	
	(* check *)
	let sta = Unix.gettimeofday () in
	for _k = 0 to 18 do (
		ignore(simdb_query sdb ba)
	) done; 
	let dist, indx = simdb_query sdb ba in
	let fin = Unix.gettimeofday () in
	let duration = ((fin -. sta) /. 20.0) in
	Printf.printf "Execution time: %f; Bandwidth:%f GB/sec\n"
		 duration ((foi (dbSize * dbDim)) /. (duration *. 1e9)); 
	Printf.printf "Best match: %d, should be %d; Minimum distance: %f\n"
				indx rand_indx dist; 
	
	(* all done; free *)
	simdb_free sdb; 
	Printf.printf "done.\n"
	;;

(* result: Ocaml is slightly slower than C/C++ (as expected), but it's certainly fast enough & is using a majority of the GPU bandwidth.
If we want faster, can batch queries. *)
