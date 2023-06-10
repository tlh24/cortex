let dbSize = (400 * 256)
let dbDim = 900

let foi = float_of_int

open Ctypes

type imgdb = unit ptr
let imgdb : imgdb typ = ptr void

open Foreign

let libss = Dl.dlopen ~flags:[Dl.RTLD_NOW] 
	~filename:"../cuda_simsearch/libsimsearch.so" 

let simdb_allocate = foreign "simdb_allocate" 
			(int @-> returning imgdb) ~from:libss;;
let simdb_free = foreign "simdb_free" 
			(imgdb @-> returning void) ~from:libss;;
let simdb_set = foreign "simdb_set" 
			(imgdb @-> int @-> ptr void @-> returning void) ~from:libss;;
let simdb_get = foreign "simdb_get" 
			(imgdb @-> int @-> ptr void @-> returning void) ~from:libss;;
let simdb_query = foreign "simdb_query" 
			(imgdb @-> ptr void @-> ptr void @-> ptr void @-> returning void) ~from:libss;;
let simdb_checksum = foreign "simdb_checksum"
			(imgdb @-> returning double) ~from:libss;;
let simdb_clear = foreign "simdb_clear"
			(imgdb @-> returning void) ~from:libss;;
(* Note: 
	https://github.com/dbuenzli/tsdl/blob/master/src/tsdl.ml
	is a good reference for Ctypes FFI, 
	including Bigarray accesses *)
	
let new_ba_row () = 
	(* only allocates a bigarray *)
	Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout dbDim
	;;
	
let mutex = Mutex.create () 
(* I don't fully unserstand why, 
	But it's clear that multiple 'query' calls from simultaneous threads
	leads to invalid results. 
	This may be a function of how Ctypes interfaces with .so -
	should be through system threads, but maybe not? 
	maybe it expects everything to be from the main thread? 
	Irregardless, this works and doesn't seem to slow things down appreciably. 
	If we need to speed things further, can batch the queries. 
	*)

let rowset sdb i ba = 
	let len = Bigarray.Array1.dim ba in
	if len <> dbDim then 
		invalid_arg (Printf.sprintf "array length %d not %d" len dbDim)
	else (
		let ps = to_voidp (bigarray_start array1 ba) in
		Mutex.lock mutex; 
		simdb_set sdb i ps; 
		Mutex.unlock mutex
	);;
	
let rowget sdb i = 
	let ba = new_ba_row () in
	let ps = to_voidp (bigarray_start array1 ba) in
	Mutex.lock mutex; 
	simdb_get sdb i ps; 
	Mutex.unlock mutex; 
	ba
	;;
		
let query sdb ba = 
	let len = Bigarray.Array1.dim ba in
	if len <> dbDim then (
		invalid_arg (Printf.sprintf "array length %d not %d" len dbDim)
	) else (
		let ps = to_voidp (bigarray_start array1 ba) in
		let dist = allocate float (-1.0) in
		let indx = allocate int (-1) in
		Mutex.lock mutex; 
		simdb_query sdb ps (to_voidp dist) (to_voidp indx); 
		Mutex.unlock mutex; 
		!@dist, !@indx
	);;

let checksum sdb = 
	Mutex.lock mutex; 
	let res = simdb_checksum sdb in
	Mutex.unlock mutex;
	res
	;;
	
let clear sdb = 
	Mutex.lock mutex; 
	simdb_clear sdb; 
	Mutex.unlock mutex; 
	()
	;;
	
let init count = 
	if count > dbSize then (
		invalid_arg (Printf.sprintf "count %d > %d" count dbDim)
	) else (
		Mutex.lock mutex; 
		let sdb = simdb_allocate dbSize in
		Mutex.unlock mutex; 
		sdb 
	);;

let test () =
	let sdb = init dbSize in
	let ba = new_ba_row () in
	for k = 0 to dbSize-1 do (
		for j = 0 to dbDim-1 do (
			ba.{j} <- Random.int 256
		) done; 
		rowset sdb k ba
	) done; 
	(* make a new random query *)
	let rand_indx = Random.int dbSize in
	for j = 0 to dbDim-1 do (
		ba.{j} <- Random.int 256
	) done; 
	rowset sdb rand_indx ba; 
	
	(* check *)
	let sta = Unix.gettimeofday () in
	for _k = 0 to 18 do (
		ignore(query sdb ba)
	) done; 
	let dist, indx = query sdb ba in
	let fin = Unix.gettimeofday () in
	let duration = ((fin -. sta) /. 20.0) in
	Logs.debug (fun m->m "simdb Time: %f; Bandwidth:%f GB/sec"
		 duration ((foi (dbSize * dbDim)) /. (duration *. 1e9))); 
	Logs.debug (fun m->m "simdb Best match: %d, should be %d; Min dist: %f"
				indx rand_indx dist ); 
	
	(* all done; free *)
	simdb_free sdb; 
	Printf.printf "done.\n"
	;;
