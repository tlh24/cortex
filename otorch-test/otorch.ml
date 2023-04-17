open Torch

(*open Ctypes
module C = Torch_bindings.C (Torch_generated)
Caml.Gc.finalise C.Tensor.free d;
Caml.Gc.finalise C.Tensor.free d2; *)

let image_count = 1*2048
let image_res = 30

(* 
test the ocaml equivalent of (python): 
	dbf = th.ones(image_count, image_res, image_res)
	d = th.sum((dbf - a)**2, (1,2))
	mindex = th.argmin(d)
	dist = d[mindex]
*)
let foi = float_of_int

let image_dist_a dbf img = (* max 2044 MB *)
	let d = Tensor.( (dbf - img) ) in
	(* per-element square and sum *)
	let d2 = Tensor.einsum ~equation:"ijk, ijk -> i" [d;d] ~path:None in
	let mindex = Tensor.argmin d2 ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d2 mindex |> Tensor.float_value in
	dist,mindex

let image_dist_b dbf img =  (* max 2302 MB *)
	let dbfn,h,_ = Tensor.shape3_exn dbf in
	let b = Tensor.expand img ~implicit:true ~size:[dbfn;h;h] in
	let d = Tensor.(sum_dim_intlist (square(dbf - b)) 
			~dim:(Some [1;2]) ~keepdim:false ~dtype:(T Float) ) in
	let mindex = Tensor.argmin d ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d mindex |> Tensor.float_value in
	dist,mindex
	
let run_nvidiasmi () = 
	let (ocaml_stdout, ocaml_stdin, ocaml_stderr) = 
		Unix.open_process_full "nvidia-smi | grep exe" [||] in
	close_out ocaml_stdin;
	print_chan ocaml_stdout;
	print_chan ocaml_stderr;

let () = 
	Printf.printf "cuda available: %b\n%!" (Cuda.is_available ());
	let device = Torch.Device.cuda_if_available () in
	(* dbf is a tensor of images to be compared (MSE) against *)
	let dbf = Tensor.( (ones [image_count; image_res; image_res] ) * (f (-1.0)))
		|> Tensor.to_device ~device in
	let siz = image_count * image_res * image_res * 4 in
	Printf.printf "dbf size: %d bytes %f MB\n" siz ((foi siz) /. 1e6); 
	let start = Unix.gettimeofday () in
	for i = 0 to 30 do (
		(* generate a random image *)
		let img = Tensor.(randn [image_res; image_res] ) 
			|> Tensor.to_device ~device in
		ignore( image_dist_a dbf img ); 
		(* in the actual program, we do something with dist,mindex *)
		if i mod 10 = 9 then (
			Caml.Gc.full_major(); 
		); 
		run_nvidiasmi ()
	) done; 
	let stop = Unix.gettimeofday () in
	Printf.printf "1k image_dist calc time: %fs\n%!" 
		(stop -. start);
	ignore(read_line ()); 
	Printf.printf "Check nvidia-smi for memory usage" 

