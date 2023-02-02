open Torch
open Unix

let image_count = 5*2048
let image_res = 30

(* 
test the ocaml equivalent of (python): 
	dbf = th.ones(image_count, image_res, image_res)
	d = th.sum((dbf - a)**2, (1,2))
	mindex = th.argmin(d)
	dist = d[mindex]
*)

let image_dist dbf img = 
	let d = Tensor.( (dbf - img) ) in
	(* per-element square and sum *)
	let d2 = Tensor.einsum ~equation:"ijk, ijk -> i" [d;d] ~path:None in
	let mindex = Tensor.argmin d2 ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d2 mindex |> Tensor.float_value in
	dist,mindex
	
let measure_torch_copy_speed device = 
	let start = Unix.gettimeofday () in
	let nimg = 6*2048*2 in
	let dbf = Tensor.( zeros [nimg; image_res; image_res] ) in
	for i = 0 to nimg/2-1 do (
		let k = Tensor.ones [image_res; image_res] in
		Tensor.copy_ (Tensor.narrow dbf ~dim:0 ~start:i ~length:1) ~src:k; 
	) done; 
	let y = Tensor.to_device dbf ~device in
	let z = Tensor.sum y in
	let stop = Unix.gettimeofday () in
	Printf.printf "%d image_copy time: %fs\n%!" (nimg/2) (stop -. start);
	Printf.printf "%f\n%!" (Tensor.float_value z) 
	(* this is working just as fast or faster than python.*)
	(* something else must be going on in the larger program *)

let () = 
	Unix.clear_nonblock stdin; 
	Printf.printf "cuda available: %b\n%!" (Cuda.is_available ());
	let device = Torch.Device.cuda_if_available () in
	(* dbf is a tensor of images to be compared (MSE) against *)
	let dbf = Tensor.( 
		( ones [image_count; image_res; image_res] ) * (f (-1.0))) 
		|> Tensor.to_device ~device in
	let start = Unix.gettimeofday () in
	for i = 0 to 10_000 do (
		(* generate a random image *)
		let img = Tensor.(randn [image_res; image_res] ) 
			|> Tensor.to_device ~device in
		ignore( image_dist dbf img ); 
		if i mod 30 = 29 then 
			Caml.Gc.major()
		(* in the actual program, we do something with dist,mindex *)
	) done; 
	let stop = Unix.gettimeofday () in
	Printf.printf "10k image_dist calc time: %fs\n%!" 
		(stop -. start);
		
	(* measure the image copy speed *)
	for _i = 0 to 100 do
		measure_torch_copy_speed device
	done

