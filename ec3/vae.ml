(* Variational Auto-Encoder on MNIST.
   The implementation is based on:
     https://github.com/pytorch/examples/blob/master/vae/main.py

   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data' directory.
     train-images-idx3-ubyte.gz
     train-labels-idx1-ubyte.gz
     t10k-images-idx3-ubyte.gz
     t10k-labels-idx1-ubyte.gz
*)
open Torch
let image_res = 30
let image_siz = image_res*image_res

module VAE = struct
  type t =
    { fc1 : Layer.t
    ; fc21 : Layer.t
    ; fc22 : Layer.t
    ; fc3 : Layer.t
    ; fc4 : Layer.t
    }

  let create vs =
    { fc1 = Layer.linear vs ~input_dim:image_siz 400
    ; fc21 = Layer.linear vs ~input_dim:400 20
    ; fc22 = Layer.linear vs ~input_dim:400 20
    ; fc3 = Layer.linear vs ~input_dim:20 400
    ; fc4 = Layer.linear vs ~input_dim:400 image_siz
    }

  let encode t xs =
    let h1 = Layer.forward t.fc1 xs |> Tensor.relu in
    Layer.forward t.fc21 h1, Layer.forward t.fc22 h1

  let decode t zs =
    Layer.forward t.fc3 zs |> Tensor.relu |> Layer.forward t.fc4 |> Tensor.sigmoid

  let forward t xs =
    let mu, logvar = encode t (Tensor.view xs ~size:[ -1; image_siz ]) in
    let std_ = Tensor.(exp (logvar * f 0.5)) in
    let eps = Tensor.randn_like std_ in
    decode t Tensor.(mu + (eps * std_)), mu, logvar
end

let loss ~recon_x ~x ~mu ~logvar =
  let bce =
    Tensor.bce_loss recon_x ~targets:(Tensor.view x ~size:[ -1; image_siz ]) ~reduction:Sum
  in
  let kld = Tensor.(f (-0.5) * (f 1.0 + logvar - (mu * mu) - exp logvar) |> sum) in
  Tensor.( + ) bce kld

let write_samples samples ~filename =
  let samples = Tensor.(samples * f 256.) in
  List.init 8 (fun i ->
      List.init 8 (fun j ->
          Tensor.narrow samples ~dim:0 ~start:((4 * i) + j) ~length:1)
      |> Tensor.cat ~dim:2)
  |> Tensor.cat ~dim:3
  |> Torch_vision.Image.write_image ~filename

let train images device batch_size =
  let nimg,_,_ = Tensor.shape3_exn images in
  let vs = Var_store.create ~name:"vae" ~device () in
  let vae = VAE.create vs in
  let opt = Optimizer.adam vs ~learning_rate:1e-3 in
  for epoch_idx = 1 to 10 do
    let train_loss = ref 0. in
    let samples = ref 0. in
    for _i=0 to 200 do
      let index = Tensor.randint ~size:[batch_size;] ~high:nimg ~options:(T Int64, device) in
      let batch = Tensor.index_select images ~dim:0 ~index in
      let recon_x, mu, logvar = VAE.forward vae batch in
      let loss = loss ~recon_x ~x:batch ~mu ~logvar in
      Optimizer.backward_step ~loss opt;
      train_loss := !train_loss +. Tensor.float_value loss;
      samples := !samples +. (Tensor.shape batch |> List.hd |> Float.of_int)
    done ; 
    Caml.Gc.full_major () ; 
    Printf.printf "epoch %4d  loss: %12.6f\n%!" epoch_idx (!train_loss /. !samples);
    Tensor.randn [ 64; 20 ] ~device
    |> VAE.decode vae
    |> Tensor.to_device ~device:Cpu
    |> Tensor.view ~size:[ -1; 1; image_res; image_res ]
    |> write_samples ~filename:(Printf.sprintf "/tmp/png/vae_samp_%d.png" epoch_idx)
  done; 
  vae

(* external interface *)

let dummy_ext () = 
  let device = Torch.Device.Cpu in
  let vs = Var_store.create ~name:"vae" ~device () in
  VAE.create vs

let encode_ext vae v = 
	let mean,logvar = VAE.encode vae v in
	let std = Tensor.(exp (logvar * f 0.5)) in
	let nn,cols = Tensor.shape2_exn mean in
	let both = Tensor.zeros [nn;cols*2] in
	Tensor.copy_ ~src:mean (Tensor.narrow both ~dim:1 ~start:0 ~length:cols); 
	Tensor.copy_ ~src:std (Tensor.narrow both ~dim:1 ~start:cols ~length:cols);
	both
	
let encode1_ext vae v = 
  (* encode a 1d vector *)
	let mean,logvar = VAE.encode vae v in
	let std = Tensor.(exp (logvar * f 0.5)) in
	let cols = Tensor.shape1_exn mean in
	let both = Tensor.zeros [cols*2] in
	Tensor.copy_ ~src:mean (Tensor.narrow both ~dim:0 ~start:0 ~length:cols); 
	Tensor.copy_ ~src:std (Tensor.narrow both ~dim:0 ~start:cols ~length:cols);
	both
	
let train_ext dbf mnist device batch_size= 
	(* make a new tensor with both mnist and generated images *)
	let ndbf,rows,cols = Tensor.shape3_exn dbf in
	let nmnist,_,_ = Tensor.shape3_exn mnist in (* 60k but w/e *)
	let n = ndbf + nmnist in
	let m = Tensor.zeros [n;rows;cols] ~device in
	Tensor.copy_ ~src:dbf (Tensor.narrow m ~dim:0 ~start:0 ~length:ndbf) ; 
	Tensor.copy_ ~src:mnist (Tensor.narrow m ~dim:0 
			~start:ndbf ~length:nmnist) ;
	let vae = train m device batch_size in
	(* the Vae converts an image into a mean and log variance (or std) *)
	(* to select where to start, find the closest image & modify it *)
	let dbf_enc = encode_ext vae (Tensor.view dbf ~size:[ndbf;-1]) in
	let mnist_enc = encode_ext vae ((Tensor.view mnist ~size:[nmnist;-1]) |> Tensor.to_device ~device ) in
	vae,dbf_enc,mnist_enc
