open Base
open Torch

(* This should reach ~97% accuracy. *)
let hidden_nodes = 128
let epochs = 10000
let learning_rate = 1e-3

let () =
  Stdio.printf "cuda available: %b\n%!" (Cuda.is_available ());
  Stdio.printf "cudnn available: %b\n%!" (Cuda.cudnn_is_available ());
  let device = Torch.Device.cuda_if_available () in
  let mnist = Mnist_helper.read_files () in
  let { Dataset_helper.train_images; train_labels; _ } = mnist in
  let vs = Var_store.create ~name:"nn" ~device () in
  let linear1 =
    Layer.linear vs hidden_nodes ~activation:Relu ~input_dim:Mnist_helper.image_dim
  in
  let linear2 = Layer.linear vs Mnist_helper.label_count ~input_dim:hidden_nodes in
  let adam = Optimizer.adam vs ~learning_rate ~weight_decay:5e-5 in
  let model xs =
    Layer.forward linear1 xs
    |> Layer.forward linear2 in
  let img = train_images |> Torch.Tensor.to_device ~device in
  let lab = train_labels |> Torch.Tensor.to_device ~device in
  for index = 1 to epochs do
    (* Compute the cross-entropy loss. *)
    let loss =
      Tensor.cross_entropy_for_logits (model img) ~targets:lab
    in
    Optimizer.backward_step adam ~loss;
    if index % 50 = 0
    then (
      (* Compute the validation error. *)
      let test_accuracy =
        Dataset_helper.batch_accuracy ~device mnist `test ~batch_size:1000 ~predict:model
      in
      Stdio.printf
        "%d %f %.2f%%\n%!"
        index
        (Tensor.float_value loss)
        (100. *. test_accuracy));
    Caml.Gc.full_major ()
  done
