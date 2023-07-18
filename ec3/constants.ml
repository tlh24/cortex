let pi = 3.1415926
(*let image_count = ref 0 *)
let image_alloc = 4*1024 (*6*2048*2*) (* need to make this a parameter *)
let all_alloc = 4*1024 (* including equivalents *)
let image_res = 30
let batch_size = ref 512
let toklen = 30
let p_ctx = 96
let poslen = 7 (* 7 graycode; 128 -> 7 bits + flag *)
let p_indim = toklen + 1 + 1*poslen (* 31 + 56 = 87 *)
let e_indim = 5 + toklen + poslen (* edits only use absolute position *)

let soi = string_of_int
let ios = int_of_string
let iof = int_of_float
let foi = float_of_int
