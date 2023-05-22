open Ctypes

type window = unit ptr
let window : window typ = ptr void

type imgdb = unit ptr
let imgdb : imgdb typ = ptr void

open Foreign

let libncurses = Dl.(dlopen ~filename:"libncurses.so.6.4" ~flags:[RTLD_NOW])
let libss = Dl.(dlopen ~filename:"./simsearch.so" ~flags:[RTLD_NOW])

let initscr = foreign "initscr" (void @-> returning window) ~from:libncurses

let newwin =
  foreign "newwin" (int @-> int @-> int @-> int @-> returning window) ~from:libncurses
  
let newsimdb =
  foreign "simdb_allocate" (int @-> returning imgdb) ~from:libss

let endwin = foreign "endwin" (void @-> returning void) ~from:libncurses
let refresh = foreign "refresh" (void @-> returning void) ~from:libncurses
let wrefresh = foreign "wrefresh" (window @-> returning void) ~from:libncurses
let addstr = foreign "addstr" (string @-> returning void) ~from:libncurses


let mvwaddch =
  foreign
    "mvwaddch"
    (window @-> int @-> int @-> char @-> returning void) ~from:libncurses

let mvwaddstr =
  foreign
    "mvwaddstr"
    (window @-> int @-> int @-> string @-> returning void) ~from:libncurses

let box = foreign "box" (window @-> char @-> char @-> returning void) ~from:libncurses
let cbreak = foreign "cbreak" (void @-> returning int) ~from:libncurses
