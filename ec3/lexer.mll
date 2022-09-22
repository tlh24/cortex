{
open Parser

exception SyntaxError of string

let remove_first_char s =
  String.sub s 1 ((String.length s) - 1) 
}


let int = '-'? ['0'-'9'] ['0'-'9']*

(* going to keep this around since it's smart *)
let digit = ['0'-'9']
let frac = '.' digit*
let exp = ['e' 'E'] ['-' '+']? digit+
let float = digit* frac? exp?

let white = [' ' '\t']+
let newline = '\r' | '\n' | "\r\n"
let id = ['a'-'z' 'A'-'Z' '_'] ['a'-'z' 'A'-'Z' '0'-'9' '_']*
let var = 'v' digit
(*let fun = 'f' digit+
let call = 'c' digit+*)

rule read =
  parse
  | white    { read lexbuf }
  | newline  { Lexing.new_line lexbuf; read lexbuf }
  | var		 { 
      VAR (int_of_string (remove_first_char (Lexing.lexeme lexbuf))) }
  | int      { INT (int_of_string (Lexing.lexeme lexbuf)) }
(*  | float    { FLOAT (float_of_string (Lexing.lexeme lexbuf)) }*)
  | "move"  { MOVE }
  | "loop"  { LOOP }
  | "ua"		{ UNIT_ANGLE }
  | "ul"		{ UNIT_LENGTH }
  | '*'		{ MULT }
  | '/'		{ DIVI }
  | '+'		{ PLUS }
  | '-'		{ MINUS }
  | '('      { LEFT_PAREN }
  | ')'      { RIGHT_PAREN }
  | ';'      { SEMICOLON }
  | ','      { COMMA }
  | _ { raise (SyntaxError ("Unexpected char: " ^ Lexing.lexeme lexbuf)) }
  | eof      { EOF }
