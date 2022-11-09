{
open Parser
open Lexing

exception SyntaxError of string

let remove_first_char s =
  String.sub s 1 ((String.length s) - 1) 
  
let ios_tok s =
  remove_first_char s |> int_of_string

let next_line lexbuf =
  let pos = lexbuf.lex_curr_p in
  lexbuf.lex_curr_p <-
    { pos with pos_bol = lexbuf.lex_curr_pos;
              pos_lnum = pos.pos_lnum + 1
    }
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
let call = 'c' digit+
let def = 'd' digit+ (* lazy but i don't care *)

rule read =
  parse
  | white    { read lexbuf }
  | newline  { Lexing.new_line lexbuf; read lexbuf }
  | var		 { VAR  (ios_tok (Lexing.lexeme lexbuf)) }
  | call		 { CALL (ios_tok (Lexing.lexeme lexbuf)) }
  | def		 { DEF  (ios_tok (Lexing.lexeme lexbuf)) }
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
  | ':'      { COLON }
  | ';'      { SEMICOLON }
  | ','      { COMMA }
  | '='     { EQUALS }
  | "//"    { read_single_line_comment lexbuf }
  | _ { raise (SyntaxError ("Unexpected char: " ^ Lexing.lexeme lexbuf)) }
  | eof      { EOF }

and read_single_line_comment = parse
  | newline { next_line lexbuf; read lexbuf }
  | eof { EOF }
  | _ { read_single_line_comment lexbuf }
