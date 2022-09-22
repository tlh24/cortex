
(* The type of tokens. *)

type token = 
  | VAR of (int)
  | UNIT_LENGTH
  | UNIT_ANGLE
  | SEMICOLON
  | RIGHT_PAREN
  | PLUS
  | MULT
  | MOVE
  | MINUS
  | LOOP
  | LEFT_PAREN
  | INT of (int)
  | EOF
  | DIVI
  | COMMA

(* This exception is raised by the monolithic API functions. *)

exception Error

(* The monolithic API. *)

val parse_prog: (Lexing.lexbuf -> token) -> Lexing.lexbuf -> (Logo.prog)
