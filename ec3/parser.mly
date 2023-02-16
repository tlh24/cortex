%token <int> VAR
%token <int> INT
%token <string> STR
%token MOVE
%token LOOP
%token <int> CALL
%token <int> DEF
%token UNIT_ANGLE
%token UNIT_LENGTH
%token PEN
%token MULT
%token DIVI
%token PLUS
%token MINUS
%token LEFT_PAREN
%token RIGHT_PAREN
%token COLON
%token SEMICOLON
%token COMMA
%token EQUALS
%token EOF

%start <Logo.prog> parse_prog

%%

parse_prog: value EOF { $1 }


value:
  | LEFT_PAREN; obj = seq_elements; RIGHT_PAREN 
		{ `Seq(obj,0)  }
  | v = VAR                                
		{ `Var(v,0) }
  | MOVE; a = value; COMMA ; b = value 
      { `Move( a, b, 0) }
  | LOOP; a = INT; COMMA ; b = value ; COMMA ; c = value
      { `Loop( a, b, c, 0 ) }
  | a = CALL; obj = call_elements
      { `Call(a, obj, 0) }
  | a = DEF; COLON; b = value
      { `Def(a, b, 0) }
  | PEN; a = value
      { `Pen(a,0) }
  | UNIT_ANGLE
		{ `Const( 8.0 *. atan 1.0 , 0 ) }
  | UNIT_LENGTH
		{ `Const( 1.0 , 0) } 
  | a = value; MULT ; b = value
      { `Binop(a, "*", ( *. ), b, 0) }
  | a = value; DIVI ; b = value
      { `Binop(a, "/", ( /. ), b, 0) }
  | a = value; PLUS ; b = value
      { `Binop(a, "+", ( +. ), b, 0) }
  | a = value; MINUS ; b = value
      { `Binop(a, "-", ( -. ), b, 0) }
  | v = VAR; EQUALS; b = value
      { `Save(v, b, 0) }
  | i = INT                                  
		{ `Const( float_of_int i , 0)   }
/*| s = STRING                               
		{ `String s   }*/
  ; 
  
seq_elements:
  obj = separated_list(SEMICOLON, value)    
    { obj } ;

call_elements:
  obj = separated_list(COMMA, value)    
    { obj } ;
