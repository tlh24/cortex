%token <int> VAR
%token <int> INT
%token MOVE
%token LOOP
%token UNIT_ANGLE
%token UNIT_LENGTH
%token MULT
%token DIVI
%token PLUS
%token MINUS
%token LEFT_PAREN
%token RIGHT_PAREN
%token SEMICOLON
%token COMMA
%token EOF


%start <Logo.prog> parse_prog

%%

parse_prog: value EOF { $1 }


value:
  | LEFT_PAREN; obj = seq_elements; RIGHT_PAREN 
		{ `Seq obj  }
  | v = VAR                                
		{ `Var v   }
  | MOVE; obj = two_elements; 
      { `Move( (fst obj), (snd obj) ) }
  | LOOP; obj = three_elements; 
      { `Loop( obj ) }
  | UNIT_ANGLE
		{ `Const( 8.0 *. atan 1.0  ) }
  | UNIT_LENGTH
		{ `Const( 1.0 ) } 
  | a = value; MULT ; b = value
      { `Binop(a, ( *. ), b) }
  | a = value; DIVI ; b = value
      { `Binop(a, ( /. ), b) }
  | a = value; PLUS ; b = value
      { `Binop(a, ( +. ), b) }
  | a = value; MINUS ; b = value
      { `Binop(a, ( -. ), b) }
  | i = INT                                  
		{ `Const( float_of_int i )   }
/*| s = STRING                               
		{ `String s   }*/
  ; 
  
seq_elements:
  obj = separated_list(SEMICOLON, value)    
    { obj } ;

two_elements: 
  a  = value; l = value 
    { (a, l) } 
    
three_elements: 
  i = INT; a  = value; l = value 
    { (i, a, l) }
