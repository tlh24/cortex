open Program


let usage_msg = "logofilerun.exe <logo_file>"
let input_files = ref []
let output_file = ref ""
let imageres = ref 256 
let anon_fun filename = (* just incase we need later *)
  input_files := filename :: !input_files
let speclist =
  [("-r", Arg.Set_int imageres, "output resolution (default 256)");
   ("-g", Arg.Set gdebug, "Turn on debug");]

let () = 
	Arg.parse speclist anon_fun usage_msg;
	Random.self_init (); 
	gdebug := true; (*can't imagine when you'd not want error msgs*)
	Logs_threaded.enable ();
	let () = Logs.set_reporter (Logs.format_reporter ()) in
	let () = Logs.set_level 
		(if !gdebug then Some Logs.Debug else Some Logs.Info) in
	if !gdebug then Logs.debug (fun m -> m "Debug logging enabled.")
	else Logs.info (fun m -> m "Debug logging disabled.") ; 
	
	if List.length !input_files > 0 then (
		List.iter (fun fname -> 
		Logs.info (fun m->m "opening %s" fname); 
		Logoext.run_logo_file fname !imageres !gdebug) !input_files
	) else (
		Logs.info (fun m->m "please specify a logo file"); 
	)
	
