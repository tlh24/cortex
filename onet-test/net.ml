open Lwt

(* Shared mutable counter *)
let counter = ref 0

let listen_address = Unix.inet_addr_loopback
let port = 9000
let backlog = 10


let handle_message msg =
	match msg with
	| "read" -> string_of_int !counter
	| "inc"  -> counter := !counter + 1; "Counter has been incremented"
	| _      -> "Unknown command"

let rec handle_connection ic oc () =
	Lwt_io.read_line_opt ic >>=
	(fun msg ->
		match msg with
		| Some msg -> 
			let reply = handle_message msg in
			Lwt_io.write_line oc reply >>= handle_connection ic oc
		| None -> Logs_lwt.info (fun m -> m "Connection closed") >>= return)

let accept_connection conn =
	let fd, _ = conn in
	let ic = Lwt_io.of_fd ~mode:Lwt_io.Input fd  in
	let oc = Lwt_io.of_fd ~mode:Lwt_io.Output fd in
	Lwt.on_failure (handle_connection ic oc ()) (fun e -> Logs.err (fun m -> m "%s" (Printexc.to_string e) ));
	Logs_lwt.info (fun m -> m "New connection") >>= return
 
let create_socket () =
	let open Lwt_unix in
	let sock = socket PF_INET SOCK_STREAM 0 in
	let adr = ADDR_INET(listen_address, port) in
	Lwt.async( fun () -> bind sock adr );
	listen sock backlog;
	sock

let create_server sock =
	let rec serve () =
		Lwt_unix.accept sock >>= accept_connection >>= serve
	in serve

let () =
	let () = Logs.set_reporter (Logs.format_reporter ()) in
	let () = Logs.set_level (Some Logs.Info) in
	let sock = create_socket () in
	let serve = create_server sock in
	Lwt_main.run @@ serve ()
