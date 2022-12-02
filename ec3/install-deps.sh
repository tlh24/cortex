# commands for getting lambda labs up and running 
sudo chmod +rw /usr/bin
sudo apt-get install ocaml
# upgrade opam
bash -c "sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)"
# add a flambda build, with the latest ocaml compiler
sudo apt-get install make gcc unzip bubblewrap
opam init
opam update
opam switch create myswitch ocaml-variants.4.14.0+options ocaml-option-flambda
eval $(opam env --switch=myswitch)
opam update
opam install core core_unix vg cairo2 pbrt vector ocaml-protoc

# for accessing remotely: 
# sshfs -o allow_other,default_permissions ubuntu@104.171.203.63:/home/ubuntu/cortex/ /home/tlh24/remote/
