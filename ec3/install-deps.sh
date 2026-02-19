# Install conda
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local && rm Miniconda3-latest-Linux-x86_64.sh && /usr/local/bin/conda init


# commands for getting lambda labs up and running
sudo chmod +rw /usr/bin
sudo apt-get update
sudo apt-get install -y make gcc unzip bubblewrap libpcre2-dev menhir

# Create and activate the Conda environment
conda env create -f environment.yml
source activate ec3

# upgrade opam
opam init
opam update --confirm-level=unsafe-yes
opam switch create myswitch ocaml-variants.5.0.0+options ocaml-option-flambda
# flambda speeds up executution, at the cost of longer compilation.
eval $(opam env --switch=myswitch)
opam update --confirm-level=unsafe-yes

# need to install libtorch
wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip
mv libtorch-cxx11-abi-shared-with-deps-1.13.1+cu116.zip ~
unzip ~/libtorch-cxx11-abi-shared-with-deps-1.13.1+cu116.zip
mv libtorch ~
export LIBTORCH=~/libtorch

opam install --confirm-level=unsafe-yes vg cairo2 vector lwt logs torch domainslib ocamlgraph psq ctypes utop
eval $(opam env --switch=myswitch)
dune build

# need to install MNIST data
mkdir ../otorch-test/data/
cd ../otorch-test/data/
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

# for accessing remotely:  (e.g.)
# sshfs -o allow_other,default_permissions ubuntu@104.171.203.63:/home/ubuntu/cortex/ /home/tlh24/remote/
