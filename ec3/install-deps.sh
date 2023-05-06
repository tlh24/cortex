# Install conda
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local && rm Miniconda3-latest-Linux-x86_64.sh && /usr/local/bin/conda init


# commands for getting lambda labs up and running
sudo chmod +rw /usr/bin
sudo apt-get update
sudo apt-get install -y make gcc unzip bubblewrap

# Create and activate the Conda environment
conda env create -f environment.yml
source activate ec3

# upgrade opam
opam init --disable-sandboxing --yes
opam update --confirm-level=unsafe-yes
cd eval $(opam env --switch=myswitch)
opam update --confirm-level=unsafe-yes

# need to install libtorch
wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip
mv libtorch-cxx11-abi-shared-with-deps-1.13.1+cu116.zip ~
unzip ~/libtorch-cxx11-abi-shared-with-deps-1.13.1+cu116.zip
mv libtorch ~
export LIBTORCH=~/libtorch

opam install --confirm-level=unsafe-yes vg cairo2 pbrt vector lwt logs pcre torch domainslib ocamlgraph
eval $(opam env --switch=myswitch)
dune build



# Get the path to the opam switch
OPAM_SWITCH_ROOT=$(opam var switch:root)

# Create the activate.d directory if it doesn't exist
ACTIVATE_DIR=$CONDA_PREFIX/etc/conda/activate.d
sudo mkdir -p $ACTIVATE_DIR

# Create the deactivate.d directory if it doesn't exist
DEACTIVATE_DIR=$CONDA_PREFIX/etc/conda/deactivate.d
sudo mkdir -p $DEACTIVATE_DIR

# Create the activate script
ACTIVATE_SCRIPT=$ACTIVATE_DIR/opam-activate.sh
echo "#!/bin/bash" > $ACTIVATE_SCRIPT
echo "source $OPAM_SWITCH_ROOT/activate" >> $ACTIVATE_SCRIPT
chmod +x $ACTIVATE_SCRIPT

# Create the deactivate script
DEACTIVATE_SCRIPT=$DEACTIVATE_DIR/opam-deactivate.sh
echo "#!/bin/bash" > $DEACTIVATE_SCRIPT
echo "source $OPAM_SWITCH_ROOT/deactivate" >> $DEACTIVATE_SCRIPT
chmod +x $DEACTIVATE_SCRIPT

echo "Opam switch '$OPAM_SWITCH_NAME' will now be automatically activated when you activate the '$CONDA_DEFAULT_ENV' conda environment."


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
