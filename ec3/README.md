# Setup

## Prerequisites

Conda should be installed on your system. You can install Miniconda by running the following commands:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local
rm Miniconda3-latest-Linux-x86_64.sh
/usr/local/bin/conda init
```

## Run the setup script

This repository contains a setup script (`install-deps.sh`) to help you get started. The script performs the following tasks:

1. Installs system packages (make, gcc, unzip, and bubblewrap).
2. Creates and activates a Conda environment named `ec3` using the `environment.yml` file.
3. Installs and configures `opam` and OCaml with the flambda option.
4. Installs libtorch and sets the LIBTORCH environment variable.
5. Installs additional OCaml packages using `opam`.
6. Builds the project using `dune`.
7. Downloads and extracts the MNIST dataset.

To run the setup script, run the following commands:

```bash
cd ec3
./install-deps.sh
```

# Use

`cd ec3`
Run the executable: `./run.sh -b 512 -g -p`
-b : batch size (change based on your gpu memory)
-g : (optional) debug logging
-p : parallel.  (turn it off when debugging )

Training: in a separate terminal, run `python ec34.py -b 512`
Batch size needs to be the same.

Dreaming: Once it writes out a model, you can start dreaming in yet another terminal:
python ec34.py -b 512 -d
where -d : dreaming
