# Setup

`cd ec3`
`sh install-deps.sh` script to build the ocaml executable.

## python env setup

To get a virtual env using `pyenv` and `poetry`
`cd ec3`
`pyenv local 3.10`
`poetry install`

To use, run `poetry shell` to activate the virtual env or prepend `poetry run` to the python commands below.

# Use

`cd ec3`
Run the executable: `./run.sh -b 512 -g -p`
-b : batch size (change based on your gpu memory)
-g : (optional) debug logging
-p : parallel.  (turn it off when debugging )

Training: in a separate terminal, run `python ec33.py -b 512`
Batch size needs to be the same.

Dreaming: Once it writes out a model, you can start dreaming in yet another terminal:
python ec33.py -b 512 -d
where -d : dreaming