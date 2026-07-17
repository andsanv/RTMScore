# rtmscore_pytorch

A pure PyTorch port of RTMScore with all DGL dependencies removed. The scope is doing inference on a pre-trained model.

## Changes compared to the DGL model

The original model uses DGL to represent each protein/ligand as a graph and to run its message-passing and batching through DGL's own graph engine. This port keeps the same architecture and trained weights, but replaces every DGL util with plain PyTorch

For example, DGL graphs become Python dictionaries and batching is just a tensor concatenation operation.


The results behave like the original, with errors in the order of float32's precision.

## Installation

The packages required for this project are found in the 'requirements.txt' file.

They can be installed with:

```bash
pip install -r requirements.txt
```

or, in case a local environment is preferred, with:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

The OpenBabel package (used for pocket generation) needs a native install alongside the Python bindings. For Apple Silicon:

```bash
brew install open-babel swig
curl -L -o openbabel-3.1.1.1.tar.gz "<link from https://pypi.org/project/openbabel/#files>"
tar -xzvf openbabel-3.1.1.1.tar.gz && rm -f openbabel-3.1.1.1.tar.gz && cd openbabel-3.1.1.1
python setup.py build_ext -I/opt/homebrew/include/openbabel3 -L/opt/homebrew/lib && python setup.py install && cd ..
```

## Usage

Run the CLI entry point `src/main.py` from the repository root. It defaults to `trained_models/rtmscore.pth` and writes output to `rtmscore_pytorch/output/`, regardless of the working directory.

Example of usages:

```bash
# input is a full protein (pocket extracted first, needs a reference ligand + cutoff)
python rtmscore_pytorch/src/main.py -p example/1qkt_p.pdb -l example/1qkt_decoys.sdf -rl example/1qkt_l.sdf -gen_pocket -c 10.0

# input is an already-extracted pocket PDB
python rtmscore_pytorch/src/main.py -p example/1qkt_p_pocket_10.0.pdb -l example/1qkt_decoys.sdf

# atom-level score contributions
python rtmscore_pytorch/src/main.py -p example/1qkt_p_pocket_10.0.pdb -l example/1qkt_decoys.sdf -ac

# residue-level score contributions
python rtmscore_pytorch/src/main.py -p example/1qkt_p_pocket_10.0.pdb -l example/1qkt_decoys.sdf -rc
```

Alternatively, `example/run_with_pytorch.sh` runs all four modes above end to end.


## Arguments

The script can take several arguments in input, depending on the specific usage:

| flag | meaning |
|---|---|
| `-p/--prot` | input protein file (`.pdb`) |
| `-l/--lig` | input ligand file (`.sdf`/`.mol2`, multi-pose supported) |
| `-m/--model` | checkpoint path (default `trained_models/rtmscore.pth`) |
| `-o/--outprefix` | output file prefix (default `rtmscore_pytorch/output/out`) |
| `-gen_pocket` | generate the pocket from `-p`/`-rl`/`-c` instead of using a pre-extracted pocket |
| `-c/--cutoff` | pocket/interaction cutoff in Å (default 10.0) |
| `-rl/--reflig` | reference ligand used to define the pocket (required with `-gen_pocket`) |
| `-pl/--parallel` | featurize poses in parallel (large datasets may run out of memory with this on) |
| `-ac/--atom_contribution` | decompose the score at atom level (mutually exclusive with `-rc`) |
| `-rc/--res_contribution` | decompose the score at residue level (mutually exclusive with `-ac`) |

Some notes:
- parameters `-ac`/`-rc` are mutually exclusive and can be used once at a time, and without either one score per pose is produced;
- `-gen_pocket` needs `-rl` (and usually `-c`) to know which residues count as "the pocket".

