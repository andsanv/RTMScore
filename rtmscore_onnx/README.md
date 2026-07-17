# rtmscore_onnx

A C++ implementation of the RTMScore inference pipeline. RDKit parses
the molecular files, generates the pocket, and builds the graph tensors; ONNX
Runtime executes the exported neural network. No Python or DGL runtime is
required.

The application supports CPU and NVIDIA CUDA inference, multiple ligand poses,
and batching. It can start from a full protein, an existing pocket, or
pre-featurized tensors.

## Features

This implementation replaces the Python runtime with:

- RDKit C++ for PDB/SDF parsing and stereochemistry;
- in-memory pocket generation from a full receptor and reference ligand;
- ligand atom-graph and protein residue-graph featurization;
- disjoint-union graph batching;
- ONNX Runtime CPU or CUDA inference.

The neural-network layers and trained parameters are stored in
`trained_models/rtmscore.onnx`.

## Installation

The build uses a separate Mamba/Conda environment from the Python
`.venv`. Do not activate both environments together.

Dependencies include a C++20 compiler, CMake, Ninja, RDKit, Boost development
libraries, and ONNX Runtime C++.

### CPU

```bash
mamba env create -f rtmscore_onnx/environment-cpu.yml
mamba activate rtmscore-cpp-cpu
```

### NVIDIA CUDA

The supplied CUDA environment targets Linux x86-64, ONNX Runtime 1.26, CUDA
12.9, and cuDNN 9. A compatible NVIDIA driver is required.

```bash
mamba env create -f rtmscore_onnx/environment-cuda.yml
mamba activate rtmscore-cpp-cuda
```

## Build

From the repository root:

```bash
cmake -S rtmscore_onnx -B rtmscore_onnx/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"

cmake --build rtmscore_onnx/build --parallel
```

Check the ONNX Runtime providers available in the current build:

```bash
./rtmscore_onnx/build/interaction --list-providers
```

The convenience script configures, builds, and smoke-tests the application:

```bash
bash example/run_with_onnx.sh cpu
# or
bash example/run_with_onnx.sh cuda
```

Set `RTMSCORE_BUILD_DIR=/path/to/build` to keep separate CPU and CUDA build
directories.

## Usage

The executable is `rtmscore_onnx/build/interaction`. It has three scoring
modes, depending on how much preprocessing has already been performed.

Run the examples from the repository root.

### Mode 1: full protein and reference ligand

Use this mode when the input is a complete receptor PDB. The reference ligand
defines the binding pocket, while the ligand SDF contains the pose or poses to
score.

```bash
./rtmscore_onnx/build/interaction trained_models/rtmscore.onnx \
  --protein example/1qkt_p.pdb \
  --reflig example/1qkt_l.sdf \
  --ligands example/1qkt_decoys.sdf \
  --single-model trained_models/rtmscore_single.onnx
```

The application:

1. loads the full protein and the first molecule in the reference-ligand SDF;
2. keeps every complete protein residue having any atom within the pocket
   cutoff of any reference-ligand atom;
3. removes protein hydrogens and builds the protein residue graph;
4. builds one ligand atom graph per scored SDF record;
5. runs ONNX inference and prints one score per pose.

The default pocket and graph cutoffs are both 10 Å. They can be set
independently:

```bash
./rtmscore_onnx/build/interaction trained_models/rtmscore.onnx \
  --protein example/1qkt_p.pdb \
  --reflig example/1qkt_l.sdf \
  --ligands example/1qkt_decoys.sdf \
  --pocket-cutoff 8.0 \
  --graph-cutoff 10.0 \
  --single-model trained_models/rtmscore_single.onnx
```

`--pocket-cutoff` controls which residues become graph nodes.
`--graph-cutoff` controls which selected residues are connected by graph edges.
`--cutoff C` sets both values to `C`.

The reference ligand currently must be an SDF. If it contains multiple
records, only the first record defines the pocket.

### Mode 2: existing pocket

Use this mode when the pocket PDB has already been extracted. The two values
after `--featurize` are the pocket PDB and ligand SDF.

```bash
./rtmscore_onnx/build/interaction trained_models/rtmscore.onnx \
  --featurize example/1qkt_p_pocket_10.0.pdb \
  example/1qkt_decoys.sdf \
  --single-model trained_models/rtmscore_single.onnx
```

Pocket selection is skipped. RDKit featurizes the supplied pocket and every
ligand record, then the application runs inference. In this mode,
`--cutoff C` controls the protein residue-edge cutoff only.

### Mode 3: pre-featurized tensor bundle

Use bundle mode to bypass RDKit and run tensors that have already been
generated. This mode is primarily for regression testing, cross-language
validation, and integrations that produce the ONNX inputs themselves.

```bash
./rtmscore_onnx/build/interaction \
  trained_models/rtmscore.onnx \
  rtmscore_onnx/fixtures/<bundle_id>
```

A bundle contains:

```text
manifest.txt
<tensor_name>.bin
pose_ids.txt        optional
expected.txt        optional
```

The manifest records each tensor's name, dtype, rank, and shape. If
`expected.txt` exists, the application checks the resulting scores against its
reference values.

Bundle mode accepts `--device`, `--cuda-device`, and `--profile`. Pose
selection, live batching, benchmarking, and dumping do not apply because the
bundle already contains the complete model inputs.

## Arguments

#### Mode and input arguments

| Argument | Meaning |
|---|---|
| `<model.onnx>` | General model; normally `trained_models/rtmscore.onnx`. |
| `--protein <protein.pdb>` | Full receptor used by mode 1. |
| `--reflig <reference.sdf>` | Reference ligand that defines the mode 1 pocket. |
| `--ligands <poses.sdf>` | Single- or multi-pose SDF scored in mode 1. |
| `--featurize <pocket.pdb> <poses.sdf>` | Select mode 2 and provide its inputs. |
| `<bundle_dir>` | Pre-featurized input directory for mode 3. |

#### Selection and graph arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--pocket-cutoff C` | `10.0` | Mode 1 residue-selection cutoff in Å. |
| `--graph-cutoff C` | `10.0` | Mode 1 residue-edge cutoff in Å. |
| `--cutoff C` | `10.0` | Set both mode 1 cutoffs, or the mode 2 graph cutoff. |
| `--pose N` | all | Score only zero-based SDF record `N`. |

#### Inference and diagnostics

| Argument | Default | Meaning |
|---|---:|---|
| `--device cpu\|cuda` | `cpu` | Select the ONNX Runtime execution provider. |
| `--cuda-device N` | `0` | Select the CUDA device. |
| `--threads N` | `1` | Set CPU intra-operation threads; `0` lets ONNX Runtime choose. |
| `--batch-size N` | `1` | Number of poses per inference call. |
| `--single-model <path>` | none | Use the optimized batch-one model for one-pose chunks. |
| `--benchmark` | off | Report inference-only timings and graph sizes. |
| `--profile <prefix>` | off | Write an ONNX Runtime Chrome-trace profile. |
| `--dump <directory>` | off | Write generated tensors as validation bundles. |
| `--list-providers` | — | Print the providers compiled into ONNX Runtime. |

`--threads`, `--batch-size`, `--single-model`, `--benchmark`, and `--dump`
apply to modes 1 and 2.

## Additional examples

### Score one pose

`--pose` is a zero-based SDF record index:

```bash
./rtmscore_onnx/build/interaction trained_models/rtmscore.onnx \
  --featurize example/1qkt_p_pocket_10.0.pdb \
  example/1qkt_decoys.sdf \
  --pose 2 \
  --single-model trained_models/rtmscore_single.onnx
```

### Run on CUDA

```bash
./rtmscore_onnx/build/interaction trained_models/rtmscore.onnx \
  --featurize example/1qkt_p_pocket_10.0.pdb \
  example/1qkt_decoys.sdf \
  --device cuda \
  --cuda-device 0 \
  --single-model trained_models/rtmscore_single.onnx
```

If CUDA is requested but `CUDAExecutionProvider` is unavailable, the
application reports the available providers and exits instead of silently
running the entire model on CPU.

### Batch poses on CUDA

```bash
./rtmscore_onnx/build/interaction trained_models/rtmscore.onnx \
  --featurize example/1qkt_p_pocket_10.0.pdb \
  example/1qkt_decoys.sdf \
  --device cuda \
  --batch-size 8 \
  --single-model trained_models/rtmscore_single.onnx
```

The single model is also useful with larger batches: a final chunk containing
one pose is routed to it automatically.

### Benchmark CPU inference

```bash
./rtmscore_onnx/build/interaction trained_models/rtmscore.onnx \
  --featurize example/1qkt_p_pocket_10.0.pdb \
  example/1qkt_decoys.sdf \
  --threads 8 \
  --benchmark \
  --single-model trained_models/rtmscore_single.onnx
```

Benchmark timing covers `session.Run()` only, excluding file loading,
pocket generation, and featurization. The first call is reported separately as
warm-up.

### Dump generated tensors

```bash
./rtmscore_onnx/build/interaction trained_models/rtmscore.onnx \
  --featurize example/1qkt_p_pocket_10.0.pdb \
  example/1qkt_decoys.sdf \
  --pose 0 \
  --dump /tmp/rtmscore_tensors \
  --single-model trained_models/rtmscore_single.onnx
```

## Batching information

Two models are provided from the same checkpoint:

- `rtmscore.onnx` supports dynamic batch sizes;
- `rtmscore_single.onnx` is optimized for batch size 1.

For normal single-pose execution, use:

```text
--batch-size 1 --single-model trained_models/rtmscore_single.onnx
```

Without `--single-model`, a one-pose call uses the general model's slower
padding and masking path.


## Model input

The general model accepts ten tensors:

| Tensor | Shape | Dtype |
|---|---|---|
| `l_ndata_atom` | `[N_l, 41]` | f32 |
| `l_edata_bond` | `[E_l, 10]` | f32 |
| `l_edge_index` | `[2, E_l]` | i64 |
| `l_ndata_pos` | `[N_l, 3]` | f32 |
| `l_batch_num_nodes` | `[B]` | i64 |
| `p_ndata_feats` | `[N_p, 41]` | f32 |
| `p_edata_feats` | `[E_p, 5]` | f32 |
| `p_edge_index` | `[2, E_p]` | i64 |
| `p_ndata_pos` | `[N_p, 24, 3]` | f32 |
| `p_batch_num_nodes` | `[B]` | i64 |

`N_l`, `N_p`, `E_l`, and `E_p` are totals across the batch. The output is
`score [B]` with dtype float64.

All floating-point inputs must be float32. Edge indices and batch node counts
must be int64. Node order must remain consistent across features, positions,
and edge indices. Protein residue coordinates are NaN-padded to 24 atoms.


## Regenerating the ONNX models

After changing the PyTorch model, regenerate both exports with:

```bash
.venv/bin/python scripts/export_onnx.py
```

This writes the general `rtmscore.onnx` and optimized
`rtmscore_single.onnx`, then cross-checks the exports.
