#!/usr/bin/env bash
set -euo pipefail

device="${1:-cpu}"
if [[ "$device" != "cpu" && "$device" != "cuda" ]]; then
  echo "usage: $0 [cpu|cuda]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${RTMSCORE_BUILD_DIR:-$repo_root/rtmscore_onnx/build}"

cmake_args=(
  -S "$repo_root/rtmscore_onnx"
  -B "$build_dir"
  -DCMAKE_BUILD_TYPE=Release
)
if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
  cmake_args+=(-G Ninja)
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  cmake_args+=("-DCMAKE_PREFIX_PATH=$CONDA_PREFIX")

  onnx_config="$(find "$CONDA_PREFIX" -name onnxruntimeConfig.cmake -print -quit)"
  rdkit_config="$(find "$CONDA_PREFIX" -name rdkit-config.cmake -print -quit)"
  if [[ -z "$onnx_config" || -z "$rdkit_config" ]]; then
    echo "The active Mamba environment lacks ONNX Runtime or RDKit C++ CMake metadata." >&2
    echo "Create it from rtmscore_onnx/environment-cpu.yml or rtmscore_onnx/environment-cuda.yml." >&2
    exit 1
  fi
  cmake_args+=("-Donnxruntime_DIR=$(dirname "$onnx_config")")
  cmake_args+=("-Drdkit_DIR=$(dirname "$rdkit_config")")
fi

cmake "${cmake_args[@]}"
cmake --build "$build_dir" --parallel

echo "Available ONNX Runtime providers:"
"$build_dir/interaction" --list-providers

"$build_dir/interaction" \
  "$repo_root/trained_models/rtmscore.onnx" \
  --protein "$repo_root/example/1qkt_p.pdb" \
  --reflig "$repo_root/example/1qkt_l.sdf" \
  --ligands "$repo_root/example/1qkt_decoys.sdf" \
  --cutoff 10 \
  --pose 0 \
  --device "$device" \
  --cuda-device "${CUDA_DEVICE:-0}"
