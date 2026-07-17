#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
main="$repo_root/rtmscore_pytorch/src/main.py"

# input is protein (needs to be converted to pocket)
python "$main" -p "$repo_root/example/1qkt_p.pdb" -l "$repo_root/example/1qkt_decoys.sdf" -rl "$repo_root/example/1qkt_l.sdf" -gen_pocket -c 10.0

# input is pocket
python "$main" -p "$repo_root/example/1qkt_p_pocket_10.0.pdb" -l "$repo_root/example/1qkt_decoys.sdf"


# calculate the atom contributions of the score
python "$main" -p "$repo_root/example/1qkt_p_pocket_10.0.pdb" -l "$repo_root/example/1qkt_decoys.sdf" -ac


# calculate the residue contributions of the score
python "$main" -p "$repo_root/example/1qkt_p_pocket_10.0.pdb" -l "$repo_root/example/1qkt_decoys.sdf" -rc
