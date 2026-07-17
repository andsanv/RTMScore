#ifndef INTERACTION_MAIN_HDR
#define INTERACTION_MAIN_HDR

#include <cstdint>
#include <string>
#include <vector>

// One input tensor loaded from a bundle produced by scripts/dump_inputs.py.
struct Tensor {
  std::string name;             // ONNX input name, e.g. "l_ndata_atom"
  std::string dtype;            // "f32" or "i64"
  std::vector<int64_t> shape;   // row-major dimensions
  std::vector<float> f32;       // populated when dtype == "f32"
  std::vector<int64_t> i64;     // populated when dtype == "i64"

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;  // total element count, product of all dimensions
    return n;
  }
};

// Load a complex bundle (manifest.txt + <name>.bin files) from a directory.
std::vector<Tensor> load_bundle(const std::string &dir);

#endif  // INTERACTION_MAIN_HDR
