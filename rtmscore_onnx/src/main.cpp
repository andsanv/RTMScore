// RTMScore ONNX-Runtime driver.
//
// Three modes:
//
//   (a) Load a pre-featurized bundle (Python scripts/dump_inputs.py) and score:
//         ./interaction <model.onnx> <bundle_dir>
//
//   (b) Featurize a protein pocket + ligand poses in C++ (RDKit) and score:
//         ./interaction <model.onnx> --featurize <pocket.pdb> <ligs.sdf>
//                       [--pose N] [--cutoff C] [--dump <out_dir>]
//       Scores every pose in the SDF (or just pose N). --dump writes each pose's
//       8 input tensors as a bundle (manifest.txt + <name>.bin), byte-comparable
//       against the Python golden data in rtmscore_onnx/fixtures for validation.
//
//   (c) Generate a pocket from a full protein + reference ligand, then score:
//         ./interaction <model.onnx> --protein <protein.pdb>
//                       --reflig <reference.sdf> --ligands <poses.sdf>
//                       [--pocket-cutoff C] [--graph-cutoff C]
//                       [--pose N] [--dump <out_dir>]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#elif __has_include(<core/session/onnxruntime_cxx_api.h>)
// conda-forge's onnxruntime-cpp target exposes <prefix>/include/onnxruntime,
// while storing public session headers one directory below it.
#include <core/session/onnxruntime_cxx_api.h>
#else
#error "ONNX Runtime C++ headers were not found"
#endif

#include <GraphMol/GraphMol.h>
#include <RDGeneral/types.h>

#include "batch.hpp"
#include "featurize.hpp"
#include "main.hpp"
#include "pocket.hpp"

namespace {

struct RuntimeConfig {
  std::string device = "cpu";
  int cuda_device_id = 0;
  // Pinned to 1 by default so CPU runs stay deterministic. Can be changed with --threads N (N = 0 ORT auto-detect).
  int intra_op_threads = 1;
  // Non-empty enables ORT's built-in per-node profiler. ORT appends "_<pid>_<timestamp>.json" to this prefix
  std::string profile_prefix;
};

std::string available_providers_text() {
  const std::vector<std::string> providers = Ort::GetAvailableProviders();
  std::ostringstream out;
  for (std::size_t i = 0; i < providers.size(); ++i) {
    if (i) out << ", ";
    out << providers[i];
  }
  return out.str();
}

void configure_execution_provider(Ort::SessionOptions &options,
                                  const RuntimeConfig &runtime) {
  if (runtime.device == "cpu") {
    std::cerr << "Execution provider: CPUExecutionProvider" << std::endl;
    return;  // nothing more to configure, ORT defaults to CPU
  }
  if (runtime.device != "cuda")
    throw std::runtime_error("--device must be 'cpu' or 'cuda'");
  if (runtime.cuda_device_id < 0)
    throw std::runtime_error("--cuda-device must be non-negative");

  const std::vector<std::string> providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") ==
      providers.end()) {
    throw std::runtime_error(  // this ORT build was compiled without CUDA support
        "CUDAExecutionProvider is not available in this ONNX Runtime build; "
        "available providers: " + available_providers_text());
  }

  // Appending CUDA first gives supported nodes to CUDA while having ORT's normal CPU fallback.
  Ort::CUDAProviderOptions cuda_options;
  cuda_options.Update(
      {{"device_id", std::to_string(runtime.cuda_device_id)}});
  options.AppendExecutionProvider_CUDA_V2(*cuda_options);
  std::cerr << "Execution provider: CUDAExecutionProvider (device "
            << runtime.cuda_device_id << ")" << std::endl;
}

// Ends ORT profiling (if it was enabled) and prints the trace file path.
void finish_profiling(Ort::Session &session, const RuntimeConfig &runtime) {
  if (runtime.profile_prefix.empty()) return;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr path = session.EndProfilingAllocated(allocator);
  std::cerr << "ONNX Runtime profile written to: " << path.get() << std::endl;
}

std::vector<char> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);  // "ate" seeks to the end so tellg gives the file size
  if (!f) throw std::runtime_error("cannot open " + path);
  const std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);  // rewind before actually reading
  std::vector<char> buf(static_cast<size_t>(n));
  if (n > 0 && !f.read(buf.data(), n))
    throw std::runtime_error("cannot read " + path);
  return buf;
}

// Names the given session actually declares as inputs. 
// Needed as ONNX optimizes batch size == 1. Ask the loaded session which inputs it requires.
std::vector<std::string> declared_input_names(Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> names;
  const size_t n = session.GetInputCount();
  names.reserve(n);
  for (size_t i = 0; i < n; ++i)
    names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
  return names;
}

// Run the model on the batched input tensors.
std::vector<double> run_model_batch(Ort::Session &session, Ort::MemoryInfo &mem,
                                    std::vector<Tensor> &tensors) {
  const std::vector<std::string> declared = declared_input_names(session);
  std::vector<const char *> input_names;
  std::vector<Ort::Value> input_values;
  input_names.reserve(tensors.size());
  input_values.reserve(tensors.size());
  for (Tensor &t : tensors) {
    if (std::find(declared.begin(), declared.end(), t.name) == declared.end())
      continue;  // this session's graph doesn't need this tensor, skip it
    input_names.push_back(t.name.c_str());
    if (t.dtype == "f32") {
      input_values.push_back(Ort::Value::CreateTensor<float>(
          mem, t.f32.data(), t.f32.size(), t.shape.data(), t.shape.size()));  // wraps the existing buffer, no copy
    } else {
      input_values.push_back(Ort::Value::CreateTensor<int64_t>(
          mem, t.i64.data(), t.i64.size(), t.shape.data(), t.shape.size()));
    }
  }
  const char *output_names[] = {"score"};
  std::vector<Ort::Value> outputs = session.Run(
      Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
      input_values.size(), output_names, 1);  // forward pass
  const double *data = outputs[0].GetTensorData<double>();
  const std::vector<int64_t> shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  const size_t n = shape.empty() ? 1 : static_cast<size_t>(shape[0]);  // one score per pose in the batch
  return std::vector<double>(data, data + n);
}

// Write a bundle (manifest.txt + <name>.bin + pose_ids.txt) in dump_inputs.py's format.
void write_bundle(const std::string &dir, const std::vector<Tensor> &tensors,
                  const std::vector<std::string> &pose_ids) {
  std::string mkdir = "mkdir -p '" + dir + "'";
  if (std::system(mkdir.c_str()) != 0)
    throw std::runtime_error("cannot create " + dir);
  std::ofstream manifest(dir + "/manifest.txt");
  if (!manifest) throw std::runtime_error("cannot write " + dir + "/manifest.txt");
  
  for (const Tensor &t : tensors) {
    manifest << t.name << " " << t.dtype << " " << t.shape.size();  // one line per tensor: name, dtype, rank
    for (int64_t d : t.shape) manifest << " " << d;  // followed by each dimension
    manifest << "\n";
    std::ofstream bin(dir + "/" + t.name + ".bin", std::ios::binary);
    if (t.dtype == "f32")
      bin.write(reinterpret_cast<const char *>(t.f32.data()),
                static_cast<std::streamsize>(t.f32.size() * sizeof(float)));
    else
      bin.write(reinterpret_cast<const char *>(t.i64.data()),
                static_cast<std::streamsize>(t.i64.size() * sizeof(int64_t)));
  }
  if (!pose_ids.empty()) {
    std::ofstream ids(dir + "/pose_ids.txt");
    for (const std::string &id : pose_ids) ids << id << "\n";
  }
}

}  // namespace

std::vector<Tensor> load_bundle(const std::string &dir) {
  std::ifstream manifest(dir + "/manifest.txt");
  if (!manifest) throw std::runtime_error("cannot open " + dir + "/manifest.txt");

  std::vector<Tensor> tensors;
  std::string line;
  while (std::getline(manifest, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    Tensor t;
    int ndim = 0;
    ss >> t.name >> t.dtype >> ndim;  // write_bundle() format
    for (int i = 0; i < ndim; ++i) {
      int64_t d = 0;
      ss >> d;
      t.shape.push_back(d);
    }

    const std::vector<char> raw = read_file(dir + "/" + t.name + ".bin");
    const int64_t n = t.numel();
    if (t.dtype == "f32") {
      if (raw.size() != static_cast<size_t>(n) * sizeof(float))
        throw std::runtime_error("size mismatch for " + t.name);
      t.f32.resize(static_cast<size_t>(n));
      std::memcpy(t.f32.data(), raw.data(), raw.size());  // reinterpret the raw bytes as float32
    } else if (t.dtype == "i64") {
      if (raw.size() != static_cast<size_t>(n) * sizeof(int64_t))
        throw std::runtime_error("size mismatch for " + t.name);
      t.i64.resize(static_cast<size_t>(n));
      std::memcpy(t.i64.data(), raw.data(), raw.size());
    } else {
      throw std::runtime_error("unknown dtype '" + t.dtype + "' for " + t.name);
    }
    tensors.push_back(std::move(t));
  }
  return tensors;
}

namespace {

// Execution path when bundle is used as input.
int run_bundle_mode(const std::string &model_path, const std::string &bundle_dir,
                    const RuntimeConfig &runtime) {
  std::vector<Tensor> tensors = load_bundle(bundle_dir);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rtmscore");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(runtime.intra_op_threads);
  configure_execution_provider(opts, runtime);
  if (!runtime.profile_prefix.empty())
    opts.EnableProfiling(runtime.profile_prefix.c_str());
  Ort::Session session(env, model_path.c_str(), opts);
  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const std::vector<double> scores = run_model_batch(session, mem, tensors);  // inference

  std::vector<std::string> pose_ids;
  std::ifstream pose_ids_file(bundle_dir + "/pose_ids.txt");
  if (pose_ids_file) {  // optional, only present if the bundle was written with names attached
    std::string line;
    while (std::getline(pose_ids_file, line))
      if (!line.empty()) pose_ids.push_back(line);
  }
  for (size_t i = 0; i < scores.size(); ++i) {
    std::cout << "[" << i << "] ";
    if (i < pose_ids.size()) std::cout << pose_ids[i] << "   ";
    std::cout << "RTMScore = " << scores[i] << std::endl;
  }
  finish_profiling(session, runtime);

  std::ifstream exp(bundle_dir + "/expected.txt");
  if (exp) {  // optional reference values from dgl implementation, used by the validation scripts
    std::vector<double> expected;
    double v = 0.0;
    while (exp >> v) expected.push_back(v);
    if (expected.size() != scores.size()) {
      std::cerr << "expected.txt has " << expected.size()
                << " score(s) but the model produced " << scores.size()
                << std::endl;
      return EXIT_FAILURE;
    }
    bool any_mismatch = false;
    for (size_t i = 0; i < scores.size(); ++i) {
      const double diff = std::abs(scores[i] - expected[i]);
      const bool ok = diff < 1e-3;  // loose tolerance, only meant to catch real regressions
      if (!ok) any_mismatch = true;
      std::cout << "  [" << i << "] expected = " << expected[i]
                << "   |diff| = " << diff << (ok ? "   [OK]" : "   [MISMATCH]")
                << std::endl;
    }
    if (any_mismatch) return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

// Prints per-batch inference latency (excluding featurization) and a summary.
void report_timing(const std::vector<double> &ms) {
  if (ms.empty()) return;
  std::vector<double> steady(ms.begin() + (ms.size() > 1 ? 1 : 0), ms.end());  // drop the first (warm-up) call, if there's more than one
  std::sort(steady.begin(), steady.end());
  const double sum = std::accumulate(steady.begin(), steady.end(), 0.0);
  const double mean = sum / static_cast<double>(steady.size());
  const double median = steady[steady.size() / 2];
  std::cerr << "\nInference timing (ms), n=" << ms.size()
            << (ms.size() > 1 ? " (first call reported separately as warm-up)"
                               : "")
            << std::endl;
  if (ms.size() > 1) std::cerr << "  first (warm-up) = " << ms.front() << std::endl;
  std::cerr << "  mean   = " << mean << std::endl;
  std::cerr << "  median = " << median << std::endl;
  std::cerr << "  min    = " << steady.front() << std::endl;
  std::cerr << "  max    = " << steady.back() << std::endl;
}

int score_poses(const std::string &model_path, const RDKit::ROMol &pocket,
                const std::string &sdf, int pose, double graph_cutoff,
                const std::string &dump_dir, const RuntimeConfig &runtime,
                bool benchmark, int batch_size,
                const std::string &single_model_path) {
  // Protein features depend only on the pocket and graph cutoff, so compute them once and reuse them for every ligand pose in the input SDF.
  std::vector<Tensor> prot_tensors_single =
      featurize_protein(pocket, graph_cutoff);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rtmscore");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(runtime.intra_op_threads);
  configure_execution_provider(opts, runtime);
  if (!runtime.profile_prefix.empty())
    opts.EnableProfiling(runtime.profile_prefix.c_str());
  Ort::Session session(env, model_path.c_str(), opts);

  // Optional second session for single-pose (batch_size==1) calls.
  // Since batching has overhead, this session improves latency on single batch calls.
  std::unique_ptr<Ort::Session> single_session;
  if (!single_model_path.empty())
    single_session = std::make_unique<Ort::Session>(env, single_model_path.c_str(), opts);

  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const std::vector<std::string> blocks = split_sdf(sdf);
  if (blocks.empty()) throw std::runtime_error("no ligand poses in " + sdf);

  // Indices of poses to process, respecting --pose N (single-pose filter).
  std::vector<int> indices;
  for (int i = 0; i < static_cast<int>(blocks.size()); ++i)
    if (pose < 0 || i == pose) indices.push_back(i);  // pose < 0 means "score every pose"

  std::vector<double> batch_ms_all; // run time for benchmarking

  for (size_t start = 0; start < indices.size(); start += static_cast<size_t>(batch_size)) {
    const size_t end = std::min(start + static_cast<size_t>(batch_size), indices.size());

    std::vector<std::vector<Tensor>> lig_batch, prot_batch;
    std::vector<std::string> pose_ids;
    // N_l,E_l,N_p,E_p per pose, so --benchmark can report per-pose graph sizes even though poses now run together in one session.Run() call
    std::vector<std::array<int64_t, 4>> sizes;
    for (size_t k = start; k < end; ++k) {
      const int i = indices[k];
      std::unique_ptr<RDKit::ROMol> lig;
      try {
        lig = load_ligand_block(blocks[static_cast<size_t>(i)]);
      } catch (const std::exception &e) {
        std::cerr << "  [" << i << "] skipped: " << e.what() << std::endl;
        continue;  // one bad pose shouldn't stop the whole batch
      }
      std::string name;
      lig->getPropIfPresent(RDKit::common_properties::_Name, name);
      pose_ids.push_back((name.empty() ? "lig" : name) + "-" + std::to_string(i));

      std::vector<Tensor> lig_tensors = featurize_ligand(*lig);
      sizes.push_back({lig_tensors[0].shape[0], lig_tensors[1].shape[0],
                       prot_tensors_single[0].shape[0], prot_tensors_single[1].shape[0]});
      lig_batch.push_back(std::move(lig_tensors));
      prot_batch.push_back(prot_tensors_single);  // same protein tensors repeated for every pose in the batch
    }
    if (lig_batch.empty()) continue;  // every pose in this chunk failed to parse

    std::vector<Tensor> l_merged = batch_ligand_tensors(lig_batch);
    std::vector<Tensor> p_merged = batch_protein_tensors(prot_batch);
    std::vector<Tensor> tensors = l_merged;
    tensors.insert(tensors.end(), p_merged.begin(), p_merged.end());  // ligand tensors first, then protein, matching ONNX input order

    const bool use_single = single_session && lig_batch.size() == 1;  // route lone poses to the cheaper batch=1 model (it's also fine to always build the full 10-tensor set)

    std::vector<double> scores;
    double batch_ms = 0.0;  // run time for the single batch
    if (benchmark) {
      const auto t0 = std::chrono::steady_clock::now();
      scores = run_model_batch(use_single ? *single_session : session, mem, tensors);
      const auto t1 = std::chrono::steady_clock::now();
      batch_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();  // wall time for just this inference call
      batch_ms_all.push_back(batch_ms);
    } else {
      scores = run_model_batch(use_single ? *single_session : session, mem, tensors);
    }

    for (size_t k = 0; k < scores.size(); ++k) {
      std::cout << "  [" << indices[start + k] << "] " << pose_ids[k]
                << "   RTMScore = " << scores[k];
      if (benchmark) {
        std::cout << "   N_l=" << sizes[k][0] << " E_l=" << sizes[k][1]
                  << " N_p=" << sizes[k][2] << " E_p=" << sizes[k][3]
                  << "   batch_size=" << scores.size()
                  << "   inference_ms(batch)=" << batch_ms
                  << "   inference_ms(avg/pose)="
                  << batch_ms / static_cast<double>(scores.size());
      }
      std::cout << std::endl;
    }

    if (!dump_dir.empty()) {
      const std::string out_dir =
          batch_size == 1
              ? dump_dir + "/" + pose_ids[0]  // one folder per pose, named after it
              : dump_dir + "/batch" + std::to_string(batch_size) + "_" +
                    std::to_string(start);  // one folder per batch chunk
      write_bundle(out_dir, tensors, pose_ids);
    }
  }
  if (benchmark) report_timing(batch_ms_all);
  finish_profiling(session, runtime);
  if (single_session) finish_profiling(*single_session, runtime);
  return EXIT_SUCCESS;
}

int run_featurize_mode(const std::string &model_path, const std::string &pdb,
                       const std::string &sdf, int pose, double graph_cutoff,
                       const std::string &dump_dir,
                       const RuntimeConfig &runtime, bool benchmark,
                       int batch_size, const std::string &single_model_path) {
  std::unique_ptr<RDKit::ROMol> pocket = load_protein_pdb(pdb);
  return score_poses(model_path, *pocket, sdf, pose, graph_cutoff, dump_dir,
                     runtime, benchmark, batch_size, single_model_path);
}

int run_full_protein_mode(const std::string &model_path,
                          const std::string &protein_path,
                          const std::string &reference_ligand_path,
                          const std::string &ligands_path, int pose,
                          double pocket_cutoff, double graph_cutoff,
                          const std::string &dump_dir,
                          const RuntimeConfig &runtime, bool benchmark,
                          int batch_size, const std::string &single_model_path) {
  std::unique_ptr<RDKit::ROMol> protein =
      load_protein_pdb(protein_path, false);  // "false" as hydrogens needed for the cutoff distance check
  std::unique_ptr<RDKit::ROMol> reference_ligand =
      load_reference_ligand_sdf(reference_ligand_path);
  PocketResult pocket =
      generate_pocket(*protein, *reference_ligand, pocket_cutoff);

  std::cout << "Generated pocket: " << pocket.residue_count << " residues, "
            << pocket.atom_count << " heavy atoms (cutoff " << pocket_cutoff
            << " A)" << std::endl;
  return score_poses(model_path, *pocket.molecule, ligands_path, pose,
                     graph_cutoff, dump_dir, runtime, benchmark, batch_size,
                     single_model_path);
}

void print_usage(const char *program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " --list-providers\n"
      << "  " << program
      << " <model.onnx> <bundle_dir> [--device cpu|cuda]"
         " [--cuda-device N] [--profile <file_prefix>]\n"
      << "  " << program
      << " <model.onnx> --featurize <pocket.pdb> <ligs.sdf>"
         " [--pose N] [--cutoff C] [--dump <dir>] [--benchmark]"
         " [--batch-size N] [--single-model <fast_path.onnx>]"
         " [--device cpu|cuda] [--cuda-device N]"
         " [--threads N] [--profile <file_prefix>]\n"
      << "  " << program
      << " <model.onnx> --protein <protein.pdb> --reflig <reference.sdf>"
         " --ligands <poses.sdf> [--pocket-cutoff C] [--graph-cutoff C]"
         " [--cutoff C] [--pose N] [--dump <dir>] [--benchmark]"
         " [--batch-size N] [--single-model <fast_path.onnx>]"
         " [--device cpu|cuda] [--cuda-device N]"
         " [--threads N] [--profile <file_prefix>]\n";
}

}  // namespace

int main(int argc, char *argv[]) {
  try {
    if (argc == 2 && std::strcmp(argv[1], "--list-providers") == 0) {
      std::cout << available_providers_text() << std::endl;
      return EXIT_SUCCESS;
    }

    if (argc >= 3 && std::strcmp(argv[2], "--featurize") == 0) {  // mode (b): featurize an already-extracted pocket
      if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> --featurize <pocket.pdb> <ligs.sdf>"
                     " [--pose N] [--cutoff C] [--dump <dir>]\n";
        return EXIT_FAILURE;
      }
      const std::string model_path = argv[1];
      const std::string pdb = argv[3];
      const std::string sdf = argv[4];
      int pose = -1;          // -1 => all poses
      double cutoff = 10.0;
      std::string dump_dir;
      RuntimeConfig runtime;
      bool benchmark = false;
      int batch_size = 1;
      std::string single_model_path;
      for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pose" && i + 1 < argc) pose = std::atoi(argv[++i]);
        else if (a == "--cutoff" && i + 1 < argc) cutoff = std::atof(argv[++i]);
        else if (a == "--dump" && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--device" && i + 1 < argc) runtime.device = argv[++i];
        else if (a == "--cuda-device" && i + 1 < argc)
          runtime.cuda_device_id = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)
          runtime.intra_op_threads = std::atoi(argv[++i]);
        else if (a == "--profile" && i + 1 < argc)
          runtime.profile_prefix = argv[++i];
        else if (a == "--batch-size" && i + 1 < argc)
          batch_size = std::atoi(argv[++i]);
        else if (a == "--single-model" && i + 1 < argc)
          single_model_path = argv[++i];
        else if (a == "--benchmark") benchmark = true;
        else { std::cerr << "unknown arg: " << a << "\n"; return EXIT_FAILURE; }
      }
      if (batch_size < 1) {
        std::cerr << "--batch-size must be >= 1\n";
        return EXIT_FAILURE;
      }
      return run_featurize_mode(model_path, pdb, sdf, pose, cutoff, dump_dir,
                                runtime, benchmark, batch_size, single_model_path);
    }

    if (argc >= 3 && std::strcmp(argv[2], "--protein") == 0) {  // mode (c): generate the pocket, then featurize and score
      const std::string model_path = argv[1];
      std::string protein_path;
      std::string reference_ligand_path;
      std::string ligands_path;
      std::string dump_dir;
      int pose = -1;
      double pocket_cutoff = 10.0;
      double graph_cutoff = 10.0;
      RuntimeConfig runtime;
      bool benchmark = false;
      int batch_size = 1;
      std::string single_model_path;

      // Named arguments make the distinction between the reference ligand
      // (pocket definition) and scored ligand poses explicit.
      for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--protein" && i + 1 < argc) protein_path = argv[++i];
        else if (a == "--reflig" && i + 1 < argc)
          reference_ligand_path = argv[++i];
        else if (a == "--ligands" && i + 1 < argc) ligands_path = argv[++i];
        else if (a == "--pocket-cutoff" && i + 1 < argc)
          pocket_cutoff = std::atof(argv[++i]);
        else if (a == "--graph-cutoff" && i + 1 < argc)
          graph_cutoff = std::atof(argv[++i]);
        else if (a == "--cutoff" && i + 1 < argc) {
          // Compatibility shorthand matching Python's single -c option.
          pocket_cutoff = graph_cutoff = std::atof(argv[++i]);
        } else if (a == "--pose" && i + 1 < argc) {
          pose = std::atoi(argv[++i]);
        } else if (a == "--dump" && i + 1 < argc) {
          dump_dir = argv[++i];
        } else if (a == "--device" && i + 1 < argc) {
          runtime.device = argv[++i];
        } else if (a == "--cuda-device" && i + 1 < argc) {
          runtime.cuda_device_id = std::atoi(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
          runtime.intra_op_threads = std::atoi(argv[++i]);
        } else if (a == "--profile" && i + 1 < argc) {
          runtime.profile_prefix = argv[++i];
        } else if (a == "--batch-size" && i + 1 < argc) {
          batch_size = std::atoi(argv[++i]);
        } else if (a == "--single-model" && i + 1 < argc) {
          single_model_path = argv[++i];
        } else if (a == "--benchmark") {
          benchmark = true;
        } else {
          std::cerr << "unknown or incomplete arg: " << a << "\n";
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
      }
      if (protein_path.empty() || reference_ligand_path.empty() ||
          ligands_path.empty()) {
        std::cerr << "--protein, --reflig, and --ligands are required\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      if (batch_size < 1) {
        std::cerr << "--batch-size must be >= 1\n";
        return EXIT_FAILURE;
      }
      return run_full_protein_mode(
          model_path, protein_path, reference_ligand_path, ligands_path, pose,
          pocket_cutoff, graph_cutoff, dump_dir, runtime, benchmark, batch_size,
          single_model_path);
    }

    if (argc >= 3) {  // mode (a): score a bundle of tensors already dumped by dump_inputs.py
      RuntimeConfig runtime;
      for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) runtime.device = argv[++i];
        else if (a == "--cuda-device" && i + 1 < argc)
          runtime.cuda_device_id = std::atoi(argv[++i]);
        else if (a == "--profile" && i + 1 < argc)
          runtime.profile_prefix = argv[++i];
        else {
          std::cerr << "unknown or incomplete arg: " << a << "\n";
          return EXIT_FAILURE;
        }
      }
      return run_bundle_mode(argv[1], argv[2], runtime);
    }

    print_usage(argv[0]);
    return EXIT_FAILURE;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
