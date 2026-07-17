Scripts supporting the pure-PyTorch model and the ONNX/C++ export pipeline:

- `export_onnx.py` — exports the trained PyTorch checkpoint to
  `trained_models/rtmscore.onnx` (general, any batch size) and
  `trained_models/rtmscore_single.onnx` (fast path, batch_size=1 only).
- `verify_port.py` — validates the pure-PyTorch port against the mathematical
  DGL reference semantics (attention layer, dense batching, end-to-end batch
  invariance with the real trained checkpoint).
- `verify_onnx_dataset.py` — validates the exported ONNX model against eager
  PyTorch on real data, including batch>1 invariance.
- `dump_inputs.py` — featurizes protein/ligand complexes in Python and dumps
  the ONNX input tensors to disk as bundles the C++ binary can load directly
  (`./rtmscore_onnx/build/interaction trained_models/rtmscore.onnx rtmscore_onnx/fixtures/<bundle_id>`),
  bypassing the C++ RDKit featurization for validation purposes.
- `benchmark_inference.py` — benchmarks the pure-PyTorch model's inference
  latency across batch sizes
