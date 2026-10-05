# Third-party sources

Dependencies and model binaries are downloaded into ignored directories. Preserve
their upstream license files and record immutable revisions and SHA-256 hashes.

- Proton NPU / Ara BSP: Apache-2.0; https://github.com/cerebralchips/proton-npu
- IREE compiler/runtime: Apache-2.0 with LLVM exceptions;
  https://github.com/iree-org/iree. [License](licenses/IREE.txt).
- llama2.c reference and tokenizer format: MIT;
  https://github.com/karpathy/llama2.c. [License](licenses/llama2.c-MIT.txt).
- Stories260K checkpoint/tokenizer: https://huggingface.co/karpathy/tinyllamas
  The pinned repository metadata declares MIT. The checkpoint readme, immutable
  revision and hashes are retained locally. This does not characterize the
  provenance or licensing of every item in the training dataset.
- flatcc: Apache-2.0; https://github.com/dvidelabs/flatcc. Used for IREE schemas.
- eyalroz/printf: MIT; https://github.com/eyalroz/printf. Used for integer/hex UART
  formatting and IREE status formatting. Upstream license files remain in the
  pinned source archives and extracted dependencies.

The runtime embedding follows IREE's static-library and custom-dispatch examples.
Decoder equations and checkpoint/tokenizer layouts follow llama2.c; the NumPy
reference, packing, Proton kernels, deployment and evidence tooling are local
implementations. The native upstream reference is built unchanged for comparison.
