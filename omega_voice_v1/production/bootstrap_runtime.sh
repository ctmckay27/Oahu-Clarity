#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/runtime"
if [ ! -d "$HERE/runtime/qwen3-tts/.git" ]; then
  git clone https://github.com/gabriele-mastrapasqua/qwen3-tts.git "$HERE/runtime/qwen3-tts"
fi
git -C "$HERE/runtime/qwen3-tts" fetch origin e391ec5467b0218eeb175f4888ad65b259d1e7c7
git -C "$HERE/runtime/qwen3-tts" checkout --detach e391ec5467b0218eeb175f4888ad65b259d1e7c7
make -C "$HERE/runtime/qwen3-tts" blas SIMD=portable
cd "$HERE/runtime/qwen3-tts"
sed 's|/resolve/main|/resolve/0c0e3051f131929182e2c023b9537f8b1c68adfe|g' download_model.sh > pinned_cv_download.sh
bash pinned_cv_download.sh --model large --dir "$HERE/runtime/qwen3-tts-model"
echo "Omega Voice / Mari Voice v1 runtime ready."