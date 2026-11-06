#!/bin/bash
# NIMBLE's SGLang integration: clone the public SGLang, check out the commit the patches are made for (BASE) and apply
# them. The serving patches add the nimble and nccl_a2a MoE all-to-all backends, which import NIMBLE's dispatcher from
# the nimble package (nimble.integrations.sglang).
#
#   apply.sh DEST      DEST: a new directory
#
# The clone carries SGLang's history and tags (its version is computed from them) but downloads file contents only for
# BASE, and writes them 16 files at a time. Then install DEST's SGLang as usual, for example
# SGLANG_BUILD_RUST_EXTS=none pip install -e DEST/python (no Rust needed).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:?usage: apply.sh DEST}
[ $# = 1 ] || { echo "usage: apply.sh DEST" >&2; exit 2; }
BASE=$(head -1 "$HERE/BASE")
echo "apply.sh: SGLang ${BASE:0:9} with NIMBLE's patches, into $DEST"
echo "[1/3] cloning SGLang's history from GitHub (file contents come in step 2)"
git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang "$DEST"
echo "[2/3] checking out ${BASE:0:9}: downloading and writing its files"
git -C "$DEST" -c advice.detachedHead=false -c checkout.workers=16 checkout "$BASE"
echo "[3/3] applying NIMBLE's patches"
for p in "$HERE"/serving/*.patch; do git -C "$DEST" apply "$p"; echo "      $(basename "$p")"; done
echo "apply.sh: SGLang ${BASE:0:9} with NIMBLE's patches in $DEST ($((SECONDS / 60))m $((SECONDS % 60))s)"
echo "next: install it as usual, for example SGLANG_BUILD_RUST_EXTS=none pip install -e $DEST/python"
