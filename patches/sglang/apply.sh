#!/bin/bash
# NIMBLE's SGLang integration: clone the public SGLang, check out the commit the patches are made for (BASE) and apply
# them. The serving patches add the nimble and nccl_a2a MoE all-to-all backends, which import NIMBLE's dispatcher from
# the nimble package (nimble.integrations.sglang). --harness also applies NIMBLE's CI harness: seeded prompts, the
# greedy tokens' dump and the per-layer timing hooks, with their timing instrument.
#
#   apply.sh DEST [--harness]      DEST: a new directory
#
# The clone carries SGLang's history and tags (its version is computed from them) but downloads file contents only for
# BASE, and writes them once, 16 files at a time (a parallel file system creates files faster side by side). Then
# install DEST's SGLang as usual, for example pip install -e DEST/python.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:?usage: apply.sh DEST [--harness]}
case "${2:-}" in "" | --harness) ;; *) echo "usage: apply.sh DEST [--harness]" >&2; exit 2 ;; esac
BASE=$(head -1 "$HERE/BASE")
N=3; [ "${2:-}" = --harness ] && N=4
echo "apply.sh: SGLang ${BASE:0:9} with NIMBLE's patches, into $DEST"
echo "[1/$N] cloning SGLang's history from GitHub (file contents come in step 2)"
git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang "$DEST"
echo "[2/$N] checking out ${BASE:0:9}: downloading and writing its files"
git -C "$DEST" -c advice.detachedHead=false -c checkout.workers=16 checkout "$BASE"
echo "[3/$N] applying NIMBLE's patches"
for p in "$HERE"/serving/*.patch; do git -C "$DEST" apply "$p"; echo "      $(basename "$p")"; done
if [ "${2:-}" = --harness ]; then
  echo "[4/$N] applying NIMBLE's CI harness"
  for p in "$HERE"/harness/*.patch; do git -C "$DEST" apply "$p"; echo "      $(basename "$p")"; done
  cp "$HERE/harness/a2a_timing.py" "$DEST/python/sglang/srt/layers/moe/token_dispatcher/"
fi
echo "apply.sh: SGLang ${BASE:0:9} with NIMBLE's patches in $DEST ($((SECONDS / 60))m $((SECONDS % 60))s)"
echo "next: install it as usual, for example pip install -e $DEST/python"
