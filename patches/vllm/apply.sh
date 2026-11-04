#!/bin/bash
# NIMBLE's backends in vLLM -- `--all2all-backend nimble` (NIMBLE's MoE dispatcher over its AllToAllV) and `nccl_a2a`
# (the same dispatcher over NCCL's all-to-all, the baseline): patch a vLLM package (BASE's version only). The patch
# series (patches/*.patch, applied in order to vLLM's own files) adds the backends' names, the manager's construction,
# the prepare / finalize factory entry, the CUDA-graph switch and the TRT-LLM experts' admission; the classes come from
# the nimble package (nimble.integrations.vllm): put <NIMBLE>/python on PYTHONPATH.
#   apply.sh SITE                      patch the vllm/ package under SITE (e.g. <venv>/lib/python3.12/site-packages)
#   apply.sh SITE --from-wheel WHEEL   first unpack BASE's wheel (sha256-checked) into a new directory SITE: an overlay
#                                      to put ahead of a venv's site-packages on PYTHONPATH, leaving the venv untouched
#   apply.sh SITE --check              verify only (non-zero when the series is not applied)
# Never run it while a job uses the installation.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SP=${1:?usage: apply.sh SITE [--check | --from-wheel WHEEL]}
MODE=${2:-apply}
STEP=
if [ "$MODE" = "--from-wheel" ]; then
  WHEEL=${3:?usage: apply.sh SITE --from-wheel WHEEL}
  [ ! -e "$SP" ] || { echo "ABORT: $SP exists; --from-wheel makes a new directory"; exit 2; }
  echo "[1/3] checking $(basename "$WHEEL") against BASE's sha256"
  want_sha=$(sed -n 's/^sha256=//p' "$HERE/BASE")
  [ "$(sha256sum "$WHEEL" | cut -c1-64)" = "$want_sha" ] || { echo "ABORT: $WHEEL is not BASE's wheel"; exit 2; }
  echo "[2/3] unpacking it into $SP"
  mkdir -p "$SP"
  python3 - "$WHEEL" "$SP" <<'EOP'
# python3 -m zipfile -e, 16 files at a time (a parallel file system creates files faster side by side), with a
# progress bar on a terminal: files written, time spent, time left
import sys, threading, time, zipfile
from concurrent.futures import ThreadPoolExecutor
wheel, site = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel) as z:
    members = z.infolist()
local = threading.local()
def extract(m):
    if not hasattr(local, "z"):
        local.z = zipfile.ZipFile(wheel)
    try:
        local.z.extract(m, site)
    except FileExistsError:             # another writer made the same directory first
        local.z.extract(m, site)
bar, t0, n = sys.stdout.isatty(), time.monotonic(), len(members)
with ThreadPoolExecutor(16) as pool:
    for i, _ in enumerate(pool.map(extract, members), 1):
        if bar and (i % 25 == 0 or i == n):
            s = int(time.monotonic() - t0)
            left = s * (n - i) // i
            print(f"\r      [{'#' * (25 * i // n):<25}] {100 * i // n:3d}%  {i}/{n} files  "
                  f"{s // 60}m{s % 60:02d}s, ~{left // 60}m{left % 60:02d}s left ", end="", flush=True)
if bar:
    print()
EOP
  echo "unpacked $(basename "$WHEEL") into $SP"
  MODE=apply STEP="[3/3] "
fi
V=$SP/vllm
[ -d "$V" ] || { echo "ABORT: no vllm package under $SP"; exit 2; }
want=$(sed -n 's/^vllm==//p' "$HERE/BASE")
have=$(sed -n 's/^Version: //p' "$SP"/vllm-*.dist-info/METADATA 2>/dev/null | head -1)
[ "$have" = "$want" ] || { echo "ABORT: vLLM $have under $SP; the series is for $want (BASE)"; exit 2; }
applied() {  # every patch of the series applies in reverse: it is in
  local p
  for p in "$HERE"/patches/*.patch; do
    patch -d "$SP" -p1 -R --dry-run -s -f < "$p" > /dev/null 2>&1 || return 1
  done
}
if [ "$MODE" = "--check" ]; then
  applied || { echo "vLLM under $SP: the series is not applied"; exit 1; }
  echo "vLLM $have under $SP: NIMBLE's series is applied (the classes come from the nimble package)"
  exit 0
fi
[ "$MODE" = apply ] || { echo "apply.sh: unknown option $MODE" >&2; exit 2; }
echo "${STEP}applying NIMBLE's patch series to vLLM $have under $SP"
if applied; then
  echo "the series is already applied"
else
  for p in "$HERE"/patches/*.patch; do
    patch -d "$SP" -p1 --dry-run -s -f < "$p" > /dev/null 2>&1 || {
      echo "ABORT: $(basename "$p") does not apply (another series is in: reinstall BASE's wheel)"; exit 2; }
  done
  for p in "$HERE"/patches/*.patch; do patch -d "$SP" -p1 -s -f < "$p"; echo "applied $(basename "$p")"; done
fi
find "$V/distributed/device_communicators" "$V/model_executor/layers/fused_moe" "$V/config" -name __pycache__ -type d \
  -prune -exec rm -rf {} + 2>/dev/null || true
"$0" "$SP" --check
echo "apply.sh: done ($((SECONDS / 60))m $((SECONDS % 60))s)"
