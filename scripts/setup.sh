#!/usr/bin/env bash
# 在 rie-M920t (Ubuntu 20.04, i7-9700, 无 GPU) 上编译 Rummy 的 IVF-CPU 基线。
# 用法：把整个 rummy_cpu 文件夹拷到机器上，然后  bash setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HOME/research/rummy}"

echo "== 1/4 安装编译依赖 (需要 sudo 密码) =="
sudo apt-get update
sudo apt-get install -y build-essential git wget python3-pip libopenblas-dev

echo "== 2/4 安装 CMake >= 3.17 (Ubuntu 20.04 自带 3.16 不够) =="
python3 -m pip install --user --upgrade "cmake>=3.17,<4"
export PATH="$HOME/.local/bin:$PATH"
grep -q 'HOME/.local/bin' "$HOME/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
cmake --version | head -1

echo "== 3/4 获取 Rummy 源码 (cpu 分支) 并打补丁 =="
mkdir -p "$WORK"
if [ -f "$HERE/rummy_src.tar.gz" ]; then
  # 离线源码包 = cpu 分支 + 已打好的补丁（网络慢、git clone 失败时用）
  if [ ! -f "$WORK/Rummy/eval/bench_cpu_small.cpp" ]; then
    rm -rf "$WORK/Rummy"
    tar xzf "$HERE/rummy_src.tar.gz" -C "$WORK"
    echo "extracted source from rummy_src.tar.gz"
  fi
  cd "$WORK/Rummy"
else
  if [ ! -d "$WORK/Rummy" ]; then
    git clone --depth 1 -b cpu --single-branch https://github.com/pkusys/Rummy.git "$WORK/Rummy"
  fi
  cd "$WORK/Rummy"
  PATCH="$HERE/bench_cpu_small.patch"; [ -f "$PATCH" ] || PATCH="$HERE/../bench/bench_cpu_small.patch"
  if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "patch applied"
  else
    echo "patch already applied (or conflicts) — skipping"
  fi
fi

echo "== 4/4 编译 Faiss (CPU, AVX2, OpenBLAS) 和基准程序 =="
cmake -B build . \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF \
  -DBLA_VENDOR=OpenBLAS -DFAISS_OPT_LEVEL=avx2 -DCMAKE_BUILD_TYPE=Release
make -C build -j"$(nproc)" faiss_avx2
make -C build -j"$(nproc)" bench_cpu_small
ls -la build/eval/bench_cpu_small
echo "完成。下一步： bash $HERE/get_sift1m.sh"
