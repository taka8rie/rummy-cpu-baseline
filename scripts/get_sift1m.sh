#!/usr/bin/env bash
# 下载 SIFT1M 到 $DATA/sift（默认 ~/research/rummy/data/sift）。
# 默认用 Hugging Face 镜像（HTTPS，可断点续传）；官方 TEXMEX FTP 见文末注释。
set -euo pipefail
DATA="${DATA:-$HOME/research/rummy/data}"
mkdir -p "$DATA/sift" && cd "$DATA/sift"
BASE=https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main
for f in sift_base.fvecs sift_learn.fvecs sift_query.fvecs sift_groundtruth.ivecs; do
  wget -c "$BASE/$f"
done
ls -l
# 期望字节数：sift_base 516000000 · sift_learn 51600000 · sift_query 5160000 · sift_groundtruth 4040000
#
# 官方来源（http://corpus-texmex.irisa.fr/），FTP，较慢；遇到 "425" 时加 --disable-epsv：
#   curl --disable-epsv -C - -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz
