#!/usr/bin/env bash
# (可选) SIFT10M = BIGANN(SIFT1B) 的前 1000 万条。
# 不下载整个 92 GB 的压缩包：边下边解压，取够 10M 条 (1.32 GB) 就停。
# 运行时内存峰值约 5.1 GB 索引 + 0.5 GB 分块，16 GB 机器够用。
set -euo pipefail
DATA="${DATA:-$HOME/research/rummy/data}"
mkdir -p "$DATA/bigann" && cd "$DATA/bigann"
URL=ftp://ftp.irisa.fr/local/texmex/corpus
N=10000000; REC=132   # bvecs 每条: 4 字节维度 + 128 字节

if [ ! -f base10M.bvecs ] || [ "$(stat -c %s base10M.bvecs)" -ne $((N*REC)) ]; then
  # head 读够字节后关闭管道，wget/gunzip 会收到 SIGPIPE 退出，属正常
  (wget -qO- "$URL/bigann_base.bvecs.gz" | gunzip | head -c $((N*REC)) > base10M.bvecs) || true
fi
[ -f bigann_query.bvecs ] || (wget -c "$URL/bigann_query.bvecs.gz" && gunzip bigann_query.bvecs.gz)
[ -f gnd/idx_10M.ivecs ]  || (wget -c "$URL/bigann_gnd.tar.gz" && tar xzf bigann_gnd.tar.gz)
ls -la . gnd | head -30
echo "base10M.bvecs 应为 $((N*REC)) 字节"
