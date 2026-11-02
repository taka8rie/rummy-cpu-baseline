#!/usr/bin/env bash
# IVF-CPU 基线扫参：对应论文 Fig.10 / Fig.11 里 IVF-CPU 那一半（缩小规模版）。
#   batch size: 8 (online 小批) 和 2048 (offline 大批)，论文同款
#   nprobe:     1..32 (论文 Fig.11 的取值)，nlist=256 (论文 SIFT 的 ncentroids)
# 用法:  bash run_sweep.sh sift1m      或     bash run_sweep.sh sift10m
set -euo pipefail
DS="${1:-sift1m}"
R="${R:-$HOME/research/rummy}"
BIN="$R/Rummy/build/eval/bench_cpu_small"
D="$R/data"
OUT="$R/results"; mkdir -p "$OUT"
THREADS="${THREADS:-$(nproc)}"

case "$DS" in
  sift1m)
    ARGS=(--base "$D/sift/sift_base.fvecs" --learn "$D/sift/sift_learn.fvecs"
          --query "$D/sift/sift_query.fvecs" --gt "$D/sift/sift_groundtruth.ivecs") ;;
  sift10m)
    ARGS=(--base "$D/bigann/base10M.bvecs" --ntrain 500000
          --query "$D/bigann/bigann_query.bvecs" --gt "$D/bigann/gnd/idx_10M.ivecs") ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
IDX="$R/${DS}_ivf256.index"
LOG="$OUT/${DS}_$(date +%Y%m%d_%H%M%S).log"
CSV="$OUT/${DS}.csv"
[ -f "$CSV" ] || echo "dataset,nlist,nprobe,bs,k,threads,nq,batch_ms,query_ms,qps,recall" > "$CSV"

# 第一次运行会训练+建索引并缓存到 $IDX，之后直接加载
"$BIN" "${ARGS[@]}" --nlist 256 --nprobe 8 --bs 8 --k 10 --threads "$THREADS" --index "$IDX" | tee -a "$LOG"

for BS in 8 2048; do
  for NP in 1 2 3 4 5 6 8 10 12 16 20 24 28 32; do
    LINE=$("$BIN" "${ARGS[@]}" --nlist 256 --nprobe "$NP" --bs "$BS" --k 10 \
                  --threads "$THREADS" --index "$IDX" | grep '^RESULT' | tee -a "$LOG")
    echo "$LINE"
    # RESULT nlist=.. nprobe=.. bs=.. k=.. threads=.. nq=.. batches=.. batch_ms=.. query_ms=.. qps=.. recall@10=..%
    echo "$LINE" | awk -v ds="$DS" '{
        for (i = 2; i <= NF; i++) { split($i, kv, "="); sub(/%$/, "", kv[2]); v[kv[1]] = kv[2] }
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", ds, v["nlist"], v["nprobe"], v["bs"], v["k"],
               v["threads"], v["nq"], v["batch_ms"], v["query_ms"], v["qps"], v["recall@10"] }' >> "$CSV"
  done
done
echo "日志: $LOG"
echo "汇总: $CSV"
