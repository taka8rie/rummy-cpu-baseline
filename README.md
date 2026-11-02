# rummy-cpu-baseline

Reproducing the **IVF-CPU baseline** of *Rummy* (NSDI '24) on a small CPU-only machine.

> Zili Zhang, Fangyue Liu, Gang Huang, Xuanzhe Liu, Xin Jin. **Fast Vector Query Processing for Large Datasets Beyond GPU Memory with Reordered Pipelining.** NSDI 2024.
> [Paper](https://www.usenix.org/conference/nsdi24/presentation/zhang-zili-pipelining) · [Official code](https://github.com/pkusys/Rummy)

本仓库复现论文中 **CPU 对照组（IVF-CPU）** 的实验，对应 §6.1 **Fig. 10、Fig. 11 的 CPU 那一半**，使用官方仓库的 `cpu` 分支。Rummy 本体需要 NVIDIA GPU，不在本仓库范围内。

![SIFT1M results](results/sift1m_recall_latency.png)

## 复现了什么

| 论文 | 内容 | 本仓库 |
|---|---|---|
| Fig. 9 | Rummy vs IVF-Rotation vs IVF-CUM | ✗（需要 GPU） |
| **Fig. 10** | Rummy vs IVF-CPU，batch 8 / 2048 | **IVF-CPU 部分** |
| **Fig. 11** | 精度 vs 每条查询耗时，nprobe 1–32 | **IVF-CPU 部分** |
| Fig. 12–15、Table 3 | 各项技术的消融实验 | ✗（需要 GPU） |

| 设置 | 论文（IVF-CPU） | 本仓库 |
|---|---|---|
| 代码 | `cpu` 分支 `eval/overall.cpp` | 同一分支，改写为 `bench_cpu_small`，搜索逻辑相同 |
| 索引 | `IVF256,Flat` | 相同 |
| nprobe | 1–32 | 相同 |
| batch | 8、2048 | 相同 |
| top-k | 10 | 相同 |
| 数据 | SIFT1B（10 亿，480 GB） | SIFT1M（100 万）；冒烟测试用 siftsmall（1 万） |
| 机器 | AWS x1.16xlarge，64 vCPU，976 GB | Intel i7-9700（8 核 8 线程），16 GB，Ubuntu 20.04 |

数据规模和 CPU 核数都比论文小很多，**绝对数值不能和论文直接比较，只比较曲线形状和趋势**。

## 结果：SIFT1M

完整数据见 [`results/sift1m.csv`](results/sift1m.csv)。

| nprobe | 搜索数据比例 | recall@10 | 每条查询 (ms)，batch 8 | 每条查询 (ms)，batch 2048 |
|---:|---:|---:|---:|---:|
| 1 | 0.4% | 49.40% | 0.066 | 0.099 |
| 2 | 0.8% | 67.71% | 0.130 | 0.152 |
| 4 | 1.6% | 83.08% | 0.254 | 0.270 |
| 8 | 3.1% | 93.51% | 0.497 | 0.503 |
| 16 | 6.3% | 98.37% | 0.967 | 0.962 |
| 32 | 12.5% | 99.75% | 1.878 | 1.865 |

（recall 列为 batch 8 的结果；batch 2048 只用了 10000 条查询中的 8192 条，recall 相差不到 0.2 个百分点。）

**观察**

1. **只搜少量簇就有很高的精度**：nprobe = 8（全库的 3%）时 recall@10 已达 93.5%，nprobe = 32（12.5%）时达 99.75%。与论文 §2.1 “只搜约 1% 数据即可达约 99% 准确率”所描述的现象一致。
2. **耗时与 nprobe 近似成正比**：nprobe 增大 32 倍，耗时增大约 28 倍，耗时主要来自扫描簇内向量。
3. **batch 大小几乎不影响 CPU 上每条查询的耗时**：Faiss CPU 按查询并行，每条查询各自扫描自己的簇，同一个簇会被同批多条查询重复扫描。Rummy 的 cluster-based retrofitting 正是利用大 batch 中查询共享簇来减少重复传输，因此论文中 batch 越大，Rummy 相对 CPU 的优势越明显（Fig. 10）。这一点是本仓库结合论文设计所做的解读。

冒烟测试（siftsmall，nlist = 64）的结果在 [`results/siftsmall.csv`](results/siftsmall.csv)。该数据集太小，一次实验只有十几毫秒，耗时主要是线程开销，**只有 recall 一列有参考价值**。

## 目录

```
bench/
  bench_cpu_small.cpp     基准程序（基于 cpu 分支 eval/overall.cpp）
  bench_cpu_small.patch   打到官方 cpu 分支上的补丁（新增上面的文件 + CMake 目标）
scripts/
  setup.sh                安装依赖、获取 cpu 分支、打补丁、编译
  get_sift1m.sh           下载 SIFT1M
  get_sift10m.sh          （可选）从 SIFT1B 截取前 1000 万条
  run_sweep.sh            扫 nprobe 1–32 × batch 8/2048，输出 CSV
results/                  实验结果和图
```

`bench_cpu_small` 与原 `eval/overall.cpp` 的区别：搜索逻辑不变（`index_factory("IVF<nlist>,Flat")` → `train` → `add` → 按 batch `search` → recall@k），只是把数据路径、规模、线程数改成命令行参数；支持 `.fvecs` / `.bvecs`；分块读入数据以节省内存；缓存索引文件，扫参时不必重复训练；使用全部查询并加一次不计时的预热。

## 运行

默认项目根目录是 `~/research/rummy`，可以用环境变量 `R`（run_sweep）、`WORK`（setup）、`DATA`（下载脚本）修改。

```bash
git clone https://github.com/taka8rie/rummy-cpu-baseline.git && cd rummy-cpu-baseline
bash scripts/setup.sh                # apt 装依赖，pip 装 CMake ≥ 3.17，克隆官方 cpu 分支并打补丁，编译
bash scripts/get_sift1m.sh           # 下载 SIFT1M（Hugging Face 镜像，约 577 MB）
bash scripts/run_sweep.sh sift1m     # 约十几分钟；结果在 ~/research/rummy/results/sift1m.csv
```

单独运行一次：

```bash
B=~/research/rummy/Rummy/build/eval/bench_cpu_small; D=~/research/rummy/data/sift
$B --base $D/sift_base.fvecs --learn $D/sift_learn.fvecs --query $D/sift_query.fvecs \
   --gt $D/sift_groundtruth.ivecs --nlist 256 --nprobe 8 --bs 8 --k 10 --threads 8
```

## 复现时遇到的问题

- Ubuntu 20.04 自带 CMake 3.16，低于项目要求的 3.17；`setup.sh` 用 pip 安装 CMake 3.x。
- 官方 `cpu` 分支的 `eval/overall.cpp` 把数据路径（`/billion-data/...`）和线程数（40 / 64）写死了，所以另写了 `bench_cpu_small`。
- 用 gcc 13 编译官方代码会报 `uint8_t has not been declared`（缺 `#include <cstdint>`），可加 `-DCMAKE_CXX_FLAGS="-include cstdint"`；Ubuntu 20.04 的 gcc 9 无此问题。
- TEXMEX 官方 FTP 速度慢，改用 Hugging Face 镜像。
- 网络慢时 `git clone` 中途断开；`setup.sh` 支持把源码包 `rummy_src.tar.gz` 放在 `scripts/` 下离线解压。

## 数据来源

- SIFT1M / siftsmall：[TEXMEX corpus](http://corpus-texmex.irisa.fr/)（H. Jégou, M. Douze, C. Schmid）。下载脚本默认使用第三方镜像 [qbo-odp/sift1m](https://huggingface.co/datasets/qbo-odp/sift1m)，文件大小与官方格式一致，且 recall 随 nprobe 收敛到接近 100%，说明数据与 ground truth 配套。

## License

MIT。`bench/` 下的代码改编自 [pkusys/Rummy](https://github.com/pkusys/Rummy)（基于 Faiss，MIT License）。
