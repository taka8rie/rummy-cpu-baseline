/**
 * IVF-CPU baseline (Rummy NSDI'24, branch `cpu`) adapted for a small machine.
 *
 * Same index and search loop as eval/overall.cpp on the `cpu` branch
 * (index_factory "IVF<nlist>,Flat", batched index->search, recall@k vs GT),
 * but with:
 *   - data paths / sizes from the command line instead of hard-coded /billion-data
 *   - .fvecs (float) and .bvecs (uint8, BIGANN/SIFT1B) readers, optional --nb limit
 *   - thread count from the command line (paper machine used 40-64 threads)
 *   - optional index cache so nprobe / batch-size sweeps don't retrain
 *   - per-batch and per-query latency, and all nq queries used (not just 64/2048)
 *
 * Example (SIFT1M):
 *   ./bench_cpu_small --base sift/sift_base.fvecs --learn sift/sift_learn.fvecs \
 *       --query sift/sift_query.fvecs --gt sift/sift_groundtruth.ivecs \
 *       --nlist 256 --nprobe 0.03125 --bs 8 --k 10 --threads 8 --index sift1m_ivf256.index
 */
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <omp.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

static double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static bool file_exist(const std::string& p) {
    struct stat st;
    return !p.empty() && stat(p.c_str(), &st) == 0;
}

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
            s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Reads up to max_n vectors from .fvecs / .bvecs / .ivecs into a float (or int) array.
// Each record = int32 dim + dim * elem_size bytes.
static float* vecs_read(const std::string& fname, size_t* d_out, size_t* n_out,
                        size_t max_n = 0, size_t start = 0) {
    FILE* f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname.c_str());
        perror("");
        abort();
    }
    bool is_b = ends_with(fname, ".bvecs");
    size_t esz = is_b ? 1 : 4;
    int d;
    FAISS_THROW_IF_NOT(fread(&d, sizeof(int), 1, f) == 1);
    FAISS_THROW_IF_NOT_MSG(d > 0 && d < 1000000, "unreasonable dimension");
    struct stat st;
    fstat(fileno(f), &st);
    size_t rec = 4 + d * esz;
    FAISS_THROW_IF_NOT_MSG(st.st_size % rec == 0, "weird file size");
    size_t n = st.st_size / rec;
    FAISS_THROW_IF_NOT(start <= n);
    n -= start;
    if (max_n > 0 && max_n < n) n = max_n;
    fseeko(f, (off_t)(start * rec), SEEK_SET);

    float* x = new float[n * d];
    std::vector<uint8_t> buf(rec);
    for (size_t i = 0; i < n; i++) {
        FAISS_THROW_IF_NOT(fread(buf.data(), 1, rec, f) == rec);
        if (is_b) {
            for (int j = 0; j < d; j++) x[i * d + j] = float(buf[4 + j]);
        } else {
            memcpy(x + i * d, buf.data() + 4, d * 4);  // int32 for .ivecs, bit-copied
        }
    }
    fclose(f);
    *d_out = d;
    *n_out = n;
    return x;
}

static double inter_sec(const faiss::Index::idx_t* res, const int* gt, int k) {
    double hit = 0.;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            if (res[i] == gt[j]) { hit += 1.; break; }
    return hit / k;
}

static void usage() {
    printf("usage: bench_cpu_small --base B --query Q --gt G [--learn L]\n"
           "         [--nb N] [--ntrain N] [--nlist 256] [--nprobe 0.03125|8]\n"
           "         [--bs 8] [--k 10] [--threads 8] [--nq N] [--ip] [--index cache.index]\n");
}

int main(int argc, char** argv) {
    std::string base, learn, query, gtf, index_path;
    size_t nb_limit = 0, ntrain = 500000, nq_limit = 0;
    int nlist = 256, bs = 8, k = 10, threads = 8;
    float nprobe_arg = 0.03125f;
    bool ip = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(1); }
            return argv[++i];
        };
        if (a == "--base") base = next();
        else if (a == "--learn") learn = next();
        else if (a == "--query") query = next();
        else if (a == "--gt") gtf = next();
        else if (a == "--index") index_path = next();
        else if (a == "--nb") nb_limit = std::stoull(next());
        else if (a == "--ntrain") ntrain = std::stoull(next());
        else if (a == "--nq") nq_limit = std::stoull(next());
        else if (a == "--nlist") nlist = std::stoi(next());
        else if (a == "--nprobe") nprobe_arg = std::stof(next());
        else if (a == "--bs") bs = std::stoi(next());
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--ip") ip = true;
        else { usage(); return 1; }
    }
    if (base.empty() || query.empty() || gtf.empty()) { usage(); return 1; }
    FAISS_THROW_IF_NOT_MSG(k > 0 && k <= 100, "k must be in (0, 100]");

    // Same rule as the original: <1 means a fraction of nlist, >=1 means an absolute nprobe.
    int nprobe = nprobe_arg >= 1 - 1e-6 ? int(nprobe_arg + 0.5f)
                                        : std::max(1, int(nprobe_arg * nlist));
    omp_set_num_threads(threads);
    double t0 = elapsed();

    faiss::Index* index = nullptr;
    size_t d = 0;
    if (file_exist(index_path)) {
        printf("[%.3f s] Loading cached index %s\n", elapsed() - t0, index_path.c_str());
        index = faiss::read_index(index_path.c_str());
        d = index->d;
    } else {
        // Train: on --learn if given, else on the first --ntrain base vectors.
        size_t nt;
        float* xt = vecs_read(learn.empty() ? base : learn, &d, &nt, ntrain);
        std::string desc = "IVF" + std::to_string(nlist) + ",Flat";
        index = faiss::index_factory(d, desc.c_str(),
                                     ip ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2);
        printf("[%.3f s] Training %s on %zu vectors (d=%zu)\n", elapsed() - t0, desc.c_str(), nt, d);
        index->train(nt, xt);
        delete[] xt;

        // Stream the base set in 1M-vector chunks so peak RAM ~= index + one chunk.
        const size_t chunk = 1000000;
        size_t added = 0;
        printf("[%.3f s] Adding database in chunks of %zu\n", elapsed() - t0, chunk);
        while (nb_limit == 0 || added < nb_limit) {
            size_t want = chunk;
            if (nb_limit > 0) want = std::min(chunk, nb_limit - added);
            size_t n, d2;
            float* xb = vecs_read(base, &d2, &n, want, added);
            FAISS_THROW_IF_NOT(d2 == d);
            if (n == 0) { delete[] xb; break; }
            index->add(n, xb);
            delete[] xb;
            added += n;
            printf("[%.3f s]   added %zu\n", elapsed() - t0, added);
            if (n < want) break;
        }
        if (!index_path.empty()) {
            faiss::write_index(index, index_path.c_str());
            printf("[%.3f s] Index saved to %s\n", elapsed() - t0, index_path.c_str());
        }
    }
    printf("[%.3f s] Index ready: ntotal=%" PRId64 "\n", elapsed() - t0, (int64_t)index->ntotal);

    size_t nq, d2;
    float* xq = vecs_read(query, &d2, &nq);
    FAISS_THROW_IF_NOT(d2 == d);
    size_t gk, nq2;
    int* gt = (int*)vecs_read(gtf, &gk, &nq2);
    FAISS_THROW_IF_NOT_MSG(nq2 == nq, "incorrect nb of ground truth entries");
    FAISS_THROW_IF_NOT_MSG((size_t)k <= gk, "k larger than ground truth depth");
    if (nq_limit > 0 && nq_limit < nq) nq = nq_limit;

    auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
    FAISS_THROW_IF_NOT(ivf);
    ivf->nprobe = nprobe;

    size_t nbatch = nq / bs;
    FAISS_THROW_IF_NOT_MSG(nbatch > 0, "batch size larger than number of queries");
    std::vector<faiss::Index::idx_t> I(nbatch * bs * k);
    std::vector<float> D(nbatch * bs * k);

    // Warm-up batch (not timed), then timed batches.
    index->search(bs, xq, k, D.data(), I.data());
    double ts = elapsed();
    for (size_t b = 0; b < nbatch; b++)
        index->search(bs, xq + d * bs * b, k, D.data() + k * bs * b, I.data() + k * bs * b);
    double total = elapsed() - ts;

    double acc = 0.;
    for (size_t j = 0; j < nbatch * bs; j++)
        acc += inter_sec(I.data() + k * j, gt + gk * j, k);

    printf("RESULT nlist=%d nprobe=%d bs=%d k=%d threads=%d nq=%zu batches=%zu "
           "batch_ms=%.3f query_ms=%.4f qps=%.1f recall@%d=%.2f%%\n",
           ivf->nlist, nprobe, bs, k, threads, nbatch * bs, nbatch,
           total * 1000 / nbatch, total * 1000 / (nbatch * bs),
           nbatch * bs / total, k, acc * 100 / (nbatch * bs));

    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}
