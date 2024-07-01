// #################################################################
// Parts of this code are based on the source code of the following research:
// Orestis Polychroniou, Arun Raghavan, and Kenneth A Ross. Rethinking simd vectorization
// for in-memory databases. In Proceedings of the 2015 ACM SIGMOD International Conference
// on Management of Data, pages 1493–1508, 2015.
// #################################################################

#include <iostream>
#include <math.h>
#include <immintrin.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <ctype.h>
#include <sched.h>
#include <time.h>
#include <pthread.h>

using namespace std;

inline __m256i _mm256_packlo_epi32(__m256i x, __m256i y)
{
    __m256 a = _mm256_castsi256_ps(x);
    __m256 b = _mm256_castsi256_ps(y);
    __m256 c = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));
    __m256i z = _mm256_castps_si256(c);
    return _mm256_permute4x64_epi64(z, _MM_SHUFFLE(3,1,2,0));
}

inline __m256i _mm256_packhi_epi32(__m256i x, __m256i y)
{
    __m256 a = _mm256_castsi256_ps(x);
    __m256 b = _mm256_castsi256_ps(y);
    __m256 c = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3,1,3,1));
    __m256i z = _mm256_castps_si256(c);
    return _mm256_permute4x64_epi64(z, _MM_SHUFFLE(3,1,2,0));
}

void *align(const void *p)
{
    while (63 & (size_t) p) p++;
    return (void*) p;
}

inline void store(uint32_t *p, uint32_t v)
{
    #ifndef __MIC
        _mm_stream_si32((int*) p, v);
    #else
        *p = v;
    #endif
}

void bind_thread(int thread, int threads)
{
    size_t size = CPU_ALLOC_SIZE(threads);
    cpu_set_t *cpu_set = CPU_ALLOC(threads);
    assert(cpu_set != NULL);
    CPU_ZERO_S(size, cpu_set);
    CPU_SET_S(thread, size, cpu_set);
    assert(pthread_setaffinity_np(pthread_self(), size, cpu_set) == 0);
    CPU_FREE(cpu_set);
}

namespace vecht_experimental
{
    typedef struct {
        uint32_t key_;
        uint32_t val_;
    } bucket;

    typedef struct {
        pthread_t id;
        int threads;
        int thread;
        size_t outer_size;
        size_t  join_size;
        uint32_t *outer;
        pthread_barrier_t *barrier;
        uint32_t *result;
        size_t max_size;
        bucket* data;
        uint32_t hash_factor;
        uint8_t shift;
        int group_size;
    } info_t;

    static constexpr uint64_t perm[256] = {0x0706050403020100ull,
     0x0007060504030201ull, 0x0107060504030200ull, 0x0001070605040302ull,
     0x0207060504030100ull, 0x0002070605040301ull, 0x0102070605040300ull,
     0x0001020706050403ull, 0x0307060504020100ull, 0x0003070605040201ull,
     0x0103070605040200ull, 0x0001030706050402ull, 0x0203070605040100ull,
     0x0002030706050401ull, 0x0102030706050400ull, 0x0001020307060504ull,
     0x0407060503020100ull, 0x0004070605030201ull, 0x0104070605030200ull,
     0x0001040706050302ull, 0x0204070605030100ull, 0x0002040706050301ull,
     0x0102040706050300ull, 0x0001020407060503ull, 0x0304070605020100ull,
     0x0003040706050201ull, 0x0103040706050200ull, 0x0001030407060502ull,
     0x0203040706050100ull, 0x0002030407060501ull, 0x0102030407060500ull,
     0x0001020304070605ull, 0x0507060403020100ull, 0x0005070604030201ull,
     0x0105070604030200ull, 0x0001050706040302ull, 0x0205070604030100ull,
     0x0002050706040301ull, 0x0102050706040300ull, 0x0001020507060403ull,
     0x0305070604020100ull, 0x0003050706040201ull, 0x0103050706040200ull,
     0x0001030507060402ull, 0x0203050706040100ull, 0x0002030507060401ull,
     0x0102030507060400ull, 0x0001020305070604ull, 0x0405070603020100ull,
     0x0004050706030201ull, 0x0104050706030200ull, 0x0001040507060302ull,
     0x0204050706030100ull, 0x0002040507060301ull, 0x0102040507060300ull,
     0x0001020405070603ull, 0x0304050706020100ull, 0x0003040507060201ull,
     0x0103040507060200ull, 0x0001030405070602ull, 0x0203040507060100ull,
     0x0002030405070601ull, 0x0102030405070600ull, 0x0001020304050706ull,
     0x0607050403020100ull, 0x0006070504030201ull, 0x0106070504030200ull,
     0x0001060705040302ull, 0x0206070504030100ull, 0x0002060705040301ull,
     0x0102060705040300ull, 0x0001020607050403ull, 0x0306070504020100ull,
     0x0003060705040201ull, 0x0103060705040200ull, 0x0001030607050402ull,
     0x0203060705040100ull, 0x0002030607050401ull, 0x0102030607050400ull,
     0x0001020306070504ull, 0x0406070503020100ull, 0x0004060705030201ull,
     0x0104060705030200ull, 0x0001040607050302ull, 0x0204060705030100ull,
     0x0002040607050301ull, 0x0102040607050300ull, 0x0001020406070503ull,
     0x0304060705020100ull, 0x0003040607050201ull, 0x0103040607050200ull,
     0x0001030406070502ull, 0x0203040607050100ull, 0x0002030406070501ull,
     0x0102030406070500ull, 0x0001020304060705ull, 0x0506070403020100ull,
     0x0005060704030201ull, 0x0105060704030200ull, 0x0001050607040302ull,
     0x0205060704030100ull, 0x0002050607040301ull, 0x0102050607040300ull,
     0x0001020506070403ull, 0x0305060704020100ull, 0x0003050607040201ull,
     0x0103050607040200ull, 0x0001030506070402ull, 0x0203050607040100ull,
     0x0002030506070401ull, 0x0102030506070400ull, 0x0001020305060704ull,
     0x0405060703020100ull, 0x0004050607030201ull, 0x0104050607030200ull,
     0x0001040506070302ull, 0x0204050607030100ull, 0x0002040506070301ull,
     0x0102040506070300ull, 0x0001020405060703ull, 0x0304050607020100ull,
     0x0003040506070201ull, 0x0103040506070200ull, 0x0001030405060702ull,
     0x0203040506070100ull, 0x0002030405060701ull, 0x0102030405060700ull,
     0x0001020304050607ull, 0x0706050403020100ull, 0x0007060504030201ull,
     0x0107060504030200ull, 0x0001070605040302ull, 0x0207060504030100ull,
     0x0002070605040301ull, 0x0102070605040300ull, 0x0001020706050403ull,
     0x0307060504020100ull, 0x0003070605040201ull, 0x0103070605040200ull,
     0x0001030706050402ull, 0x0203070605040100ull, 0x0002030706050401ull,
     0x0102030706050400ull, 0x0001020307060504ull, 0x0407060503020100ull,
     0x0004070605030201ull, 0x0104070605030200ull, 0x0001040706050302ull,
     0x0204070605030100ull, 0x0002040706050301ull, 0x0102040706050300ull,
     0x0001020407060503ull, 0x0304070605020100ull, 0x0003040706050201ull,
     0x0103040706050200ull, 0x0001030407060502ull, 0x0203040706050100ull,
     0x0002030407060501ull, 0x0102030407060500ull, 0x0001020304070605ull,
     0x0507060403020100ull, 0x0005070604030201ull, 0x0105070604030200ull,
     0x0001050706040302ull, 0x0205070604030100ull, 0x0002050706040301ull,
     0x0102050706040300ull, 0x0001020507060403ull, 0x0305070604020100ull,
     0x0003050706040201ull, 0x0103050706040200ull, 0x0001030507060402ull,
     0x0203050706040100ull, 0x0002030507060401ull, 0x0102030507060400ull,
     0x0001020305070604ull, 0x0405070603020100ull, 0x0004050706030201ull,
     0x0104050706030200ull, 0x0001040507060302ull, 0x0204050706030100ull,
     0x0002040507060301ull, 0x0102040507060300ull, 0x0001020405070603ull,
     0x0304050706020100ull, 0x0003040507060201ull, 0x0103040507060200ull,
     0x0001030405070602ull, 0x0203040507060100ull, 0x0002030405070601ull,
     0x0102030405070600ull, 0x0001020304050706ull, 0x0607050403020100ull,
     0x0006070504030201ull, 0x0106070504030200ull, 0x0001060705040302ull,
     0x0206070504030100ull, 0x0002060705040301ull, 0x0102060705040300ull,
     0x0001020607050403ull, 0x0306070504020100ull, 0x0003060705040201ull,
     0x0103060705040200ull, 0x0001030607050402ull, 0x0203060705040100ull,
     0x0002030607050401ull, 0x0102030607050400ull, 0x0001020306070504ull,
     0x0406070503020100ull, 0x0004060705030201ull, 0x0104060705030200ull,
     0x0001040607050302ull, 0x0204060705030100ull, 0x0002040607050301ull,
     0x0102040607050300ull, 0x0001020406070503ull, 0x0304060705020100ull,
     0x0003040607050201ull, 0x0103040607050200ull, 0x0001030406070502ull,
     0x0203040607050100ull, 0x0002030406070501ull, 0x0102030406070500ull,
     0x0001020304060705ull, 0x0506070403020100ull, 0x0005060704030201ull,
     0x0105060704030200ull, 0x0001050607040302ull, 0x0205060704030100ull,
     0x0002050607040301ull, 0x0102050607040300ull, 0x0001020506070403ull,
     0x0305060704020100ull, 0x0003050607040201ull, 0x0103050607040200ull,
     0x0001030506070402ull, 0x0203050607040100ull, 0x0002030506070401ull,
     0x0102030506070400ull, 0x0001020305060704ull, 0x0405060703020100ull,
     0x0004050607030201ull, 0x0104050607030200ull, 0x0001040506070302ull,
     0x0204050607030100ull, 0x0002040506070301ull, 0x0102040506070300ull,
     0x0001020405060703ull, 0x0304050607020100ull, 0x0003040506070201ull,
     0x0103040506070200ull, 0x0001030405060702ull, 0x0203040506070100ull,
     0x0002030405060701ull, 0x0102030405060700ull, 0x0001020304050607ull};

    class lp_map
    {
        size_t max_size_;
        uint8_t shift_;
        uint32_t hash_factor_;
        uint32_t group_size_ = 64;
        bucket* data = nullptr;
        int threads_ = 1;

        private:
        

        public:

        lp_map(const size_t& max_size, const uint32_t& hash_factor, int threads=1)
        {
            max_size_ = 1 << size_t(ceil(log2(max_size)));
            hash_factor_ = hash_factor;
            shift_ = 32 - log2(max_size_);
            threads_ = threads;
            data = new bucket[max_size_];
            for (size_t i = 0 ; i != max_size_ ; ++i)
                data[i].key_ = -1;
        }

        ~lp_map()
        {
            delete [] data;
        }

        size_t get_buckets_count()
        {
            return max_size_;
        }

        // Sequential APIs ===========================================================

        inline bool insert(const uint32_t& key, const uint32_t& val)
        {
            try
            {              
                size_t h = ((uint32_t) (key * hash_factor_)) >> shift_;
                while (data[h].key_ != -1)
                    h = (h + 1) & (max_size_ - 1);
                data[h].key_ = key;
                data[h].val_ = val;
            }
            catch(const std::exception& e)
            {
                return false;
            }
            return true;
        }

        inline bool insert_all(uint32_t* keys, uint32_t* vals, const size_t& size)
        {
            try
            {              
                if (size < 1 || size >= max_size_)
                    return false;
                for (size_t i = 0 ; i != size ; ++i) {
                    uint32_t key = keys[i];
                    uint32_t val = vals[i];
                    size_t h = ((uint32_t) (key * hash_factor_)) >> shift_;
                    while (data[h].key_ != -1)
                        h = (h + 1) & (max_size_ - 1);
                    data[h].key_ = key;
                    data[h].val_ = val;
                }
            }
            catch(const std::exception& e)
            {
                return false;
            }
            return true;
        }

        inline size_t find_all_scalar(uint32_t* keys, const size_t& size, uint32_t* result)
        {
            size_t i;
            uint32_t *vals_orig = result;
            assert(sizeof(data[0]) == 8);
            const size_t buckets = max_size_;
            const uint64_t *table_64 = (const uint64_t*) data;
            for (i = 0 ; i != size ; ++i) {
                uint32_t key = keys[i];
                size_t h = ((uint32_t) (key * hash_factor_)) >> shift_;
                uint64_t tab = table_64[h];
                if (key == (uint32_t) tab)
                    store(result++, tab >> 32);
                else while (-1 != (uint32_t) tab) {
                    h = (h + 1) & (buckets - 1);
                    tab = table_64[h];
                    if (key == (uint32_t) tab) {
                        store(result++, tab >> 32);
                        break;
                    }
                }
            }
            return result - vals_orig;
        }

        // Parallel APIs Inner Tasks =================================================
        private:

        inline size_t find_parallel_executer(uint32_t* keys, const size_t& size, uint32_t** result,  void *(*__start_routine)(void *))
        {
            int b;
            pthread_barrier_t barrier[2];
            for (b = 0 ; b != 2; ++b)
                pthread_barrier_init(&barrier[b], NULL, threads_);
            info_t info[threads_];
            for (int t = 0 ; t != threads_ ; ++t) {
                size_t beg = ((size / threads_) ) * t;
                size_t end = ((size / threads_) ) * (t + 1);
                if (t + 1 == threads_) end = size;
                info[t].threads = threads_;
                info[t].thread = t;
                info[t].outer_size = end-beg;
                info[t].outer = &(keys[beg]);
                info[t].barrier = barrier;
                info[t].result = result[t];
                info[t].max_size = max_size_;
                info[t].data = data;
                info[t].hash_factor = hash_factor_;
                info[t].shift = shift_;
                info[t].group_size = group_size_;
                pthread_create(&info[t].id, NULL, __start_routine, (void*) &info[t]);
            }
            size_t j = 0;
            for (int t = 0 ; t != threads_ ; ++t) {
                pthread_join(info[t].id, NULL);
                j += info[t].join_size;
            }
            for (b = 0 ; b != 2 ; ++b)
                pthread_barrier_destroy(&barrier[b]);
            return j;
        }

        inline static void* find_all_scalar_inner(void* arg)
        {

            info_t *d = (info_t*) arg;
            assert(pthread_equal(pthread_self(), d->id));
            bind_thread(d->thread, d->threads);
            size_t j, b = 0;
            uint32_t *outer = d->outer;
            size_t outer_size = d->outer_size;

            pthread_barrier_wait(&d->barrier[b++]);

            size_t i;
            uint32_t *vals_orig = d->result;
            assert(sizeof(data[0]) == 8);
            const size_t buckets = d->max_size;
            const uint64_t *table_64 = (const uint64_t*) d->data;
            for (i = 0 ; i != outer_size ; ++i) {
                uint32_t key = outer[i];
                size_t h = ((uint32_t) (key * d->hash_factor)) >> d->shift;

                uint64_t tab = table_64[h];
                if (key == (uint32_t) tab)
                    store((d->result)++, tab >> 32);
                else while (-1 != (uint32_t) tab) {
                    h = (h + 1) & (buckets - 1);
                    tab = table_64[h];
                    if (key == (uint32_t) tab) {
                        store((d->result)++, tab >> 32);
                        break;
                    }
                }
            }

            d->join_size = (d->result) - vals_orig;
       
        	pthread_barrier_wait(&d->barrier[b++]);
            pthread_exit(NULL);
        }

        inline static void* find_all_scalar_prefetching_inner(void* arg)
        {
            info_t *d = (info_t*) arg;
            assert(pthread_equal(pthread_self(), d->id));
            bind_thread(d->thread, d->threads);
            size_t j, b = 0;
            uint32_t *outer = d->outer;
            size_t outer_size = d->outer_size;

            pthread_barrier_wait(&d->barrier[b++]);

            int i;
            uint32_t *vals_orig = d->result;
            const uint64_t *table_64 = (const uint64_t*) d->data;
            size_t hashs[d->group_size];

            for (i=0; i<outer_size; i+=d->group_size) {
                for (int g=0; g<d->group_size; g++)
                {
                    if (i+g >= outer_size) break;
                    hashs[g] = ((uint32_t) (d->outer[i+g] * d->hash_factor)) >> d->shift; 
                    _mm_prefetch(&table_64[hashs[g]], _MM_HINT_T0);
                }

                for (int g=0; g<(d->group_size); g++)
                {
                    if (i+g >= outer_size) break;
                    size_t h = hashs[g];
                    uint64_t tab = table_64[h];
                    if (d->outer[i+g] == (uint32_t) tab)
                        store(d->result++, tab >> 32);
                    else while (-1 != (uint32_t) tab) {
                        h = (h + 1) & (d->max_size - 1);
                        tab = table_64[h];
                        if (d->outer[i+g] == (uint32_t) tab) {
                            store(d->result++, tab >> 32);
                            break;
                        }
                    }
                }
            }

            d->join_size = (d->result) - vals_orig;

        	pthread_barrier_wait(&d->barrier[b++]);
            pthread_exit(NULL);
        }

        inline static void* find_all_simd_inner(void* arg)
        {
            info_t *d = (info_t*) arg;
            assert(pthread_equal(pthread_self(), d->id));
            bind_thread(d->thread, d->threads);
            size_t joins, bar = 0;
            uint32_t *outer = d->outer;
            size_t outer_size = d->outer_size;

            pthread_barrier_wait(&d->barrier[bar++]);

            size_t i = 0, o = 0, b = 0;
                assert(sizeof(d->data[0]) == 8);
                const size_t buckets = d->max_size;
                const __m128i shift = _mm_cvtsi32_si128(d->shift);
                const __m256i factor = _mm256_set1_epi32(d->hash_factor);
                const __m256i empty = _mm256_set1_epi32(-1);
                const __m256i buckets_minus_1 = _mm256_set1_epi32(d->max_size - 1);
                const __m256i mask_1 = _mm256_set1_epi32(1);
            #if defined __INTEL_COMPILER &&  __INTEL_COMPILER < 1600
                const long *table_64 = (const long*) table;
            #else
                const long long *table_64 = (const long long*) d->data;
            #endif
                const size_t buf_size = 128;
                int buf_space[buf_size + 8 + 15];
                int *buf = (int*)align((void*) buf_space);
                __m256i key, off, inv = _mm256_set1_epi32(-1);

                while (i + 8 <= outer_size) {
                    // load new items (mask out reloads)
                    __m256i new_key = _mm256_maskload_epi32((const int*) &(d->outer[i]), inv);
                    key = _mm256_andnot_si256(inv, key);
                    key = _mm256_or_si256(key, new_key);
                    // hash
                    __m256i h = _mm256_mullo_epi32(key, factor);
                    off = _mm256_add_epi32(off, mask_1);
                    off = _mm256_andnot_si256(inv, off);
                    h = _mm256_srl_epi32(h, shift);
                    h = _mm256_add_epi32(h, off);
                    h = _mm256_and_si256(h, buckets_minus_1);
                    // gather
                    __m256i tab_lo = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_hi = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    __m256i tab_key = _mm256_packlo_epi32(tab_lo, tab_hi);
                    __m256i tab_val = _mm256_packhi_epi32(tab_lo, tab_hi);
                    // update count & sum
                    inv = _mm256_cmpeq_epi32(tab_key, empty);
                    __m256i out = _mm256_cmpeq_epi32(tab_key, key);
                    inv = _mm256_or_si256(inv, out);
                    // load permutation masks
                    size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                    size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));
                    __m128i perm_inv_comp = _mm_loadl_epi64((__m128i*) &perm[j]);
                    __m128i perm_out_comp = _mm_loadl_epi64((__m128i*) &perm[k ^ 255]);
                    __m256i perm_inv = _mm256_cvtepi8_epi32(perm_inv_comp);
                    __m256i perm_out = _mm256_cvtepi8_epi32(perm_out_comp);
                    // permutation for invalid
                    inv = _mm256_permutevar8x32_epi32(inv, perm_inv);
                    key = _mm256_permutevar8x32_epi32(key, perm_inv);
                    off = _mm256_permutevar8x32_epi32(off, perm_inv);
                    i += _mm_popcnt_u64(j);
                    // permutation for output
                    tab_val = _mm256_permutevar8x32_epi32(tab_val, perm_out);
                    out = _mm256_permutevar8x32_epi32(out, perm_out);
                    _mm256_maskstore_epi32(&buf[b], out, tab_val);
                    b += _mm_popcnt_u64(k);
                    // flush buffer
                    if (b > buf_size) {
                        size_t b_i = 0;
                        do {
                            __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                            _mm256_stream_si256((__m256i*) &(d->result[o]), x);
                            b_i += 8;
                            o += 8;
                        } while (b_i != buf_size);
                        __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                        _mm256_store_si256((__m256i*) &buf[0], x);
                        b -= buf_size;
                    }
                }

                // clean buffer
                size_t b_i = 0;
                while (b_i != b) _mm_stream_si32(&((int*) d->result)[o++], buf[b_i++]);
                // extract last keys
                uint32_t l_keys[8];
                _mm256_storeu_si256((__m256i*) l_keys, key);
                size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                j = 8 - _mm_popcnt_u64(j);
                i += j;
                while (i != outer_size) l_keys[j++] = d->outer[i++];
                // process last keys
                const uint8_t s = 32 - log2(buckets);
                for (i = 0 ; i != j ; ++i) {
                    uint32_t k = l_keys[i];
                    size_t h = ((uint32_t) (k * d->hash_factor)) >> s;
                    while (-1 != d->data[h].key_) {
                        if (k == d->data[h].key_) {
                            _mm_stream_si32(&((int*) d->result)[o++], d->data[h].val_);
                            break;
                        }
                        h = (h + 1) & (buckets - 1);
                    }
                }

            d->join_size = o;

        	pthread_barrier_wait(&d->barrier[bar++]);
            pthread_exit(NULL);
        };

        inline static void* find_all_simd_prefetching_inner_0(void* arg)
        {
            
            info_t *d = (info_t*) arg;
            assert(pthread_equal(pthread_self(), d->id));
            bind_thread(d->thread, d->threads);
            size_t joins, bar = 0;
            uint32_t *outer = d->outer;
            size_t outer_size = d->outer_size;

            pthread_barrier_wait(&d->barrier[bar++]);

            int step = d->group_size;
            size_t out_count = 0;        

            size_t b = 0;
            const size_t buf_size = 128;
            int buf_space[buf_size + 8 + 15];
            int *buf = (int*)align((void*) buf_space);

            for (size_t read_idx = 0; read_idx<d->outer_size; read_idx+=d->group_size) 
            {
                if ((d->outer_size) - read_idx < (d->group_size))
                    step = (d->outer_size)%(d->group_size);

                //========================================================================================================
                auto keys = &(d->outer[read_idx]);
                auto& size = step;
                size_t i = 0, o = 0;
                assert(sizeof(d->data[0]) == 8);
                const size_t buckets = d->max_size;
                const __m128i shift = _mm_cvtsi32_si128(d->shift);
                const __m256i factor = _mm256_set1_epi32(d->hash_factor);
                const __m256i empty = _mm256_set1_epi32(-1);
                const __m256i buckets_minus_1 = _mm256_set1_epi32(buckets - 1);
                const __m256i mask_1 = _mm256_set1_epi32(1);
                #if defined __INTEL_COMPILER &&  __INTEL_COMPILER < 1600  
                    const long *table_64 = (const long*) table;
                #else
                    const long long *table_64 = (const long long*) d->data;
                #endif
                __m256i key, h, off, inv = _mm256_set1_epi32(-1);
                __m256i fixed_off = _mm256_set1_epi32(-1);
                uint32_t hashs[size];
                __m256i vec;

                for (size_t i=0; i<size; i+=8)
                {
                    __m256i vec = _mm256_maskload_epi32((const int*) &(keys[i]), fixed_off);
                    vec = _mm256_mullo_epi32(vec, factor);
                    vec = _mm256_srl_epi32(vec, shift);
                    vec = _mm256_and_si256(vec, buckets_minus_1);   
                    for (size_t j = 0; j < 8; j++)
                    {
                        hashs[i+j] = _mm256_extract_epi32(vec, j);
                        _mm_prefetch(&table_64[hashs[i+j]], _MM_HINT_T0);
                    }
                }

                auto is_buf_flushed = false;
                while (i + 8 <= size) {
                    is_buf_flushed = false;
                    // load new items (mask out reloads)
                    __m256i new_key = _mm256_maskload_epi32((const int*) &(keys[i]), inv);
                    key = _mm256_andnot_si256(inv, key);
                    key = _mm256_or_si256(key, new_key);
                    // hash
                    off = _mm256_set1_epi32(1);
                    off = _mm256_andnot_si256(inv, off);

                    __m256i new_hash = _mm256_maskload_epi32((const int*) &hashs[i], inv);
                    h = _mm256_andnot_si256(inv, h);
                    h = _mm256_or_si256(h, new_hash);
                    h = _mm256_add_epi32(h, off);
                    h = _mm256_and_si256(h, buckets_minus_1);
                    // gather
                    __m256i tab_lo = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_hi = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_key = _mm256_packlo_epi32(tab_lo, tab_hi);
                    __m256i tab_val = _mm256_packhi_epi32(tab_lo, tab_hi);
                    // update count & sum
                    __m256i out = _mm256_cmpeq_epi32(tab_key, key);
                    inv = _mm256_cmpeq_epi32(tab_key, empty);
                    inv = _mm256_or_si256(inv, out);
                    // load permutation masks
                    size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                    size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));
                    __m128i perm_inv_comp = _mm_loadl_epi64((__m128i*) &perm[j]);
                    __m128i perm_out_comp = _mm_loadl_epi64((__m128i*) &perm[k ^ 255]);
                    __m256i perm_inv = _mm256_cvtepi8_epi32(perm_inv_comp);
                    __m256i perm_out = _mm256_cvtepi8_epi32(perm_out_comp);
                    // permutation for invalid
                    inv = _mm256_permutevar8x32_epi32(inv, perm_inv);
                    key = _mm256_permutevar8x32_epi32(key, perm_inv);
                    h = _mm256_permutevar8x32_epi32(h, perm_inv);
                    off = _mm256_permutevar8x32_epi32(off, perm_inv);
                    i += _mm_popcnt_u64(j);
                    // permutation for output
                    out = _mm256_permutevar8x32_epi32(out, perm_out);
                    tab_val = _mm256_permutevar8x32_epi32(tab_val, perm_out);
                    _mm256_maskstore_epi32(&buf[b], out, tab_val);
                    b += _mm_popcnt_u64(k);

                    // flush buffer
                    if (b > buf_size) {
                        size_t b_i = 0;
                        do {
                            __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                            _mm256_stream_si256((__m256i*)(&(d->result)[out_count]), x);
                            b_i += 8;
                            out_count += 8;
                        } while (b_i != buf_size);
                        __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                        _mm256_store_si256((__m256i*) &buf[0], x);
                        b -= buf_size;
                        is_buf_flushed = true;
                    }
                }

                // extract last keys
                uint32_t l_keys[8];
                _mm256_storeu_si256((__m256i*) l_keys, key);
                size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                j = 8 - _mm_popcnt_u64(j);
                i += j;
                while (i != size) l_keys[j++] = keys[i++];
                // process last keys
                const uint8_t s = 32 - log2(buckets);
                for (i = 0 ; i != j ; ++i) {
                    uint32_t k = l_keys[i];
                    size_t h = ((uint32_t) (k * d->hash_factor)) >> s;
                    while (-1 != (d->data[h]).key_) {
                        if (k == (d->data[h]).key_) {
                            _mm_stream_si32(&buf[b++], (d->data[h]).val_);
                            break;
                        }
                        h = (h + 1) & (buckets - 1);
                    }
                }
                if (is_buf_flushed && b > 8)
                {
                    __m256i x = _mm256_load_si256((__m256i*) &buf[0]);
                    _mm256_stream_si256((__m256i*)(&(d->result)[out_count]), x);
                    out_count += 8;
                    x = _mm256_load_si256((__m256i*) &buf[8]);
                    _mm256_store_si256((__m256i*) &buf[0], x);
                    b -= 8;
                }
                else if (!is_buf_flushed && b > buf_size)
                {
                    size_t b_i = 0;
                    do {
                        __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                        _mm256_stream_si256((__m256i*)(&(d->result)[out_count]), x);
                        b_i += 8;
                        out_count += 8;
                    } while (b_i != buf_size);
                    __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                    _mm256_store_si256((__m256i*) &buf[0], x);
                    b -= buf_size;
                }


                //========================================================================================================
            }

            // final clean buffer
            size_t b_i = 0;
            while (b_i != b)_mm_stream_si32(&((int*)d->result)[out_count++], buf[b_i++]); 

            d->join_size = out_count;

        	pthread_barrier_wait(&d->barrier[bar++]);
            pthread_exit(NULL);

        }

        inline static void* find_all_simd_prefetching_inner_1(void* arg)
        {
            info_t *d = (info_t*) arg;
            assert(pthread_equal(pthread_self(), d->id));
            bind_thread(d->thread, d->threads);
            size_t joins, bar = 0;
            uint32_t *outer = d->outer;
            size_t outer_size = d->outer_size;

            pthread_barrier_wait(&d->barrier[bar++]);

            int step = d->group_size;
            int tmp_out_count = 0;
            size_t out_count = 0;        

            uint32_t tmp_joined[d->group_size];

            for (size_t read_idx = 0; read_idx<d->outer_size; read_idx+=d->group_size) 
            {
                if ((d->outer_size) - read_idx < (d->group_size))
                    step = (d->outer_size)%(d->group_size);

                //========================================================================================================
                auto keys = &(d->outer[read_idx]);
                auto& size = step;
                auto& result = tmp_joined;             

                size_t i = 0, o = 0, b = 0;
                assert(sizeof(d->data[0]) == 8);
                const size_t buckets = d->max_size;
                const __m128i shift = _mm_cvtsi32_si128(d->shift);
                const __m256i factor = _mm256_set1_epi32(d->hash_factor);
                const __m256i empty = _mm256_set1_epi32(-1);
                const __m256i buckets_minus_1 = _mm256_set1_epi32(buckets - 1);
                const __m256i mask_1 = _mm256_set1_epi32(1);
                #if defined __INTEL_COMPILER &&  __INTEL_COMPILER < 1600  
                    const long *table_64 = (const long*) table;
                #else
                    const long long *table_64 = (const long long*) d->data;
                #endif
                __m256i key, h, off, inv = _mm256_set1_epi32(-1);
                __m256i fixed_off = _mm256_set1_epi32(-1);
                uint32_t hashs[size];
                __m256i vec;


                for (size_t i=0; i<size; i+=8)
                {
                    __m256i vec = _mm256_maskload_epi32((const int*) &(keys[i]), fixed_off);
                    vec = _mm256_mullo_epi32(vec, factor);
                    vec = _mm256_srl_epi32(vec, shift);
                    vec = _mm256_and_si256(vec, buckets_minus_1);   
                    for (size_t j = 0; j < 8; j++)
                    {
                        hashs[i+j] = _mm256_extract_epi32(vec, j);
                        _mm_prefetch(&table_64[hashs[i+j]], _MM_HINT_T0);
                    }
                }

                while (i + 8 <= size) {
                    // load new items (mask out reloads)
                    __m256i new_key = _mm256_maskload_epi32((const int*) &(keys[i]), inv);
                    key = _mm256_andnot_si256(inv, key);
                    key = _mm256_or_si256(key, new_key);
                    // hash
                    off = _mm256_set1_epi32(1);
                    off = _mm256_andnot_si256(inv, off);

                    __m256i new_hash = _mm256_maskload_epi32((const int*) &hashs[i], inv);
                    h = _mm256_andnot_si256(inv, h);
                    h = _mm256_or_si256(h, new_hash);
                    h = _mm256_add_epi32(h, off);
                    h = _mm256_and_si256(h, buckets_minus_1);
                    // gather
                    __m256i tab_lo = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_hi = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_key = _mm256_packlo_epi32(tab_lo, tab_hi);
                    __m256i tab_val = _mm256_packhi_epi32(tab_lo, tab_hi);
                    // update count & sum
                    __m256i out = _mm256_cmpeq_epi32(tab_key, key);
                    inv = _mm256_cmpeq_epi32(tab_key, empty);
                    inv = _mm256_or_si256(inv, out);
                    // load permutation masks
                    size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                    size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));
                    __m128i perm_inv_comp = _mm_loadl_epi64((__m128i*) &perm[j]);
                    __m128i perm_out_comp = _mm_loadl_epi64((__m128i*) &perm[k ^ 255]);
                    __m256i perm_inv = _mm256_cvtepi8_epi32(perm_inv_comp);
                    __m256i perm_out = _mm256_cvtepi8_epi32(perm_out_comp);
                    // permutation for invalid
                    inv = _mm256_permutevar8x32_epi32(inv, perm_inv);
                    key = _mm256_permutevar8x32_epi32(key, perm_inv);
                    h = _mm256_permutevar8x32_epi32(h, perm_inv);
                    off = _mm256_permutevar8x32_epi32(off, perm_inv);
                    i += _mm_popcnt_u64(j);
                    // permutation for output
                    out = _mm256_permutevar8x32_epi32(out, perm_out);
                    tab_val = _mm256_permutevar8x32_epi32(tab_val, perm_out);
                    _mm256_maskstore_epi32(&((int*)result)[o], out, tab_val);
                    o += _mm_popcnt_u64(k);
                }

                // extract last keys
                uint32_t l_keys[8];
                // uint32_t l_pays[8];
                _mm256_storeu_si256((__m256i*) l_keys, key);
                size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                j = 8 - _mm_popcnt_u64(j);
                i += j;
                while (i != size) l_keys[j++] = keys[i++];
                // process last keys
                const uint8_t s = 32 - log2(buckets);
                for (i = 0 ; i != j ; ++i) {
                    uint32_t k = l_keys[i];
                    size_t h = ((uint32_t) (k * d->hash_factor)) >> s;
                    while (-1 != (d->data[h]).key_) {
                        if (k == (d->data[h]).key_) {
                            _mm_stream_si32(&((int*)result)[o++], (d->data[h]).val_);
                            break;
                        }
                        h = (h + 1) & (buckets - 1);
                    }
                }
                tmp_out_count = o;
                //========================================================================================================

                for (int read_idx=0; read_idx<tmp_out_count; read_idx++)
                    (d->result)[out_count + read_idx] = tmp_joined[read_idx];

                out_count += tmp_out_count;
            }
            d->join_size = out_count;

        	pthread_barrier_wait(&d->barrier[bar++]);
            pthread_exit(NULL);

        }

        inline static void* find_all_simd_prefetching_inner_2(void* arg)
        {

            info_t *d = (info_t*) arg;
            assert(pthread_equal(pthread_self(), d->id));
            bind_thread(d->thread, d->threads);
            size_t joins, bar = 0;
            uint32_t *outer = d->outer;
            size_t outer_size = d->outer_size;

            pthread_barrier_wait(&d->barrier[bar++]);

            int step = d->group_size;
            int tmp_out_count = 0;
            size_t out_count = 0;        

            uint32_t tmp_joined[d->group_size];

            for (size_t read_idx = 0; read_idx<d->outer_size; read_idx+=d->group_size) 
            {
                if ((d->outer_size) - read_idx < (d->group_size))
                    step = (d->outer_size)%(d->group_size);

                //========================================================================================================
                auto keys = &(d->outer[read_idx]);
                auto& size = step;
                auto& result = tmp_joined;             

                size_t i = 0, o = 0, b = 0;
                assert(sizeof(d->data[0]) == 8);
                const size_t buckets = d->max_size;
                const __m128i shift = _mm_cvtsi32_si128(d->shift);
                const __m256i factor = _mm256_set1_epi32(d->hash_factor);
                const __m256i empty = _mm256_set1_epi32(-1);
                const __m256i buckets_minus_1 = _mm256_set1_epi32(buckets - 1);
                const __m256i mask_1 = _mm256_set1_epi32(1);
                #if defined __INTEL_COMPILER &&  __INTEL_COMPILER < 1600  
                    const long *table_64 = (const long*) table;
                #else
                    const long long *table_64 = (const long long*) d->data;
                #endif
                __m256i key, h, off, inv = _mm256_set1_epi32(-1);
                __m256i fixed_off = _mm256_set1_epi32(-1);
                // uint32_t hashs[size];
                __m256i vec;


                for (size_t i=0; i<size; i+=8)
                {
                    __m256i vec = _mm256_maskload_epi32((const int*) &(keys[i]), fixed_off);
                    vec = _mm256_mullo_epi32(vec, factor);
                    vec = _mm256_srl_epi32(vec, shift);
                    vec = _mm256_and_si256(vec, buckets_minus_1);   
                    for (size_t j = 0; j < 8; j++)
                    {
                        // hashs[i+j] = _mm256_extract_epi32(vec, j);
                        _mm_prefetch(&table_64[_mm256_extract_epi32(vec, j)], _MM_HINT_T0);
                    }
                }

                while (i + 8 <= size) {
                    // load new items (mask out reloads)
                    __m256i new_key = _mm256_maskload_epi32((const int*) &(keys[i]), inv);
                    key = _mm256_andnot_si256(inv, key);
                    key = _mm256_or_si256(key, new_key);
                    // hash
                    // __m256i new_hash = _mm256_maskload_epi32((const int*) &hashs[i], inv);
                    // off = _mm256_set1_epi32(1);
                    // off = _mm256_andnot_si256(inv, off);
                    // h = _mm256_andnot_si256(inv, h);
                    // h = _mm256_or_si256(h, new_hash);
                    // h = _mm256_add_epi32(h, off);
                    // h = _mm256_and_si256(h, buckets_minus_1);
                    __m256i h = _mm256_mullo_epi32(key, factor);
                    off = _mm256_add_epi32(off, mask_1);
                    off = _mm256_andnot_si256(inv, off);
                    h = _mm256_srl_epi32(h, shift);
                    h = _mm256_add_epi32(h, off);
                    h = _mm256_and_si256(h, buckets_minus_1);
                    // gather
                    __m256i tab_lo = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_hi = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_key = _mm256_packlo_epi32(tab_lo, tab_hi);
                    __m256i tab_val = _mm256_packhi_epi32(tab_lo, tab_hi);
                    // update count & sum
                    __m256i out = _mm256_cmpeq_epi32(tab_key, key);
                    inv = _mm256_cmpeq_epi32(tab_key, empty);
                    inv = _mm256_or_si256(inv, out);
                    // load permutation masks
                    size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                    size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));
                    __m128i perm_inv_comp = _mm_loadl_epi64((__m128i*) &perm[j]);
                    __m128i perm_out_comp = _mm_loadl_epi64((__m128i*) &perm[k ^ 255]);
                    __m256i perm_inv = _mm256_cvtepi8_epi32(perm_inv_comp);
                    __m256i perm_out = _mm256_cvtepi8_epi32(perm_out_comp);
                    // permutation for invalid
                    inv = _mm256_permutevar8x32_epi32(inv, perm_inv);
                    key = _mm256_permutevar8x32_epi32(key, perm_inv);
                    h = _mm256_permutevar8x32_epi32(h, perm_inv);
                    off = _mm256_permutevar8x32_epi32(off, perm_inv);
                    i += _mm_popcnt_u64(j);
                    // permutation for output
                    out = _mm256_permutevar8x32_epi32(out, perm_out);
                    tab_val = _mm256_permutevar8x32_epi32(tab_val, perm_out);
                    _mm256_maskstore_epi32(&((int*)result)[o], out, tab_val);
                    o += _mm_popcnt_u64(k);
                }

                // extract last keys
                uint32_t l_keys[8];
                // uint32_t l_pays[8];
                _mm256_storeu_si256((__m256i*) l_keys, key);
                size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                j = 8 - _mm_popcnt_u64(j);
                i += j;
                while (i != size) l_keys[j++] = keys[i++];
                // process last keys
                const uint8_t s = 32 - log2(buckets);
                for (i = 0 ; i != j ; ++i) {
                    uint32_t k = l_keys[i];
                    size_t h = ((uint32_t) (k * d->hash_factor)) >> s;
                    while (-1 != (d->data[h]).key_) {
                        if (k == (d->data[h]).key_) {
                            _mm_stream_si32(&((int*)result)[o++], (d->data[h]).val_);
                            break;
                        }
                        h = (h + 1) & (buckets - 1);
                    }
                }
                tmp_out_count = o;
                //========================================================================================================

                for (int read_idx=0; read_idx<tmp_out_count; read_idx++)
                    (d->result)[out_count + read_idx] = tmp_joined[read_idx];

                out_count += tmp_out_count;
            }
            d->join_size = out_count;

        	pthread_barrier_wait(&d->barrier[bar++]);
            pthread_exit(NULL);

        }

        inline static void* find_all_simd_prefetching_inner_3(void* arg)
        {
            
            info_t *d = (info_t*) arg;
            assert(pthread_equal(pthread_self(), d->id));
            bind_thread(d->thread, d->threads);
            size_t joins, bar = 0;
            uint32_t *outer = d->outer;
            size_t outer_size = d->outer_size;

            pthread_barrier_wait(&d->barrier[bar++]);

            int step = d->group_size;
            size_t out_count = 0;        

            size_t b = 0;
            const size_t buf_size = 128;
            int buf_space[buf_size + 8 + 15];
            int *buf = (int*)align((void*) buf_space);

            for (size_t read_idx = 0; read_idx<d->outer_size; read_idx+=d->group_size) 
            {
                if ((d->outer_size) - read_idx < (d->group_size))
                    step = (d->outer_size)%(d->group_size);

                //========================================================================================================
                auto keys = &(d->outer[read_idx]);
                auto& size = step;
                size_t i = 0, o = 0;
                assert(sizeof(d->data[0]) == 8);
                const size_t buckets = d->max_size;
                const __m128i shift = _mm_cvtsi32_si128(d->shift);
                const __m256i factor = _mm256_set1_epi32(d->hash_factor);
                const __m256i empty = _mm256_set1_epi32(-1);
                const __m256i buckets_minus_1 = _mm256_set1_epi32(buckets - 1);
                const __m256i mask_1 = _mm256_set1_epi32(1);
                #if defined __INTEL_COMPILER &&  __INTEL_COMPILER < 1600  
                    const long *table_64 = (const long*) table;
                #else
                    const long long *table_64 = (const long long*) d->data;
                #endif
                __m256i key, off, inv = _mm256_set1_epi32(-1);
                __m256i fixed_off = _mm256_set1_epi32(-1);
                __m256i vec;

                for (size_t i=0; i<size; i+=8)
                {
                    __m256i vec = _mm256_maskload_epi32((const int*) &(keys[i]), fixed_off);
                    vec = _mm256_mullo_epi32(vec, factor);
                    vec = _mm256_srl_epi32(vec, shift);
                    vec = _mm256_and_si256(vec, buckets_minus_1);   
                    for (size_t j = 0; j < 8; j++)
                    {
                        _mm_prefetch(&table_64[_mm256_extract_epi32(vec, j)], _MM_HINT_T0);
                    }
                }

                auto is_buf_flushed = false;
                while (i + 8 <= size) {
                    is_buf_flushed = false;
                    // load new items (mask out reloads)
                    __m256i new_key = _mm256_maskload_epi32((const int*) &(keys[i]), inv);
                    key = _mm256_andnot_si256(inv, key);
                    key = _mm256_or_si256(key, new_key);
                    // hash
                     __m256i h = _mm256_mullo_epi32(key, factor);
                    off = _mm256_add_epi32(off, mask_1);
                    off = _mm256_andnot_si256(inv, off);
                    h = _mm256_srl_epi32(h, shift);
                    h = _mm256_add_epi32(h, off);
                    h = _mm256_and_si256(h, buckets_minus_1);
                    // gather
                    __m256i tab_lo = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_hi = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    __m256i tab_key = _mm256_packlo_epi32(tab_lo, tab_hi);
                    __m256i tab_val = _mm256_packhi_epi32(tab_lo, tab_hi);
                    // update count & sum
                    inv = _mm256_cmpeq_epi32(tab_key, empty);
                    __m256i out = _mm256_cmpeq_epi32(tab_key, key);
                    inv = _mm256_or_si256(inv, out);
                    // load permutation masks
                    size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                    size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));
                    __m128i perm_inv_comp = _mm_loadl_epi64((__m128i*) &perm[j]);
                    __m128i perm_out_comp = _mm_loadl_epi64((__m128i*) &perm[k ^ 255]);
                    __m256i perm_inv = _mm256_cvtepi8_epi32(perm_inv_comp);
                    __m256i perm_out = _mm256_cvtepi8_epi32(perm_out_comp);
                    // permutation for invalid
                    inv = _mm256_permutevar8x32_epi32(inv, perm_inv);
                    key = _mm256_permutevar8x32_epi32(key, perm_inv);
                    off = _mm256_permutevar8x32_epi32(off, perm_inv);
                    i += _mm_popcnt_u64(j);
                    // permutation for output
                    tab_val = _mm256_permutevar8x32_epi32(tab_val, perm_out);
                    out = _mm256_permutevar8x32_epi32(out, perm_out);
                    _mm256_maskstore_epi32(&buf[b], out, tab_val);
                    b += _mm_popcnt_u64(k);
              
                    // flush buffer
                    if (b > buf_size) {
                        size_t b_i = 0;
                        do {
                            __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                            _mm256_stream_si256((__m256i*)(&(d->result)[out_count]), x);
                            b_i += 8;
                            out_count += 8;
                        } while (b_i != buf_size);
                        __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                        _mm256_store_si256((__m256i*) &buf[0], x);
                        b -= buf_size;
                        is_buf_flushed = true;
                    }
                }

                // extract last keys
                uint32_t l_keys[8];
                _mm256_storeu_si256((__m256i*) l_keys, key);
                size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                j = 8 - _mm_popcnt_u64(j);
                i += j;
                while (i != size) l_keys[j++] = keys[i++];
                // process last keys
                const uint8_t s = 32 - log2(buckets);
                for (i = 0 ; i != j ; ++i) {
                    uint32_t k = l_keys[i];
                    size_t h = ((uint32_t) (k * d->hash_factor)) >> s;
                    while (-1 != (d->data[h]).key_) {
                        if (k == (d->data[h]).key_) {
                            _mm_stream_si32(&buf[b++], (d->data[h]).val_);
                            break;
                        }
                        h = (h + 1) & (buckets - 1);
                    }
                }
                if (is_buf_flushed && b > 8)
                {
                    __m256i x = _mm256_load_si256((__m256i*) &buf[0]);
                    _mm256_stream_si256((__m256i*)(&(d->result)[out_count]), x);
                    out_count += 8;
                    x = _mm256_load_si256((__m256i*) &buf[8]);
                    _mm256_store_si256((__m256i*) &buf[0], x);
                    b -= 8;
                }
                else if (!is_buf_flushed && b > buf_size)
                {
                    size_t b_i = 0;
                    do {
                        __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                        _mm256_stream_si256((__m256i*)(&(d->result)[out_count]), x);
                        b_i += 8;
                        out_count += 8;
                    } while (b_i != buf_size);
                    __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                    _mm256_store_si256((__m256i*) &buf[0], x);
                    b -= buf_size;
                }
                //========================================================================================================
            }
            // clean buffer
            size_t b_i = 0;
            while (b_i != b)_mm_stream_si32(&((int*)d->result)[out_count++], buf[b_i++]);                

            d->join_size = out_count;

        	pthread_barrier_wait(&d->barrier[bar++]);
            pthread_exit(NULL);

        }

        inline static void* find_all_simd_prefetching_inner_10(void* arg)
        {

            info_t *d = (info_t*) arg;
            assert(pthread_equal(pthread_self(), d->id));
            bind_thread(d->thread, d->threads);
            size_t joins, bar = 0;
            uint32_t *outer = d->outer;
            size_t outer_size = d->outer_size;

            pthread_barrier_wait(&d->barrier[bar++]);

            int step = d->group_size;
            int tmp_out_count = 0;
            size_t out_count = 0;        

            uint32_t tmp_joined[d->group_size];

            for (size_t read_idx = 0; read_idx<d->outer_size; read_idx+=d->group_size) 
            {
                if ((d->outer_size) - read_idx < (d->group_size))
                    step = (d->outer_size)%(d->group_size);

                //========================================================================================================
                auto keys = &(d->outer[read_idx]);
                auto& size = step;
                auto& result = tmp_joined;             

                size_t i = 0, o = 0, b = 0;
                assert(sizeof(d->data[0]) == 8);
                const size_t buckets = d->max_size;
                const __m128i shift = _mm_cvtsi32_si128(d->shift);
                const __m256i factor = _mm256_set1_epi32(d->hash_factor);
                const __m256i empty = _mm256_set1_epi32(-1);
                const __m256i buckets_minus_1 = _mm256_set1_epi32(buckets - 1);
                const __m256i mask_1 = _mm256_set1_epi32(1);
                #if defined __INTEL_COMPILER &&  __INTEL_COMPILER < 1600  
                    const long *table_64 = (const long*) table;
                #else
                    const long long *table_64 = (const long long*) d->data;
                #endif
                __m256i key, h, off, inv = _mm256_set1_epi32(-1);
                __m256i fixed_off = _mm256_set1_epi32(-1);
                uint32_t hashs[size];
                __m256i vec;


                for (size_t i=0; i<size; i+=8)
                {
                    __m256i vec = _mm256_maskload_epi32((const int*) &(keys[i]), fixed_off);
                    vec = _mm256_mullo_epi32(vec, factor);
                    vec = _mm256_srl_epi32(vec, shift);
                    vec = _mm256_and_si256(vec, buckets_minus_1);   
                    for (size_t j = 0; j < 8; j++)
                    {
                        hashs[i+j] = _mm256_extract_epi32(vec, j);
                        _mm_prefetch(&table_64[hashs[i+j]], _MM_HINT_T0);
                        _mm_prefetch(&table_64[hashs[i+j]+1], _MM_HINT_T0);
                    }
                }

                while (i + 8 <= size) {
                    // load new items (mask out reloads)
                    __m256i new_key = _mm256_maskload_epi32((const int*) &(keys[i]), inv);
                    key = _mm256_andnot_si256(inv, key);
                    key = _mm256_or_si256(key, new_key);
                    // hash
                    off = _mm256_set1_epi32(1);
                    off = _mm256_andnot_si256(inv, off);

                    __m256i new_hash = _mm256_maskload_epi32((const int*) &hashs[i], inv);
                    h = _mm256_andnot_si256(inv, h);
                    h = _mm256_or_si256(h, new_hash);
                    h = _mm256_add_epi32(h, off);
                    h = _mm256_and_si256(h, buckets_minus_1);
                    // gather
                    __m256i tab_lo = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_hi = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_key = _mm256_packlo_epi32(tab_lo, tab_hi);
                    __m256i tab_val = _mm256_packhi_epi32(tab_lo, tab_hi);
                    // update count & sum
                    __m256i out = _mm256_cmpeq_epi32(tab_key, key);
                    inv = _mm256_cmpeq_epi32(tab_key, empty);
                    inv = _mm256_or_si256(inv, out);
                    // load permutation masks
                    size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                    size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));
                    __m128i perm_inv_comp = _mm_loadl_epi64((__m128i*) &perm[j]);
                    __m128i perm_out_comp = _mm_loadl_epi64((__m128i*) &perm[k ^ 255]);
                    __m256i perm_inv = _mm256_cvtepi8_epi32(perm_inv_comp);
                    __m256i perm_out = _mm256_cvtepi8_epi32(perm_out_comp);
                    // permutation for invalid
                    inv = _mm256_permutevar8x32_epi32(inv, perm_inv);
                    key = _mm256_permutevar8x32_epi32(key, perm_inv);
                    h = _mm256_permutevar8x32_epi32(h, perm_inv);
                    off = _mm256_permutevar8x32_epi32(off, perm_inv);
                    i += _mm_popcnt_u64(j);
                    // permutation for output
                    out = _mm256_permutevar8x32_epi32(out, perm_out);
                    tab_val = _mm256_permutevar8x32_epi32(tab_val, perm_out);
                    _mm256_maskstore_epi32(&((int*)result)[o], out, tab_val);
                    o += _mm_popcnt_u64(k);
                }

                // extract last keys
                uint32_t l_keys[8];
                // uint32_t l_pays[8];
                _mm256_storeu_si256((__m256i*) l_keys, key);
                size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                j = 8 - _mm_popcnt_u64(j);
                i += j;
                while (i != size) l_keys[j++] = keys[i++];
                // process last keys
                const uint8_t s = 32 - log2(buckets);
                for (i = 0 ; i != j ; ++i) {
                    uint32_t k = l_keys[i];
                    size_t h = ((uint32_t) (k * d->hash_factor)) >> s;
                    while (-1 != (d->data[h]).key_) {
                        if (k == (d->data[h]).key_) {
                            _mm_stream_si32(&((int*)result)[o++], (d->data[h]).val_);
                            break;
                        }
                        h = (h + 1) & (buckets - 1);
                    }
                }
                tmp_out_count = o;
                //========================================================================================================

                for (int read_idx=0; read_idx<tmp_out_count; read_idx++)
                    (d->result)[out_count + read_idx] = tmp_joined[read_idx];

                out_count += tmp_out_count;
            }
            d->join_size = out_count;

        	pthread_barrier_wait(&d->barrier[bar++]);
            pthread_exit(NULL);

        }

        inline static void* find_all_simd_prefetching_inner_20(void* arg)
        {
            info_t *d = (info_t*) arg;
            assert(pthread_equal(pthread_self(), d->id));
            bind_thread(d->thread, d->threads);
            size_t joins, bar = 0;
            uint32_t *outer = d->outer;
            size_t outer_size = d->outer_size;

            pthread_barrier_wait(&d->barrier[bar++]);

            size_t i = 0, o = 0, b = 0;
                assert(sizeof(d->data[0]) == 8);
                const size_t buckets = d->max_size;
                const __m128i shift = _mm_cvtsi32_si128(d->shift);
                const __m256i factor = _mm256_set1_epi32(d->hash_factor);
                const __m256i empty = _mm256_set1_epi32(-1);
                const __m256i buckets_minus_1 = _mm256_set1_epi32(d->max_size - 1);
                const __m256i mask_1 = _mm256_set1_epi32(1);
            #if defined __INTEL_COMPILER &&  __INTEL_COMPILER < 1600
                const long *table_64 = (const long*) table;
            #else
                const long long *table_64 = (const long long*) d->data;
            #endif
                const size_t buf_size = 128;
                int buf_space[buf_size + 8 + 15];
                int *buf = (int*)align((void*) buf_space);
                __m256i key, off, inv = _mm256_set1_epi32(-1);
                while (i + 8 <= outer_size) {
                    // load new items (mask out reloads)
                    __m256i new_key = _mm256_maskload_epi32((const int*) &(d->outer[i]), inv);
                    key = _mm256_andnot_si256(inv, key);
                    key = _mm256_or_si256(key, new_key);
                    // hash
                    __m256i h = _mm256_mullo_epi32(key, factor);
                    off = _mm256_add_epi32(off, mask_1);
                    off = _mm256_andnot_si256(inv, off);
                    h = _mm256_srl_epi32(h, shift);
                    h = _mm256_add_epi32(h, off);
                    h = _mm256_and_si256(h, buckets_minus_1);


                    // Naiive Prefetch
                    for (size_t j = 0; j < 8; j++)
                    {
                        _mm_prefetch(&table_64[_mm256_extract_epi32(h, j)], _MM_HINT_T0);
                    }


                    // gather
                    __m256i tab_lo = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    h = _mm256_permute4x64_epi64(h, _MM_SHUFFLE(1,0,3,2));
                    __m256i tab_hi = _mm256_i32gather_epi64(table_64, _mm256_castsi256_si128(h), 8);
                    __m256i tab_key = _mm256_packlo_epi32(tab_lo, tab_hi);
                    __m256i tab_val = _mm256_packhi_epi32(tab_lo, tab_hi);
                    // update count & sum
                    inv = _mm256_cmpeq_epi32(tab_key, empty);
                    __m256i out = _mm256_cmpeq_epi32(tab_key, key);
                    inv = _mm256_or_si256(inv, out);
                    // load permutation masks
                    size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                    size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));
                    __m128i perm_inv_comp = _mm_loadl_epi64((__m128i*) &perm[j]);
                    __m128i perm_out_comp = _mm_loadl_epi64((__m128i*) &perm[k ^ 255]);
                    __m256i perm_inv = _mm256_cvtepi8_epi32(perm_inv_comp);
                    __m256i perm_out = _mm256_cvtepi8_epi32(perm_out_comp);
                    // permutation for invalid
                    inv = _mm256_permutevar8x32_epi32(inv, perm_inv);
                    key = _mm256_permutevar8x32_epi32(key, perm_inv);
                    off = _mm256_permutevar8x32_epi32(off, perm_inv);
                    i += _mm_popcnt_u64(j);
                    // permutation for output
                    tab_val = _mm256_permutevar8x32_epi32(tab_val, perm_out);
                    out = _mm256_permutevar8x32_epi32(out, perm_out);
                    _mm256_maskstore_epi32(&buf[b], out, tab_val);
                    b += _mm_popcnt_u64(k);
                    // flush buffer
                    if (b > buf_size) {
                        size_t b_i = 0;
                        do {
                            __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                            _mm256_stream_si256((__m256i*) &(d->result[o]), x);
                            b_i += 8;
                            o += 8;
                        } while (b_i != buf_size);
                        __m256i x = _mm256_load_si256((__m256i*) &buf[b_i]);
                        _mm256_store_si256((__m256i*) &buf[0], x);
                        b -= buf_size;
                    }
                }
                // clean buffer
                size_t b_i = 0;
                while (b_i != b) _mm_stream_si32(&((int*) d->result)[o++], buf[b_i++]);
                // extract last keys
                uint32_t l_keys[8];
                _mm256_storeu_si256((__m256i*) l_keys, key);
                size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                j = 8 - _mm_popcnt_u64(j);
                i += j;
                while (i != outer_size) l_keys[j++] = d->outer[i++];
                // process last keys
                const uint8_t s = 32 - log2(buckets);
                for (i = 0 ; i != j ; ++i) {
                    uint32_t k = l_keys[i];
                    size_t h = ((uint32_t) (k * d->hash_factor)) >> s;
                    while (-1 != d->data[h].key_) {
                        if (k == d->data[h].key_) {
                            _mm_stream_si32(&((int*) d->result)[o++], d->data[h].val_);
                            break;
                        }
                        h = (h + 1) & (buckets - 1);
                    }
                }

            d->join_size = o;
       
        	pthread_barrier_wait(&d->barrier[bar++]);
            pthread_exit(NULL);
        };

        // Parallel APIs =============================================================
        public:

        inline size_t find_all_scalar(uint32_t* keys, const size_t& size, uint32_t** result)
        {
            return find_parallel_executer(keys, size, result, find_all_scalar_inner);
        }

        inline size_t find_all_scalar_prefetching(uint32_t* keys, const size_t& size, uint32_t** result)
        {
            return find_parallel_executer(keys, size, result, find_all_scalar_prefetching_inner);
        }

        inline size_t find_all_simd(uint32_t* keys, const size_t& size, uint32_t** result)
        {
            return find_parallel_executer(keys, size, result, find_all_simd_inner);
        };

        inline size_t find_all_simd_prefetching(uint32_t* keys, const size_t& size, uint32_t** result, int type)
        {
            if (type == 0) // Memo + Buffer
                return find_parallel_executer(keys, size, result, find_all_simd_prefetching_inner_0);
            else if (type == 1) // Memo + No Buffer (Selected)
                return find_parallel_executer(keys, size, result, find_all_simd_prefetching_inner_1);
            else if (type == 2) // No Memo + No Buffer
                return find_parallel_executer(keys, size, result, find_all_simd_prefetching_inner_2);
            else if (type == 3) // No Memo + Buffer
                return find_parallel_executer(keys, size, result, find_all_simd_prefetching_inner_3);
            else if (type == 10) // Pessimistic Prefetch
                return find_parallel_executer(keys, size, result, find_all_simd_prefetching_inner_10);
            else if (type == 20) // Naive Prefetch
                return find_parallel_executer(keys, size, result, find_all_simd_prefetching_inner_20);
            else
                throw std::runtime_error("invalid simd type!!!");
        }

    };

}