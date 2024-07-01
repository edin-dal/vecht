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
#include <vector>
#include <thread>
#include <iomanip>
#include <cmath>
#include "tbb/tbb.h"

namespace vecht
{
    using K = uint32_t; using V = uint32_t; using P = uint32_t;
    using no_func_type = nullptr_t;
    
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

    static void *mamalloc(size_t size)
    {
        void *p = NULL;
        return posix_memalign(&p, 64, size) ? NULL : p;
    }

    // =================================================================================================

    typedef struct {
        K key_;
        V val_;
    } bucket;

    class iter_batch
    {
    private:

        size_t max_size_;
        size_t threads_;
        std::vector<size_t> threads_res_size_;
    
    public:
        bool for_zip_;
        uint32_t** keys_ = nullptr;
        uint32_t** values_ = nullptr;
        uint32_t** payloads_ = nullptr;

        iter_batch (size_t max_size, size_t threads, bool for_zip=false)
        {
            threads_ = threads;
            max_size_ = max_size;
            for_zip_ = for_zip;

            keys_ = new uint32_t*[threads_];
            values_ = new uint32_t*[threads_];
            if (for_zip)
                payloads_ = new uint32_t*[threads_];

            for (int i = 0; i < threads_; i++)
            {
                keys_[i] = (uint32_t*) mamalloc(max_size * sizeof(uint32_t));
                values_[i] = (uint32_t*) mamalloc(max_size * sizeof(uint32_t));
                if (for_zip)
                    payloads_[i] = (uint32_t*) mamalloc(max_size * sizeof(uint32_t));

                memset(keys_[i], 0x00, max_size * sizeof(uint32_t));
                memset(values_[i], 0x00, max_size * sizeof(uint32_t));
                if (for_zip)
                    memset(payloads_[i], 0x00, max_size * sizeof(uint32_t));
            }          
          
            threads_res_size_.resize(threads_);
        }

        ~iter_batch ()
        {
            for (int i = 0; i < threads_; i++)
            {
                delete[] keys_[i];
                delete[] values_[i];
                if (for_zip_)
                    delete[] payloads_[i];
            }

            delete[] keys_;
            delete[] values_;
            if (for_zip_)
                delete[] payloads_;
        }

        void print ()
        {
            for (size_t i=0; i<threads_; i++)
            {
                std::cout << "Thread " << i << " (" << threads_res_size_[i] << "): " << std::endl;
                for (size_t j=0; j<max_size_; j++)
                    if (for_zip_)
                        std::cout << "K: " << keys_[i][j] << " | V: " << values_[i][j] << " | P: " << payloads_[i][j] << std::endl;
                    else
                        std::cout << "K: " << keys_[i][j] << " | V: " << values_[i][j] << std::endl;
            }
        }

        void reset ()
        {
            for (int i = 0; i < threads_; i++)
            {
                memset(keys_[i], 0x00, max_size_ * sizeof(uint32_t));
                memset(values_[i], 0x00, max_size_ * sizeof(uint32_t));
                if (for_zip_)
                    memset(payloads_[i], 0x00, max_size_ * sizeof(uint32_t));
            }  

            for (size_t i=0; i<threads_; i++)
                threads_res_size_[i] = 0;
        }

        inline void set_thread_size (size_t thread_id, size_t size)
        {
            threads_res_size_[thread_id] += size;
        }
    
        inline void foreach (std::function<void(K& key, V& value)> func)
        {
            if (for_zip_ == true)
                throw std::runtime_error("iter_batch::foreach_parallel: for_zip must be false.");
            for (size_t i=0; i<threads_; i++)
                for (size_t j=0; j<threads_res_size_[i]; j++)
                    func(keys_[i][j], values_[i][j]);
        }
    
        inline void foreach (std::function<void(K& key, V& value, P& payload)> func)
        {
            if (for_zip_ == false)
                throw std::runtime_error("iter_batch::foreach_parallel: for_zip must be true.");
            for (size_t i=0; i<threads_; i++)
            {
                for (size_t j=0; j<threads_res_size_[i]; j++)
                {
                    func(keys_[i][j], values_[i][j], payloads_[i][j]);
                }
            }
        }
    
        inline void foreach_parallel (std::function<void(K& key, V& value)> func)
        {
            if (for_zip_ == true)
                throw std::runtime_error("iter_batch::foreach_parallel: for_zip must be false.");
            if (threads_ == 1)
                throw std::runtime_error("iter_batch::foreach_parallel: threads must be > 1.");

            auto range = std::vector<size_t>(threads_);
            for (size_t i=0; i<threads_; i++) range[i] = i;
            tbb::parallel_for_each (range, [&](size_t thread_id)
            {
                for (size_t j=0; j<threads_res_size_[thread_id]; j++)
                    func(keys_[thread_id][j], values_[thread_id][j]);
            });
        }
    
        inline void foreach_parallel (std::function<void(K& key, V& value, P& payload)> func)
        {
            if (for_zip_ == false)
                throw std::runtime_error("iter_batch::foreach_parallel: for_zip must be true.");
            if (threads_ == 1)
                throw std::runtime_error("iter_batch::foreach_parallel: threads must be > 1.");

            auto range = std::vector<size_t>(threads_);
            for (size_t i=0; i<threads_; i++) range[i] = i;
            tbb::parallel_for_each(range, [&](size_t thread_id)
            {
                for (size_t j=0; j<threads_res_size_[thread_id]; j++)
                    func(keys_[thread_id][j], values_[thread_id][j], payloads_[thread_id][j]);
            });
        }
    };

    class lp_map
    {
    private:

        size_t size_;
        size_t threads_;
        uint32_t group_size_;
        uint32_t hash_factor_;
        uint8_t shift_;
        uint32_t empty_key_;
        bucket * entries_;

        template<typename FUNC_TYPE>
        inline size_t find_batch_inner(uint32_t* keys, uint32_t* payloads, size_t size, bool complement, FUNC_TYPE func, iter_batch* res_iter, int tid=0)
        {
            size_t joins, bar = 0;
            uint32_t *outer = keys;
            size_t outer_size = size;

            int step = group_size_;
            int tmp_out_count = 0;
            size_t out_count = 0;        

            bool is_zip = (payloads==NULL) ? false : true; 

            uint32_t tmp_joined_keys[group_size_];
            uint32_t tmp_joined_vals[group_size_];
            uint32_t tmp_joined_pays[group_size_];
            
            for (size_t read_idx = 0; read_idx<outer_size; read_idx+=group_size_) 
            {
                if ((outer_size) - read_idx < (group_size_))
                    step = (outer_size)%(group_size_);

                //========================================================================================================
                auto keys = &(outer[read_idx]);

                P* pays = NULL;
                if (is_zip)
                    pays = &(payloads[read_idx]);
                
                auto& size = step;

                auto& result_keys = tmp_joined_keys;
                auto& result_vals = tmp_joined_vals;
                auto& result_pays = tmp_joined_pays;             

                size_t i = 0, o = 0, b = 0;

                const size_t buckets = size_;
                const __m128i shift = _mm_cvtsi32_si128(shift_);
                const __m256i factor = _mm256_set1_epi32(hash_factor_);
                const __m256i empty = _mm256_set1_epi32(-1);
                const __m256i buckets_minus_1 = _mm256_set1_epi32(buckets - 1);
                const __m256i mask_1 = _mm256_set1_epi32(1);
                #if defined __INTEL_COMPILER &&  __INTEL_COMPILER < 1600  
                    const long *table_64 = (const long*) table;
                #else
                    const long long *table_64 = (const long long*) entries_;
                #endif
                __m256i key, pay, key_out, val_out, pay_out, h, off, inv = _mm256_set1_epi32(-1);
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

                    __m256i new_pay;
                    if (is_zip)
                    {
                        new_pay = _mm256_maskload_epi32((const int*) &(pays[i]), inv);
                        pay = _mm256_andnot_si256(inv, pay);
                        pay = _mm256_or_si256(pay, new_pay);
                    }

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
                    __m256i out;

                    if (complement)
                    {
                        inv = _mm256_cmpeq_epi32(tab_key, key);
                        out = _mm256_cmpeq_epi32(tab_key, empty);
                    }
                    else
                    {
                        inv = _mm256_cmpeq_epi32(tab_key, empty);
                        out = _mm256_cmpeq_epi32(tab_key, key);
                    }

                    inv = _mm256_or_si256(inv, out);
                    // load permutation masks
                    size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                    size_t k = _mm256_movemask_ps(_mm256_castsi256_ps(out));
                    __m128i perm_inv_comp = _mm_loadl_epi64((__m128i*) &perm[j]);
                    __m128i perm_out_comp = _mm_loadl_epi64((__m128i*) &perm[k ^ 255]);
                    __m256i perm_inv = _mm256_cvtepi8_epi32(perm_inv_comp);
                    __m256i perm_out = _mm256_cvtepi8_epi32(perm_out_comp);

                    h = _mm256_permutevar8x32_epi32(h, perm_inv);
                    off = _mm256_permutevar8x32_epi32(off, perm_inv);
                    i += _mm_popcnt_u64(j);
                    // permutation for output
                    out = _mm256_permutevar8x32_epi32(out, perm_out);

                    key_out = _mm256_permutevar8x32_epi32(key, perm_out);
                    val_out = _mm256_permutevar8x32_epi32(tab_val, perm_out);                
                    pay_out = _mm256_permutevar8x32_epi32(pay, perm_out);

                    // permutation for invalid
                    inv = _mm256_permutevar8x32_epi32(inv, perm_inv);
                    key = _mm256_permutevar8x32_epi32(key, perm_inv);
                    pay = _mm256_permutevar8x32_epi32(pay, perm_inv);
    
                    _mm256_maskstore_epi32(&((int*)result_keys)[o], out, key_out);
                    _mm256_maskstore_epi32(&((int*)result_vals)[o], out, val_out);
                    if (is_zip)
                        _mm256_maskstore_epi32(&((int*)result_pays)[o], out, pay_out);
                    o += _mm_popcnt_u64(k);
                }

                // extract last keys
                uint32_t l_keys[8];
                uint32_t l_pays[8];
                _mm256_storeu_si256((__m256i*) l_keys, key);
                if (is_zip)
                    _mm256_storeu_si256((__m256i*) l_pays, pay);
                size_t j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
                j = 8 - _mm_popcnt_u64(j);
                i += j;
                while (i != size) 
                {
                    l_keys[j] = keys[i];
                    if (is_zip)
                        l_pays[j] = pays[i];
                    i++;j++;
                }
                // process last keys
                const uint8_t s = 32 - log2(buckets);

                if (!complement)
                {
                    for (i = 0 ; i != j ; ++i) {
                        uint32_t k = l_keys[i];
                        size_t h = ((uint32_t) (k * hash_factor_)) >> s;
                        while (-1 != (entries_[h]).key_) {
                            if (k == (entries_[h]).key_) {
                                _mm_stream_si32(&((int*)result_keys)[o], l_keys[i]);
                                _mm_stream_si32(&((int*)result_vals)[o], (entries_[h]).val_);
                                if (is_zip)
                                    _mm_stream_si32(&((int*)result_pays)[o], l_pays[i]);
                                o++;
                                break;
                            }
                            h = (h + 1) & (buckets - 1);
                        }
                    }
                }
                else
                {
                    for (i = 0 ; i != j ; ++i) {
                        uint32_t k = l_keys[i];
                        size_t h = ((uint32_t) (k * hash_factor_)) >> s;
                        while (-1 != (entries_[h]).key_) {
                            if (k == (entries_[h]).key_) {
                                break;
                            }
                            h = (h + 1) & (buckets - 1);
                        }
                        if (-1 == (entries_[h]).key_)
                        {
                            _mm_stream_si32(&((int*)result_keys)[o], l_keys[i]);
                            _mm_stream_si32(&((int*)result_vals)[o], (entries_[h]).val_);
                            if (is_zip)
                                _mm_stream_si32(&((int*)result_pays)[o], l_pays[i]);
                            o++;
                        }
                    }
                }
                tmp_out_count = o;
                //========================================================================================================

                for (int read_idx=0; read_idx<tmp_out_count; read_idx++)
                {
                    if constexpr (std::is_same_v<FUNC_TYPE, std::function<void(K&, V&)>const&>)
                        func(tmp_joined_keys[read_idx], tmp_joined_vals[read_idx]);
                    else if constexpr (std::is_same_v<FUNC_TYPE, std::function<void(K&, V&, P&)>const&>)
                        func(tmp_joined_keys[read_idx], tmp_joined_vals[read_idx], tmp_joined_pays[read_idx]);
                    else if constexpr (std::is_same_v<FUNC_TYPE, no_func_type>)
                    {                    
                        (res_iter->keys_)[tid][out_count + read_idx] = tmp_joined_keys[read_idx];
                        (res_iter->values_)[tid][out_count + read_idx] = tmp_joined_vals[read_idx];
                        if (is_zip)
                            (res_iter->payloads_)[tid][out_count + read_idx] = tmp_joined_pays[read_idx];
                    }
                }

                out_count += tmp_out_count;
            }
            
            if (threads_ == 1 and res_iter!=nullptr)
                res_iter->set_thread_size(0, out_count);
                

            return out_count;
        }

        template<typename FUNC_TYPE>
        inline size_t parallel_dispatcher(uint32_t* keys, uint32_t* payloads, size_t size, bool complement, FUNC_TYPE func, iter_batch* res_iter)
        {

            if constexpr (std::is_same_v<FUNC_TYPE, no_func_type>)
            {
                std::vector<size_t> counts(threads_);
                
                std::vector<size_t> thread_ids(threads_);
                for (int i=0; i<threads_; i++)
                    thread_ids[i] = i;

                tbb::parallel_for_each(thread_ids, [&](const size_t& tid) 
                {
                    size_t start =  tid * (size/threads_);
                    size_t local_size = size/threads_;
                    if (tid == threads_-1)
                        local_size = size - start;

                    if (payloads == NULL)   
                        counts[tid] += find_batch_inner<no_func_type>((uint32_t*)&keys[start], NULL, local_size, complement, nullptr, res_iter, tid);
                    else
                        counts[tid] += find_batch_inner<no_func_type>((uint32_t*)&keys[start], (uint32_t*)&payloads[start], local_size, complement, nullptr, res_iter, tid);

                    res_iter->set_thread_size(tid, counts[tid]);
                });
                
                size_t total_count = 0;
                
                for (auto& c : counts)
                    total_count += c;
                
                return total_count;
            }
            else 
            {
                tbb::enumerable_thread_specific<size_t> counts(0);
                
                tbb::parallel_for(tbb::blocked_range < size_t > (0, size), [&](const tbb::blocked_range<size_t>& range) 
                {
                    auto& count = counts.local();
                    size_t start = range.begin();
                    size_t local_size = range.end() - start;

                    if constexpr (std::is_same_v<FUNC_TYPE, std::function<void(K&, V&)>const&>)
                    {
                        if (payloads == NULL) 
                            count += find_batch_inner<std::function<void(K&, V&)>const&>((uint32_t*)&keys[start], NULL, local_size, complement, func, nullptr, -1);
                        else
                            count += find_batch_inner<std::function<void(K&, V&)>const&>((uint32_t*)&keys[start], (uint32_t*)&payloads[start], local_size, complement, func, nullptr, -1);
                    }
                    else if constexpr (std::is_same_v<FUNC_TYPE, std::function<void(K&, V&, P&)>const&>)
                    {
                        if (payloads == NULL) 
                            count += find_batch_inner<std::function<void(K&, V&, P&)>const&>((uint32_t*)&keys[start], NULL, local_size, complement, func, nullptr, -1);
                        else
                            count += find_batch_inner<std::function<void(K&, V&, P&)>const&>((uint32_t*)&keys[start], (uint32_t*)&payloads[start], local_size, complement, func, nullptr, -1);
                    }
                });
                size_t total_count = 0;
                
                for (auto& c : counts)
                    total_count += c;
                
                return total_count;
            }
        }

    public:

        lp_map (size_t size , size_t group_size=64, size_t threads=1)
        {
            size_ = 1 << size_t(ceil(log2(size)));
            hash_factor_ = (rand() << 1) | 1;
            shift_ = 32 - log2(size_);
            threads_ = threads;
            entries_ = new bucket[size_];
            group_size_ = group_size;
            for (size_t i = 0 ; i != size_ ; ++i)
            {
                entries_[i].key_ = -1;
                entries_[i].val_ = -1;
            }
        }

        ~lp_map()
        {
            delete [] entries_;
        }

        void print()
        {
            std::cout << "size: " << size_ << std::endl;
            std::cout << "threads: " << threads_ << std::endl;
            for (int i=0; i<size_; i++)
                std::cout << entries_[i].key_ << " | " << entries_[i].val_ << std::endl;
        }

        void clear()
        {
            delete [] entries_;
            entries_ = new bucket[size_];
            for (size_t i = 0 ; i != size_ ; ++i)
                entries_[i].key_ = -1;
        }

        // ---- Non - Batch APIs - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        inline bool insert (const K& key, const V& value)
        {
            try
            {              
                size_t h = ((uint32_t) (key * hash_factor_)) >> shift_;
                while (entries_[h].key_ != -1)
                    h = (h + 1) & (size_ - 1);
                entries_[h].key_ = key;
                entries_[h].val_ = value;
            }
            catch(const std::exception& e)
            {
                return false;
            }
            return true;
        }

        bucket find (const K& key)
        {
            size_t h = ((uint32_t) (key * hash_factor_)) >> shift_;
            bucket tmp = entries_[h];
            if (key == tmp.key_)
                return tmp;
            else while (-1 != tmp.key_) {
                h = (h + 1) & (size_ - 1);
                tmp = entries_[h];
                if (key == tmp.key_)
                    return tmp;
            }
            return tmp;
        }

        // ---- Batch APIs - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        inline size_t insert_batch (uint32_t* keys, uint32_t* values, size_t size)
        {
            try
            {
                for (size_t i = 0 ; i != size ; ++i) {
                    K key = keys[i];
                    V val = values[i];
                    size_t h = ((uint32_t) (key * hash_factor_)) >> shift_;
                    while (entries_[h].key_ != -1)
                        h = (h + 1) & (size_ - 1);
                    entries_[h].key_ = key;
                    entries_[h].val_ = val;
                }
            }
            catch(const std::exception& e)
            {
                return false;
            }
            return true;
        }

        inline size_t find_batch (uint32_t* keys, size_t size, bool complement, iter_batch* res_it)
        {
            if (threads_ == 1)
                return find_batch_inner<no_func_type>(keys, NULL, size, complement, nullptr, res_it);
            else
                return parallel_dispatcher<no_func_type>(keys, NULL, size, complement, nullptr, res_it);
        }
        
        inline size_t find_batch_apply(uint32_t* keys, size_t size, bool complement, std::function<void(K& key, V& value)>const& func)
        {
            if (threads_ == 1)
                return find_batch_inner<std::function<void(K& key, V& value)>const&>(keys, NULL, size, complement, func, nullptr);
            else
                return parallel_dispatcher<std::function<void(K& key, V& value)>const&>(keys, NULL, size, complement, func, nullptr);
        }

        inline size_t zip(uint32_t* keys, uint32_t* payloads, size_t size, bool complement, iter_batch* res_it)
        {
            if (threads_ == 1)
                return find_batch_inner<no_func_type>(keys, payloads, size, complement, nullptr, res_it);
            else
                return parallel_dispatcher<no_func_type>(keys, payloads, size, complement, nullptr, res_it);
        }
        
        inline size_t zip_apply(uint32_t* keys, uint32_t* payloads, size_t size, bool complement, std::function<void(K& key, V& value, P& payload)>const& func)
        {
            if (threads_ == 1)
                return find_batch_inner<std::function<void(K& key, V& value, P& payload)>const&>(keys, payloads, size, complement, func, nullptr);
            else
                return parallel_dispatcher<std::function<void(K& key, V& value, P& payload)>const&>(keys, payloads, size, complement, func, nullptr);
        }
    
   };
}