#include <iostream>
#include <iomanip>
#include <numeric>
#include "tbb/tbb.h"


#include "../include/inner_outer.h"
#include "../include/parallel_hashmap/phmap.h"
#include "../include/libcuckoo/libcuckoo/cuckoohash_map.hh"

#include "../include/high_precision_timer.h"
#include "../src/vecht.hpp"
#include "../src/vecht_experimental.hpp"

using namespace std;

size_t iterations = 5;
size_t warmup = 5;
bool verbose = false;

uint64_t sum(const uint32_t *data, size_t size)
{
	size_t i;
	uint64_t sum = 0;
	for (i = 0 ; i != size ; ++i)
    {
		sum += data[i];
    }
	return sum;
}

uint64_t sumsum(uint32_t **data, size_t size, int threads)
{
	int i, j=0;
	uint64_t sum = 0;
	for (i = 0 ; i != threads ; ++i)
    {
        for (j=0;j<size;j++)
        {
            sum += data[i][j];
        }
    }
	return sum;
}

uint64_t sumsum_vector(tbb::enumerable_thread_specific<vector<uint32_t>> data)
{
    int64_t sum = 0;
	for (auto& local : data)
    {
        for (auto& i : local)
            sum += i;
    }
	return sum;
}

void print(string title, uint64_t outer, uint64_t time, bool verbose=true)
{
    if (verbose)
        cout << title << ": " << (outer/1000000.0) / (time/1000.0) << endl;
    else
        cout << (outer/1000000.0) / (time/1000.0) << ",";
}

static void *mamalloc(size_t size)
{
	void *p = NULL;
	return posix_memalign(&p, 64, size) ? NULL : p;
}

int main(int argc, char *argv[])
{

    phmap::flat_hash_map<string, vector<float>> times;
    phmap::flat_hash_map<string, float> final_times;
    int log_inner_bytes = stoi(argv[1]); // 20;
    int outer_size = stoi(argv[2]);// 1000 * 1000 * 1000;
    bool is_first_call = stoi(argv[6]);// 1;

    //========================================================================================

    for (int iteration = 0; iteration < iterations+warmup; iteration++)
    {        
        float selectivity = stof(argv[3]);// 1;
        float load_factor = stof(argv[4]);// 0.5;
        int threads = stoi(argv[5]);// 1-8;
        uint32_t hash_factor = (rand() << 1);
        size_t size = (1 << (log_inner_bytes-3));
        
        tbb::task_scheduler_init scheduler(threads);
        //========================================================================================
        auto ht = vecht_experimental::lp_map(size, (rand() << 1) | 1, threads);
        auto ht_pphmap = phmap::parallel_flat_hash_map<uint32_t, uint32_t, phmap::priv::hash_default_hash<uint32_t>, phmap::priv::hash_default_eq<uint32_t>, std::allocator<std::pair<const uint32_t, uint32_t>>, 4, std::mutex>(size);
        auto ht_tbb = tbb::concurrent_unordered_map<uint32_t, uint32_t, std::hash<uint32_t>, std::equal_to<uint32_t>>(size);
        auto ht_cuckoo = libcuckoo::cuckoohash_map<uint32_t, uint32_t>(size);
        auto ht_phmap = phmap::flat_hash_map<uint32_t, uint32_t>(size);
        //========================================================================================
        size_t inner_size = load_factor * ht.get_buckets_count();
        uint32_t *inner = NULL;
        uint32_t *outer = NULL;
        size_t join_size = inner_outer(inner_size, outer_size, selectivity, &inner, &outer);
        
        vector<uint32_t> outer_vec(outer_size);
        for (int i = 0; i < outer_size; i++)
            outer_vec[i] = outer[i];

        uint32_t *res = (uint32_t*) mamalloc(join_size * sizeof(uint32_t));
        memset(res, 0x00, join_size * sizeof(uint32_t));
        
        uint32_t** res_parallel = new uint32_t*[threads];
        for (int i = 0; i < threads; i++)
        {
            res_parallel[i] = (uint32_t*) mamalloc(join_size * sizeof(uint32_t));
            memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));
        }

        vecht::iter_batch iter(join_size, threads);

        //========================================================================================
        uint64_t time;
        HighPrecisionTimer t;

        ht.insert_all(inner, inner, inner_size);
        time = t.GetElapsedTime();

        for (int i = 0; i < inner_size; i++)
        {
            ht_pphmap.emplace(inner[i], inner[i]);
            ht_tbb.emplace(inner[i], inner[i]);
            ht_cuckoo.insert(inner[i], inner[i]);
            ht_phmap.emplace(inner[i], inner[i]);
        }
        //========================================================================================


        auto checksize = ht.find_all_scalar(outer, outer_size, res);
        auto checksum = sum(res, checksize);
        
        t.Reset();
        auto size1 = ht.find_all_scalar(outer, outer_size, res_parallel);
        time = t.GetElapsedTime();
        times["Scalar"].push_back(time);
        auto sum1 = sumsum(res_parallel, size1, threads);
        assert(size1 == checksize);
        assert(sum1 == checksum);
        for (int i = 0; i < threads; i++) memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));


        t.Reset();
        auto size2 = ht.find_all_simd(outer, outer_size, res_parallel);
        time = t.GetElapsedTime();
        times["SIMD"].push_back(time);
        auto sum2 = sumsum(res_parallel, size1, threads);
        assert(size2 == checksize);
        assert(sum2 == checksum);
        for (int i = 0; i < threads; i++) memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));


        t.Reset();
        auto size3 = ht.find_all_scalar_prefetching(outer, outer_size, res_parallel);
        time = t.GetElapsedTime();
        times["Scalar_Prefetching"].push_back(time);
        auto sum3 = sumsum(res_parallel, size3, threads);
        assert(size3 == checksize);
        assert(sum3 == checksum);
        for (int i = 0; i < threads; i++) memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));


        t.Reset();
        auto size4 = ht.find_all_simd_prefetching(outer, outer_size, res_parallel, 1);
        time = t.GetElapsedTime();
        times["SIMD_Prefetching_Memo_NoBuff"].push_back(time);
        auto sum4 = sumsum(res_parallel, size4, threads);
        assert(size4 == checksize);
        assert(sum4 == checksum);
        for (int i = 0; i < threads; i++) memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));


        t.Reset();
        auto size40 = ht.find_all_simd_prefetching(outer, outer_size, res_parallel, 0);
        time = t.GetElapsedTime();
        times["SIMD_Prefetching_Memo_Buff"].push_back(time);
        auto sum40 = sumsum(res_parallel, size40, threads);
        assert(size40 == checksize);
        assert(sum40 == checksum);
        for (int i = 0; i < threads; i++) memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));


        t.Reset();
        auto size42 = ht.find_all_simd_prefetching(outer, outer_size, res_parallel, 2);
        time = t.GetElapsedTime();
        times["SIMD_Prefetching_NoMemo_NoBuff"].push_back(time);
        auto sum42 = sumsum(res_parallel, size42, threads);
        assert(size42 == checksize);
        assert(sum42 == checksum);
        for (int i = 0; i < threads; i++) memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));


        t.Reset();
        auto size43 = ht.find_all_simd_prefetching(outer, outer_size, res_parallel, 3);
        time = t.GetElapsedTime();
        times["SIMD_Prefetching_NoMemo_Buff"].push_back(time);
        auto sum43 = sumsum(res_parallel, size43, threads);
        assert(size43 == checksize);
        assert(sum43 == checksum);
        for (int i = 0; i < threads; i++) memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));

        t.Reset();
        auto size101 = ht.find_all_simd_prefetching(outer, outer_size, res_parallel, 10);
        time = t.GetElapsedTime();
        times["SIMD_Prefetching_Pessimistic"].push_back(time);
        auto sum101 = sumsum(res_parallel, size101, threads);
        assert(size101 == checksize);
        assert(sum101 == checksum);
        for (int i = 0; i < threads; i++) memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));

        t.Reset();
        auto size201 = ht.find_all_simd_prefetching(outer, outer_size, res_parallel, 20);
        time = t.GetElapsedTime();
        times["SIMD_NaivePrefetching"].push_back(time);
        auto sum201 = sumsum(res_parallel, size201, threads);
        assert(size201 == checksize);
        assert(sum201 == checksum);
        for (int i = 0; i < threads; i++) memset(res_parallel[i], 0x00, join_size * sizeof(uint32_t));

        // ============================================================================================

        for (int i = 0; i < threads; i++) free(res_parallel[i]);
        free(res_parallel);
    
        tbb::enumerable_thread_specific<vector<uint32_t>> res_parallel_other;
        for (auto& local : res_parallel_other) 
        {
            local.reserve(join_size);
            std::fill(local.begin(), local.end(), 0);
        }

        // phmap =======================================================================================
        t.Reset();
        auto size5 = tbb::parallel_reduce(tbb::blocked_range <size_t> (0, outer_size), 0, [&](const tbb::blocked_range < size_t > & r, const size_t& total)
        {
            auto sum = total;
            for (size_t i = r.begin(), end = r.end(); i != end; ++i) 
            {
                auto& local = res_parallel_other.local();
                const auto& found = ht_pphmap.find(outer[i]);
                if (found != ht_pphmap.end())
                {
                    local.emplace_back(found->second);
                    sum++;
                }
            }
            return sum;
        }, plus<uint32_t>());
        time = t.GetElapsedTime();
        times["pphmap"].push_back(time);
        auto sum5 = sumsum_vector(res_parallel_other);
        assert(size5 == checksize);
        assert(sum5 == checksum);
        for (auto& local : res_parallel_other) std::fill(local.begin(), local.end(), 0);
        // TBB =======================================================================================
        t.Reset();
        auto size6 = tbb::parallel_reduce(tbb::blocked_range <size_t> (0, outer_size), 0, [&](const tbb::blocked_range < size_t > & r, const size_t& total)
        {
            auto sum = total;
            for (size_t i = r.begin(), end = r.end(); i != end; ++i) 
            {
                auto& local = res_parallel_other.local();
                const auto& found = ht_tbb.find(outer[i]);
                if (found != ht_tbb.end())
                {
                    local.emplace_back(found->second);
                    sum++;
                }
            }
            return sum;
        }, plus<uint32_t>());
        time = t.GetElapsedTime();
        times["TBB"].push_back(time);
        auto sum6 = sumsum_vector(res_parallel_other);
        assert(size6 == checksize);
        assert(sum6 == checksum);
        for (auto& local : res_parallel_other) std::fill(local.begin(), local.end(), 0);
        // Cuckoo==================================================================================
        t.Reset();
        auto size7 = tbb::parallel_reduce(tbb::blocked_range <size_t> (0, outer_size), 0, [&](const tbb::blocked_range < size_t > & r, const size_t& total)
        {
            auto sum = total;
            uint32_t val;
            for (size_t i = r.begin(), end = r.end(); i != end; ++i) 
            {
                auto& local = res_parallel_other.local();
                if (ht_cuckoo.find(outer[i], val))
                {
                    local.emplace_back(val);
                    sum++;
                }
            }
            return sum;
        }, plus<uint32_t>());
        time = t.GetElapsedTime();
        times["Cuckoo"].push_back(time);
        auto sum7 = sumsum_vector(res_parallel_other);
        assert(size7 == checksize);
        assert(sum7 == checksum);
        for (auto& local : res_parallel_other) std::fill(local.begin(), local.end(), 0);
        // phmap =======================================================================================
        t.Reset();

        auto size8 = tbb::parallel_reduce(tbb::blocked_range <size_t> (0, outer_size), 0, [&](const tbb::blocked_range < size_t > & r, const size_t& total)
        {
            auto sum = total;
            for (size_t i = r.begin(), end = r.end(); i != end; ++i) 
            {
                auto& local = res_parallel_other.local();
                const auto& found = ht_phmap.find(outer[i]);
                if (found != ht_phmap.end())
                {
                    local.emplace_back(found->second);
                    sum++;
                }
            }
            return sum;
        }, plus<uint32_t>());


        time = t.GetElapsedTime();
        times["phmap"].push_back(time);
        auto sum8 = sumsum_vector(res_parallel_other);
        assert(size8 == checksize);
        assert(sum8 == checksum);
        for (auto& local : res_parallel_other) std::fill(local.begin(), local.end(), 0);
        // ========================================================================================
    
        free(inner);
        free(outer);
    }

    for (auto& p : times)
    {
        float sum = 0.0;
        for (int i = 0; i < p.second.size(); i++)
        {
            if (i>=warmup) sum += p.second[i];
        }
        final_times[p.first] = sum/(iterations+0.0);
    }

    if (!verbose and is_first_call)
    {
        cout << "size,";
        auto counter = 0;
        for (auto& p : final_times)
        {
            if (counter == final_times.size()-1)
                cout << p.first;
            else
                cout << p.first << ",";
            counter++;
        }
        cout << endl;
    }

    cout << log_inner_bytes << ',';

    if (verbose)
        cout << "\n";

    for (auto& p : final_times)
    {
        if (!verbose)
            cout << (outer_size/1000000.0) / (p.second/1000.0) << ",";
        else 
            cout << p.first << " : " << (outer_size/1000000.0) / (p.second/1000.0) << endl;
    }

    if (!verbose)
        cout << '\n';

    return 0;
}