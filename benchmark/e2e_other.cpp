#include <ctime>
#include <vector>
#include <iomanip>
#include <numeric>
#include <iostream>
#include <algorithm>

#include "tbb/tbb.h"
#include "../include/parallel_hashmap/phmap.h"
#include "../include/libcuckoo/libcuckoo/cuckoohash_map.hh"

#include "../include/high_precision_timer.h"
#include "../src/vecht.hpp"

using namespace std;

static void *mamalloc(size_t size)
{
	void *p = NULL;
	return posix_memalign(&p, 64, size) ? NULL : p;
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

int main(int argc, char *argv[])
{
    size_t iterations = 5;
    size_t warmup = 5;

    phmap::flat_hash_map<string, vector<float>> times;
    phmap::flat_hash_map<string, float> final_times;
    HighPrecisionTimer t;
    long time;
    //========================================================================================
    bool verbose = false;
    int size1 = stoi(argv[1]);      
    int size2 = stoi(argv[2]);      
    float sparsity1 = stof(argv[3]);  
    float sparsity2 = stof(argv[4]);  
    uint32_t max_vec_val = stoi(argv[5]); 
    int threads = stoi(argv[6]);
    bool is_first_call = stoi(argv[7]);
    tbb::task_scheduler_init scheduler(threads);
    if (verbose)
        cout << "size1: " << size1 << " | size2: " << size2 << " | sparsity1: " << sparsity1 << " | sparsity2: " << sparsity2 << " | max_val: " << max_vec_val << " | threads: " << threads << endl; 
    //========================================================================================
    int actual_size1 = floor(sparsity1 * size1);
    int actual_size2 = floor(sparsity2 * size2);

    srand(unsigned(std::time(nullptr)));

    phmap::flat_hash_map<uint32_t, uint32_t> v1;
    vector<uint32_t> v1_key;
    vector<uint32_t> v1_val;

    phmap::flat_hash_map<uint32_t, uint32_t> v2;

    while (v1.size() != actual_size1)
    {
        auto tmp = rand()%size1;
        if (!v1.contains(tmp))
        {
            int val = 1+(rand()%max_vec_val);
            v1[tmp] = val;
            v1_key.push_back(tmp);
            v1_val.push_back(val);
        }
    }

    while (v2.size() != actual_size2)
    {
        auto tmp = rand()%size2;
        if (!v2.contains(tmp))
            v2[tmp] = 1+(rand()%max_vec_val);
    }

    phmap::flat_hash_map<uint32_t, uint32_t> res_elem_prod;    
    phmap::flat_hash_map<uint32_t, bool> diff;    
    uint64_t inner_prod_sum = 0;
    for (auto& p : v1)
        if (v2.contains(p.first))
        {
            inner_prod_sum += p.second * v2[p.first];
            res_elem_prod[p.first] = p.second * v2[p.first];
        }
        else
        {
            diff[p.first] = true;
        }

    uint64_t sum_intersection_keys = 0;
    uint64_t sum_diff_keys = 0;
    for (auto& p : res_elem_prod) sum_intersection_keys += p.first;
    for (auto& p : diff) sum_diff_keys += p.first;

    int intersection_size = res_elem_prod.size();

    if (verbose)
    {
        cout << "intersection size:     " << intersection_size << endl;
        cout << "sum intersection keys: " << sum_intersection_keys << endl;
        cout << "diff size:             " << diff.size() << endl;
        cout << "sum diff keys:         " << sum_diff_keys << endl;
        cout << "inner product:         " << inner_prod_sum << endl;
    }

    for (int iteration = 0; iteration < iterations+warmup; iteration++)
    {
        // phmap ========================================================================================

        auto v1_phmap = phmap::flat_hash_map<uint32_t, uint32_t>(2*actual_size1);
        auto v2_phmap = phmap::flat_hash_map<uint32_t, uint32_t>(2*actual_size2);
        auto s1_phmap = phmap::flat_hash_map<uint32_t, uint32_t>(2*actual_size1);
        auto s2_phmap = phmap::flat_hash_map<uint32_t, uint32_t>(2*actual_size2);

        for (auto& p : v1)
        {
            v1_phmap[p.first] = p.second;
            s1_phmap[p.first] = 1;
        }

        for (auto& p : v2)
        {
            v2_phmap[p.first] = p.second;
            s2_phmap[p.first] = 1;
        }

        phmap::flat_hash_map<uint32_t, uint32_t> phmap_res_elem_prod (2*intersection_size);
        phmap::flat_hash_map<uint32_t, uint32_t> phmap_diff (2*(actual_size1-intersection_size));
        uint64_t phmap_sum_intersection_keys = 0;
        uint64_t phmap_sum_diff_keys = 0;
        uint64_t phmap_inner_prod_sum = 0;
        int phmap_intersection_size = 0; 
        
        tbb::enumerable_thread_specific<uint64_t> phmap_ts1;
        tbb::enumerable_thread_specific<phmap::flat_hash_map<uint32_t, uint32_t>> phmap_ts2;
        tbb::enumerable_thread_specific<phmap::flat_hash_map<uint32_t, uint32_t>> phmap_ts3;
        tbb::enumerable_thread_specific<phmap::flat_hash_map<uint32_t, uint32_t>> phmap_ts4;


        t.Reset();

        phmap_inner_prod_sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v1_key.size()), 0, [&](const tbb::blocked_range<size_t>& r, const uint64_t& total){
            auto l = total;
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                const auto& found = v2_phmap.find(v1_key[i]); 
                if (found != v2_phmap.end())
                {
                    l += v1_val[i] * found->second;
                }
            }
            return l;
        }, plus<uint64_t>());


        time = t.GetElapsedTime(); times["phmap-inner"].push_back(time); t.Reset();

        // elem mul
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                auto& local = phmap_ts2.local();
                const auto& found = v2_phmap.find(v1_key[i]); 
                if (found != v2_phmap.end())
                {
                    local.emplace(v1_key[i], v1_val[i] * found->second);
                }
            }
        });
        for (auto& local : phmap_ts2)
            for (auto& p : local)
                phmap_res_elem_prod.emplace(p.first, p.second);

        time = t.GetElapsedTime(); times["phmap-elem"].push_back(time); t.Reset();

        // intersection
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                auto& local = phmap_ts3.local();
                const auto& found = v2_phmap.find(v1_key[i]); 
                if (found != v2_phmap.end())
                    local.emplace(v1_key[i], 1);
            }
        });
        for (auto& local : phmap_ts3)
            for (auto& p : local)
                phmap_res_elem_prod.emplace(p.first, p.second);

        time = t.GetElapsedTime(); times["phmap-inter"].push_back(time); t.Reset();

        // diff
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                auto& local = phmap_ts4.local();
                const auto& found = v2_phmap.find(v1_key[i]); 
                if (found == v2_phmap.end())
                    local.emplace(v1_key[i], 1);
            }
        });

        for (auto& local : phmap_ts4)
            for (auto& p : local)
                phmap_diff.emplace(p.first, p.second);

        time = t.GetElapsedTime(); times["phmap-diff"].push_back(time); t.Reset();
        
        for (auto& p : phmap_res_elem_prod) phmap_sum_intersection_keys += p.first;
        for (auto& p : phmap_diff) phmap_sum_diff_keys += p.first;
        phmap_intersection_size = phmap_res_elem_prod.size();

        assert(phmap_intersection_size == intersection_size);
        assert(phmap_sum_intersection_keys == sum_intersection_keys);
        assert(phmap_inner_prod_sum == inner_prod_sum);
        assert(phmap_sum_diff_keys == sum_diff_keys);

        // pphmap ========================================================================================

        auto v1_pphmap = phmap::parallel_flat_hash_map<uint32_t, uint32_t, phmap::priv::hash_default_hash<uint32_t>, phmap::priv::hash_default_eq<uint32_t>, std::allocator<std::pair<const uint32_t, uint32_t>>, 4, std::mutex>(2*actual_size1);
        auto v2_pphmap = phmap::parallel_flat_hash_map<uint32_t, uint32_t, phmap::priv::hash_default_hash<uint32_t>, phmap::priv::hash_default_eq<uint32_t>, std::allocator<std::pair<const uint32_t, uint32_t>>, 4, std::mutex>(2*actual_size2);
        auto s1_pphmap = phmap::parallel_flat_hash_map<uint32_t, uint32_t, phmap::priv::hash_default_hash<uint32_t>, phmap::priv::hash_default_eq<uint32_t>, std::allocator<std::pair<const uint32_t, uint32_t>>, 4, std::mutex>(2*actual_size1);
        auto s2_pphmap = phmap::parallel_flat_hash_map<uint32_t, uint32_t, phmap::priv::hash_default_hash<uint32_t>, phmap::priv::hash_default_eq<uint32_t>, std::allocator<std::pair<const uint32_t, uint32_t>>, 4, std::mutex>(2*actual_size2);

        for (auto& p : v1)
        {
            v1_pphmap[p.first] = p.second;
            s1_pphmap[p.first] = 1;
        }

        for (auto& p : v2)
        {
            v2_pphmap[p.first] = p.second;
            s2_pphmap[p.first] = 1;
        }

        auto pphmap_res_elem_prod = phmap::parallel_flat_hash_map<uint32_t, uint32_t, phmap::priv::hash_default_hash<uint32_t>, phmap::priv::hash_default_eq<uint32_t>, std::allocator<std::pair<const uint32_t, uint32_t>>, 4, std::mutex> (2*intersection_size); 
        auto pphmap_diff = phmap::parallel_flat_hash_map<uint32_t, uint32_t, phmap::priv::hash_default_hash<uint32_t>, phmap::priv::hash_default_eq<uint32_t>, std::allocator<std::pair<const uint32_t, uint32_t>>, 4, std::mutex> (2*(actual_size1-intersection_size));
        uint64_t pphmap_sum_intersection_keys = 0;
        uint64_t pphmap_sum_diff_keys = 0;
        uint64_t pphmap_inner_prod_sum = 0;
        int pphmap_intersection_size = 0;

        tbb::enumerable_thread_specific<uint64_t> pphmap_ts1;

        t.Reset();

        pphmap_inner_prod_sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v1_key.size()), 0, [&](const tbb::blocked_range<size_t>& r, const uint64_t& total){
            auto l = total;
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                const auto& found = v2_pphmap.find(v1_key[i]); 
                if (found != v2_pphmap.end())
                {
                    l += v1_val[i] * found->second;
                }
            }
            return l;
        }, plus<uint64_t>());

        time = t.GetElapsedTime(); times["pphmap-inner"].push_back(time); t.Reset();

        // elem mul
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                const auto& found = v2_pphmap.find(v1_key[i]); 
                if (found != v2_pphmap.end())
                {
                    pphmap_res_elem_prod.emplace(v1_key[i], v1_val[i] * found->second);
                }
            }
        });

        time = t.GetElapsedTime(); times["pphmap-elem"].push_back(time); t.Reset();

        // intersection
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                const auto& found = v2_pphmap.find(v1_key[i]); 
                if (found != v2_pphmap.end())
                    pphmap_res_elem_prod.emplace(v1_key[i], 1);
            }
        });

        time = t.GetElapsedTime(); times["pphmap-inter"].push_back(time); t.Reset();

        // diff
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                const auto& found = v2_pphmap.find(v1_key[i]); 
                if (found == v2_pphmap.end())
                    pphmap_diff.emplace(v1_key[i], 1);
            }
        });

        time = t.GetElapsedTime(); times["pphmap-diff"].push_back(time); t.Reset();
        
        for (auto& p : pphmap_res_elem_prod) pphmap_sum_intersection_keys += p.first;
        for (auto& p : pphmap_diff) pphmap_sum_diff_keys += p.first;
        pphmap_intersection_size = pphmap_res_elem_prod.size();

        assert(pphmap_intersection_size == intersection_size);
        assert(pphmap_sum_intersection_keys == sum_intersection_keys);
        assert(pphmap_inner_prod_sum == inner_prod_sum);
        assert(pphmap_sum_diff_keys == sum_diff_keys);

        // tbb ========================================================================================

        auto v1_tbb = tbb::concurrent_unordered_map<uint32_t, uint32_t, std::hash<uint32_t>, std::equal_to<uint32_t>>(2*actual_size1);
        auto v2_tbb = tbb::concurrent_unordered_map<uint32_t, uint32_t, std::hash<uint32_t>, std::equal_to<uint32_t>>(2*actual_size2);
        auto s1_tbb = tbb::concurrent_unordered_map<uint32_t, uint32_t, std::hash<uint32_t>, std::equal_to<uint32_t>>(2*actual_size1);
        auto s2_tbb = tbb::concurrent_unordered_map<uint32_t, uint32_t, std::hash<uint32_t>, std::equal_to<uint32_t>>(2*actual_size2);

        for (auto& p : v1)
        {
            v1_tbb[p.first] = p.second;
            s1_tbb[p.first] = 1;
        }

        for (auto& p : v2)
        {
            v2_tbb[p.first] = p.second;
            s2_tbb[p.first] = 1;
        }

        tbb::concurrent_unordered_map<uint32_t, uint32_t, std::hash<uint32_t>, std::equal_to<uint32_t>> tbb_res_elem_prod(2*intersection_size); 
        tbb::concurrent_unordered_map<uint32_t, uint32_t, std::hash<uint32_t>, std::equal_to<uint32_t>> tbb_diff(2*(actual_size1-intersection_size));
        uint64_t tbb_sum_intersection_keys = 0;
        uint64_t tbb_sum_diff_keys = 0;
        uint64_t tbb_inner_prod_sum = 0;
        int tbb_intersection_size = 0; 

        tbb::enumerable_thread_specific<uint64_t> tbb_ts1;

        t.Reset();

        // inner
        tbb_inner_prod_sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v1_key.size()), 0, [&](const tbb::blocked_range<size_t>& r, const uint64_t& total){
            auto l = total;
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                const auto& found = v2_tbb.find(v1_key[i]); 
                if (found != v2_tbb.end())
                {
                    l += v1_val[i] * found->second;
                }
            }
            return l;
        }, plus<uint64_t>());

        time = t.GetElapsedTime(); times["tbb-inner"].push_back(time); t.Reset();

        // elem mul
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                const auto& found = v2_tbb.find(v1_key[i]); 
                if (found != v2_tbb.end())
                {
                    tbb_res_elem_prod.emplace(v1_key[i], v1_val[i] * found->second);
                }
            }
        });

        time = t.GetElapsedTime(); times["tbb-elem"].push_back(time); t.Reset();

        // intersection
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                const auto& found = v2_tbb.find(v1_key[i]); 
                if (found != v2_tbb.end())
                    tbb_res_elem_prod.emplace(v1_key[i], 1);
            }
        });

        time = t.GetElapsedTime(); times["tbb-inter"].push_back(time); t.Reset();

        // diff
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                const auto& found = v2_tbb.find(v1_key[i]); 
                if (found == v2_tbb.end())
                    tbb_diff.emplace(v1_key[i], 1);
            }
        });

        time = t.GetElapsedTime(); times["tbb-diff"].push_back(time); t.Reset();

        
        for (auto& p : tbb_res_elem_prod) tbb_sum_intersection_keys += p.first;
        for (auto& p : tbb_diff) tbb_sum_diff_keys += p.first;
        tbb_intersection_size = tbb_res_elem_prod.size();

        assert(tbb_intersection_size == intersection_size);
        assert(tbb_sum_intersection_keys == sum_intersection_keys);
        assert(tbb_inner_prod_sum == inner_prod_sum);
        assert(tbb_sum_diff_keys == sum_diff_keys);

        // cuckoo ========================================================================================

        auto v1_cuckoo = libcuckoo::cuckoohash_map<uint32_t, uint32_t>(2*actual_size1);
        auto v2_cuckoo = libcuckoo::cuckoohash_map<uint32_t, uint32_t>(2*actual_size2);
        auto s1_cuckoo = libcuckoo::cuckoohash_map<uint32_t, uint32_t>(2*actual_size1);
        auto s2_cuckoo = libcuckoo::cuckoohash_map<uint32_t, uint32_t>(2*actual_size2);

        for (auto& p : v1)
        {
            v1_cuckoo.insert(p.first, p.second);
            s1_cuckoo.insert(p.first, 1);
        }

        for (auto& p : v2)
        {
            v2_cuckoo.insert(p.first, p.second);
            s2_cuckoo.insert(p.first, 1);
        }

        libcuckoo::cuckoohash_map<uint32_t, uint32_t> cuckoo_res_elem_prod(2*intersection_size); 
        libcuckoo::cuckoohash_map<uint32_t, uint32_t> cuckoo_diff(2*(actual_size1-intersection_size));
        uint64_t cuckoo_sum_intersection_keys = 0;
        uint64_t cuckoo_sum_diff_keys = 0;
        uint64_t cuckoo_inner_prod_sum = 0;
        int cuckoo_intersection_size = 0; 
        uint32_t found; 
        
        tbb::enumerable_thread_specific<uint64_t> cuckoo_ts1;

        t.Reset();

        cuckoo_inner_prod_sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v1_key.size()), 0, [&](const tbb::blocked_range<size_t>& r, const uint64_t& total){
            auto l = total;
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                if (v2_cuckoo.find(v1_key[i], found))
                {
                    l += v1_val[i] * found;
                }
            }
            return l;
        }, plus<uint64_t>());

        time = t.GetElapsedTime(); times["cuckoo-inner"].push_back(time); t.Reset();

        // elem mul
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                if (v2_cuckoo.find(v1_key[i], found))
                    cuckoo_res_elem_prod.insert(v1_key[i], v1_val[i] * found);
            }
        });

        time = t.GetElapsedTime(); times["cuckoo-elem"].push_back(time); t.Reset();
        cuckoo_res_elem_prod.clear();
        t.Reset();

        // intersection
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                if (v2_cuckoo.find(v1_key[i], found))
                    cuckoo_res_elem_prod.insert(v1_key[i], 1);
            }
        });

        time = t.GetElapsedTime(); times["cuckoo-inter"].push_back(time); t.Reset();

        // diff
        tbb::parallel_for(tbb::blocked_range<int>(0, v1_key.size()), [&](auto& r){
            for (int i = r.begin(), end = r.end(); i != end; ++i)
            {
                if (!v2_cuckoo.find(v1_key[i], found))
                    cuckoo_diff.insert(v1_key[i], 1);
            }
        });

        time = t.GetElapsedTime(); times["cuckoo-diff"].push_back(time); t.Reset();
    
        // assert(cuckoo_inner_prod_sum == inner_prod_sum);
        assert(cuckoo_res_elem_prod.size() == intersection_size);

        bool test_res = false;

        for (auto& p : res_elem_prod)
        {
            if (cuckoo_res_elem_prod.find(p.first, found))
                cuckoo_sum_intersection_keys += p.first;
            else
            {
                test_res = true;
                cout << "prod_res key not found: " << p.first << endl;
                break;
            }
        }

        for (auto& p : diff)
        {
            if (cuckoo_diff.find(p.first, found))
                cuckoo_sum_diff_keys += p.first;
            else
            {
                test_res = true;
                cout << "diff key not found: " << p.first << endl;
                break;
            }
        }

        assert(!test_res);

        auto v1_vecht = vecht::lp_map(2*actual_size1, 64 | 1, threads);
        auto v2_vecht = vecht::lp_map(2*actual_size2, 64 | 1, threads);
        auto s1_vecht = vecht::lp_map(2*actual_size1, 64 | 1, threads);
        auto s2_vecht = vecht::lp_map(2*actual_size2, 64 | 1, threads);

        for (auto& p : v1)
        {
            v1_vecht.insert(p.first, p.second);
            s1_vecht.insert(p.first, 1);
        }

        for (auto& p : v2)
        {
            v2_vecht.insert(p.first, p.second);
            s2_vecht.insert(p.first, 1);
        }

        auto vecht_res_elem_prod = vecht::lp_map(2*intersection_size, 64, threads); 
        auto vecht_diff = vecht::lp_map(2*(actual_size1-intersection_size), 64, threads);
        uint64_t vecht_sum_intersection_keys = 0;
        uint64_t vecht_sum_diff_keys = 0;
        uint64_t vecht_inner_prod_sum = 0;

        vecht::iter_batch b_iter(intersection_size, threads, true);

        t.Reset();

        // inner
        auto size = v2_vecht.zip(&v1_key[0], &v1_val[0], v1_key.size(), false, &b_iter);
        b_iter.foreach([&](auto& key, auto& val, auto& pay)
        {
           vecht_inner_prod_sum += val * pay; 
        });
        time = t.GetElapsedTime(); times["vecht-inner"].push_back(time); t.Reset();
        b_iter.reset();

        // elem mul
        size = v2_vecht.zip(&v1_key[0], &v1_val[0], v1_key.size(), false, &b_iter);
        b_iter.foreach([&](auto& key, auto& val, auto& pay)
        {
            vecht_res_elem_prod.insert(key, val * pay);
        });
        time = t.GetElapsedTime(); times["vecht-elem"].push_back(time); t.Reset();
        vecht_res_elem_prod.clear();
        b_iter.reset();

        t.Reset();
        // intersection
        size = v2_vecht.zip(&v1_key[0], &v1_val[0], v1_key.size(), false, &b_iter);
        b_iter.foreach([&](auto& key, auto& val, auto& pay)
        {
            vecht_res_elem_prod.insert(key, 1);
        });
        time = t.GetElapsedTime(); times["vecht-inter"].push_back(time); t.Reset();
    
        b_iter.foreach([&](auto& key, auto& val, auto& pay)
        {
            vecht_sum_intersection_keys += key;
        });
        b_iter.reset();

        assert(size == intersection_size);
        assert(vecht_inner_prod_sum == inner_prod_sum);
        assert(vecht_sum_intersection_keys == sum_intersection_keys);

        vecht::iter_batch b_iter_diff(actual_size1-intersection_size, threads, true);

        t.Reset();
        // diff
        size = v2_vecht.zip(&v1_key[0], &v1_val[0], v1_key.size(), true, &b_iter_diff);
        b_iter.foreach([&](auto& key, auto& val, auto& pay)
        {
            vecht_diff.insert(key, 1);
        });
        time = t.GetElapsedTime(); times["vecht-diff"].push_back(time); t.Reset();
 
        b_iter_diff.foreach([&](auto& key, auto& val, auto& pay)
        {
            vecht_sum_diff_keys += key;
        });
        assert(vecht_sum_intersection_keys == sum_intersection_keys);
        assert(vecht_sum_diff_keys == sum_diff_keys);
        
        //========================================================================================

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
        cout << "thread,";
        auto counter = 0;
        for (auto& p : final_times)
        {
            if (counter < final_times.size()-1)
                cout << p.first << ",";
            else
                cout << p.first;
            counter++;
        }
        cout << endl;
    }

    if (verbose)
        cout << "\n";

    cout << actual_size2 << ",";
    cout << threads << ",";
    for (auto& p : final_times)
    {
        if (!verbose)
        {
            cout << p.second << ",";
        }
        else 
            cout << p.first << " : " << p.second << endl;
    }


    cout << "\n";
}