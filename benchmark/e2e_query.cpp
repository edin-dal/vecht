#include <iostream>
#include <iomanip>
#include <tbb/tbb.h>

#include "../include/parallel_hashmap/phmap.h"

#include "../include/tpch_loader.hpp"
#include "../include/map_helper.h"
#include "../include/tuple_helper.h"
#include "../include/vector_helper.h"
#include "../include/high_precision_timer.h"

#include "../src/vecht.hpp"

using namespace std;

size_t global_threads_count;

using Q4_type = phmap::flat_hash_map < tuple < VarChar < 15 > , long > , bool >;
using Q8_type = phmap::flat_hash_map < tuple < long, double > , bool >;
using Q12_type = phmap::flat_hash_map < tuple < VarChar < 10 > , long, long > , bool >;

// =================================================================================================

Q4_type q4_sequential()
{
    HighPrecisionTimer timer;

    auto v82 = phmap::flat_hash_map < tuple < VarChar < 15 > , long > , bool > ({});
    auto v83 = l_orderkey.size();
    auto v77 = phmap::flat_hash_map < long, bool > (2 * 3793296);
    auto v80 = phmap::flat_hash_map < VarChar < 15 > , long > ({});

    timer.PrintElapsedTimeAndReset("Initialization");

    for (int v76 = 0; v76 < v83; v76++) {
    if ((l_commitdate[v76] < l_receiptdate[v76])) {
        v77[l_orderkey[v76]] = true;
    };
    }

    timer.PrintElapsedTimeAndReset("Indexing");

    const auto & li_indexed = v77;
    const auto & v78 = li_indexed;
    auto v84 = o_orderkey.size();
    for (int v79 = 0; v79 < v84; v79++) {
        if (((v78.contains(o_orderkey[v79])))) {
            v80[o_orderpriority[v79]] += (long) 1;
        }
    }

    timer.PrintElapsedTimeAndReset("Probing");

    for (auto & v81: v80) {
    v82[make_tuple((v81.first), (v81.second))] = true;
    }

    timer.PrintElapsedTimeAndReset("Finalization");

    return v82;
}

Q4_type q4_sequential_vec()
{
    HighPrecisionTimer timer;

    auto v82 = phmap::flat_hash_map < tuple < VarChar < 15 > , long > , bool > ({});
    auto v83 = l_orderkey.size();
    auto v77 = vecht::lp_map(2*3793296, 64, global_threads_count); 
    auto v80 = phmap::flat_hash_map < VarChar < 15 > , long > ({});

    timer.PrintElapsedTimeAndReset("Initialization");

    for (int v76 = 0; v76 < v83; v76++) {
    if ((l_commitdate[v76] < l_receiptdate[v76])) {
        v77.insert(l_orderkey[v76], 1);
    };
    }

    timer.PrintElapsedTimeAndReset("Indexing");

    const auto & li_indexed = v77;
    const auto & v78 = li_indexed;
    auto v84 = o_orderkey.size();

    v77.zip_apply((uint32_t*)&o_orderkey[0], (uint32_t*)&o_id[0], v84, false, [&](auto& key, auto& value, auto& pay)
    {
        v80[o_orderpriority[pay]] += 1;
    });

    timer.PrintElapsedTimeAndReset("Probing");
    
    for (auto & v81: v80) {
    v82[make_tuple((v81.first), (v81.second))] = true;
    }

    timer.PrintElapsedTimeAndReset("Finalization");

    return v82;
}

Q4_type q4_parallel()
{
    HighPrecisionTimer timer;

    auto v137 = phmap::flat_hash_map < long, bool > (2 * 3793296);
    tbb::enumerable_thread_specific < phmap::flat_hash_map < long, bool >> v1000;
    auto v140 = phmap::flat_hash_map < VarChar < 15 > , long > ({});
    tbb::enumerable_thread_specific < phmap::flat_hash_map < VarChar < 15 > , long >> v152;
    auto v142 = phmap::flat_hash_map < tuple < VarChar < 15 > , long > , bool > ({});

    timer.PrintElapsedTimeAndReset("Initialization");

    auto v143 = l_orderkey.size();

    tbb::parallel_for(tbb::blocked_range < size_t > (0, v143), [ & ](const tbb::blocked_range < size_t > & v144) {
        auto & v1001 = v1000.local();
        for (size_t v136 = v144.begin(), end = v144.end(); v136 != end; ++v136) {
        if ((l_commitdate[v136] < l_receiptdate[v136])) {
            v1001[l_orderkey[v136]] = true;
        };
        }
    });
    for (auto & local: v1000) AddMap < phmap::flat_hash_map < long, bool > , long, bool > (v137, local);

    timer.PrintElapsedTimeAndReset("Indexing");

    const auto & li_indexed = v137;
    const auto & v138 = li_indexed;

    auto v148 = o_orderkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v148), [ & ](const tbb::blocked_range < size_t > & v149) {
    auto & v153 = v152.local();
        for (size_t v139 = v149.begin(), end = v149.end(); v139 != end; ++v139) {
            if (v138.contains(o_orderkey[v139])) {
                v153[o_orderpriority[v139]] += (long) 1;
            }
        }
    });
    for (auto & local: v152) AddMap < phmap::flat_hash_map < VarChar < 15 > , long > , VarChar < 15 > , long > (v140, local);
    const auto & ord_probed = v140;

    timer.PrintElapsedTimeAndReset("Probing");

    for (auto & v141: ord_probed) {
    v142[make_tuple((v141.first), (v141.second))] = true;
    }

    timer.PrintElapsedTimeAndReset("Finalization");

    return v142;
}

Q4_type q4_parallel_vec()
{
    HighPrecisionTimer timer;

    auto v137 = vecht::lp_map(2*3793296, 64, global_threads_count);
    tbb::enumerable_thread_specific < phmap::flat_hash_map < long, bool >> v1000;
    auto v140 = phmap::flat_hash_map < VarChar < 15 > , long > ({});
    tbb::enumerable_thread_specific < phmap::flat_hash_map < VarChar < 15 > , long >> v152;
    auto v142 = phmap::flat_hash_map < tuple < VarChar < 15 > , long > , bool > ({});

    timer.PrintElapsedTimeAndReset("Initialization");

    auto v143 = l_orderkey.size();

    for (int v136 = 0; v136 < v143; v136++) {
    if ((l_commitdate[v136] < l_receiptdate[v136])) {
        v137.insert(l_orderkey[v136], 1);
    };
    }

    timer.PrintElapsedTimeAndReset("Indexing");

    const auto & li_indexed = v137;
    const auto & v138 = li_indexed;

    auto v148 = o_orderkey.size();

    v137.zip_apply((uint32_t*)&o_orderkey[0], (uint32_t*)&o_id[0], v148, false, [&](auto& key, auto& value, auto& pay)
    {
        auto & v153 = v152.local();
        v153[o_orderpriority[pay]] += 1;
    });
    for (auto & local: v152) AddMap < phmap::flat_hash_map < VarChar < 15 > , long > , VarChar < 15 > , long > (v140, local);

    const auto & ord_probed = v140;

    timer.PrintElapsedTimeAndReset("Probing");

    for (auto & v141: ord_probed) {
    v142[make_tuple((v141.first), (v141.second))] = true;
    }

    timer.PrintElapsedTimeAndReset("Finalization");

    return v142;
}

// ============================

Q8_type q8_sequential()
{
    HighPrecisionTimer timer;

    auto v206 = phmap::flat_hash_map < tuple < long, double > , bool > ({});
    auto v188 = phmap::flat_hash_map < long, tuple < long >> ({});
    auto v191 = phmap::flat_hash_map < long, bool > ({});
    auto v193 = phmap::flat_hash_map < long, tuple < VarChar < 25 >>> ({});
    auto v195 = phmap::flat_hash_map < long, tuple < long >> ({});
    vector < long > v197(200001);
    auto v199 = phmap::flat_hash_map < long, tuple < long >> (2 * 200000);
    auto v201 = phmap::flat_hash_map < long, tuple < long, long >> ({});
    auto v204 = phmap::flat_hash_map < long, tuple < double, double >> ({});

    timer.PrintElapsedTimeAndReset("Initialization");

    const auto & steel = ConstantString("ECONOMY ANODIZED STEEL", 23);
    const auto & america = ConstantString("AMERICA", 8);
    const auto & brazil = ConstantString("BRAZIL", 7);
    auto v207 = r_regionkey.size();
    for (int v187 = 0; v187 < v207; v187++) {
        if ((r_name[v187] == america)) {
        v188.emplace(r_regionkey[v187], make_tuple(r_regionkey[v187]));
        };
    }
    const auto & re_indexed = v188;
    const auto & v189 = re_indexed;
    auto v208 = n_nationkey.size();
    for (int v190 = 0; v190 < v208; v190++) {
        if (true) {
        if (((v189).contains(n_regionkey[v190]))) {
            v191.emplace(n_nationkey[v190], true);
        }
        };
    }
    const auto & na_probed = v191;
    auto v209 = n_nationkey.size();
    for (int v192 = 0; v192 < v209; v192++) {
        if (true) {
        v193.emplace(n_nationkey[v192], make_tuple(n_name[v192]));
        };
    }
    const auto & na_indexed = v193;
    auto v210 = s_suppkey.size();
    for (int v194 = 0; v194 < v210; v194++) {
        if (true) {
        v195.emplace(s_suppkey[v194], make_tuple(s_nationkey[v194]));
        };
    }
    const auto & su_indexed = v195;
    auto v211 = c_custkey.size();
    for (int v196 = 0; v196 < v211; v196++) {
        v197[c_custkey[v196]] = c_nationkey[v196];
    }
    const auto & cu_indexed = v197;

    timer.PrintElapsedTimeAndReset("Not Important1");

    auto v212 = p_partkey.size();
    for (int v198 = 0; v198 < v212; v198++) {
        v199.emplace(p_partkey[v198], make_tuple(p_partkey[v198]));
    }

    timer.PrintElapsedTimeAndReset("Indexing");

    const auto & pa_indexed = v199;
    auto v213 = o_orderkey.size();
    for (int v200 = 0; v200 < v213; v200++) {
        if (((o_orderdate[v200] >= (long) 19950101) && (o_orderdate[v200] <= (long) 19961231))) {
        v201.emplace(o_orderkey[v200], make_tuple(o_custkey[v200], o_orderdate[v200]));
        };
    }

    timer.PrintElapsedTimeAndReset("Not Important2");

    const auto & ord_indexed = v201;
    const auto & v202 = pa_indexed;
    auto v214 = l_orderkey.size();
    for (int v203 = 0; v203 < v214; v203++) {
        if (true) {
        if (((v202).contains(l_partkey[v203]))) {
            if ((((ord_indexed).contains(l_orderkey[v203])) && ((na_probed).contains(cu_indexed[ /* o_custkey */ get < 0 > ((ord_indexed).at(l_orderkey[v203]))])))) {
            v204[(( /* o_orderdate */ get < 1 > ((ord_indexed).at(l_orderkey[v203]))) / 10000)] += make_tuple((( /* n_name */ get < 0 > ((na_indexed).at( /* s_nationkey */ get < 0 > ((su_indexed).at(l_suppkey[v203])))) == brazil)) ? ((l_extendedprice[v203] * (1.0 - l_discount[v203]))) : (0.0), (l_extendedprice[v203] * (1.0 - l_discount[v203])));
            }
        }
        };
    }

    timer.PrintElapsedTimeAndReset("Probing");
    
    const auto & li_probed = v204;

    for (auto & v205: li_probed) {
        v206[make_tuple((v205.first), ( /* A */ get < 0 > ((v205.second)) / /* B */ get < 1 > ((v205.second))))] = true;
    }

    timer.PrintElapsedTimeAndReset("Finalization");

    return v206;
}

Q8_type q8_sequential_vec()
{
    HighPrecisionTimer timer;

    auto v206 = phmap::flat_hash_map < tuple < long, double > , bool > ({});
    auto v188 = phmap::flat_hash_map < long, tuple < long >> ({});
    auto v191 = phmap::flat_hash_map < long, bool > ({});
    auto v193 = phmap::flat_hash_map < long, tuple < VarChar < 25 >>> ({});
    auto v195 = phmap::flat_hash_map < long, tuple < long >> ({});
    vector < long > v197(200001);
    auto v199 = vecht::lp_map(2*200000, 64, global_threads_count); 
    auto v201 = phmap::flat_hash_map < long, tuple < long, long >> ({});
    auto v204 = phmap::flat_hash_map < long, tuple < double, double >> ({});

    timer.PrintElapsedTimeAndReset("Initialization");

    const auto & steel = ConstantString("ECONOMY ANODIZED STEEL", 23);
    const auto & america = ConstantString("AMERICA", 8);
    const auto & brazil = ConstantString("BRAZIL", 7);
    auto v207 = r_regionkey.size();
    for (int v187 = 0; v187 < v207; v187++) {
        if ((r_name[v187] == america)) {
        v188.emplace(r_regionkey[v187], make_tuple(r_regionkey[v187]));
        };
    }
    const auto & re_indexed = v188;
    const auto & v189 = re_indexed;
    auto v208 = n_nationkey.size();
    for (int v190 = 0; v190 < v208; v190++) {
        if (true) {
        if (((v189).contains(n_regionkey[v190]))) {
            v191.emplace(n_nationkey[v190], true);
        }
        };
    }
    const auto & na_probed = v191;
    auto v209 = n_nationkey.size();
    for (int v192 = 0; v192 < v209; v192++) {
        if (true) {
        v193.emplace(n_nationkey[v192], make_tuple(n_name[v192]));
        };
    }
    const auto & na_indexed = v193;
    auto v210 = s_suppkey.size();
    for (int v194 = 0; v194 < v210; v194++) {
        if (true) {
        v195.emplace(s_suppkey[v194], make_tuple(s_nationkey[v194]));
        };
    }
    const auto & su_indexed = v195;
    auto v211 = c_custkey.size();
    for (int v196 = 0; v196 < v211; v196++) {
        v197[c_custkey[v196]] = c_nationkey[v196];
    }
    const auto & cu_indexed = v197;

    timer.PrintElapsedTimeAndReset("Not Important1");

    auto v212 = p_partkey.size();

    for (int v198 = 0; v198 < v212; v198++) {
        v199.insert(p_partkey[v198], p_partkey[v198]);
    }

    timer.PrintElapsedTimeAndReset("Indexing");

    const auto & pa_indexed = v199;
    auto v213 = o_orderkey.size();
    for (int v200 = 0; v200 < v213; v200++) {
        if (((o_orderdate[v200] >= (long) 19950101) && (o_orderdate[v200] <= (long) 19961231))) {
        v201.emplace(o_orderkey[v200], make_tuple(o_custkey[v200], o_orderdate[v200]));
        };
    }

    const auto & ord_indexed = v201;
    const auto & v202 = pa_indexed;
    auto v214 = l_orderkey.size();

    timer.PrintElapsedTimeAndReset("Not Important2");

    v199.zip_apply((uint32_t*)&l_partkey[0], (uint32_t*)&l_id[0], v214, false, [&](auto& key, auto& value, auto& pay)
    {
        if ((((ord_indexed).contains(l_orderkey[pay])) && ((na_probed).contains(cu_indexed[ /* o_custkey */ get < 0 > ((ord_indexed).at(l_orderkey[pay]))])))) {
          v204[(( /* o_orderdate */ get < 1 > ((ord_indexed).at(l_orderkey[pay]))) / 10000)] += make_tuple((( /* n_name */ get < 0 > ((na_indexed).at( /* s_nationkey */ get < 0 > ((su_indexed).at(l_suppkey[pay])))) == brazil)) ? ((l_extendedprice[pay] * (1.0 - l_discount[pay]))) : (0.0), (l_extendedprice[pay] * (1.0 - l_discount[pay])));
        }
    });

    timer.PrintElapsedTimeAndReset("Probing");

    const auto & li_probed = v204;

    for (auto & v205: li_probed) {
        v206[make_tuple((v205.first), ( /* A */ get < 0 > ((v205.second)) / /* B */ get < 1 > ((v205.second))))] = true;
    }

    timer.PrintElapsedTimeAndReset("Finalization");

    return v206;
}

Q8_type q8_parallel()
{
    HighPrecisionTimer timer;

    auto v327 = phmap::flat_hash_map < long, tuple < long >> ({});
    tbb::enumerable_thread_specific < vector < pair < long, tuple < long >>> > v350;
    auto v330 = phmap::flat_hash_map < long, bool > ({});
    tbb::enumerable_thread_specific < vector < pair < long, bool >>> v357;
    auto v332 = phmap::flat_hash_map < long, tuple < VarChar < 25 >>> ({});
    tbb::enumerable_thread_specific < vector < pair < long, tuple < VarChar < 25 >>> >> v364;
    auto v334 = phmap::flat_hash_map < long, tuple < long >> ({});
    tbb::enumerable_thread_specific < vector < pair < long, tuple < long >>> > v371;
    vector < long > v336(200001);
    auto v338 = phmap::flat_hash_map < long, tuple < long >> (2 * 200000);
    tbb::enumerable_thread_specific < vector < pair < long, tuple < long >>> > v383;
    auto v340 = phmap::flat_hash_map < long, tuple < long, long >> ({});
    tbb::enumerable_thread_specific < vector < pair < long, tuple < long, long >>> > v390;
    auto v343 = phmap::flat_hash_map < long, tuple < double, double >> ({});
    tbb::enumerable_thread_specific < phmap::flat_hash_map < long, tuple < double, double >>> v397;
    auto v345 = phmap::flat_hash_map < tuple < long, double > , bool > ({});
    
    timer.PrintElapsedTimeAndReset("Initialization");

    const auto & steel = ConstantString("ECONOMY ANODIZED STEEL", 23);
    const auto & america = ConstantString("AMERICA", 8);
    const auto & brazil = ConstantString("BRAZIL", 7);

    auto v346 = r_regionkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v346), [ & ](const tbb::blocked_range < size_t > & v347) {
    auto & v351 = v350.local();
    for (size_t v326 = v347.begin(), end = v347.end(); v326 != end; ++v326) {
        if ((r_name[v326] == america)) {
        v351.emplace_back(r_regionkey[v326], make_tuple(r_regionkey[v326]));
        };
    }
    });
    for (auto & local: v350) v327.insert(local.begin(), local.end());
    const auto & re_indexed = v327;
    const auto & v328 = re_indexed;

    auto v353 = n_nationkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v353), [ & ](const tbb::blocked_range < size_t > & v354) {
    auto & v358 = v357.local();
    for (size_t v329 = v354.begin(), end = v354.end(); v329 != end; ++v329) {
        if (true) {
        if (((v328).contains(n_regionkey[v329]))) {
            v358.emplace_back(n_nationkey[v329], true);
        }
        };
    }
    });
    for (auto & local: v357) v330.insert(local.begin(), local.end());
    const auto & na_probed = v330;

    auto v360 = n_nationkey.size();;
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v360), [ & ](const tbb::blocked_range < size_t > & v361) {
    auto & v365 = v364.local();
    for (size_t v331 = v361.begin(), end = v361.end(); v331 != end; ++v331) {
        if (true) {
        v365.emplace_back(n_nationkey[v331], make_tuple(n_name[v331]));
        };
    }
    });
    for (auto & local: v364) v332.insert(local.begin(), local.end());
    const auto & na_indexed = v332;

    auto v367 = s_suppkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v367), [ & ](const tbb::blocked_range < size_t > & v368) {
    auto & v372 = v371.local();
    for (size_t v333 = v368.begin(), end = v368.end(); v333 != end; ++v333) {
        if (true) {
        v372.emplace_back(s_suppkey[v333], make_tuple(s_nationkey[v333]));
        };
    }
    });
    for (auto & local: v371) v334.insert(local.begin(), local.end());
    const auto & su_indexed = v334;

    auto v374 = c_custkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v374), [ & ](const tbb::blocked_range < size_t > & v375) {
    for (size_t v335 = v375.begin(), end = v375.end(); v335 != end; ++v335) {
        v336[c_custkey[v335]] = c_nationkey[v335];
    }
    });
    const auto & cu_indexed = v336;

    timer.PrintElapsedTimeAndReset("Not Important1");

    auto v379 = p_partkey.size();

    tbb::parallel_for(tbb::blocked_range < size_t > (0, v379), [ & ](const tbb::blocked_range < size_t > & v380) {
    auto & v384 = v383.local();
    for (size_t v337 = v380.begin(), end = v380.end(); v337 != end; ++v337) {
        v384.emplace_back(p_partkey[v337], make_tuple(p_partkey[v337]));
    }
    });
    for (auto & local: v383) v338.insert(local.begin(), local.end());

    const auto & pa_indexed = v338;

    timer.PrintElapsedTimeAndReset("Indexing");

    auto v386 = o_orderkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v386), [ & ](const tbb::blocked_range < size_t > & v387) {
    auto & v391 = v390.local();
    for (size_t v339 = v387.begin(), end = v387.end(); v339 != end; ++v339) {
        if (((o_orderdate[v339] >= (long) 19950101) && (o_orderdate[v339] <= (long) 19961231))) {
        v391.emplace_back(o_orderkey[v339], make_tuple(o_custkey[v339], o_orderdate[v339]));
        };
    }
    });
    for (auto & local: v390) v340.insert(local.begin(), local.end());
    const auto & ord_indexed = v340;
    const auto & v341 = pa_indexed;

    timer.PrintElapsedTimeAndReset("Not Important2");

    auto v393 = l_orderkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v393), [ & ](const tbb::blocked_range < size_t > & v394) {
    auto & v398 = v397.local();
    for (size_t v342 = v394.begin(), end = v394.end(); v342 != end; ++v342) {
        if (true) {
        if (((v341).contains(l_partkey[v342]))) {
            if ((((ord_indexed).contains(l_orderkey[v342])) && ((na_probed).contains(cu_indexed[ /* o_custkey */ get < 0 > ((ord_indexed).at(l_orderkey[v342]))])))) {
            v398[(( /* o_orderdate */ get < 1 > ((ord_indexed).at(l_orderkey[v342]))) / 10000)] += make_tuple((( /* n_name */ get < 0 > ((na_indexed).at( /* s_nationkey */ get < 0 > ((su_indexed).at(l_suppkey[v342])))) == brazil)) ? ((l_extendedprice[v342] * (1.0 - l_discount[v342]))) : (0.0), (l_extendedprice[v342] * (1.0 - l_discount[v342])));
            }
        }
        };
    }
    });
    for (auto & local: v397) AddMap < phmap::flat_hash_map < long, tuple < double, double >> , long, tuple < double, double >> (v343, local);
    const auto & li_probed = v343;

    timer.PrintElapsedTimeAndReset("Probing");

    for (auto & v344: li_probed) {
    v345[make_tuple((v344.first), ( /* A */ get < 0 > ((v344.second)) / /* B */ get < 1 > ((v344.second))))] = true;
    }

    timer.PrintElapsedTimeAndReset("Finalization");

    return v345;
}

Q8_type q8_parallel_vec()
{
    HighPrecisionTimer timer;

    auto v327 = phmap::flat_hash_map < long, tuple < long >> ({});
    tbb::enumerable_thread_specific < vector < pair < long, tuple < long >>> > v350;
    auto v330 = phmap::flat_hash_map < long, bool > ({});
    tbb::enumerable_thread_specific < vector < pair < long, bool >>> v357;
    auto v332 = phmap::flat_hash_map < long, tuple < VarChar < 25 >>> ({});
    tbb::enumerable_thread_specific < vector < pair < long, tuple < VarChar < 25 >>> >> v364;
    auto v334 = phmap::flat_hash_map < long, tuple < long >> ({});
    tbb::enumerable_thread_specific < vector < pair < long, tuple < long >>> > v371;
    vector < long > v336(200001);
    tbb::enumerable_thread_specific < vector < pair < long, tuple < long >>> > v383;
    auto v340 = phmap::flat_hash_map < long, tuple < long, long >> ({});
    tbb::enumerable_thread_specific < vector < pair < long, tuple < long, long >>> > v390;
    auto v343 = phmap::flat_hash_map < long, tuple < double, double >> ({});
    auto v345 = phmap::flat_hash_map < tuple < long, double > , bool > ({});
    tbb::enumerable_thread_specific < phmap::flat_hash_map < long, tuple < double, double >>> v397;
    auto v338 = vecht::lp_map(2*200000, 64, global_threads_count); 

    timer.PrintElapsedTimeAndReset("Initialization");

    const auto & steel = ConstantString("ECONOMY ANODIZED STEEL", 23);
    const auto & america = ConstantString("AMERICA", 8);
    const auto & brazil = ConstantString("BRAZIL", 7);
    auto v346 = r_regionkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v346), [ & ](const tbb::blocked_range < size_t > & v347) {
    auto & v351 = v350.local();
    for (size_t v326 = v347.begin(), end = v347.end(); v326 != end; ++v326) {
        if ((r_name[v326] == america)) {
        v351.emplace_back(r_regionkey[v326], make_tuple(r_regionkey[v326]));
        };
    }
    });
    for (auto & local: v350) v327.insert(local.begin(), local.end());
    const auto & re_indexed = v327;
    const auto & v328 = re_indexed;

    auto v353 = n_nationkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v353), [ & ](const tbb::blocked_range < size_t > & v354) {
    auto & v358 = v357.local();
    for (size_t v329 = v354.begin(), end = v354.end(); v329 != end; ++v329) {
        if (true) {
        if (((v328).contains(n_regionkey[v329]))) {
            v358.emplace_back(n_nationkey[v329], true);
        }
        };
    }
    });
    for (auto & local: v357) v330.insert(local.begin(), local.end());
    const auto & na_probed = v330;

    auto v360 = n_nationkey.size();;
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v360), [ & ](const tbb::blocked_range < size_t > & v361) {
    auto & v365 = v364.local();
    for (size_t v331 = v361.begin(), end = v361.end(); v331 != end; ++v331) {
        if (true) {
        v365.emplace_back(n_nationkey[v331], make_tuple(n_name[v331]));
        };
    }
    });
    for (auto & local: v364) v332.insert(local.begin(), local.end());
    const auto & na_indexed = v332;

    auto v367 = s_suppkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v367), [ & ](const tbb::blocked_range < size_t > & v368) {
    auto & v372 = v371.local();
    for (size_t v333 = v368.begin(), end = v368.end(); v333 != end; ++v333) {
        if (true) {
        v372.emplace_back(s_suppkey[v333], make_tuple(s_nationkey[v333]));
        };
    }
    });
    for (auto & local: v371) v334.insert(local.begin(), local.end());
    const auto & su_indexed = v334;

    auto v374 = c_custkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v374), [ & ](const tbb::blocked_range < size_t > & v375) {
    for (size_t v335 = v375.begin(), end = v375.end(); v335 != end; ++v335) {
        v336[c_custkey[v335]] = c_nationkey[v335];
    }
    });
    const auto & cu_indexed = v336;

    auto v379 = p_partkey.size();

    timer.PrintElapsedTimeAndReset("Not Important1");


    for (int v337 = 0; v337 < v379; v337++) {
        v338.insert(p_partkey[v337], p_partkey[v337]);
    }

    timer.PrintElapsedTimeAndReset("Indexing");

    const auto & pa_indexed = v338;

    auto v386 = o_orderkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v386), [ & ](const tbb::blocked_range < size_t > & v387) {
    auto & v391 = v390.local();
    for (size_t v339 = v387.begin(), end = v387.end(); v339 != end; ++v339) {
        if (((o_orderdate[v339] >= (long) 19950101) && (o_orderdate[v339] <= (long) 19961231))) {
        v391.emplace_back(o_orderkey[v339], make_tuple(o_custkey[v339], o_orderdate[v339]));
        };
    }
    });
    for (auto & local: v390) v340.insert(local.begin(), local.end());
    const auto & ord_indexed = v340;
    const auto & v341 = pa_indexed;

    auto v393 = l_orderkey.size();

    timer.PrintElapsedTimeAndReset("Not Important2");

    v338.zip_apply((uint32_t*)&l_partkey[0], (uint32_t*)&l_id[0], v393, false, [&](auto& key, auto& value, auto& pay)
    {
        auto& v398 = v397.local();
        if ((((ord_indexed).contains(l_orderkey[pay])) && ((na_probed).contains(cu_indexed[ /* o_custkey */ get < 0 > ((ord_indexed).at(l_orderkey[pay]))])))) {
            v398[(( /* o_orderdate */ get < 1 > ((ord_indexed).at(l_orderkey[pay]))) / 10000)] += make_tuple((( /* n_name */ get < 0 > ((na_indexed).at( /* s_nationkey */ get < 0 > ((su_indexed).at(l_suppkey[pay])))) == brazil)) ? ((l_extendedprice[pay] * (1.0 - l_discount[pay]))) : (0.0), (l_extendedprice[pay] * (1.0 - l_discount[pay])));
        }
    });
    for (auto & local: v397) AddMap < phmap::flat_hash_map < long, tuple < double, double >> , long, tuple < double, double >> (v343, local);
    
    const auto & li_probed = v343;

    timer.PrintElapsedTimeAndReset("Probing");

    for (auto & v344: li_probed) {
    v345[make_tuple((v344.first), ( /* A */ get < 0 > ((v344.second)) / /* B */ get < 1 > ((v344.second))))] = true;
    }

    timer.PrintElapsedTimeAndReset("Finalization");

    return v345;
}

// ============================

Q12_type q12_sequential()
{
    HighPrecisionTimer timer;

    auto v330 = phmap::flat_hash_map < long, VarChar < 15 >> (2 * 1500000);
    auto v335 = phmap::flat_hash_map < tuple < VarChar < 10 > , long, long > , bool > ({});
    auto v333 = phmap::flat_hash_map < tuple < VarChar < 10 >> , tuple < long, long >> ({});
    const auto & mail = ConstantString("MAIL", 5);
    const auto & ship = ConstantString("SHIP", 5);
    const auto & urgent = ConstantString("1-URGENT", 9);
    const auto & high = ConstantString("2-HIGH", 7);
    
    timer.PrintElapsedTimeAndReset("Initialization");

    auto v336 = o_orderkey.size();
    for (int v329 = 0; v329 < v336; v329++) {
        v330.emplace(o_orderkey[v329], o_orderpriority[v329]);
    }
    const auto & ord_indexed = v330;
    const auto & v331 = ord_indexed;

    timer.PrintElapsedTimeAndReset("Indexing");

    auto v337 = l_orderkey.size();
    for (int v332 = 0; v332 < v337; v332++) {
        if (((v331).contains(l_orderkey[v332]))) {
            v333[make_tuple(l_shipmode[v332])] += make_tuple(((((v331).at(l_orderkey[v332]) == urgent) || ((v331).at(l_orderkey[v332]) == high))) ? ((long) 1) : ((long) 0), ((((v331).at(l_orderkey[v332]) != urgent) && ((v331).at(l_orderkey[v332]) != high))) ? ((long) 1) : ((long) 0));
        }
    }

    timer.PrintElapsedTimeAndReset("Probing");

    const auto & li_probed = v333;

    for (auto & v334: li_probed) {
    v335[tuple_cat((v334.first), move((v334.second)))] = true;
    }
    const auto & results = v335;

    timer.PrintElapsedTimeAndReset("Finalization");

    return results;   
}

Q12_type q12_sequential_vec()
{
    HighPrecisionTimer timer;

    auto v330 = vecht::lp_map(2 * 1500000, 64, global_threads_count);
    auto v335 = phmap::flat_hash_map < tuple < VarChar < 10 > , long, long > , bool > ({});
    auto v333 = phmap::flat_hash_map < tuple < VarChar < 10 >> , tuple < long, long >> ({});
    const auto & mail = ConstantString("MAIL", 5);
    const auto & ship = ConstantString("SHIP", 5);
    const auto & urgent = ConstantString("1-URGENT", 9);
    const auto & high = ConstantString("2-HIGH", 7);
    
    timer.PrintElapsedTimeAndReset("Initialization");

    auto v336 = o_orderkey.size();
    for (int v329 = 0; v329 < v336; v329++) {
        v330.insert(o_orderkey[v329], o_id[v329]);
    }
    const auto & ord_indexed = v330;
    const auto & v331 = ord_indexed;

    timer.PrintElapsedTimeAndReset("Indexing");

    auto v337 = l_orderkey.size();

    v330.zip_apply((uint32_t*)&l_orderkey[0], (uint32_t*)&l_id[0], v337, false, [&](auto& key, auto& value, auto& pay) 
    {
        v333[make_tuple(l_shipmode[pay])] += make_tuple((((o_orderpriority[value] == urgent) || (o_orderpriority[value] == high))) ? ((long) 1) : ((long) 0), (((o_orderpriority[value] != urgent) && (o_orderpriority[value] != high))) ? ((long) 1) : ((long) 0));
    }); 

    timer.PrintElapsedTimeAndReset("Probing");

    const auto & li_probed = v333;

    for (auto & v334: li_probed) {
    v335[tuple_cat((v334.first), move((v334.second)))] = true;
    }
    const auto & results = v335;

    timer.PrintElapsedTimeAndReset("Finalization");

    return results;   
}

Q12_type q12_parallel()
{
    HighPrecisionTimer timer;

    auto v602 = phmap::flat_hash_map < long, VarChar < 15 >> (2 * 1500000);
    tbb::enumerable_thread_specific < vector < pair < long, VarChar < 15 >>> > v612;
    auto v605 = phmap::flat_hash_map < tuple < VarChar < 10 >> , tuple < long, long >> ({});
    tbb::enumerable_thread_specific < phmap::flat_hash_map < tuple < VarChar < 10 >> , tuple < long, long >>> v619;
    auto v607 = phmap::flat_hash_map < tuple < VarChar < 10 > , long, long > , bool > ({});

    const auto & mail = ConstantString("MAIL", 5);
    const auto & ship = ConstantString("SHIP", 5);
    const auto & urgent = ConstantString("1-URGENT", 9);
    const auto & high = ConstantString("2-HIGH", 7);
    auto v608 = o_orderkey.size();

    timer.PrintElapsedTimeAndReset("Initialization");

    tbb::parallel_for(tbb::blocked_range < size_t > (0, v608), [ & ](const tbb::blocked_range < size_t > & v609) {
    auto & v613 = v612.local();
    for (size_t v601 = v609.begin(), end = v609.end(); v601 != end; ++v601) {
        v613.emplace_back(o_orderkey[v601], o_orderpriority[v601]);
    }
    });
    for (auto & local: v612) v602.insert(local.begin(), local.end());
    
    const auto & ord_indexed = v602;
    const auto & v603 = ord_indexed;

    timer.PrintElapsedTimeAndReset("Indexing");

    auto v615 = l_orderkey.size();
    tbb::parallel_for(tbb::blocked_range < size_t > (0, v615), [ & ](const tbb::blocked_range < size_t > & v616) {
    auto & v620 = v619.local();
    for (size_t v604 = v616.begin(), end = v616.end(); v604 != end; ++v604) {
            if (((v603).contains(l_orderkey[v604]))) {
                v620[make_tuple(l_shipmode[v604])] += make_tuple(((((v603).at(l_orderkey[v604]) == urgent) || ((v603).at(l_orderkey[v604]) == high))) ? ((long) 1) : ((long) 0), ((((v603).at(l_orderkey[v604]) != urgent) && ((v603).at(l_orderkey[v604]) != high))) ? ((long) 1) : ((long) 0));
            }
    }
    });
    for (auto & local: v619) AddMap < phmap::flat_hash_map < tuple < VarChar < 10 >> , tuple < long, long >> , tuple < VarChar < 10 >> , tuple < long, long >> (v605, local);

    timer.PrintElapsedTimeAndReset("Probing");

    const auto & li_probed = v605;

    for (auto & v606: li_probed) {
    v607[tuple_cat((v606.first), move((v606.second)))] = true;
    }
    const auto & results = v607;

    timer.PrintElapsedTimeAndReset("Finalization");

    return results;
}

Q12_type q12_parallel_vec()
{
    HighPrecisionTimer timer;

    auto v602 = vecht::lp_map(2 * 1500000, 64, global_threads_count);
    tbb::enumerable_thread_specific < vector < pair < long, VarChar < 15 >>> > v612;
    auto v605 = phmap::flat_hash_map < tuple < VarChar < 10 >> , tuple < long, long >> ({});
    tbb::enumerable_thread_specific < phmap::flat_hash_map < tuple < VarChar < 10 >> , tuple < long, long >>> v619;
    auto v607 = phmap::flat_hash_map < tuple < VarChar < 10 > , long, long > , bool > ({});

    const auto & mail = ConstantString("MAIL", 5);
    const auto & ship = ConstantString("SHIP", 5);
    const auto & urgent = ConstantString("1-URGENT", 9);
    const auto & high = ConstantString("2-HIGH", 7);
    auto v608 = o_orderkey.size();

    timer.PrintElapsedTimeAndReset("Initialization");

    for (size_t v601 = 0; v601 < v608; ++v601) {
        v602.insert(o_orderkey[v601], o_id[v601]);
    }
  
    const auto & ord_indexed = v602;
    const auto & v603 = ord_indexed;

    timer.PrintElapsedTimeAndReset("Indexing");

    auto v615 = l_orderkey.size();

    v602.zip_apply((uint32_t*)&l_orderkey[0], (uint32_t*)&l_id[0], v615, false, [&](auto& key, auto& value, auto& pay) 
    {
        auto & v620 = v619.local();
        v620[make_tuple(l_shipmode[pay])] += make_tuple((((o_orderpriority[value] == urgent) || (o_orderpriority[value] == high))) ? ((long) 1) : ((long) 0), (((o_orderpriority[value] != urgent) && (o_orderpriority[value] != high))) ? ((long) 1) : ((long) 0));
    });
    for (auto & local: v619) AddMap < phmap::flat_hash_map < tuple < VarChar < 10 >> , tuple < long, long >> , tuple < VarChar < 10 >> , tuple < long, long >> (v605, local);

    timer.PrintElapsedTimeAndReset("Probing");

    const auto & li_probed = v605;

    for (auto & v606: li_probed) {
    v607[tuple_cat((v606.first), move((v606.second)))] = true;
    }
    const auto & results = v607;

    timer.PrintElapsedTimeAndReset("Finalization");

    return results;
}

// =================================================================================================

template <typename func_type>
inline void time(string name, func_type(*func)(), size_t threads, size_t iterations = 10, bool verbose = false)
{

    tbb::task_scheduler_init scheduler(threads);
    global_threads_count = threads;
    size_t warmup = 5;
    for (int i = 0; i < 5; i++)
        func(); // warmup

    if (verbose)
        cout << "=== Start: " << name << " =================" << endl;

    HighPrecisionTimer timer;

    for (int i = 0; i < iterations; i++)
    {
        if (verbose)
            cout << name << ": Iteration " << i << " ... \n";

        tbb::task_scheduler_init scheduler(threads);
        global_threads_count = threads;
        timer.Reset();
        func();
        timer.StoreElapsedTime(0);

        if (verbose)
            cout << "Done" << endl;

    }

    if (verbose)
    {
        cout << "=== End: " << name << " ================" << endl;
        cout << func() << endl;
        cout << "===================" << endl;
        cout << "Mean: " << timer.GetMean(0) << " ms | " << "StdDev: " << timer.GetStDev(0) << " ms" << endl;
        cout << "=========================================" << endl;
    }
    else
    {
        cout << name << "\t" << timer.GetMean(0) << endl;
    }
}

// =================================================================================================

int main (int argc, char *argv[])
{
    bool verbose = true;

    string dataset_path = "";
    if (argc > 1)
        dataset_path = argv[1];
    populate(dataset_path);
    size_t iterations = 5;

    // =================================================================================================
    // QueryName, QueryFunction, Threads, Iterations, Verbose
    // ===============================================================

    time<Q4_type>("Q4-1T", q4_sequential, 1, iterations, verbose);
    time<Q4_type>("Q4-1T-VecHT", q4_sequential_vec, 1, iterations, verbose);
    time<Q4_type>("Q4-4T", q4_parallel, 4, iterations, verbose);
    time<Q4_type>("Q4-4T-VecHT", q4_parallel_vec, 4, iterations, verbose);

    time<Q8_type>("Q8-1T", q8_sequential, 1, iterations, verbose);
    time<Q8_type>("Q8-1T-VecHT", q8_sequential_vec, 1, iterations, verbose);
    time<Q8_type>("Q8-4T", q8_parallel, 4, iterations, verbose);
    time<Q8_type>("Q8-4T-VecHT", q8_parallel_vec, 4, iterations, verbose);

    time<Q12_type>("Q12-1T", q12_sequential, 1, iterations, verbose);
    time<Q12_type>("Q12-1T-VecHT", q12_sequential_vec, 1, iterations, verbose);
    time<Q12_type>("Q12-4T", q12_parallel, 4, iterations, verbose);
    time<Q12_type>("Q12-4T-VecHT", q12_parallel_vec, 4, iterations, verbose);

    return 0;
}