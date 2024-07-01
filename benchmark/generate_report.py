import os
import sys
if sys.version_info < (3, 10):
    print("Python 3.10 or newer is required")
    sys.exit(1)
import math
import pandas as pd
import matplotlib.pyplot as plt

def e2e_query():
    print("e2e_query report...")

    results = pd.DataFrame(columns=["query", "thread", "version", "build", "probe", "total"])

    data = ""
    with open("results/e2e_query.txt", "r") as f:
        data = f.read()

    data = data.split("\n")
    current_query = ""
    current_iteration = -1
    current_times = []
    for i in range(len(data)):
        line = data[i]
        if line.startswith("=== Start: "):
            current_query = line[11:-18]
        elif line.startswith(current_query + ": Iteration "):
            current_iteration = int(line.split(" ")[2])
        elif line.startswith(">> "):
            if current_query == "":
                continue
            time = float(line.split(" ")[1])
            token = line.split("\t")[2]
            if current_iteration == len(current_times):
                current_times.append({})
            current_times[current_iteration][token] = time
        elif line.startswith("Mean: "):
            build = 0
            probe = 0
            total = 0
            for j in range(len(current_times)):
                for (k, v) in current_times[j].items():
                    if k == "Indexing":
                        build += v
                    elif k == "Probing":
                        probe += v
                    total += v
            build /= (0.0+len(current_times))
            probe /= (0.0+len(current_times))
            total /= (0.0+len(current_times))
            thread = current_query.split("-")[1][0]
            version = "vec" if len(current_query.split("-")) == 3 else "std"
            query = int(current_query.split("-")[0][1:])
            results = pd.concat([results, pd.DataFrame({"query": [query], "thread": [thread], "version": [version], "build": [build], "probe": [probe], "total": [total]})])
            current_times = []
            current_query = ""
            current_iteration = -1
        elif line in ("Done\n"):
            continue

    results = pd.pivot_table(results, index=["query", "thread"], columns="version", values=["build", "probe", "total"])
    results = results.reset_index()
    results.columns = ["query", "thread", "build_std", "build_vec", "probe_std", "probe_vec", "total_std", "total_vec"]
    results["build_speedup"] = round(results["build_std"] / results["build_vec"], 2)
    results["probe_speedup"] = round(results["probe_std"] / results["probe_vec"], 2)
    results["total_speedup"] = round(results["total_std"] / results["total_vec"], 2)
    results["query"] = results["query"].apply(lambda x: "q" + str(x))  

    print("Saving results to results/e2e_query_summary.csv")
    results.to_csv("results/e2e_query_summary.csv", index=False)    

def e2e_other():
    print("e2e_other report...")

    fixed_size = 16777216 
    data = pd.read_csv("results/e2e_other.csv", index_col=False)
    data["size"] = data["size"].apply(lambda x: math.log2(x) - math.log2(fixed_size))

    # Vector Inner Product data
    data_vip = data[["size", "thread", "vecht-inner", "phmap-inner", "pphmap-inner", "tbb-inner", "cuckoo-inner"]]
    data_vip.columns = ["size", "thread", "vecht", "phmap", "pphmap", "tbb", "cuckoo"]
    data_vip_1 = data_vip[data_vip["thread"] == 1].drop(columns=["thread"])
    data_vip_4 = data_vip[data_vip["thread"] == 4].drop(columns=["thread"])

    # Vector Pairwise Multiplication data
    data_vpm = data[["size", "thread", "vecht-elem", "phmap-elem", "pphmap-elem", "tbb-elem", "cuckoo-elem"]]
    data_vpm.columns = ["size", "thread", "vecht", "phmap", "pphmap", "tbb", "cuckoo"]
    data_vpm_1 = data_vpm[data_vpm["thread"] == 1].drop(columns=["thread"])
    data_vpm_4 = data_vpm[data_vpm["thread"] == 4].drop(columns=["thread"])

    # Set Intersection data
    data_si = data[["size", "thread", "vecht-inter", "phmap-inter", "pphmap-inter", "tbb-inter", "cuckoo-inter"]]
    data_si.columns = ["size", "thread", "vecht", "phmap", "pphmap", "tbb", "cuckoo"]
    data_si_1 = data_si[data_si["thread"] == 1].drop(columns=["thread"])
    data_si_4 = data_si[data_si["thread"] == 4].drop(columns=["thread"])

    # Set Difference data
    data_sd = data[["size", "thread", "vecht-diff", "phmap-diff", "pphmap-diff", "tbb-diff", "cuckoo-diff"]]
    data_sd.columns = ["size", "thread", "vecht", "phmap", "pphmap", "tbb", "cuckoo"]
    data_sd_1 = data_sd[data_sd["thread"] == 1].drop(columns=["thread"])
    data_sd_4 = data_sd[data_sd["thread"] == 4].drop(columns=["thread"])

    # chart styles
    colors = ["purple", "lightgreen", "brown", "lightskyblue", "darkgreen"]
    markers = ["o", "o", "o", "o", "o"]
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linewidth'] = 0.5
    
    fig = plt.figure(figsize=(10, 10))
    handles, labels = [], []

    # Vector Inner Product plots
    plt.subplot(4, 2, 1)
    for i in range(5):
        line, = plt.plot(data_vip_1["size"], data_vip_1.iloc[:, i+1], color=colors[i], marker=markers[i], label=data_vip_1.columns[i+1])
        handles.append(line)
        labels.append(data_vip_1.columns[i+1])

    plt.title("Vector Inner Product (Thread=1)")
    plt.xlabel("Log of V2 Density")
    plt.ylabel("Run Time (ms)")
    ##
    plt.subplot(4, 2, 2)
    for i in range(5):
        plt.plot(data_vip_4["size"], data_vip_4.iloc[:, i+1], color=colors[i], marker=markers[i], label=data_vip_4.columns[i+1])
    plt.title("Vector Inner Product (Thread=4)")
    plt.xlabel("Log of V2 Density")
    plt.ylabel("Run Time (ms)")

    # Vector Pairwise Multiplication plots
    plt.subplot(4, 2, 3)
    for i in range(5):
        plt.plot(data_vpm_1["size"], data_vpm_1.iloc[:, i+1], color=colors[i], marker=markers[i], label=data_vpm_1.columns[i+1])
    plt.title("Vector Pairwise Multiplication (Thread=1)")
    plt.xlabel("Log of V2 Density")
    plt.ylabel("Run Time (ms)")
    ##
    plt.subplot(4, 2, 4)
    for i in range(5):
        plt.plot(data_vpm_4["size"], data_vpm_4.iloc[:, i+1], color=colors[i], marker=markers[i], label=data_vpm_4.columns[i+1])
    plt.title("Vector Pairwise Multiplication (Thread=4)")
    plt.xlabel("Log of V2 Density")
    plt.ylabel("Run Time (ms)")

    # Set Difference plots
    plt.subplot(4, 2, 5)
    for i in range(5):
        plt.plot(data_sd_1["size"], data_sd_1.iloc[:, i+1], color=colors[i], marker=markers[i], label=data_sd_1.columns[i+1])
    plt.title("Set Difference (Thread=1)")
    plt.xlabel("Log of S2 Density")
    plt.ylabel("Run Time (ms)")

    ##
    plt.subplot(4, 2, 6)
    for i in range(5):
        plt.plot(data_sd_4["size"], data_sd_4.iloc[:, i+1], color=colors[i], marker=markers[i], label=data_sd_4.columns[i+1])
    plt.title("Set Difference (Thread=4)")
    plt.xlabel("Log of S2 Density")
    plt.ylabel("Run Time (ms)")

    # Set Intersection plots
    plt.subplot(4, 2, 7)
    for i in range(5):
        plt.plot(data_si_1["size"], data_si_1.iloc[:, i+1], color=colors[i], marker=markers[i], label=data_si_1.columns[i+1])
    plt.title("Set Intersection (Thread=1)")
    plt.xlabel("Log of S2 Density")
    plt.ylabel("Run Time (ms)")

    ##
    plt.subplot(4, 2, 8)
    for i in range(5):
        plt.plot(data_si_4["size"], data_si_4.iloc[:, i+1], color=colors[i], marker=markers[i], label=data_si_4.columns[i+1])
    plt.title("Set Intersection (Thread=4)")
    plt.xlabel("Log of S2 Density")
    plt.ylabel("Run Time (ms)")

    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.savefig("results/e2e_other.pdf", bbox_inches='tight')
    plt.close()

    print("Plots are saved in results/e2e_other.pdf")

def prepare_micro_data():

    data = pd.DataFrame(columns=["thread", "sel", "size", "scalar_pref", "simd_pref_mem_nbuf", "simd_pref_nmem_nbuf", "simd_pref_mem_buf", "simd_prefn", "tbb", "simd_pref_nmem_buf", "pphmap", "simd_prefp", "simd", "scalar", "phmap", "cuckoo"])

    files = sorted([f for f in os.listdir("results/micro") if f.endswith(".csv")])
    for f in files:
        tokens = f.split("-")
        thread = tokens[1]
        sel = tokens[3][:-4]
        try:
            file_content = pd.read_csv("results/micro/" + f, index_col=False)
        except:
            print("Error reading " + f)
            continue
        
        file_content.insert(0, "thread", thread)
        file_content.insert(1, "sel", sel)
        file_content.columns = ["thread", "sel", "size", "scalar_pref", "simd_pref_mem_nbuf", "simd_pref_nmem_nbuf", "simd_pref_mem_buf", "simd_prefn", "tbb", "simd_pref_nmem_buf", "pphmap", 
                                "simd_prefp", "simd", "scalar", "phmap", "cuckoo"]
        file_content.iloc[:, 3:] = file_content.iloc[:, 3:].apply(lambda x: round(x, 0))
        data = pd.concat([data, file_content], ignore_index=True)

    return data

def micro_join(data):
    # Join Micro-Benchmark data
    join_data = data.copy()
    join_data = join_data[['thread', 'sel', 'size', 'simd_pref_mem_nbuf', 'phmap', 'pphmap', 'tbb', 'cuckoo']]
    join_data.rename(columns={"simd_pref_mem_nbuf": "vecht"}, inplace=True)

    # chart styles
    colors = ["darkslateblue", "lightgreen", "brown", "lightblue", "darkgreen"]
    markers = ["o", "o", "o", "o", "o"]
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linewidth'] = 0.5

    fig = plt.figure(figsize=(10, 8))
    handles, labels = [], []

    ### thread=1 | sel=0.1 
    plt.subplot(4, 2, 1)
    plt_data = join_data[(join_data["thread"] == "1") & (join_data["sel"] == "0.1")]
    for i in range(5):
        line, = plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
        handles.append(line)
        labels.append(join_data.columns[i+3])
    plt.title("Selectivity=0.1 | Thread=1")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### thread=4 | sel=0.1
    plt.subplot(4, 2, 2)
    plt_data = join_data[(join_data["thread"] == "4") & (join_data["sel"] == "0.1")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=0.1 | Thread=4")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### thread=1 | sel=0.5
    plt.subplot(4, 2, 3)
    plt_data = join_data[(join_data["thread"] == "1") & (join_data["sel"] == "0.5")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=0.5 | Thread=1")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### thread=4 | sel=0.5
    plt.subplot(4, 2, 4)
    plt_data = join_data[(join_data["thread"] == "4") & (join_data["sel"] == "0.5")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=0.5 | Thread=4")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### thread=1 | sel=1
    plt.subplot(4, 2, 5)
    plt_data = join_data[(join_data["thread"] == "1") & (join_data["sel"] == "1")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=1 | Thread=1")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### thread=4 | sel=1
    plt.subplot(4, 2, 6)
    plt_data = join_data[(join_data["thread"] == "4") & (join_data["sel"] == "1")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=1 | Thread=4")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.savefig("results/micro_join.pdf", bbox_inches='tight')
    plt.close()

    print("Plots are saved in results/micro_join.pdf")

def micro_pref_memo(data):
    # Prefetch Memo Micro-Benchmark data
    memo_data = data.copy()
    memo_data = memo_data[['thread', 'sel', 'size', 'scalar', 'scalar_pref', 'simd', 'simd_pref_nmem_nbuf', 'simd_pref_mem_nbuf']] 
    memo_data.rename(columns={"simd_pref_nmem_nbuf": "simd_prefetching_nomemo"}, inplace=True)
    memo_data.rename(columns={"simd_pref_mem_nbuf": "simd_prefetching_memo"}, inplace=True)

    # chart styles
    colors = ["green", "darkblue", "gold", "orangered", "lightskyblue"]
    markers = ["o", "o", "o", "o", "o"]
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linewidth'] = 0.5

    fig = plt.figure(figsize=(10, 8))
    handles, labels = [], []

    ### sel=0.1 
    plt.subplot(4, 2, 1)
    plt_data = memo_data[(memo_data["sel"] == "0.1") & (memo_data["thread"] == "1")]
    for i in range(5):
        line, = plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
        handles.append(line)
        labels.append(memo_data.columns[i+3])
    plt.title("Selectivity=0.1")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### sel=0.5
    plt.subplot(4, 2, 2)
    plt_data = memo_data[(memo_data["sel"] == "0.5") & (memo_data["thread"] == "1")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=0.5")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### sel=1
    plt.subplot(4, 2, 3)
    plt_data = memo_data[(memo_data["sel"] == "1") & (memo_data["thread"] == "1")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=1")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.savefig("results/micro_memo.pdf", bbox_inches='tight')
    plt.close()
    print("Plots are saved in results/micro_memo.pdf")

def micro_pref_optipessi(data):
    # Prefetch Optimistic-Pessimistic Micro-Benchmark data
    optipessi_data = data.copy()
    optipessi_data = optipessi_data[['thread', 'sel', 'size', 'scalar', 'scalar_pref', 'simd', 'simd_pref_mem_nbuf', 'simd_prefp']]
    optipessi_data.rename(columns={"simd_pref_mem_nbuf": "simd_prefetching_optimistic"}, inplace=True)
    optipessi_data.rename(columns={"simd_prefp": "simd_prefetching_pessimistic"}, inplace=True)

    # chart styles
    colors = ["green", "darkblue", "gold", "lightskyblue", "orangered"]
    markers = ["o", "o", "o", "o", "o"]
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linewidth'] = 0.5

    fig = plt.figure(figsize=(10, 8))
    handles, labels = [], []

    ### sel=0.1
    plt.subplot(4, 2, 1)
    plt_data = optipessi_data[(optipessi_data["sel"] == "0.1") & (optipessi_data["thread"] == "1")]
    for i in range(5):
        line, = plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
        handles.append(line)
        labels.append(optipessi_data.columns[i+3])
    plt.title("Selectivity=0.1")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### sel=0.5
    plt.subplot(4, 2, 2)
    plt_data = optipessi_data[(optipessi_data["sel"] == "0.5") & (optipessi_data["thread"] == "1")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=0.5")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### sel=1
    plt.subplot(4, 2, 3)
    plt_data = optipessi_data[(optipessi_data["sel"] == "1") & (optipessi_data["thread"] == "1")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=1")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.savefig("results/micro_optipessi.pdf", bbox_inches='tight')
    plt.close()
    print("Plots are saved in results/micro_optipessi.pdf")

def micro_pref_stdgrp(data):
    # Prefetch Standard vs Group Micro-Benchmark data
    stdgrp_data = data.copy()
    stdgrp_data = stdgrp_data[['thread', 'sel', 'size', 'scalar', 'scalar_pref', 'simd', 'simd_prefn', 'simd_pref_nmem_buf']]
    stdgrp_data.rename(columns={"simd_prefn": "simd_prefetching_standard"}, inplace=True)
    stdgrp_data.rename(columns={"simd_pref_nmem_buf": "simd_prefetching_group"}, inplace=True)

    # chart styles
    colors = ["green", "darkblue", "gold", "orangered", "lightskyblue"]
    markers = ["o", "o", "o", "o", "o"]
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linewidth'] = 0.5

    fig = plt.figure(figsize=(10, 8))
    handles, labels = [], []

    ### sel=0.1
    plt.subplot(4, 2, 1)
    plt_data = stdgrp_data[(stdgrp_data["sel"] == "0.1") & (stdgrp_data["thread"] == "1")]
    for i in range(5):
        line, = plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
        handles.append(line)
        labels.append(stdgrp_data.columns[i+3])
    plt.title("Selectivity=0.1")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### sel=0.5
    plt.subplot(4, 2, 2)
    plt_data = stdgrp_data[(stdgrp_data["sel"] == "0.5") & (stdgrp_data["thread"] == "1")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=0.5")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    ### sel=1
    plt.subplot(4, 2, 3)
    plt_data = stdgrp_data[(stdgrp_data["sel"] == "1") & (stdgrp_data["thread"] == "1")]
    for i in range(5):
        plt.plot(plt_data["size"], plt_data.iloc[:, i+3], color=colors[i], marker=markers[i], label=plt_data.columns[i+3])
    plt.title("Selectivity=1")
    plt.xlabel("Log of Hash Table Size (Bytes)")
    plt.ylabel("Tuple per Second (Million)")
    plt.xlim(20, 29)

    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.savefig("results/micro_stdgrp.pdf", bbox_inches='tight')
    plt.close()

###############################################################################################

def main():
    print("Generating report...")
    e2e_query()
    e2e_other()
    micro_data = prepare_micro_data()
    micro_join(micro_data)
    micro_pref_memo(micro_data)
    micro_pref_optipessi(micro_data)
    micro_pref_stdgrp(micro_data)

if __name__ == "__main__":
    main()