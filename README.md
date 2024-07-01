# VecHT: An Efficient Hash Table for Batch Processing

As an implementation of our ([research work](https://drops.dagstuhl.de/storage/00lipics/lipics-vol263-ecoop2023/LIPIcs.ECOOP.2023.27/LIPIcs.ECOOP.2023.27.pdf)) presented at ECOOP'23, we propose VecHT, a novel hash table that leverages parallelization, vectorization (SIMD), and prefetching to improve the performance of batch lookups. VecHT offers a batch interface for the main operations on hash tables in a bulk fashion. For more information on the motivation, design, and benchmarks please refere to the referenced article.

## Disclaimer
This project is a research prototype. While it has been tested to the best of the authors' abilities, it is not guaranteed to be bug-free or suitable for deployment in production environments. Users are encouraged to thoroughly test the software in their own environments before any serious use.

The implementation details and performance characteristics are subject to change as the project evolves. The authors welcome feedback and contributions from the community to improve VecHT.

## Acknowledgments
This project reuses code from other open-source/research projects, and we have made sure to acknowledge this reuse in each related file where applicable. We are grateful to the original authors for their contributions to the open-source community, which have significantly aided in the development of this project. For specific details on the reused code and its origins, please refer to the header comments or documentation within each respective file.

## Project Requirements
This prototype is tested in the following environment:
* __Hardware__
    * At the moment, the project uses Intel TBB library and AVX2 instruction set and can be used on Intel CPUs.
* __Software__
    * Ubuntu 20.04.6 LTS
    * gcc 9.4.0
    * libtbb-dev 2020.1-2
    * Python 3.10 
    * TPC-H dataset with scaling factor of 1.

* __Notes:__
    * Install the software requirements before moving to the benchmarking section.
    * To prepare TPC-H dataset, download and follow the guidance on: `https://github.com/electrum/tpch-dbgen`


## Running Benchmarks
* Clone the repository: `git clone https://github.com/edin-dal/vecht.git`
* Export your TPC-H (SF1) dataset path: `export TPCH_PATH=[your_tpch_path]` 
* Change directory to benchmark folder: `cd benchmark`
* Build the source and benchmark files: `./build_all.sh`
* Execute the (end-to-end and micro) benchmarks: `./run_all.sh`
* Generate reports: `python generate_report.py`
* Reveiw the results stored in the `results` folder.