/* Copyright (c) 2015
 * The Trustees of Columbia University in the City of New York
 * All rights reserved.
 *
 * Author:  Orestis Polychroniou  (orestis@cs.columbia.edu)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "rand.h"


static int hardware_threads(void)
{
	char name[64];
	struct stat st;
	int threads = -1;
	do {
		sprintf(name, "/sys/devices/system/cpu/cpu%d", ++threads);
	} while (stat(name, &st) == 0);
	return threads;
}

static void *mamalloc(size_t size)
{
	void *p = NULL;
	return posix_memalign(&p, 64, size) ? NULL : p;
}

typedef struct {
	pthread_t id;
	int seed;
	int thread;
	int threads;
	uint32_t hash_factor;
	uint32_t invalid_key;
	uint32_t *inner;
	uint32_t *outer;
	volatile uint32_t *table;
	size_t inner_size;
	size_t outer_size;
	size_t table_size;
	size_t join_size;
	double selectivity;
	pthread_barrier_t *barrier;
} info_t;

static void *run(void *arg)
{
	info_t *d = (info_t*) arg;
	assert(pthread_equal(pthread_self(), d->id));
	int thread = d->thread;
	int threads = d->threads;
	uint32_t hash_factor = d->hash_factor;
	uint32_t invalid_key = d->invalid_key;
	uint32_t *inner = d->inner;
	uint32_t *outer = d->outer;
	volatile uint32_t *table = d->table;
	size_t i, o, t, h;
	size_t inner_size = d->inner_size;
	size_t outer_size = d->outer_size;
	size_t table_size = d->table_size;
	size_t inner_beg = (inner_size / threads) *  thread;
	size_t inner_end = (inner_size / threads) * (thread + 1);
	size_t outer_beg = (outer_size / threads) *  thread;
	size_t outer_end = (outer_size / threads) * (thread + 1);
	size_t table_beg = (table_size / threads) *  thread;
	size_t table_end = (table_size / threads) * (thread + 1);
	if (thread + 1 == threads) {
		inner_end = inner_size;
		outer_end = outer_size;
		table_end = table_size;
	}
	for (t = table_beg ; t != table_end ; ++t)
		table[t] = invalid_key;
	pthread_barrier_wait(&d->barrier[0]);
	rand32_t *gen = rand32_init(d->seed);
	for (i = inner_beg ; i != inner_end ; ++i) {
		int new_key_inserted = 0;
		uint32_t key;
		do {
			do {
				key = rand32_next(gen);
			} while (key == invalid_key);
			h = (uint32_t) (key * hash_factor);
			h = (h * table_size) >> 32;
			for (;;) {
				if (table[h] == invalid_key &&
				    __sync_bool_compare_and_swap(&table[h], invalid_key, key)) {
				  	new_key_inserted = 1;
					break;
				}
				if (table[h] == key) break;
				if (++h == table_size) h = 0;
			}
		} while (new_key_inserted == 0);
		inner[i] = key;
	}
	pthread_barrier_wait(&d->barrier[1]);
	size_t join_size = 0;
	uint32_t limit = ~0;
	limit *= d->selectivity;
	for (o = outer_beg ; o != outer_end ; ++o) {
		uint32_t key;
		if (rand32_next(gen) <= limit) {
			i = rand32_next(gen);
			i = (i * inner_size) >> 32;
			key = inner[i];
			join_size++;
		} else do {
			do {
				key = rand32_next(gen);
			} while (key == invalid_key);
			h = (uint32_t) (key * hash_factor);
			h = (h * table_size) >> 32;
			while (table[h] != invalid_key) {
				if (table[h] == key) break;
				if (++h == table_size) h = 0;
			}
		} while (table[h] == key);
		outer[o] = key;
	}
	free(gen);
	d->join_size = join_size;
	pthread_exit(NULL);
}

size_t inner_outer(size_t inner_size, size_t outer_size, double selectivity,
                   uint32_t **inner_p, uint32_t **outer_p)
{
	srand(time(NULL));
	int t, threads = hardware_threads();
	// input arguments
	assert(inner_size <= 1000 * 1000 * 1000);
	assert(selectivity >= 0.0 && selectivity <= 1.0);
	// tables
	uint32_t *inner = (uint32_t*) mamalloc((inner_size + 1) * sizeof(uint32_t));
	uint32_t *outer = (uint32_t*) mamalloc(outer_size * sizeof(uint32_t));
	size_t table_size = inner_size / 0.7;
	uint32_t *table = (uint32_t*) malloc(table_size * sizeof(uint32_t));
	// constants
	uint32_t hash_factor = (rand() << 1) | 1;
	uint32_t invalid_key = rand() * rand();
	// barriers
	int b, barriers = 2;
	pthread_barrier_t barrier[barriers];
	for (b = 0 ; b != barriers ; ++b)
		pthread_barrier_init(&barrier[b], NULL, threads);
	// run threads
	info_t info[threads];
	for (t = 0 ; t != threads ; ++t) {
		info[t].seed = rand();
		info[t].thread = t;
		info[t].threads = threads;
		info[t].hash_factor = hash_factor;
		info[t].invalid_key = invalid_key;
		info[t].selectivity = selectivity;
		info[t].inner = inner;
		info[t].outer = outer;
		info[t].table = table;
		info[t].inner_size = inner_size;
		info[t].outer_size = outer_size;
		info[t].table_size = table_size;
		info[t].barrier = barrier;
		pthread_create(&info[t].id, NULL, run, (void*) &info[t]);
	}
	size_t join_size = 0;
	for (t = 0 ; t != threads ; ++t) {
		pthread_join(info[t].id, NULL);
		join_size += info[t].join_size;
	}
	// cleanup
	for (b = 0 ; b != barriers ; ++b)
		pthread_barrier_destroy(&barrier[b]);
	free(table);
	// pass output
	inner[inner_size] = invalid_key;
	*inner_p = inner;
	*outer_p = outer;
	return join_size;
}

#ifdef _MAIN

int main(int argc, char **argv)
{
	size_t i, j, o;
	uint32_t *inner = NULL;
	uint32_t *outer = NULL;
	size_t inner_size = argc > 1 ? atoll(argv[1]) : 100;
	size_t outer_size = argc > 2 ? atoll(argv[2]) : 10000;
	double selectivity = argc > 3 ? atof(argv[3]) : 0.5;
	if (selectivity < 0.0 || selectivity > 1.0) {
		fprintf(stderr, "Invalid selectivity\n");
		return EXIT_FAILURE;
	}
	size_t join_size = inner_outer(inner_size, outer_size, selectivity, &inner, &outer);
	fprintf(stderr, "Selectivity: %.2f%%\n", join_size * 100.0 / outer_size);
	for (j = o = 0 ; o != outer_size ; ++o) {
		for (i = 0 ; i != inner_size ; ++i)
			if (inner[i] == outer[o]) break;
		if (i != inner_size) j++;
	}
	if (j != join_size)
		fprintf(stderr, "Error: %ld != %ld\n", j, join_size);
	free(inner);
	free(outer);
	return j == join_size ? EXIT_SUCCESS : EXIT_FAILURE;
}

#endif
