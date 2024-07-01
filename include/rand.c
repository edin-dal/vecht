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


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>


typedef struct rand_state_64 {
	uint64_t num[313];
	size_t index;
} rand64_t;

rand64_t *rand64_init(uint64_t seed)
{
	rand64_t *state = (rand64_t*) malloc(sizeof(rand64_t));
	uint64_t *n = state->num;
	size_t i;
	n[0] = seed;
	for (i = 0 ; i != 311 ; ++i)
		n[i + 1] = 6364136223846793005ull *
		           (n[i] ^ (n[i] >> 62)) + i + 1;
	state->index = 312;
	return state;
}

uint64_t rand64_next(rand64_t *state)
{
	uint64_t x, *n = state->num;
	if (state->index == 312) {
		size_t i = 0;
		do {
			x = n[i] & 0xffffffff80000000ull;
			x |= n[i + 1] & 0x7fffffffull;
			n[i] = n[i + 156] ^ (x >> 1);
			n[i] ^= 0xb5026f5aa96619e9ull & -(x & 1);
		} while (++i != 156);
		n[312] = n[0];
		do {
			x = n[i] & 0xffffffff80000000ull;
			x |= n[i + 1] & 0x7fffffffull;
			n[i] = n[i - 156] ^ (x >> 1);
			n[i] ^= 0xb5026f5aa96619e9ull & -(x & 1);
		} while (++i != 312);
		state->index = 0;
	}
	x = n[state->index++];
	x ^= (x >> 29) & 0x5555555555555555ull;
	x ^= (x << 17) & 0x71d67fffeda60000ull;
	x ^= (x << 37) & 0xfff7eee000000000ull;
	x ^= (x >> 43);
	return x;
}

typedef struct rand_state_32 {
	uint32_t num[625];
	size_t index;
} rand32_t;

rand32_t *rand32_init(uint32_t seed)
{
	rand32_t *state = (rand32_t*) malloc(sizeof(rand32_t));
	uint32_t *n = state->num;
	size_t i;
	n[0] = seed;
	for (i = 0 ; i != 623 ; ++i)
		n[i + 1] = 0x6c078965 * (n[i] ^ (n[i] >> 30));
	state->index = 624;
	return state;
}

uint32_t rand32_next(rand32_t *state)
{
	uint32_t y, *n = state->num;
	if (state->index == 624) {
		size_t i = 0;
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i + 397] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 227);
		n[624] = n[0];
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i - 227] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 624);
		state->index = 0;
	}
	y = n[state->index++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680;
	y ^= (y << 15) & 0xefc60000;
	y ^= (y >> 18);
	return y;
}
