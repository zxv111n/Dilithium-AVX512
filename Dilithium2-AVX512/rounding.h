#ifndef ROUNDING_H
#define ROUNDING_H

#include <stdint.h>
#include "params.h"
#include <immintrin.h>




void power2round_avx(__m512i *a1, __m512i *a0, const __m512i *a);
void decompose_avx(__m512i *a1, __m512i *a0, const __m512i *a);
unsigned int make_hint_avx(__m512i * restrict hint, const __m512i * restrict a0, const __m512i * restrict a1);
void use_hint_avx(__m512i *b, const __m512i *a, const __m512i * restrict hint);

#endif
