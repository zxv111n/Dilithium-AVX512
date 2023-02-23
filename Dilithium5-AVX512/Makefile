CC ?= /usr/bin/cc
CFLAGS += -Wall -Wextra -Wpedantic -Wmissing-prototypes -Wredundant-decls \
  -Wshadow -Wpointer-arith -mavx2 -mavx512f -mavx512vbmi -mavx512bw -mavx512cd -mavx512vl -mpopcnt -maes \
  -march=native -mtune=native -O3
NISTFLAGS += -Wno-unused-result -O3
SOURCES = sign.c packing.c polyvec.c poly.c rounding.c align.h consts.c \
  ntt.S invntt.S pointwise.S rejsample.c smallntt.S shuffle8.S shuffle.S tailoredntt.S
HEADERS = config.h params.h api.h sign.h packing.h polyvec.h poly.h ntt.h consts.h\
   rounding.h symmetric.h randombytes.h shuffle.inc rejsample.h shuffle8.h
KECCAK_SOURCES = $(SOURCES) fips202.c symmetric-shake.c fips202x8.c keccak8x/KeccakP-1600-times8-SIMD512.o
KECCAK_HEADERS = $(HEADERS) fips202.h fips202x8.h
AES_SOURCES = $(SOURCES) fips202.c aes256ctr.c symmetric-aes.c
AES_HEADERS = $(HEADERS) fips202.h aes256ctr.h

.PHONY: all 

all: keccak8x/KeccakP-1600-times8-SIMD512.o test/test_dilithium5 test/test_speed5 PQCgenKAT_sign


test/test_dilithium5: test/test_dilithium.c randombytes.c $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=5 \
	  -o $@ $< randombytes.c $(KECCAK_SOURCES)

test/test_speed5: test/test_speed.c test/speed_print.c test/speed_print.h \
  test/cpucycles.c test/cpucycles.h randombytes.c $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=5 \
	  -o $@ $< test/speed_print.c test/cpucycles.c randombytes.c \
	  $(KECCAK_SOURCES)

PQCgenKAT_sign: PQCgenKAT_sign.c rng.c rng.h $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -o $@ $< rng.c $(KECCAK_SOURCES) $(CFLAGS) -lcrypto


keccak8x/KeccakP-1600-times8-SIMD512.o: keccak8x/align.h \
										keccak8x/brg_endian.h \
										keccak8x/KeccakP-1600-times8-SIMD512.c \
										keccak8x/KeccakP-1600-times8-SnP.h \
										keccak8x/KeccakP-1600-unrolling.macros \
										keccak8x/SIMD512-config.h
	$(CC) $(CFLAGS) -c keccak8x/KeccakP-1600-times8-SIMD512.c -o $@

clean:
	rm -f test/test_dilithium5
	rm -f test/test_speed5
	rm -f PQCgenKAT_sign