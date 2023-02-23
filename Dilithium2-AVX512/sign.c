#include <stdint.h>
#include "params.h"
#include "sign.h"
#include "packing.h"
#include "polyvec.h"
#include "poly.h"
#include "randombytes.h"
#include "symmetric.h"
#include "fips202.h"
#include <string.h>

/*************************************************
* Name:        crypto_sign_keypair
*
* Description: Generates public and private key.
*
* Arguments:   - uint8_t *pk: pointer to output public key (allocated
*                             array of CRYPTO_PUBLICKEYBYTES bytes)
*              - uint8_t *sk: pointer to output private key (allocated
*                             array of CRYPTO_SECRETKEYBYTES bytes)
*
* Returns 0 (success)
**************************************************/
int crypto_sign_keypair(uint8_t *pk, uint8_t *sk) {
  uint8_t seedbuf[2*SEEDBYTES + CRHBYTES];
  const uint8_t *rho, *rhoprime, *key;
  int i;
  polyvecl mat[K];
  polyvecl s1;
  polyveck s2, t1, t0;
  /* Get randomness for rho, rhoprime and key */
  randombytes(seedbuf, SEEDBYTES);
  shake256(seedbuf, 2*SEEDBYTES + CRHBYTES, seedbuf, SEEDBYTES);
  rho = seedbuf;
  rhoprime = rho + SEEDBYTES;
  key = rhoprime + CRHBYTES;

  /* Store rho, key */
  memcpy(pk, rho, SEEDBYTES);
  memcpy(sk, rho, SEEDBYTES);
  memcpy(sk + SEEDBYTES, key, SEEDBYTES);

  /* Expand matrix */
  polyvec_matrix_expand(mat, rho);

  /* Sample short vectors s1 and s2 */
  #if K == 4 && L == 4
    poly_uniform_eta_8x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], &s2.vec[0], &s2.vec[1], &s2.vec[2], &s2.vec[3], rhoprime, 0, 1, 2, 3, 4, 5, 6, 7);
  #elif K == 6 && L == 5
    poly tmp;
    poly_uniform_eta_8x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], &s1.vec[4], &s2.vec[0], &s2.vec[1], &s2.vec[2], rhoprime, 0, 1, 2, 3, 4, 5, 6, 7);
    poly_uniform_eta_8x(&s2.vec[3], &s2.vec[4], &s2.vec[5], &tmp, &tmp, &tmp, &tmp, &tmp, rhoprime, 8, 9, 10, 11, 12, 13, 14, 15);
  #elif K == 8 && L == 7
    poly tmp;
    poly_uniform_eta_8x(&s1.vec[0], &s1.vec[1], &s1.vec[2], &s1.vec[3], &s1.vec[4], &s1.vec[5], &s1.vec[6], &tmp, rhoprime, 0, 1, 2, 3, 4, 5, 6, 7);
    poly_uniform_eta_8x(&s2.vec[0], &s2.vec[1], &s2.vec[2], &s2.vec[3], &s2.vec[4], &s2.vec[5], &s2.vec[6], &s2.vec[7], rhoprime, 8, 9, 10, 11, 12, 13, 14, 15);
  #else
  #error
  #endif
   /* Pack secret vectors */
  for(i = 0; i < L; i++)
    polyeta_pack(sk + 3*SEEDBYTES + i*POLYETA_PACKEDBYTES, &s1.vec[i]);
  for(i = 0; i < K; i++)
    polyeta_pack(sk + 3*SEEDBYTES + (L + i)*POLYETA_PACKEDBYTES, &s2.vec[i]);


  /* Matrix-vector multiplication */
  
  polyvecl_smallntt(&s1);
  
  polyvec_matrix_pointwise_montgomery(&t1, mat, &s1);
  
  polyveck_invntt_tomont(&t1);

  /* Add error vector s2 */
  polyveck_add(&t1, &t1, &s2);

  /* Extract t1 and write public key */
  polyveck_caddq(&t1);
  polyveck_power2round(&t1, &t0, &t1);
  
  for(i = 0; i < K; i ++)
  {
    polyt1_pack(pk + SEEDBYTES + i*POLYT1_PACKEDBYTES, &t1.vec[i]);
    polyt0_pack(sk + 3*SEEDBYTES + (L+K)*POLYETA_PACKEDBYTES + i*POLYT0_PACKEDBYTES, &t0.vec[i]);
  }
  /* Compute H(rho, t1) and write secret key */
  shake256(sk + 2*SEEDBYTES, SEEDBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  

  return 0;
}

/*************************************************
* Name:        crypto_sign_signature
*
* Description: Computes signature.
*
* Arguments:   - uint8_t *sig:   pointer to output signature (of length CRYPTO_BYTES)
*              - size_t *siglen: pointer to output length of signature
*              - uint8_t *m:     pointer to message to be signed
*              - size_t mlen:    length of message
*              - uint8_t *sk:    pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int crypto_sign_signature(uint8_t *sig,
                          size_t *siglen,
                          const uint8_t *m,
                          size_t mlen,
                          const uint8_t *sk)
{
  unsigned int n;
  unsigned int i;
  
  uint8_t seedbuf[3*SEEDBYTES + 2*CRHBYTES];
  uint8_t *rho, *tr, *key, *mu, *rhoprime;
  uint16_t nonce = 0;
  polyvecl mat[K], s1, y, z;
  polyveck t0, s2, w1, w0, h;
  poly cp;
  poly tmp;
  keccak_state state;
  
  rho = seedbuf;
  tr = rho + SEEDBYTES;
  key = tr + SEEDBYTES;
  mu = key + SEEDBYTES;
  rhoprime = mu + CRHBYTES;
  unpack_sk(rho, tr, key, &t0, &s1, &s2, sk);
  
  /* Compute CRH(tr, msg) */
  shake256_init(&state);
  shake256_absorb(&state, tr, SEEDBYTES);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

#ifdef DILITHIUM_RANDOMIZED_SIGNING
  randombytes(rhoprime, CRHBYTES);
#else
  shake256(rhoprime, CRHBYTES, key, SEEDBYTES + CRHBYTES);
#endif

  /* Expand matrix and transform vectors */
  polyvec_matrix_expand(mat, rho);
  polyvecl_smallntt(&s1);
  polyveck_smallntt(&s2);
  polyveck_tailoredntt(&t0);
   

rej:
  /* Sample intermediate vector y */
  #if L == 4
    poly_uniform_gamma1_8x(&y.vec[0], &y.vec[1], &y.vec[2], &y.vec[3], &tmp, &tmp, &tmp, &tmp,rhoprime, nonce, nonce + 1, nonce + 2, nonce + 3, 0, 0, 0, 0);  
    nonce += 4;
  #elif L == 5
    poly_uniform_gamma1_8x(&y.vec[0], &y.vec[1], &y.vec[2], &y.vec[3], &y.vec[4], &tmp, &tmp, &tmp,rhoprime, nonce, nonce + 1, nonce + 2, nonce + 3, nonce + 4, 0, 0, 0);  
    nonce += 5;
  #elif L == 7
    poly_uniform_gamma1_8x(&y.vec[0], &y.vec[1], &y.vec[2], &y.vec[3], &y.vec[4], &y.vec[5], &y.vec[6], &tmp,rhoprime, nonce, nonce + 1, nonce + 2, nonce + 3, nonce + 4, nonce + 5, nonce + 6, 0);  
    nonce += 7;
  #else
  #error
  #endif

  /* Matrix-vector multiplication */
  z = y;
  
  polyvecl_ntty(&z);
  polyvec_matrix_pointwise_montgomery(&w1, mat, &z);
  polyveck_reduce(&w1);
  polyveck_invntt_tomont(&w1);

  /* Decompose w and call the random oracle */
  polyveck_caddq(&w1);
  polyveck_decompose(&w1, &w0, &w1);
  polyveck_pack_w1(sig, &w1);

  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, sig, K*POLYW1_PACKEDBYTES);
  shake256_finalize(&state);
  shake256_squeeze(sig, SEEDBYTES, &state);
  poly_challenge(&cp, sig);
  
  poly_smallntt(&cp);
  /* Compute z, reject if it reveals secret */
  
  for(i = 0; i < L ; i ++)
  {
    poly_pointwise_montgomery(&z.vec[i],&cp,&s1.vec[i]);
    poly_invntt_tomont(&z.vec[i]);
    poly_add(&z.vec[i],&z.vec[i],&y.vec[i]);
    poly_reduce(&z.vec[i]);
    if(poly_chknorm(&z.vec[i], GAMMA1 - BETA))
    {
      goto rej;
    }
  }
 
  /* Check that subtracting cs2 does not change high bits of w and low bits
   * do not reveal secret information */
  
  for(i = 0; i < K ; i ++)
  {
    poly_pointwise_montgomery(&h.vec[i],&cp,&s2.vec[i]);
    poly_invntt_tomont(&h.vec[i]);
    poly_sub(&w0.vec[i],&w0.vec[i],&h.vec[i]);
    poly_reduce(&w0.vec[i]);
    if(poly_chknorm(&w0.vec[i], GAMMA2 - BETA))
    {
      goto rej;
    }
  }
 

  /* Compute hints for w1 */
   
  for(i = 0; i < K ; i ++)
  {
    poly_pointwise_montgomery(&h.vec[i],&cp,&t0.vec[i]);
    poly_invntt_tomont(&h.vec[i]);
    poly_reduce(&h.vec[i]);
    if(poly_chknorm(&h.vec[i], GAMMA2))
    {
      goto rej;
    }
  }
  

  polyveck_add(&w0, &w0, &h);
  n = polyveck_make_hint(&h, &w0, &w1);
  if(n > OMEGA)
    goto rej;

  /* Write signature */
  pack_sig(sig, sig, &z, &h);
  *siglen = CRYPTO_BYTES;
  return 0;
}

/*************************************************
* Name:        crypto_sign
*
* Description: Compute signed message.
*
* Arguments:   - uint8_t *sm: pointer to output signed message (allocated
*                             array with CRYPTO_BYTES + mlen bytes),
*                             can be equal to m
*              - size_t *smlen: pointer to output length of signed
*                               message
*              - const uint8_t *m: pointer to message to be signed
*              - size_t mlen: length of message
*              - const uint8_t *sk: pointer to bit-packed secret key
*
* Returns 0 (success)
**************************************************/
int crypto_sign(uint8_t *sm,
                size_t *smlen,
                const uint8_t *m,
                size_t mlen,
                const uint8_t *sk)
{
  size_t i;

  for(i = 0; i < mlen; ++i)
    sm[CRYPTO_BYTES + mlen - 1 - i] = m[mlen - 1 - i];
  crypto_sign_signature(sm, smlen, sm + CRYPTO_BYTES, mlen, sk);
  *smlen += mlen;
  return 0;
}

/*************************************************
* Name:        crypto_sign_verify
*
* Description: Verifies signature.
*
* Arguments:   - uint8_t *m: pointer to input signature
*              - size_t siglen: length of signature
*              - const uint8_t *m: pointer to message
*              - size_t mlen: length of message
*              - const uint8_t *pk: pointer to bit-packed public key
*
* Returns 0 if signature could be verified correctly and -1 otherwise
**************************************************/
int crypto_sign_verify(const uint8_t *sig,
                       size_t siglen,
                       const uint8_t *m,
                       size_t mlen,
                       const uint8_t *pk)
{
  unsigned int i;
  uint8_t buf[K*POLYW1_PACKEDBYTES];
  uint8_t rho[SEEDBYTES];
  uint8_t mu[CRHBYTES];
  uint8_t c[SEEDBYTES];
  uint8_t c2[SEEDBYTES];
  poly cp;
  polyvecl mat[K], z;
  polyveck t1, w1, h;
  keccak_state state;
  if(siglen != CRYPTO_BYTES)
    return -1;

  unpack_pk(rho, &t1, pk);
  if(unpack_sig(c, &z, &h, sig))
    return -1;
  if(polyvecl_chknorm(&z, GAMMA1 - BETA))
    return -1;

  /* Compute CRH(H(rho, t1), msg) */
  shake256(mu, SEEDBYTES, pk, CRYPTO_PUBLICKEYBYTES);
  shake256_init(&state);
  shake256_absorb(&state, mu, SEEDBYTES);
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  /* Matrix-vector multiplication; compute Az - c2^dt1 */
  poly_challenge(&cp, c);
  polyvec_matrix_expand(mat, rho);

  polyvecl_ntty(&z);
  polyvec_matrix_pointwise_montgomery(&w1, mat, &z);

  
  poly_smallntt(&cp);
  polyveck_shiftl(&t1);
  polyveck_ntt(&t1);
  polyveck_pointwise_poly_montgomery(&t1, &cp, &t1);

  polyveck_sub(&w1, &w1, &t1);
  polyveck_reduce(&w1);
  polyveck_invntt_tomont(&w1);

  /* Reconstruct w1 */
  polyveck_caddq(&w1);
  polyveck_use_hint(&w1, &w1, &h);
  polyveck_pack_w1(buf, &w1);

  /* Call random oracle and verify challenge */
  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, buf, K*POLYW1_PACKEDBYTES);
  shake256_finalize(&state);
  shake256_squeeze(c2, SEEDBYTES, &state);
  for(i = 0; i < SEEDBYTES; ++i)
    if(c[i] != c2[i])
      return -1;

  return 0;
}

/*************************************************
* Name:        crypto_sign_open
*
* Description: Verify signed message.
*
* Arguments:   - uint8_t *m: pointer to output message (allocated
*                            array with smlen bytes), can be equal to sm
*              - size_t *mlen: pointer to output length of message
*              - const uint8_t *sm: pointer to signed message
*              - size_t smlen: length of signed message
*              - const uint8_t *pk: pointer to bit-packed public key
*
* Returns 0 if signed message could be verified correctly and -1 otherwise
**************************************************/
int crypto_sign_open(uint8_t *m,
                     size_t *mlen,
                     const uint8_t *sm,
                     size_t smlen,
                     const uint8_t *pk)
{
  size_t i;

  if(smlen < CRYPTO_BYTES)
    goto badsig;

  *mlen = smlen - CRYPTO_BYTES;
  if(crypto_sign_verify(sm, CRYPTO_BYTES, sm + CRYPTO_BYTES, *mlen, pk))
    goto badsig;
  else {
    /* All good, copy msg, return 0 */
    for(i = 0; i < *mlen; ++i)
      m[i] = sm[CRYPTO_BYTES + i];
    return 0;
  }

badsig:
  /* Signature verification failed */
  *mlen = -1;
  for(i = 0; i < smlen; ++i)
    m[i] = 0;

  return -1;
}
