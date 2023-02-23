This repository accompanies the paper **Highly Vectorized Implementation of CRYSTAL-Dilithium Using AVX-512**  submitted to TCHES Volume 2023/3.
# Installation

## Cloning the code
Clone the code 

```
git clone https://github.com/zxv111n/Dilithium-AVX512.git
```

### Hardware Configuration

- Intel(R) Core(TM) i7-11700F CPU at 2.5GHz with TurBo Boost and Hyperthreading disabled. The processor should support AVX-512 and AVX512 VBMI.

### Software Version
- Ubuntu 20.04.5
- gcc  9.4.0
- openssl
### Install openssl
```
sudo apt-get install openssl
sudo apt-get install libssl-dev
```
To compile the code
```
make
```
Run the code
```
./test/test_speed2
./test/test_dilithium2
./test/test_speed3
./test/test_dilithium3
./test/test_speed5
./test/test_dilithium5
```




