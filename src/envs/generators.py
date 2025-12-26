# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import hashlib
import os
import numpy as np
import math
from scipy.linalg import circulant
from logging import getLogger

logger = getLogger()

class Generator(ABC):
    def __init__(self, params):
        pass

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass


class ModularMultiply(Generator):
    def __init__(self, params, secret):
        super().__init__(params)
        self.Q = params.Q
        self.S = secret
        self.N = params.N
        assert len(self.S) == self.N

    def generate(self, rng):
        a = rng.randint(0, self.Q, self.N)
        result = [np.dot(a, self.S) % self.Q]
        return a, result

    def evaluate(self, src, tgt, hyp):
        return 1 if hyp == tgt else 0

#### RLWE DATA ####

class RLWE(Generator):
    def __init__(self, params, rng):
        super().__init__(params)
        self.N = params.N
        self.Q = params.Q
        self.rng = rng
        self.sparsity= params.sparsity
        self.density = params.density
        self.hamming = params.hamming #if not self.hamming_curriculum else 1
        self.error = params.error
        self.sigma = params.sigma
        self.maxQ_prob = params.maxQ_prob
        self.percQ_bound = params.percQ_bound
        self.correctQ = params.correctQ
        self.q2_correction = np.vectorize(self.q2_correct)
        # Hallucination key params (optional).
        self.use_hallucination = getattr(params, "use_hallucination", False)
        self.hallucination_k_seed = getattr(params, "hallucination_k_seed", -1)
        self.hallucination_k_bits = getattr(params, "hallucination_k_bits", 128)
        self.hallucination_degrees = [
            d for d in self._parse_int_list(getattr(params, "hallucination_degrees", "1,3,5")) if d > 0
        ]
        if not self.hallucination_degrees:
            self.hallucination_degrees = [1]
        self.hallucination_coeff_choices = self._parse_int_list(
            getattr(params, "hallucination_coeff_choices", "-1,1")
        )
        if not self.hallucination_coeff_choices:
            self.hallucination_coeff_choices = [-1, 1]
        self.hallucination_k = None
        self.hallucination_coeffs = None
        self.secret_raw = None

        # if density is greater than 0, set hamming weight by it. 
        if self.density > 0: 
            ham = round(self.N * self.density) 
            self.hamming = ham

        # curriculum parameters
        self.secrets = self.getSecrets(params)
        logger.info(f'secrets: {self.secrets}')

        # reuse data? 
        self.reuse = params.reuse
        if self.reuse:
            self.reuse_samples = np.zeros(shape=(params.num_reuse_samples,self.N,self.N+1)) # N+1 allows space to store B
            self.reuse_counter = np.zeros(shape=params.num_reuse_samples) - 1
            self.times_reused = params.times_reused
            self.K = params.K
        else:
            self.reuse_samples, self.times_reused, self.reuse_counter = None, None, None

    def getSecrets(self, params):
        s = self.genSecretKey(params.secrettype, self.N)
        if not self.use_hallucination:
            return [s]
        # Apply k-based obfuscation to produce s'.
        k_bits, k_bytes = self._make_hallucination_k()
        s_prime, coeffs = self._maclaurin_obfuscate(s, k_bytes)
        self.secret_raw = s
        self.hallucination_k = k_bits
        self.hallucination_coeffs = coeffs
        return [s_prime]

    def genSecretKey(self, secret, N):
        if secret == "b":
            # sample secret uniformly from {0, 1}
            if self.hamming == 0:
                s = np.vectorize(lambda x: 1 if x <= self.sparsity else 0)(self.rng.uniform(size=N))
                while self.N > 1 and np.sum(s) < 2: # make sure you have at least 2 nonzero elements.
                    s[self.rng.integers(N)] = 1
            else:
                s = np.zeros(shape=N, dtype=np.int64)
                for _ in range(self.hamming):
                    setit = False
                    while not setit:
                        idx = self.rng.integers(N, size=1)
                        if s[idx] != 1:
                            s[idx] = 1
                            setit = True
        elif secret == "g":
            s = self.rng.normal(0, self.sigma, size=N).round()
        elif secret == "u":
            s = self.rng.integers(0, self.Q-1, endpoint=True, size=N)
        elif secret == "t":
            # sample secret uniformly from {-1, 0, 1}
            s = self.rng.integers(-1, 1, endpoint=True, size=N)
        return s

    @staticmethod
    def _parse_int_list(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [int(v) for v in value]
        if value is None:
            return []
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip() != ""]
            return [int(p) for p in parts]
        return [int(value)]

    def _make_hallucination_k(self):
        bits = self.hallucination_k_bits if self.hallucination_k_bits > 0 else 128
        nbytes = (bits + 7) // 8
        if self.hallucination_k_seed is not None and self.hallucination_k_seed >= 0:
            seed_int = int(self.hallucination_k_seed)
            seed_len = max(1, (seed_int.bit_length() + 7) // 8)
            seed_bytes = seed_int.to_bytes(seed_len, "big", signed=False)
            k_bytes = hashlib.shake_256(b"HKD|k|" + seed_bytes).digest(nbytes)
        else:
            k_bytes = os.urandom(nbytes)
        k_bits = np.unpackbits(np.frombuffer(k_bytes, dtype=np.uint8))[:bits].astype(np.int64)
        return k_bits, k_bytes

    def _xof_uint32_stream(self, k_bytes, label):
        counter = 0
        buf = b""
        while True:
            if len(buf) < 4:
                h = hashlib.shake_256()
                h.update(b"HKD|" + label + b"|" + counter.to_bytes(4, "big") + b"|" + k_bytes)
                buf += h.digest(64)
                counter += 1
            val = int.from_bytes(buf[:4], "big")
            buf = buf[4:]
            yield val

    def _xof_choice_indices(self, k_bytes, label, count, mod):
        if mod <= 0:
            raise ValueError("mod must be positive")
        limit = (1 << 32) - ((1 << 32) % mod)
        out = []
        for v in self._xof_uint32_stream(k_bytes, label):
            if v < limit:
                out.append(v % mod)
                if len(out) >= count:
                    break
        return out

    def _negacyclic_convolve(self, a, b):
        # Negacyclic convolution: mod x^N + 1 to match get_sample().
        conv = np.convolve(a, b)
        n = len(a)
        res = conv[:n].astype(np.int64, copy=True)
        tail = conv[n:]
        if tail.size:
            res[:tail.size] -= tail
        return res % self.Q

    def _maclaurin_obfuscate(self, s, k_bytes):
        s = s.astype(np.int64)
        s_prime = np.zeros_like(s, dtype=np.int64)
        coeffs = {}
        coeff_idx = self._xof_choice_indices(
            k_bytes, b"coeff", len(self.hallucination_degrees), len(self.hallucination_coeff_choices)
        )
        for d in self.hallucination_degrees:
            idx = coeff_idx.pop(0)
            coeffs[d] = int(self.hallucination_coeff_choices[idx])
            if d == 1:
                term = s.copy()
            else:
                term = s.copy()
                for _ in range(d - 1):
                    term = self._negacyclic_convolve(term, s)
            s_prime = (s_prime + coeffs[d] * term) % self.Q
        return s_prime, coeffs

    def generate(self, rng, idx, currN=-1):
        if self.reuse:
            if self.K > 1:
                return self.combine_reused_samples(rng, idx, currN)
            else:
                return self.get_reused_sample(rng, idx, currN)
        else:
            return self.get_sample(rng, idx, currN)
        
    def combine_reused_samples(self, rng, idx, currN):
        '''
        Combines the reused samples depending on the K level.
        '''
        A_s = np.zeros(shape=(self.K, self.N, self.N), dtype=np.int64)
        B_s = np.zeros(shape=(self.K, self.N), dtype=np.int64)
        for i in range(self.K):
            a,b = self.get_reused_sample(rng, idx, currN)
            A_s[i,:,:] = a
            B_s[i,:] = b
        k_s = rng.choice([-1,0,1], self.K, replace=True).reshape((-1,) + (1,)*(2)).astype(np.int64)
        while np.all(k_s == 0):
            k_s = rng.choice([-1,0,1], self.K, replace=True).reshape((-1,) + (1,)*(2)).astype(np.int64)
        return np.sum(A_s * k_s, axis=0) % self.Q, np.sum(B_s * np.squeeze(k_s, axis=1), axis=0) % self.Q

    def get_reused_sample(self, rng, idx, currN=-1):
        ''' 
        Code to faciliate sample reuse. 
        '''
        # Choose a random sample
        sample_idx = rng.randint(0, self.reuse_samples.shape[0])
        curr_count = self.reuse_counter[sample_idx]
        # If the reuse counter is -1 or times_reused, generate a new sample and put it in the reuse samples array at this index
        if (curr_count == -1) or (curr_count >= self.times_reused):
            A, B = self.get_sample(rng, idx, currN)
            self.reuse_samples[sample_idx, :, :self.N] = A
            self.reuse_samples[sample_idx, :,self.N:] = np.expand_dims(B,1)
            self.reuse_counter[sample_idx] = 0
        # Return the sample at this index
        self.reuse_counter[sample_idx] += 1 / self.K
        a,b = self.reuse_samples[sample_idx, :, :self.N].astype(np.int64), np.squeeze(self.reuse_samples[sample_idx, :, self.N:]).astype(np.int64)
        return a,b 

    def q2_correct(self, x):
        if x <= -self.Q/2:
            x = x+self.Q
        elif x >= self.Q/2:
            x = x-self.Q
        return x

    def get_sample(self, rng, idx, currN=-1):
        # Use passed-in N if it isn't 0.
        N = currN if currN > 0 else self.N
        if (self.rng.uniform() < self.maxQ_prob):
            maxQ = self.Q
        else: 
            maxQ = self.percQ_bound * self.Q

        # sample a uniformly from Z_q^n
        a = rng.randint(0, maxQ, size=N, dtype=np.int64)

        # do the circulant:
        c = circulant(a)
        tri = np.triu_indices(N, 1)
        c[tri] *= -1
        if self.correctQ:
            c = self.q2_correction(c)

        c = c % self.Q

        assert (np.min(c) >= 0) and (np.max(c) < self.Q)

        if self.error:
            e = np.int64(rng.normal(0, self.sigma, size = self.N).round())
            b = (np.inner(c, self.secrets[idx]) + e) % self.Q
        else:
            b = np.inner(c, self.secrets[idx]) % self.Q

        if self.correctQ:
            b = self.q2_correction(b)

        return c,b

    def evaluate(self, src, tgt, hyp):
        return 1 if hyp == tgt else 0

    def get_difference(self, tgt, hyp):
        return abs(hyp[0]-tgt[0])

    def evaluate_bitwise(self, tgt, hyp):
        return [int(str(e1)==str(e2)) for e1,e2 in zip(tgt,hyp)]
