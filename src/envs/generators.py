# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod  # 추상 베이스 클래스 상속을 위한 모듈
import numpy as np  # 수치 계산용
import math  
from scipy.linalg import circulant  # 순환 행렬(circulant) 생성
from logging import getLogger  # 로그 출력

logger = getLogger()  # module-level logger

class Generator(ABC):
    # 기본 Generator 인터페이스.
    # 각 연산 환경(예: ModularMultiply, RLWE 등)은 이 인터페이스를 구현해야 함.
    def __init__(self, params):
        pass  # 서브클래스에서 필요한 초기화 수행

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass


class ModularMultiply(Generator):
    # 단순한 모듈러 곱셈 문제 생성기
    # 입력 벡터 a에 대해 비밀 키 S와 내적을 계산하여 결과를 생성
    def __init__(self, params, secret):
        super().__init__(params)
        self.Q = params.Q  # 모듈러스 q
        self.S = secret  # 비밀 키 벡터
        self.N = params.N  # 벡터 길이 = 차원수
        assert len(self.S) == self.N  # S 길이 검증

    def generate(self, rng):
        a = rng.randint(0, self.Q, self.N)  # Z_q^N에서 균등 샘플
        result = [np.dot(a, self.S) % self.Q]  # a·S mod q
        return a, result

    def evaluate(self, src, tgt, hyp):
        # 정확히 맞추면 1, 아니면 0
        return 1 if hyp == tgt else 0

#### RLWE DATA ####

class RLWE(Generator):
    # 순환 링-LWE(circumetential RLWE) 샘플 생성기
    # - a는 Z_q^n에서 균등 샘플
    # - c는 a로부터 생성된 순환 행렬
    # - b는 c * s + e (mod q) 형태의 관측값
    # 이 환경은 입력으로 c를 주고, 출력으로 b를 예측하는 문제를 다룸.
    def __init__(self, params, rng):
        super().__init__(params)
        self.N = params.N  # 차원 (폴리노미얼 차수)
        self.Q = params.Q  # 모듈러스 q
        self.rng = rng  # numpy 랜덤 생성기

        # 비밀키 관련 파라미터
        #  - sparsity: 비밀키에서 1이 될 확률 (hamming==0일 때)
        #  - density: 비밀키에서 1이 차지하는 비율 (hamming 값으로 변환)
        #  - hamming: 1의 개수 (밀도)
        self.sparsity = params.sparsity
        self.density = params.density
        self.hamming = params.hamming  # if not self.hamming_curriculum else 1

        # 노이즈 관련 파라미터
        self.error = params.error  # 노이즈 사용 여부
        self.sigma = params.sigma  # 노이즈 표준편차

        # Q 범위 및 샘플링 제어
        self.maxQ_prob = params.maxQ_prob  # Q 전체 범위 샘플링 확률
        self.percQ_bound = params.percQ_bound  # 일부 범위 샘플링 비율
        self.correctQ = params.correctQ  # Q 범위를 -Q/2..Q/2로 보정할지 여부
        self.q2_correction = np.vectorize(self.q2_correct)  # 벡터화된 보정 함수

        # density가 지정되어 있으면 해밍 무게(hamming)를 density 비율로 설정
        if self.density > 0:
            ham = round(self.N * self.density)  # 비밀 벡터에서 1의 개수를 결정
            self.hamming = ham

        # secrets: 비밀 키 벡터 리스트 (현재는 하나만 사용)
        self.secrets = self.getSecrets(params)
        logger.info(f'secrets: {self.secrets}')

        # 샘플 재사용(reuse) 설정
        self.reuse = params.reuse
        if self.reuse:
            # reuse_samples: (num_reuse_samples, N, N+1) 크기의 배열로, 각 샘플에 대해 A와 B를 함께 저장
            self.reuse_samples = np.zeros(shape=(params.num_reuse_samples, self.N, self.N + 1))
            # reuse_counter: 각 슬롯이 마지막으로 사용된 횟수를 기록
            self.reuse_counter = np.zeros(shape=params.num_reuse_samples) - 1
            self.times_reused = params.times_reused  # 각 슬롯이 재사용되는 최대 횟수
            self.K = params.K  # K > 1이면 여러 샘플을 섞어서 반환
        else:
            self.reuse_samples, self.times_reused, self.reuse_counter = None, None, None

    def getSecrets(self, params):
        # secret type에 따라 비밀 키 벡터를 하나 생성하여 리스트로 반환
        secrets = [self.genSecretKey(params.secrettype, self.N)]
        return secrets

    def genSecretKey(self, secret, N):
        # 비밀 키 벡터 s 생성
        # secret 타입에 따라 생성 방식이 달라짐
        if secret == "b":
            # {0,1} 기반 비밀키 (binary secret)
            # hamming이 0이라면 sparsity 확률로 1을 선택하고, 해밍 무게가 지정돼 있으면 정확히 hamming 만큼 1을 채움
            if self.hamming == 0:
                s = np.vectorize(lambda x: 1 if x <= self.sparsity else 0)(self.rng.uniform(size=N))
                while self.N > 1 and np.sum(s) < 2:  # 최소 2개 이상의 1이 있도록 보장
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
            # 가우시안 분포 기반 비밀키
            s = self.rng.normal(0, self.sigma, size=N).round()
        elif secret == "u":
            # 균등 분포 비밀키 (0..Q-1)
            s = self.rng.integers(0, self.Q-1, endpoint=True, size=N)
        elif secret == "t":
            # {-1, 0, 1} 분포 비밀키
            s = self.rng.integers(-1, 1, endpoint=True, size=N)
        # elif 비밀키 관련 타입이 더 있다면 여기에 추가 (우지안거 구현해야함)
        return s

    def generate(self, rng, idx, currN=-1):
        # 샘플을 생성하는 진입점
        # 재사용(reuse) 설정이 켜져 있으면 미리 생성된 샘플을 활용
        if self.reuse:
            if self.K > 1:
                return self.combine_reused_samples(rng, idx, currN)
            else:
                return self.get_reused_sample(rng, idx, currN)
        else:
            return self.get_sample(rng, idx, currN)
        
    def combine_reused_samples(self, rng, idx, currN):
        '''
        K개의 재사용 샘플을 조합하여 하나의 샘플을 반환.
        - A_s: 각 샘플의 순환 행렬 A를 저장 (K x N x N)
        - B_s: 각 샘플의 벡터 b를 저장 (K x N)
        - k_s: [-1,0,1]에서 랜덤하게 선택한 계수 벡터로 샘플을 조합
        '''
        A_s = np.zeros(shape=(self.K, self.N, self.N), dtype=np.int64)
        B_s = np.zeros(shape=(self.K, self.N), dtype=np.int64)
        for i in range(self.K):
            a, b = self.get_reused_sample(rng, idx, currN)
            A_s[i, :, :] = a
            B_s[i, :] = b

        # -1, 0, 1로 이루어진 계수를 샘플링 (모두 0이면 다시 샘플)
        k_s = rng.choice([-1, 0, 1], self.K, replace=True).reshape((-1,) + (1,) * 2).astype(np.int64)
        while np.all(k_s == 0):
            k_s = rng.choice([-1, 0, 1], self.K, replace=True).reshape((-1,) + (1,) * 2).astype(np.int64)

        # A_s, B_s를 k_s로 가중합하여 반환
        return np.sum(A_s * k_s, axis=0) % self.Q, np.sum(B_s * np.squeeze(k_s, axis=1), axis=0) % self.Q

    def get_reused_sample(self, rng, idx, currN=-1):
        '''
        샘플 재사용 로직
        - 매 호출마다 reuse_samples 중 하나를 선택
        - 해당 슬롯이 만료되었으면 새로운 샘플을 생성하여 갱신
        - 선택된 샘플을 반환
        '''
        # 랜덤하게 reuse 슬롯 선택
        sample_idx = rng.randint(0, self.reuse_samples.shape[0])
        curr_count = self.reuse_counter[sample_idx]

        # 슬롯이 비어있거나 재사용 횟수를 초과하면 새로운 샘플 생성
        if (curr_count == -1) or (curr_count >= self.times_reused):
            A, B = self.get_sample(rng, idx, currN)
            self.reuse_samples[sample_idx, :, :self.N] = A
            self.reuse_samples[sample_idx, :, self.N:] = np.expand_dims(B, 1)
            self.reuse_counter[sample_idx] = 0

        # 선택된 슬롯의 카운터 증가 (K가 1이면 1씩, K>1이면 1/K씩 증가)
        self.reuse_counter[sample_idx] += 1 / self.K

        # 저장된 A, B를 꺼내서 반환
        a = self.reuse_samples[sample_idx, :, :self.N].astype(np.int64)
        b = np.squeeze(self.reuse_samples[sample_idx, :, self.N:]).astype(np.int64)
        return a, b

    def q2_correct(self, x):
        # q에서 -Q/2 .. Q/2 범위로 값을 보정 (mod Q 값이 0 중심으로 보정됨)
        # 예: Q=10일 때 7 -> -3, 9 -> -1, 5 -> -5
        if x <= -self.Q/2:
            x = x + self.Q
        elif x >= self.Q/2:
            x = x - self.Q
        return x

    def get_sample(self, rng, idx, currN=-1):
        # currN가 지정되어 있으면 그 크기를 사용, 아니면 기본 N 사용
        N = currN if currN > 0 else self.N

        # maxQ_prob 확률로 Q를 그대로 사용하고, 그렇지 않으면 percQ_bound에 비례하는 maxQ 사용
        # (학습 안정성을 위해 Q를 전 범위에서 샘플링하지 않도록 할 때 사용)
        if (self.rng.uniform() < self.maxQ_prob):
            maxQ = self.Q
        else:
            maxQ = self.percQ_bound * self.Q

        # Z_q^n에서 균등 샘플 a 생성
        a = rng.randint(0, maxQ, size=N, dtype=np.int64)

        # a로부터 순환 행렬(circulant matrix) c 생성
        c = circulant(a)
        tri = np.triu_indices(N, 1)
        # upper triangular 부분은 -1을 곱해서 반대 방향으로 값이 이어지게 함
        c[tri] *= -1

        # correctQ가 켜져있으면 값들을 -Q/2..Q/2 범위로 이동시켜
        # 학습 시 0 중심 분포를 만들도록 함.
        if self.correctQ:
            c = self.q2_correction(c)

        # mod Q 범위로 다시 조정
        c = c % self.Q

        assert (np.min(c) >= 0) and (np.max(c) < self.Q)

        # b = c * s (+ e) mod q 형태로 계산
        #  - c: 순환 행렬 (N x N)
        #  - s: 비밀 벡터 (길이 N)
        #  - e: Gaussian 노이즈
        if self.error:
            # 노이즈가 있는 경우
            e = np.int64(rng.normal(0, self.sigma, size=self.N).round())
            b = (np.inner(c, self.secrets[idx]) + e) % self.Q
        else:
            # 노이즈 없는 경우
            b = np.inner(c, self.secrets[idx]) % self.Q

        # correctQ가 켜져 있으면 -Q/2..Q/2 범위로 보정
        if self.correctQ:
            b = self.q2_correction(b)

        return c, b

    def evaluate(self, src, tgt, hyp):
        return 1 if hyp == tgt else 0

    def get_difference(self, tgt, hyp):
        return abs(hyp[0]-tgt[0])

    def evaluate_bitwise(self, tgt, hyp):
        return [int(str(e1)==str(e2)) for e1,e2 in zip(tgt,hyp)]
