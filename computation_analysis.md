# NCRL 계산 비용과 병목 분석

작성 2026-10-24. 측정 환경: RTX 4070 12GB (가용 11.58 GiB), Ryzen 5 7500F, Ubuntu 22.04, conda `ncrl_env` (Python 3.10.22, PyTorch 2.13.0, CUDA 13.0, cuDNN 9.27, TF32 matmul 꺼짐(기본값), cuDNN TF32 켜짐(기본값)). 표기는 `architecture_analysis.md`와 같습니다: **[측정] [코드] [논문] [추정]**.

## 0. 결론 먼저

1. **학습 wall-clock의 99%는 업데이트다.** Quadruped Walk 기준 epoch당 env 수집 4초, 업데이트 362초 **[측정]**. 따라서 env step 수를 줄이는 것(sample efficiency)과 업데이트 연산을 줄이는 것(computational efficiency)은 별개의 성과다.
2. **업데이트 1회 = 2.92초 = WM 업데이트 1.21초(41%) + 상상·actor-critic 1.71초(59%)** **[측정, 4070 + 메모리 절약 패치]**. 논문 저자 환경(메모리 충분, recompute 없음)에서는 상상·AC가 약 1.15초로 추정되어 WM 비중이 약 51%가 된다 **[추정: 128/256 시작점 측정치 선형 외삽]**.
3. **FLOPs와 시간의 순위가 다르다.** FLOPs로는 상상·AC가 WM 업데이트의 약 4배(13.6 vs 3.45 TFLOP)지만, 시간으로는 WM 업데이트가 비슷한 크기다. WM 학습은 배치 16짜리 64단계 **순차 scan**이라 대역폭과 launch 지연에 묶이고(달성 2.9 TFLOP/s), 상상은 1024행 병렬이라 연산에 묶인다(달성 약 11.5 TFLOP/s) **[측정 + 추정]**.
4. **WM 업데이트를 통째로 생략해도 업데이트 시간은 최대 32%만 줄어든다**(4070 기준. 저자 환경 추정 약 40%). 상상 시작점을 만들기 위한 encoder + RSSM forward(274 ms)는 남기 때문이다 **[측정]**. 남는 비용의 대부분은 상상·actor backward다.
5. **반복되는 계산(핵심 질문의 답)**:
   - (a) 같은 리플레이 chunk를 매번 0 상태부터 다시 encode하고 64단계 scan한다. online 전이 하나가 평균 약 144번 재처리된다.
   - (b) WM 업데이트가 이미 본 online 데이터를 외우는 쪽으로 흘러간다. 40k 시점에 본 에피소드의 open-loop 오차는 10.8, 바로 다음에 수집한 새 에피소드는 73.0으로 7배 차이 **[측정, §5]**.
   - (c) rehearsal 4행은 retrieved 데이터의 오차를 줄이지 못하고 유지만 한다(9.6 → 10.0) **[측정]**.
   - (d) 1024개의 서로 비슷한 시작점에서 모두 상상한다. 256개만 써도 −48% 시간, −4.7% return(1 seed) **[측정, 이전 세션]**.

---

## 1. 측정 방법 **[측정 설정]**
- 스크립트: `tools/profile_update.py`. NCRL 클래스를 그대로 쓰고 `DreamerAgent.update`와 같은 연산 순서를 단계별로 나눠 CUDA event로 시간을 잰다. 워밍업 3회, 측정 8회, 중앙값. 각 반복 시작 전 `torch.cuda.synchronize()`, cuDNN benchmark 켬
- 가중치: 공개 pretrained `dmcontrol.pt`. 데이터: 리플레이 shape의 무작위 텐서(B 16, T 64, action 21 패딩, 실제 12). 비용은 픽셀 값에 무관하다
- FLOPs: `torch.utils.flop_counter.FlopCounterMode`. **합계만 사용한다.** 모듈별 귀속은 backward 연산이 엉뚱한 모듈로 잡혀서(예: 0.04M짜리 `_in_action`에 1.3 TFLOP) 신뢰할 수 없다. 모듈별 비중은 forward를 해석적으로 계산했다(행당 MAC × 행 수)
- 실제 학습 중 시간: `ncrl_runs/branch_qw40k/full1024/branch_log.jsonl` (원본 업데이트 + 결과가 같은 checkpoint 패치)
- 로그: `results/profile_update.log`, `results/profile_update_run1.json`

## 2. 업데이트 1회의 단계별 비용 **[측정]**

### 2.1 WM 업데이트 (B16 × T64, 285.7M 파라미터 전부 학습)

| 단계 | ms | 비중 |
|---|---:|---:|
| encoder forward | 25.2 | 2% |
| RSSM observe forward (64 step scan) | 252.4 | 21% |
| KL | 3.4 | 0% |
| decoder forward | 48.1 | 4% |
| recon loss | 0.5 | 0% |
| **backward (전체)** | **820.9** | **68%** |
| grad clip (285.7M) | 8.2 | 1% |
| weight decay (전체 재할당) | 5.4 | 0% |
| Adam step | 48.9 | 4% |
| **합계** | **1212.9** | |

- 피크 메모리 8.90 GiB (모델 + Adam 상주분 3.44 GiB 포함) → 12GB에 들어감
- FLOPs 3.45 TFLOP. forward 1.15 TFLOP의 해석적 분해 **[추정]**: encoder 126.5 GMAC(22%), RSSM observe 215.0 GMAC(37%), decoder 235.6 GMAC(41%)
- forward **시간** 비중은 RSSM 77%, decoder 15%, encoder 8%로, FLOPs 비중과 순위가 반대다

### 2.2 Reward head + 상상 + actor-critic

| 단계 | 원본 128 시작점 | 원본 256 | 원본 1024 | checkpoint 패치 1024 |
|---|---:|---:|---:|---:|
| reward head 업데이트 | 8.5 | 8.6 | | 9.0 |
| 상상 rollout forward (16 step) | 77.5 | 136.5 | | 521.1 |
| reward / value / λ-return | 3.1 | 6.2 | | 25.2 |
| actor loss forward | 1.4 | 3.0 | | 11.2 |
| **actor backward (RSSM 통과)** | 78.5 | 150.6 | | **1111.3** (recompute 포함) |
| actor / critic Adam | 1.7 | 1.6 | | 1.6 |
| critic forward + backward | 4.0 | 7.6 | | 30.7 |
| slow target | 0.3 | 0.3 | | 0.3 |
| **합계 (ms)** | **174.9** | **314.4** | **OOM** | **1710.4** |
| 피크 메모리 (GiB) | 5.46 | 7.23 | OOM (512도 OOM) | 8.20 |

- FLOPs: 128 시작점에서 1.70 TFLOP → 1024개면 약 13.6 TFLOP. checkpoint 패치는 forward를 한 번 더 하므로 실제로는 약 19.7 TFLOP **[추정]**
- 원본 1024(recompute 없음) 시간은 기울기로 외삽하면 약 1.15초 **[추정]**: 상상 forward 0.46 ms/시작점, actor backward 0.56 ms/시작점

### 2.3 업데이트 1회와 실행 전체
- 4070, 패치 적용: 1.21 + 1.71 = **2.92초/업데이트**. 실제 학습 로그: 362.5초/125 업데이트 = 2.90초 **[측정]**
- epoch(QW 1000 frame): 업데이트 362초 + 수집 4초 + 평가(10k frame마다 5–10 에피소드)
- 150k frame 전체 업데이트 시간 ≈ 150 epoch × 362초 ≈ **15시간**, 여기에 BC 사전학습 약 75분 추가 **[측정 기반 추정]**. Meta-World는 epoch당 25 업데이트, 약 3.2초/업데이트로 150k frame에 **약 17시간** **[측정, 이전 세션]**
- 논문: fine-tuning 1회 약 8 GPU-h(MI250X, 150k env step), Fig. 3 전체 1728 GPU-h **[논문 App. F]**

## 3. FLOPs 대비 시간: 무엇이 무엇에 묶여 있나 **[추정, 측정치 기반]**

| 구간 | FLOPs | 시간 | 달성 처리량 | 성격 |
|---|---:|---:|---:|---|
| WM 업데이트 | 3.45 TF | 1.21 s | 2.9 TF/s | 순차 64 step × 16행. 매 step 가중치 약 0.84 GB를 다시 읽음 |
| 상상 + AC (패치) | ~19.7 TF | 1.71 s | ~11.5 TF/s | 16 step × 1024행. 연산에 묶임 |

4070 FP32 최대 성능은 약 29 TFLOP/s. WM scan에서 weight 읽기만으로도 forward 64 step × 0.84 GB ≈ 54 GB가 들고, 이는 504 GB/s에서 최소 107 ms다. backward는 weight 읽기와 grad 누적을 합쳐 그 몇 배다.

**함의**:
- WM 업데이트 시간은 대략 **(시퀀스 길이 T) × (파라미터 바이트)**에 비례하고 배치 B에는 거의 무관할 것이다. 배치를 줄여도 시간은 거의 줄지 않고, 시퀀스를 줄이거나 업데이트 횟수를 줄여야 줄어든다. 이 가설은 B = 8/16/32 시간 측정으로 확인할 예정이다(`experiment_plan.md` E0).
- 상상·AC 시간은 **시작점 수 × horizon × img_step 크기**에 비례한다.

## 4. 모듈 freeze와 업데이트 생략의 계산 절감 상한 **[측정]**
`profile_update.py` Part B: 지정한 모듈만 학습하고 새 Adam은 학습 대상 파라미터에만 만든다(frozen 파라미터는 weight decay도 받지 않음).

| WM 업데이트 구성 | ms | WM 대비 | 학습 파라미터 | TFLOP |
|---|---:|---:|---:|---:|
| 전체 (원본) | 1207.9 | — | 281.4M | 3.45 |
| encoder freeze | 1151.9 | −5% | 275.2M | 2.96 |
| decoder freeze | 1163.0 | −4% | 216.3M | 2.98 |
| encoder + decoder freeze (RSSM만) | 1099.9 | −9% | 210.1M | 2.48 |
| **RSSM GRU core freeze** | **794.6** | **−34%** | 125.6M | 3.13 |
| RSSM 전체 freeze (enc + dec만) | 657.8 | −46% | 71.3M | 3.02 |
| prior + posterior head만 | 693.5 | −43% | 54.3M | 2.16 |
| decoder만 | 402.9 | −67% | 65.1M | 2.07 |
| encoder만 | 616.0 | −49% | 6.2M | 2.55 |
| **WM 업데이트 없음 (시작점용 forward만)** | **274.0** | **−77%** | 0 | 0.68 |

이 표의 피크 메모리 열은 이전 구성의 Adam 상태가 남아 있던 버그로 오염되어 싣지 않았다. 수정 후 재측정 예정.

**업데이트 1회 전체(2.92초) 기준 절감** (상상·AC 1.71초는 그대로):

| 방식 | WM 평균 ms | 업데이트 1회 | 절감 |
|---|---:|---:|---:|
| 원본 | 1208 | 2.92 s | — |
| WM 1/2 빈도 | 741 | 2.45 s | −16% |
| WM 1/4 빈도 | 508 | 2.22 s | −24% |
| WM 1/8 빈도 | 391 | 2.10 s | −28% |
| WM 없음 | 274 | 1.98 s | −32% |
| GRU freeze | 795 | 2.51 s | −14% |
| encoder + decoder freeze | 1100 | 2.81 s | −4% |
| (참고) 상상 시작점 256개 | 1208 | 약 1.5 s | 약 −48% |

실제 분기 실행에서 WM 없음 팔의 epoch 시간은 255초(원본 362초 대비 −30%)로 표와 일치한다 **[측정, 진행 중인 파일럿]**.

**해석**:
- encoder나 decoder freeze는 계산 측면의 실익이 거의 없다(−4~9%). 게다가 논문 Fig. 8은 처음부터 모듈을 고정하면 sample efficiency가 크게 떨어진다고 보여 준다.
- WM 쪽에서 절감이 큰 것은 **GRU core의 weight-gradient**와 **업데이트 자체의 생략**이다.
- 상상·AC를 건드리지 않는 한, WM만 줄여서 얻을 수 있는 절감은 업데이트 시간의 최대 32%다.

## 5. Fine-tuning 중 반복·불필요 계산의 근거 (실험 C 전반부) **[측정]**
스크립트 `tools/diag_wm_error.py`. 기존 Quadruped Walk 실행(seed 1, 원본 업데이트)의 스냅샷을 사용했다: pre = pretrained, ft20 = 20k frame(약 1.25k 업데이트), ft40 = 40k frame(약 3.75k 업데이트). 데이터 세트마다 64-step chunk 96개(고정 seed), forward만 수행.
- **recon**: posterior 복원 MSE(픽셀 합)
- **kl**: KL(post‖prior), free nats 없음
- **openloop**: 16프레임 관측 후, 기록된 action으로 32 step을 상상한 이미지의 MSE
- **value_gap**: |V(open-loop prior) − V(posterior)|. ft 모델의 critic 사용

### 5.1 본 데이터와 새 데이터의 오차 (평균 ± SE)

| 모델 | on_rl (ep 15–39) | **on_new (ep 40–50, 미학습)** | prior (retrieved) | walker | cartpole |
|---|---|---|---|---|---|
| openloop pre | 73.4 ± 2.7 | 90.3 ± 2.2 | 65.3 ± 2.5 | 151 ± 6 | 5.7 ± 0.4 |
| openloop ft20 | 53.0 ± 3.7 | 82.5 ± 2.8 | 70.7 ± 2.6 | 156 ± 5 | 13.1 ± 0.7 |
| openloop ft40 | **10.8 ± 1.2** | **73.0 ± 2.9** | 66.3 ± 2.7 | 165 ± 5 | 57.3 ± 8.5 |
| kl pre | 21.9 | 22.8 | 10.2 | 11.7 | 2.1 |
| kl ft40 | **3.7** | **18.6** | 10.5 | 19.9 | 12.6 |
| recon pre | 13.4 | 14.4 | 9.6 | 8.2 | 0.66 |
| recon ft40 | **4.6** | **13.0** | 10.0 | 18.7 | 31.6 |

읽는 법:
- **(i) 암기**: ft40은 이미 학습한 online 에피소드에서는 오차가 7분의 1(73 → 11)로 떨어졌지만, 바로 다음 에피소드(on_new)에서는 pretrained 대비 19%만 줄었다(90 → 73. chunk별 중앙 비율 0.82, 75%의 chunk에서 개선). online 전이 하나가 평균 약 144번 재생된다(업데이트 3.75k × 12 chunk × 64 step ÷ online 전이 약 2만 개).
- **(ii) 추가 업데이트의 일반화 이득 체감**: ft20 → ft40(2.5k 업데이트 추가) 동안 on_new의 openloop은 82.5 → 73.0, KL은 19.0 → 18.6으로 줄었다. 반면 본 데이터의 오차는 크게 떨어졌다.
- **(iii) rehearsal은 유지만 한다**: retrieved 데이터(매 업데이트 4/16행)의 오차는 그대로다(openloop 65 → 66, KL 10.2 → 10.5).
- **(iv) 다른 embodiment는 잊는다**: cartpole recon 0.66 → 31.6, walker 8.2 → 18.7. 목표 과제와 무관한 지식이라 단일 과제 성능에는 영향이 없을 수 있지만, catastrophic forgetting의 직접 증거다.
- **주의**: on_new는 나중의 정책이 수집한 데이터라 상태 분포 자체가 어려워졌을 수 있다(pretrained 오차도 on_bc 66.6 → on_rl 73.4 → on_new 90.3으로 증가). 따라서 seen/unseen 비교는 같은 모델 안에서 해석해야 한다.

### 5.2 모듈별 파라미터 이동량 (‖Δθ‖/‖θ‖)

| 모듈 | pre → ft20 | pre → ft40 |
|---|---:|---:|
| encoder | 0.034 | 0.058 |
| GRU `_in_stoch` | 0.090 | 0.151 |
| GRU `_dynlayers` / `_dyngru` | 0.069 / 0.066 | 0.117 / 0.116 |
| GRU `_in_deter` | 0.062 | 0.110 |
| prior / posterior | 0.057 / 0.051 | 0.102 / 0.089 |
| decoder fc / conv | 0.061 / 0.064 | 0.112 / 0.118 |
| GRU `_in_action` | 0.014 | 0.022 |

특정 모듈만 따로 움직이지는 않는다. 모든 모듈이 비슷한 비율(약 10%)로 바뀌고, encoder와 action 입력층이 가장 적게 바뀐다.

### 5.3 모듈 되돌리기: ft40에서 한 모듈만 pretrained 가중치로 복원 (on_new, 평균 ± SE)

| 되돌린 모듈 | recon | kl | openloop | value_gap |
|---|---|---|---|---|
| (없음, ft40) | 13.05 ± 0.27 | 18.6 ± 0.5 | 73.0 ± 2.9 | 1.39 ± 0.16 |
| encoder | 16.68 | 18.7 | 74.3 | 1.43 |
| decoder | 18.31 | 18.6 | 75.2 | 1.39 |
| encoder + decoder | 14.97 | 18.7 | 75.3 | 1.43 |
| GRU core | 15.89 | **23.3** | **85.6** | 1.38 |
| prior head | 13.05 | 21.2 | 73.9 | 1.59 |
| posterior head | 13.86 | 21.3 | 72.6 | 1.39 |
| RSSM 전체 | 16.94 | 26.4 | 89.7 (≈ pretrained 90.3) | 1.81 ± 0.23 |

- encoder + decoder의 fine-tuning을 되돌려도 새 데이터의 동역학 예측(openloop +3%)과 value_gap(+3%)은 거의 그대로다. 새 데이터에 대한 일반화 이득은 대부분 **RSSM(특히 GRU)의 변화**에서 온다.
- value_gap의 차이는 RSSM 전체 되돌리기를 제외하면 SE 안이다. 즉 "GRU는 픽셀 오차에, prior는 value 오차에 반응한다"는 신호는 **결론을 내릴 수 없는 수준**이다.
- chunk 단위로 보면 openloop 오차와 value_gap의 Spearman 상관은 0.67(ft40), 0.57(ft20)이다. 픽셀 오차는 과제 관련 오차와 상당 부분 겹치지만 완전히 같지는 않다.
- 이 분석은 사후 분석이다. 같은 모듈을 처음부터 고정한 학습 궤적(논문 Fig. 8)과는 다르다. 예를 들어 encoder 적응은 초기에만 중요할 수 있다.

## 6. 코드 이슈의 실제 크기: action padding 미적용 **[측정]**
`tools/padded_action_probe.py` (QW ft40, 미학습 online 데이터의 posterior 4096개에서 16 step 상상, CPU 실행):
- 상상 중 action의 패딩 차원(12–20) 평균 |a| = 0.41 (실제 차원 0.69). RSSM에 들어가는 값이 작지 않다
- RSSM `_in_action` 가중치의 열 노름: 패딩 차원 4.00, 실제 차원 3.50. 사전학습 중 입력이 늘 0이어서 학습되지 않은 무작위 초기값이 거의 그대로 남아 있다
- 같은 noise로 패딩만 0으로 바꿨을 때 상상 λ-return 차이: +0.62 ± 0.02 (전체 131.8의 0.47%). 81%의 시작점에서 원본이 더 높다
- **결론**: Quadruped(패딩 9차원)에서는 효과가 작고 일관된 상상 return 부풀림이다. Meta-World(실제 4차원, 패딩 17차원)에서는 더 클 수 있지만 측정하지 않았다

## 7. 핵심 질문: "NCRL은 새 task에 적응하며 무엇을 불필요하게 반복하는가?"

| 반복 계산 | 근거 | 비용 비중 | 줄일 여지 |
|---|---|---|---|
| 같은 chunk의 encode와 64-step scan을 0 상태부터 매번 | [코드] state=None. online 전이당 약 144번 재생 | WM forward 326 ms (11%) | 캐시는 파라미터가 바뀌므로 정확하지 않음. 시작점용 forward(274 ms)는 WM 생략 시에도 남음 |
| WM 업데이트가 본 데이터를 외움 | [측정] seen 10.8 vs unseen 73.0 | WM 업데이트 1.21 s (41%) | 업데이트 빈도·조건부 생략: 최대 −32% (4070) |
| rehearsal 4행이 오차를 유지만 함 | [측정] prior 오차 변화 없음 | WM 배치의 25% | 생략하면 forgetting과 성능에 영향(논문 Fig. 6 ER)이 예상되어 직접 검증이 필요 |
| 1024개 시작점 전부에서 상상 | [측정, 이전] 256개: −48% 시간, −4.7% return(1 seed) | 1.71 s (59%) | 가장 큼. 단 WM 업데이트가 아니라 AC 쪽 |
| actor gradient의 RSSM 통과(dynamics) | [측정] actor backward 1.11 s(패치) / 약 0.58 s(원본 추정) | 상상·AC의 35–65% | REINFORCE로 바꾸면 RSSM 역전파가 없어짐(DreamerV3 방식). 이전 측정: 664 ms vs 1738 ms. NCRL 성능 영향은 미검증 |
| decoder recon(1024프레임) | [코드] 표현 학습에만 쓰임 | FLOPs 41%, 시간 4% + backward | 시간 비중이 작음 |

**"그 비용이 정말 필요한가"에 대한 현재 답**: 아직 실험 결과(return)가 없다. 지금까지의 증거는 WM 업데이트가 **이미 본 데이터의 오차를 줄이는 데** 대부분 쓰인다는 것이다. 이것이 정책 학습에 필요한지는 별도 문제다. 상상 시작점이 바로 그 본 데이터에서 나오므로, 암기가 오히려 상상 정확도에 도움이 될 수도 있다. 이 질문은 분기 파일럿(`experiment_plan.md` P1)이 처음으로 return으로 답한다.

## 8. 재현 명령
```bash
cd ~/RL/ncrl_light/tools
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl
PY=/home/yeho/miniforge3/envs/ncrl_env/bin/python
$PY -I profile_update.py --out ../results/profile_update.json        # §2, §4 (~6분)
$PY -I diag_wm_error.py --out ../results/diag_wm_error.json          # §5 (~5분)
DEV=cpu $PY -I padded_action_probe.py --out ../results/padded_action_probe.json   # §6 (CPU 약 15분)
```
