# NCRL 아키텍처 분석 (구조와 학습 과정)

작성 2026-10-24. 대상: *Efficient Reinforcement Learning by Guiding World Models with Non-Curated Data* (Zhao et al., ICLR 2026, arXiv 2502.19544v3), 공식 코드 `zhaoyi11/ncrl` (로컬 `~/RL/ncrl_src`, 수정 없음).

출처 표기:
- **[논문]** 논문 본문, 표, 부록에 적힌 내용
- **[코드]** `ncrl_src`를 직접 읽고 확인한 내용 (파일:줄 표기)
- **[측정]** 이 저장소에서 직접 잰 값. 스크립트와 로그는 `results/`에 있음
- **[추정]** 위 근거들로부터 추론한 내용. 검증된 사실이 아님

모든 수치는 논문 크기 설정(`scripts/finetune.sh`의 override) 기준입니다: `rssm.deter=12288, rssm.hidden=1536, rssm.classes=96, encoder.cnn_depth=96`, 배치 16 × 길이 64, 상상 horizon 16, fp32.

---

## 0. 한 장 요약

```
[Offline 사전학습]  비큐레이션 데이터 (DMC 5 embodiment 10k ep / MW 50 task 50k ep, reward 없음)
   └─ WM = Encoder + RSSM(GRU core + prior + posterior) + Decoder,  loss = recon + KL   (reward·continue head 없음)

[Online fine-tuning]  새 task 하나
   0) Retrieval   : 첫 프레임을 pretrained encoder로 임베딩 → Faiss L2 top-500 에피소드 → prior_buffer (1회)
   1) BC prior    : prior_buffer로 BC 정책 학습 (150k step, 1회)
   2) 매 epoch = 에피소드 1개 수집 (warm-up 15k frame은 BC가 행동, 이후 확률적으로 BC 구간 끼움 = execution guidance)
   3) epoch마다 업데이트 125회 (Quadruped 기준). 업데이트 1회 =
        a. WM 업데이트   : online 12 + retrieved 4 chunk (=rehearsal) → recon + KL → Adam (285.7M 파라미터 전부)
        b. reward head  : online 12 chunk의 posterior (detach) → two-hot CE
        c. 상상          : 1024개 posterior 상태 각각에서 16 step rollout (actor 샘플 → RSSM img_step)
        d. actor        : λ-return을 RSSM을 통과해 역전파 (actor_grad=dynamics)
        e. critic       : 상상 궤적의 λ-return 회귀, target critic EMA
```

파라미터 **[측정]** (`results/profile_update.json`):

| 그룹 | 파라미터 | 비율 |
|---|---:|---:|
| Encoder (CNN 4층) | 6.20M | 2.1% |
| RSSM GRU core (`_gru`) | 155.83M | 52.5% |
| RSSM prior head | 25.96M | 8.7% |
| RSSM posterior head | 28.32M | 9.5% |
| Decoder (FC 33.05M + ConvT 32.01M) | 65.07M | 21.9% |
| Reward head | 4.33M | 1.5% |
| **World model 합계** | **285.72M** | |
| Actor | 4.49M | |
| Critic (+ target critic 같은 크기) | 4.60M (+4.60M) | |
| **Actor-critic 합계** | **13.68M** | |

논문이 말하는 "280M RSSM"은 encoder, decoder, reward head까지 포함한 WM 전체 크기와 맞습니다. 이 설정은 DreamerV3의 `size400m` preset과 일치합니다(deter 12288, hidden 1536, classes 96, cnn 96, blocks 8) **[코드, 이전 세션에서 dreamerv3 configs.yaml과 대조]**.

---

## 1. World Model

### 1.1 Encoder — `dreamer/nets.py: Encoder`
- **역할**: 64×64×3 RGB 프레임 → 3072차원 embed (768채널 × 2×2) **[코드]**
- **구조**: Conv2d k4 s2 × 4층 (96→192→384→768채널), 각 층 뒤에 채널 LayerNorm(`ChannelNorm`, eps 1e-3) + SiLU **[코드]**
- **파라미터**: 6.20M **[측정]**
- **Fine-tuning 중 업데이트**: 예. WM loss의 backward가 posterior head를 거쳐 encoder까지 내려감 **[코드: wm.py loss]**
- **Forward/Backward**: B·T = 1024 프레임을 한 번에 처리 (시간축 병렬) **[코드]**. forward 25 ms **[측정]**
- **계산량이 생기는 곳**: 2·3층 conv가 대부분. 프레임당 0.124 GMAC, 업데이트당 126.5 GMAC (WM forward FLOPs의 22%) **[추정: 해석적 계산]**
- **독립 freeze**: 가능. embed에서 gradient를 멈추면 됨. 단 절감은 WM 업데이트 시간의 5%뿐 **[측정: §2 freeze 표]**
- **부수 용도**: retrieval key/query를 만드는 feature extractor. retrieval은 학습 시작 시 1회, **pretrained** encoder로 수행 **[코드: finetune.py `_retrieve_buffer`]**. 그래서 fine-tuning이 encoder를 바꿔도 retrieval 결과에는 영향이 없음

### 1.2 RSSM / Dynamics — `dreamer/rssm.py`
상태 = deter h (12288차원) + stoch z (32 × 96 categorical, unimix 1%) **[코드]**.

| 하위 모듈 | 구조 | 파라미터 [측정] | 행당 MAC [추정] |
|---|---|---:|---:|
| `_gru._in_deter` | Linear 12288→1536 + LN + SiLU | 18.88M | 18.9M |
| `_gru._in_stoch` | Linear 3072→1536 + LN + SiLU | 4.72M | 4.7M |
| `_gru._in_action` | Linear 21→1536 + LN + SiLU | 0.04M | 0.03M |
| `_gru._dynlayers` | BlockLinear (8 blocks) 49152→12288 + LN + SiLU | 75.53M | 75.5M |
| `_gru._dyngru` | BlockLinear (8 blocks) 12288→36864 (reset/cand/update) | 56.66M | 56.6M |
| `_prior` | MLP 12288→1536→1536→3072 logits | 25.96M | 26.0M |
| `_posterior` | MLP (12288+3072)→1536→3072 logits | 28.32M | 28.3M |

- **역할**: `img_step`(prior: h_t = f(h_{t-1}, z_{t-1}, a_{t-1}), z_t ~ p(z|h_t))와 `obs_step`(img_step + posterior: z_t ~ q(z|h_t, e_t)) **[코드]**
- **학습 시 forward**: `observe`가 길이 64의 시간축을 **Python loop로 순차 실행**. 행 수는 B = 16뿐 **[코드: dreamer_utils.static_scan]**. 매 step마다 `torch.cat`으로 출력을 누적하므로 복사 비용이 T²에 비례 **[코드]** (총량이 작아서 시간 영향은 미미 **[추정]**)
- **상상 시 forward**: `img_step`만 16 step 실행. 행 수는 1024 (모든 posterior 상태) **[코드: wm.py imagine]**
- **업데이트**: 예. dyn KL(×1.0)이 prior와 GRU로, repr KL(×0.1)과 recon이 posterior·GRU·encoder로 흐름 **[코드]**
- **계산 특성**: 학습 scan은 매 step마다 GRU 계열 가중치 약 210M개(fp32로 840 MB)를 다시 읽는데, 곱하는 행은 16개뿐이다. 그래서 FLOPs보다 **메모리 대역폭과 kernel launch 지연에 묶인다** **[추정. 근거: observe forward 252 ms가 FLOPs 비중(37%)에 비해 매우 큼, §2 참조]**
- **독립 freeze**: GRU core, prior, posterior를 각각 따로 얼릴 수 있음. 단 encoder나 posterior를 학습하는 동안에는 scan 전체의 activation gradient가 여전히 필요하므로, 얼려도 줄어드는 것은 weight-gradient 계산과 Adam뿐 **[코드 + 측정: GRU freeze 시 WM 시간 −34%]**
- **코드에서 확인한 세부 사항**:
  - `BlockGRUCell`의 입력 투영(in_deter/in_stoch/in_action)은 block 사이에 공유되고 8번 복제됨. gate는 `torch.chunk(x, 3, -1)`로 **평탄화된 벡터**에서 나뉜다. DreamerV3는 block별로 나누므로, NCRL에서는 block들이 gate를 통해 서로 섞인다 **[코드; DreamerV3와의 차이는 이전 세션에서 대조]**
  - free nats(1.0)를 batch 평균 KL에 적용함 (DreamerV3는 원소별 clip 후 평균) **[코드: rssm.py loss]**

### 1.3 Decoder — `dreamer/nets.py: Decoder`
- **역할**: (h, z) → 64×64×3 이미지 복원. recon MSE (픽셀 합, batch·time 평균) **[코드]**
- **구조**: deter → BlockLinear(12288→3072, 8 blocks) + LN + SiLU + Linear(3072→3072). stoch → Linear 3072→3072 ×2. 두 결과를 더해 3072×1×1로 만든 뒤 ConvTranspose 4층(k 5,5,6,6; 384→192→96→3채널) **[코드]**
- **파라미터**: 65.07M **[측정]**
- **업데이트**: 예 **[코드]**
- **계산**: 프레임당 0.230 GMAC (ConvT 3층이 112M으로 최대). 업데이트당 235.6 GMAC = WM forward FLOPs의 41% **[추정: 해석적 계산]**. forward 48 ms **[측정]**
- **독립 freeze**: 가중치는 얼릴 수 있음. 그러나 recon gradient가 RSSM과 encoder로 가야 하므로 decoder의 input-gradient backward는 여전히 실행된다. 절감은 WM 시간의 4% **[측정]**
- **용도 제한**: 상상, 행동, actor-critic 어디에서도 쓰이지 않음. 표현 학습용 loss에만 쓰임 **[코드]** (`report()` 영상 기록 제외)

### 1.4 Reward Predictor — `wm.reward_model` (`nets.MLPs`, 255-bin symlog two-hot)
- **역할**: (h, z) → reward 분포 **[코드]**
- **파라미터**: 4.33M **[측정]**
- **업데이트**: fine-tuning에서만 존재(`mode != 'pretrain'`). **online 12행만** 사용하고 입력 posterior는 detach. 별도 step으로 학습 **[코드: dreamer.py update]**
- **옵티마이저 공유**: WM과 같은 `model_opt`를 쓴다. 이 step에서도 WM 전체 285.7M 파라미터에 weight decay(`var.data = (1-wd)·var.data`)가 한 번 더 적용됨. 즉 업데이트 1회에 WM 전체가 weight decay를 두 번 받는다 **[코드: dreamer_utils.Optimizer]**
- **비용**: 9 ms/업데이트 **[측정]**

### 1.5 Continuation Predictor — **없음** [코드]
- NCRL WM에는 continue/termination head가 없다. 상상 중 discount는 상수 0.99이고, weight는 discount의 누적곱이다 **[코드: wm.py imagine]**
- DMC와 Meta-World는 에피소드 중간에 종료가 없으므로 실질적 영향은 없다고 봄 **[추정]**. 따라서 "continuation predictor freeze" 실험은 대상이 없어 제외함

---

## 2. Actor–Critic — `dreamer/actor_critic.py`
- **입력**: (h, z). 앞단은 MLP: deter → BlockLinear 12288→1024(8 blocks) + LN + SiLU + Linear 1024→512, stoch → Linear 3072→512, 둘을 더함 **[코드: nets.MLPs]**
- **Actor**: MLP 3층(512) → 21차원(패딩 포함) mean/std. tanh mean, std ∈ [0.1, 1], TruncatedNormal. 4.49M **[코드/측정]**
- **Critic**: 같은 trunk → 255-bin two-hot. 4.60M. target critic은 매 업데이트마다 EMA 0.02 **[코드]**
- **Actor 목적함수**: `actor_grad=dynamics`. λ-return(γ 0.99, λ 0.95)을 RSSM `img_step` 16단계를 거쳐 action까지 역전파한다. WM 파라미터는 `requires_grad=False`라 갱신되지 않지만 **activation은 모두 저장해야 한다**. 이것이 12GB OOM의 유일한 원인이다 **[코드 + 측정]**. entropy 계수 1e-4. return 정규화(`ValueEMA`)는 reinforce 모드에서만 쓰임 **[코드]**
- **Critic 목적함수**: 상상 궤적(detach) 위에서 two-hot CE, weight = discount 누적곱 **[코드]**
- DreamerV3 원본은 `ac_grads: False` + REINFORCE 조합이라 RSSM을 통한 역전파를 하지 않는다 **[이전 세션: dreamerv3 configs.yaml 확인]**. 이 부분은 NCRL만의 선택이다

### 2.1 코드에서 발견한 이슈: action padding이 샘플에 적용되지 않음
`nets.py GaussianPolicy.forward`에서 mean에는 mask가 적용되지만 샘플 줄은 오타 때문에 적용되지 않는다 (`sampe = sample * action_mask`) **[코드]**.
- 결과: 학습 모드의 행동 선택(`act` → `agent_state`)과 상상 rollout에서 RSSM이 **패딩 차원(DMC Quadruped 12–20, MW 4–20)에 0이 아닌 값을 받는다**. 리플레이와 사전학습 데이터의 action은 이 차원이 항상 0이다 **[코드: replay_buffer.py가 0으로 패딩]**
- `actor_grad=dynamics`이므로 actor는 env에 전달되지 않는 이 차원으로도 상상 return을 올리도록 gradient를 받는다 **[코드]**
- 실제 영향은 `tools/padded_action_probe.py`로 측정 중이며, 결과는 `computation_analysis.md` §6에 기록한다. 이 이슈는 경량화와는 별개지만 offline→online 분포 이동과 dip 분석에서는 교란 요인이 될 수 있다

---

## 3. Imagination Rollout — `wm.imagine`
- **시작점**: WM 업데이트 배치의 **모든** posterior 상태 B×T = 16×64 = 1024개(detach). retrieved 4행에서 나온 256개 상태도 포함된다 **[코드: dreamer.py update]**
- **진행**: step마다 actor 샘플 → `img_step`(GRU + prior, z 샘플). 16 step **[코드]**
- **계산**: 시작점당 step당 186M MAC (img_step 182M + actor 4.5M). 업데이트당 forward 3.05 TMAC = 6.1 TFLOP. 같은 양이 actor backward로 한 번 더 든다 **[추정: 해석적 계산. 전체 AC 측정치 13.6 TFLOP와 일치]**
- **메모리**: activation 저장 때문에 시작점당 약 13.6 MiB **[측정, 이전 세션]**. 1024개면 +13.7 GiB → OOM **[측정]**
- 한 chunk(64 step) 안에서 이웃한 시작점들은 서로 매우 비슷할 것이다 **[추정]**. 256개만 무작위로 써도 학습이 가능하다는 단일 seed 근거가 있다(§ computation_analysis 4.3)

---

## 4. Experience Retrieval & Rehearsal
**Retrieval** **[코드: finetune.py `_retrieve_buffer`, prepare_dataset_kv.py; 논문 Eq. 5]**
- key: 사전학습 풀의 각 에피소드 **첫 프레임**을 pretrained encoder로 임베딩(3072차원). query: 학습 env의 `reset()` 프레임 1장
- Faiss `IndexFlatL2`, top-k (`num_retrieve_episodes=500`, action 차원이 다른 에피소드는 버림) → `prior_buffer`에 복사. **학습 시작 시 1회만** 수행
- 비용: 수 초 수준 **[논문 "in seconds"]**. 다만 원본 `prepare_dataset_kv.py`는 RAM을 60 GB 넘게 쓰는 문제가 있어 `ncrl_probe/prep_db.py`로 대체했다 **[이전 세션 측정]**
- 정밀도: Quadruped Run, Assembly, Shelf Place 100%, Door Open 68% (@500) **[논문 Table 2]**. 이전 세션 측정: MW 일부 과제는 100% 목표 과제의 에피소드였다(근사 전문가 σ 0.1 포함) **[측정: memory `direction-ncrl-split`]**

**Rehearsal** **[코드: finetune.py `_setup_buffer`, dreamer.py update]**
- offline loader 배치 = `int((1-0.75)·16)` = 4 chunk. 매 업데이트마다 online 12 chunk와 이어 붙인다
- 사용처: (i) WM loss (recon + KL), (ii) 상상 시작점(256/1024). **reward head에는 쓰지 않음**
- 주의: B가 4의 배수가 아니면 `int()` 내림 때문에 online/offline 비율이 바뀜 (예: B=6 → 4 + 1 = 5행) **[코드]**

## 5. Execution Guidance — `behavior_cloning.py`, `finetune.py train`
- BC 정책: 3-frame stack(9채널) → 별도 CNN encoder → MLP 512-512 → 21차원. MSE(패딩 차원 mask), Adam 1e-4, random shift aug. `prior_buffer`로 150k step(batch 256) 사전학습 **[코드]**. 4070에서 약 75분 **[측정, 이전 세션]**
- warm-up(15k frame = QW 15 에피소드, MW 75 에피소드) 동안은 BC가 에피소드 전체를 행동 **[코드 + 실행 로그]**
- 이후 매 에피소드 확률 `schedule(jsrl_schedule, step)`로 BC 구간을 끼운다. 길이는 에피소드의 10–100% 중 무작위, 시작 위치도 무작위 **[코드]**
- schedule: repo 기본값 `linear(1,0.1,100000)`, 논문 Table 6은 DMC `linear(1,0,50000)` / MW `linear(1,0,1,150000)`. MW 쪽 표기는 인자가 4개라 코드의 parser로 파싱할 수 없음(오타로 추정) **[논문 + 코드]**. 단위는 agent step(frame/2) **[코드: global_step]**
- BC가 출력하는 패딩 차원 값도 mask되지 않고 `agent_state`(다음 obs_step 입력)로 들어간다 **[코드]**

## 6. Offline Pre-training과 Online Fine-tuning 과정

### 6.1 Pre-training — `pretrain.py`, `config/pretrain.yaml`
- 데이터: DMC 10k 궤적(5 embodiment), MW 50k 궤적(50 task, TD-MPC2 에이전트 + Gaussian noise σ 최대 2.0). reward 사용 안 함 **[논문 §4]**
- loss: recon + dyn KL + 0.1·repr KL. reward/continue head 없음. action은 21차원으로 0 패딩 **[코드]**
- benchmark마다 WM 1개(dmcontrol.pt, metaworld.pt) **[코드: finetune.py가 task prefix로 ckpt 선택]**
- 설정 불일치 **[논문 vs 코드]**:
  - 배치: Table 6은 16, `scripts/pretrain.sh`는 32
  - 단계 수: Table 6은 200k, App. F는 "150k step에 약 48 GPU-h"
  - 정밀도: pretrain은 precision 16, finetune 기본은 32
  - Table 6은 hidden 12288 / deter 1536으로 적었으나 코드는 반대(deter 12288, hidden 1536)

### 6.2 Fine-tuning 루프 — `finetune.py Workspace.train`
- 1 epoch = 에피소드 1개 수집. QW: 1000 frame = 500 agent step. MW: 200 frame = 100 step **[코드 + 실행 로그]**
- 업데이트 수/epoch = `ceil(steps / (B·T/train_ratio))` = `ceil(500/(16·64/256))` = **125** (QW), 25 (MW) **[코드 + 실행 로그 'Grad update per epoch']**. 즉 agent step 4개마다 업데이트 1회
- 업데이트는 epoch > `min(WARMUP_EPOCH, 10)`일 때 시작. QW는 11번째 epoch부터 **[코드]**
- 평가: 10k frame마다 결정적 정책(mean)으로 5 에피소드 **[코드 기본값]**
- **"환경 상호작용 횟수"와 "업데이트 횟수"는 `train_ratio`로 묶여 있다.** batch를 줄이면 업데이트 수가 자동으로 늘어 재생 비율(replay ratio)이 유지되므로, 배치 축소는 env step당 연산량을 거의 줄이지 못한다 **[코드]**

### 6.3 무엇이 학습되고 무엇이 고정되는가 (fine-tuning 기본값)

| 모듈 | 초기값 | 업데이트 | 데이터 | 비고 |
|---|---|---|---|---|
| Encoder | pretrained | 매 업데이트 | online 12 + offline 4 | |
| RSSM (GRU, prior, posterior) | pretrained | 매 업데이트 | online 12 + offline 4 | |
| Decoder | pretrained | 매 업데이트 | online 12 + offline 4 | |
| Reward head | random | 매 업데이트 | online 12 | posterior detach |
| Actor | random | 매 업데이트 | 상상(1024 시작점 × 16) | RSSM 통과 역전파 |
| Critic / target | random | 매 업데이트 / EMA | 상상 | |
| BC 정책 | random | 사전학습 후 고정 | retrieved 500 ep | |

논문 Fig. 8 (Quadruped Walk, 그림에서 읽음 **[논문]**):
- **Full** fine-tuning: 약 50k step에 정규화 점수 약 90
- **Enc & Dec**만 학습: 약 200k step에 약 90
- **Dynamics**만 학습: 400k step 이후에도 약 80, 분산 큼

따라서 "처음부터 특정 모듈을 고정"하는 방식은 sample efficiency를 크게 잃는다는 선행 근거가 이미 있다. 단 seed 수, rehearsal/guidance 포함 여부, x축 단위(frame 또는 step)는 본문에 적혀 있지 않다.

---

## 7. 재현 정보
- 파라미터와 시간: `python -I tools/profile_update.py --out results/profile_update.json` (ncrl_env, torch 2.13 + CUDA 13.0, RTX 4070)
- 코드 사실의 위치: `ncrl_src/src/dreamer/{wm,rssm,nets,actor_critic,dreamer,dreamer_utils}.py`, `finetune.py`, `replay_buffer.py`, `behavior_cloning.py`
- 논문 Fig. 8 원본: `https://arxiv.org/html/2502.19544v3/ablation_role_rssm.svg`
