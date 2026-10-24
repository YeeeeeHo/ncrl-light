# RTX 4070 12GB에서 원본 NCRL 실행 가능성

작성 2026-10-24. 표기: **[측정] [코드] [논문] [추정]**. 측정 환경은 `computation_analysis.md` §0과 같다.

## 0. 결론
- **원본 설정(논문 크기 285.7M WM, 배치 16×64, 상상 1024 시작점 × 16, fp32)을 알고리즘 변경 없이 12GB에서 돌릴 수 있다.** 필요한 변경은 상상 rollout의 step 단위 activation checkpointing 하나뿐이다. RNG 상태를 보존하므로 actor/critic의 loss, gradient, 파라미터가 원본과 비트 단위로 같다(차이 0.0) **[측정]**.
- 대가: 업데이트당 약 +0.55초(상상 forward 재계산). 업데이트 1회 2.92초, 150k frame 학습 1회에 약 15–17시간 **[측정]**.
- 배치, 시퀀스 길이 축소나 mixed precision, 모듈 freeze는 **메모리 때문에 필요하지 않다**. 이것들은 알고리즘이나 수치를 바꾸므로 baseline 재현에는 쓰지 않는다.
- 실제 학습 검증: Quadruped Walk에서 원본 업데이트로 학습이 정상 진행된다(50k frame 803점, 분기 70k frame 942점). Meta-World 재현은 아직 확인하지 못했다(§4).

## 1. 메모리가 어디서 터지는가 **[측정]**

| 상태 | 할당 (GiB) |
|---|---:|
| 모델 상주 (WM 285.7M + AC 13.7M, fp32) | 1.12 |
| + WM Adam 상태 (2 × 285.7M) | 약 3.44 |
| WM 업데이트 피크 (B16×T64) | 8.90 |
| 상상·AC 업데이트 피크, 원본 128 시작점 | 5.46 |
| 원본 256 시작점 | 7.23 |
| 원본 512 / 1024 시작점 | **OOM** |
| checkpoint 패치, 1024 시작점 | 8.20 |

- 원인: `actor_grad=dynamics`라서 16 step 상상의 RSSM activation을 전부 저장해야 한다. 시작점당 약 13.6 MiB이므로 1024개면 +13.7 GiB **[측정, 이전 세션 `mem_breakdown.py`]**
- WM 업데이트 자체(8.9 GiB)는 들어간다. 즉 **OOM의 유일한 원인은 상상 단계**다
- 가용 메모리는 11.58 GiB이고 X 서버가 약 0.4 GB를 쓴다. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 단편화를 줄여야 학습 중 OOM이 나지 않았다(학습 로그에 "mapping failed" 경고가 나오지만 allocator가 재시도해 진행됨) **[측정]**

## 2. 우선순위별 검토

### 2.1 공개 pretrained checkpoint — **사용 가능 [측정]**
- Hugging Face `zhaoyi11/ncrl`의 `models.zip`(2.09 GB) → `dmcontrol.pt`, `metaworld.pt`. 데이터셋 19 GB(DMC 10k, MW 50k 에피소드)
- 논문 크기 config로 strict `load_state_dict`가 오류 없이 된다. 즉 아키텍처가 일치한다(encoder, rssm, decoder 키와 shape 전부)
- 사전학습을 다시 할 필요가 없다. 논문 기준 사전학습 비용은 MI250X에서 약 48 GPU-h **[논문 App. F]**
- retrieval용 DB: 원본 `prepare_dataset_kv.py`는 RAM을 60 GB 넘게 쓴다. `ncrl_probe/prep_db.py`(첫 프레임만 복사)로 만든 `dataset_db_dmc.npz`, `dataset_db_mw.npz`를 쓴다 **[이전 세션]**

### 2.2 Batch size 축소 — **불필요. 쓰면 알고리즘이 바뀜**
- NCRL은 `GRAD_PER_EPOCH = ceil(steps / (B·T / train_ratio))`라서, B를 반으로 줄이면 epoch당 업데이트 수가 두 배가 된다. 재생 비율(env step당 재생 timestep 256)은 유지되지만, gradient noise와 Adam step 수가 바뀐다 **[코드]**
- online/offline 분할이 `int(0.75·B)`, `int(0.25·B)`라서 B가 4의 배수가 아니면 rehearsal 비율이 바뀐다(B=6 → 4 + 1) **[코드]**
- 상상 시작점 수 = B×T이므로 B를 줄이면 AC 학습도 바뀐다 **[코드]**
- 시간 측면: WM scan은 대역폭에 묶여 있어서 B를 줄여도 업데이트당 시간이 거의 줄지 않을 것이다. 그러면 env step당 총 시간은 오히려 늘어난다 **[추정 → E0에서 측정 예정: `tools/profile_variants.py`, 파일럿 종료 후 자동 실행]**

### 2.3 Sequence length 축소 — **불필요. 쓰면 RSSM 학습이 바뀜**
- 각 chunk는 0 상태에서 시작한다(burn-in 없음). T를 줄이면 앞부분의 부정확한 posterior 비중이 커지고, 상상 시작점(B×T)도 함께 줄어든다 **[코드 + 추정]**
- WM 업데이트 메모리와 시간은 T에 비례해 줄 것이다 **[추정 → E0 측정 예정]**

### 2.4 Mixed precision — **알고리즘 동일성이 깨짐. 메모리 이득도 없음**
- NCRL에는 `agent.precision=16` 스위치가 있다(fp16 autocast + GradScaler). 사전학습은 16, fine-tuning 기본값은 32 **[코드]**
- 이전 측정: fp16 autocast에서는 상상 메모리가 오히려 **시작점당 35 MiB**로 늘었다(fp32는 13.6 MiB). autocast가 만드는 dtype 변환 사본 때문으로 보인다 **[측정, 이전 세션 / 원인은 추정]**
- TF32 matmul(기본 꺼짐)은 메모리 변화 없이 속도만 올릴 수 있다. 다만 수치가 비트 단위로 같지 않다 **[추정 → E0 측정 예정]**
- 결론: baseline 재현에는 fp32를 유지한다. 속도 비교용 별도 팔로만 검토한다

### 2.5 Gradient checkpointing — **채택. 결과가 동일한 유일한 방법**
- `ncrl_probe/ckpt_patch.py`: `WorldModel.imagine`의 각 step(정책 샘플 → `img_step`)을 `torch.utils.checkpoint(..., preserve_rng_state=True)`로 감싼다. 하류에서 쓰지 않는 `feat`, `logits` 텐서도 저장하지 않는다. NCRL 저장소 파일은 수정하지 않고, import 시점에 메서드만 교체한다
- 동등성: 논문 아키텍처, B2×T8(16 시작점), 3회 업데이트에서 actor와 critic의 loss, gradient, 파라미터, target critic 차이가 모두 **0.0**이다 **[측정: `ncrl_probe/equiv_paper_arch.json`, `equiv_test2.py`]**
- 부분 checkpoint(앞 8 step 또는 12 step만)는 1024 시작점에서 OOM. 16 step 전체 checkpoint만 들어간다 **[측정]**
- 비용: 상상·AC 1.71초(패치) vs 약 1.15초(원본 외삽) → 업데이트당 +0.55초, 총 시간 +약 23% **[측정 + 추정]**
- WM 업데이트 쪽 checkpoint는 필요 없다(8.9 GiB로 들어감)

### 2.6 Parameter freezing — **메모리 목적으로는 불필요. 연구 대상으로만 다룸**
- freeze는 Adam 상태와 gradient 메모리를 줄이지만 학습 동작을 바꾼다. 논문 Fig. 8은 처음부터 모듈을 고정하면 sample efficiency가 크게 떨어진다고 보여 준다 **[논문]**
- 계산 절감 효과는 `computation_analysis.md` §4 참조

## 3. "OOM이 안 난다 ≠ 정상 학습" 점검 **[측정 + 코드]**

| 점검 항목 | 결과 |
|---|---|
| 업데이트 동등성 | checkpoint 패치가 비트 단위로 같음 (위 §2.5) |
| online/offline 비율 | 실행 config `buffer.online_ratio=0.75`, 배치 16 → loader 12 + 4 |
| epoch당 업데이트 수 | 로그 "Grad update per epoch: 125"(QW), 25(MW). 코드 공식과 일치 |
| warm-up | 로그 "Warmup epoch: 15"(QW), 75(MW). 업데이트는 epoch 11부터 |
| BC 사전학습 | 150k step, batch 256 (config 그대로) |
| guidance schedule | QW 실행은 논문 DMC 값 `linear(1,0,50000)`. repo 기본값(`linear(1,0.1,100000)`)과 다르므로 명시적으로 지정함 |
| pretrained 호환 | strict load 통과. `models_to_load={wm: true}`이고 actor/critic/reward는 새로 초기화 (원본과 같음) |
| 학습 곡선 | QW seed 1: 10k/20k/30k/40k/50k frame에서 103/489/463/574/803 (평가 5 ep). 40k 분기(seed 7, 원본 업데이트): 50k/60k/70k/80k에서 740/846/942/936 (평가 10 ep) |
| 논문 대조 | Fig. 8 "Full" 곡선은 약 50k step에서 정규화 약 90. 우리 결과는 50k frame에서 74–80, 70k frame에서 94. 대체로 맞지만 seed 1–2개라 재현이라 부를 수는 없음. x축이 frame인지 step인지도 논문에 명시되지 않음 |

## 4. 아직 확인하지 못한 것
- **Meta-World 성능 재현**: Shelf Place 원본 실행은 21k frame에서 중단되었다(사용자 요청). 10k/20k frame 성공률은 0이다. 논문 수치(150k frame에 0.80)와 비교할 수 있는 지점까지 가지 못했다. 1회에 약 17시간 **[측정]**
- **seed 수**: 논문은 task당 3 seed. 우리는 QW 1–2 seed
- B/T/TF32/fp16 시간과 메모리 표는 파일럿 종료 후 자동 측정된다(`results/profile_variants.json`). 결과는 이 문서에 추가한다

## 5. 연구용 실험 환경 구성안
1. **기준(baseline)**: 원본 설정 + checkpoint 패치(결과 동일). 비교는 언제나 이 설정을 기준으로 한다
2. **빠른 반복 실험**: 처음부터 학습하지 않고, 저장된 학습 상태(`*.full.pt`: 모든 파라미터, 세 Adam 상태, BC 정책, 카운터, RNG, 리플레이 manifest)에서 **분기**한다(`ncrl_probe/branch_ncrl.py`). 20 epoch 분기 1회가 1.4–2시간
3. **과제 선택**: Quadruped Walk(빠르게 학습, 1 epoch = 125 업데이트) + Meta-World 1–2개(사용자 주 관심 도메인)
4. **seed**: 결론을 내리는 실험은 최소 3 seed. 분기 파일럿은 1 seed(경향 확인용)

## 6. 재현
```bash
# 원본 fine-tuning (12GB): run_ncrl.py = 원본 + ckpt_patch + jsonl 로그 + 전체 상태 스냅샷
cd ~/RL/ncrl_runs/<dir>
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -I ~/RL/ncrl_probe/run_ncrl.py ws.use_wandb=false ws.seed=1 ws.mode=finetune ws.num_train_frames=300000 \
  ws.use_action_padding=true env.task=dmc-quadruped-walk snapshot.load_path=~/RL/ncrl_data/models/models \
  buffer.num_workers=4 buffer.online_ratio=0.75 buffer.kv_path=~/RL/ncrl_data/dataset_db_dmc.npz buffer.save_buffer=true \
  ws.use_bc=true ws.jsrl_schedule='linear(1,0,50000)' snapshot.save_every_frames=20000 \
  agent.rssm.deter=12288 agent.rssm.hidden=1536 agent.rssm.classes=96 agent.encoder.cnn_depth=96
# 동등성 테스트
python -I ~/RL/ncrl_probe/equiv_test2.py --ncrl ~/RL/ncrl_src --paper_arch --B 2 --T 8 --updates 3 --tol 0.0 --out equiv_paper_arch.json
```
(실제 실행 인자는 `ncrl_runs/baseline/logs/.../run_info.json`과 `config_resolved.yaml`에 저장되어 있다. 시작 시 retrieval과 BC 사전학습에 4488초(75분)가 걸렸다.)
