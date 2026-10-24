# Baseline 및 Ablation 실험 계획

작성 2026-10-24. 원칙:
- baseline은 **원본 NCRL 설정**이다(+ 결과가 같은 checkpoint 패치).
- 결과를 예상해서 적지 않는다. 판정 규칙은 결과를 보기 전에 정한다(아래 "판정 규칙"은 2026-10-24 17시 기준, P1의 첫 eval이 나오기 전에 작성).
- **env 상호작용 수(frame, step)와 학습 업데이트 수(WM step, AC step)를 항상 따로 적는다.**
- 계산량 절감(시간, FLOPs)과 sample efficiency(env step당 return)는 별개의 성과로 보고한다.

## 0. 용어와 단위
- frame = env 프레임. action repeat 2 → agent step = frame/2. QW 1 epoch = 1000 frame = 500 step = **125 업데이트**. MW 1 epoch = 200 frame = 100 step = 25 업데이트
- "업데이트 1회" = WM step(선택적) + reward head step + AC step(상상 포함). 이 계획에서는 **AC step 수와 env step 수를 고정하고 WM step 수만 바꾼다**(P1/P2)
- 원본: WM step = AC step = 125/epoch

## 1. 평가 지표

| 분류 | 지표 | 측정 방법 |
|---|---|---|
| 계산 효율 | wall-clock(업데이트, 수집, 평가 분리), GPU-h | `branch_log.jsonl`의 `update_s`, `collect_s` (CUDA 동기화 후 측정) |
| | 업데이트당 시간, FLOPs | `profile_update.py` (CUDA event, FlopCounterMode 합계) |
| | Peak VRAM | `torch.cuda.max_memory_allocated` |
| | WM step 수 / AC step 수 | `update_metrics.jsonl` (`wm_steps`, `updates`) |
| sample efficiency | env frame 대비 eval return (10 에피소드, mean action) | `eval/avg_episode_reward` |
| | 학습 곡선 AUC | eval 지점의 사다리꼴 적분 / 구간 길이 |
| | 목표 도달 frame (QW 800, 900) | 처음 넘는 eval 지점 |
| 안정성 | 분기 시점 대비 최저점(dip), 최대 하락폭, 회복 여부 | eval 곡선 |
| | forgetting | retrieved 데이터 및 다른 embodiment의 WM 오차 (`diag_wm_error.py`) |
| WM 상태 | 새 에피소드 오차(학습 전), 리플레이 오차, value_gap | `diag_wm_error.py` 확장(E2) |

## 2. 이미 실행한 것 (재현 정보)

| ID | 내용 | 명령/스크립트 | 결과 위치 | 상태 |
|---|---|---|---|---|
| R0 | QW 원본 baseline, seed 1, 300k frame 목표 | `ncrl_probe/run_ncrl.py` (`hardware_feasibility.md` §6) | `ncrl_runs/baseline/` | 50k frame에서 중단(사용자). eval 103/489/463/574/803. 20k·40k 전체 상태 스냅샷 있음 |
| R1 | 40k 분기, 원본 업데이트, seed 7, 40 epoch | `ncrl_probe/branch_ncrl.py` | `ncrl_runs/branch_qw40k/full1024` | 완료. eval 50k/60k/70k/80k = 740/846/942/936. epoch당 362초 |
| R2 | 40k 분기, 상상 시작점 256개 | 같음 `--starts 256` | `ncrl_runs/branch_qw40k/s256` | 완료. 701/906/866/892. epoch당 약 188초 |
| M1 | 업데이트 단계별 시간, FLOPs, freeze 표 | `tools/profile_update.py` | `results/profile_update_run1.json` | 완료 (freeze 표 메모리 열은 재측정 예정) |
| M2 | WM 오차, drift, 모듈 되돌리기 | `tools/diag_wm_error.py` | `results/diag_wm_error.json` | 완료 |
| M3 | action padding 영향 | `tools/padded_action_probe.py` (CPU) | `results/padded_action_probe.json` | 완료 |
| MW0 | MW Shelf Place 원본, seed 1 | `ncrl_probe/run_split.py` | `ncrl_runs/split_shelf/full` | 21k frame에서 중단(사용자). 10k/20k 성공률 0 |

## 3. 진행 중: 분기 파일럿 P1/P2 (실험 A·B의 첫 단계)
- **스크립트**: `tools/branch_selective.py`. `branch_ncrl.py`를 수정 없이 실행하고, `DreamerAgent.update`(WM step 생략)와 `WorldModel.update`/`update_reward`(freeze)만 교체한다. 생략된 업데이트에서도 배치는 원본과 같게 뽑는다(online 12 + retrieved 4). posterior는 no-grad forward로 만들어 reward head와 1024 상상 시작점에 쓴다. 따라서 **AC step 수, 상상 시작점 수, env step 수는 원본과 같다**
- **스모크 테스트** (2026-10-24 16:47, 1 epoch, `--wm_every 2 --freeze gru`): WM step 63/125, GRU 변화량 정확히 0.0, 나머지 모듈 0.8–1.4% 변화, epoch 290초. 첫 epoch 수집 return 506.17은 R1과 소수점까지 같다(분기의 짝 비교 성립)
- **러너**: `runs/branch40k/queue_runner.sh` + `pending.txt`(대기 중인 줄은 수정·삭제 가능). 로그: `runs/branch40k/queue.log`, 팔별 `<arm>/branch_log.jsonl`, `<arm>/update_metrics.jsonl`, `<arm>/selective_end.json`

| 팔 | 분기점 | WM step / AC step | 예상 시간 | 비교 대상 |
|---|---|---|---|---|
| `wm_off` | 40k | 0 / 2500 | 1.4 h | R1, full_rep |
| `full_rep` | 40k | 2500 / 2500 | 2.0 h | **R1과의 차이 = 같은 seed 재실행 noise floor** |
| `wm_every4` | 40k | 625 / 2500 | 1.55 h | |
| `s20_full` | 20k | 2500 / 2500 | 2.0 h | 초반 기준 |
| `s20_wm_off` | 20k | 0 / 2500 | 1.4 h | H1b |
| `s20_wm_every4` | 20k | 625 / 2500 | 1.55 h | H1b |
| `wm_every2` | 40k | 1250 / 2500 | 1.7 h | |
| `freeze_gru` | 40k | 2500(GRU 제외) / 2500 | 1.75 h | 실험 A의 유일한 계산 절감형 freeze |
| `wm_every8` | 40k | 313 / 2500 | 1.45 h | |

- 공통: seed 7, 20 epoch(= +20k frame = +10k step), 10k frame마다 eval 10 에피소드, train env seed 8, eval env seed 11(R0과 같음)
- 총 약 15시간. 2026-10-24 16:53 시작
- 실험 A에서 제외한 구성과 이유:
  - encoder freeze, decoder freeze, encoder + decoder freeze: 계산 절감이 4–9%뿐이고(M1), Fig. 8에서 해로움이 이미 보고되었으며, 되돌리기 분석(M2)에서 이 모듈들의 후반 변화가 새 데이터 오차에 거의 기여하지 않았다
  - continuation predictor: 없다

**대기열 변경 (2026-10-24 22:22, 사용자 요청)**
- 목적: WM 업데이트를 중단하는 학습 시점(20k / 40k / 60k)에 따라 성능 손실이 달라지는지 확인
- 새 순서: `s20_wm_off` → (`eval_s20`, `eval_s40`) → `full_to60k` → `eval_s60` → `s60_full` → `s60_wm_off` → `s20_wm_every4` → `wm_every8` → `wm_every2` → `freeze_gru`. 이전 목록은 `runs/branch40k/pending_before_2026-10-24_reorder.txt`에 보존
- **60k 체크포인트**: 기존 `full_rep`은 분기 도중 모델과 옵티마이저 상태를 저장하지 않았다(리플레이와 로그만 남음). 그래서 `full_to60k`로 40k→60k를 원본 업데이트로 다시 돌리고, 끝 상태를 run_ncrl 형식으로 저장한다(`--save_end_state`). 60k 분기는 이 상태에서 시작한다. `full_to60k`는 40k→60k full의 세 번째 반복도 겸한다. **`full_rep`과 같은 상태는 아니다.**
- **WM OFF 구현 차이**: 새 OFF 팔(`s20_wm_off`, `s60_wm_off`)은 `--strict_off`로 WM을 비트 단위로 고정한다(gradient 없음, weight decay 없음). 이미 끝난 `wm_off`(40k)는 NCRL reward-head 단계의 weight decay로 WM이 0.25% 줄어들었다(gradient 업데이트는 0회). 요약표의 notes 열에 "old off"와 "strict off"로 구분해 표시한다
- **초기 성능**: 분기 시점마다 스냅샷을 1회 평가한다(`--eval_only`). 학습 루프의 eval과 같은 eval env seed 11과 10 에피소드를 쓰고, 별도 프로세스라 각 팔의 RNG와 eval 순서에 영향이 없다
- **새 기록 항목**: epoch별 peak VRAM(allocated/reserved). 이 기능 추가 이전에 시작한 팔(R1, full_rep, wm_off, wm_every4, s20_full)은 미기록. 요약(`tools/summarize_pilot.py`)은 분기 시점별로 초기 성능, +10k/+20k eval, 최종 return, eval AUC(초기, +10k, +20k의 사다리꼴 / 20k), full 대비 차이, 업데이트 시간, peak VRAM, 구현 차이를 표시한다
- **CPU 검증** (GPU는 실행 중인 팔이 사용 중): `tools/test_branch_selective.py`, 결과 `results/test_branch_selective_{unit,eval,resume}.json`
  - strict off → WM 동일, reward·actor·critic만 변화
  - old off → WM이 정확히 fp32(1−1e-6)^n 비율로 감소
  - every2 → WM step [1,0,1]
  - eval_only → 40k 스냅샷 평가 동작
  - 저장한 상태 → `branch_ncrl.py`로 epoch 40에서 정상 재개(online 51 + 검색 500 에피소드 복사)

**중간 결과 (2026-10-24 20:22)**: `results/pilot_interim_20261024.md`
- noise floor δ = 121(50k), 68(60k)
- wm_off: eval 759 / 711, 학습 return은 원본보다 +61–68, 업데이트 시간 −32%
- 지표 방향이 엇갈려 판단 불가. seed 추가가 필요하다

**seed 추가 규칙** (사용자 결정 2026-10-24 20:3x: "다 끝나고 결과 보고 필요하면 추가". 규칙은 파일럿 결과 전에 기록함):
- 다음 중 하나라도 해당하면 그 비교에 분기 seed 8, 9를 추가한다
  - 원본 대비 eval 평균 차이(50k·60k, 또는 30k·40k)가 2δ보다 작다
  - eval과 학습 return의 차이 방향이 서로 다르다
- 대상과 우선순위: (40k) full, wm_off, wm_every4 → (20k) s20_full, s20_wm_off
- 두 지표가 같은 방향이고 2δ를 넘는 비교에는 추가하지 않는다
- (2026-10-24 20:4x 사용자: 자동으로 추가하지 않는다. 파일럿 종료 후 GPU는 비워 두고, 추가 여부는 사용자가 다음 날 결정)

### 판정 규칙 (P1/P2, 1 seed이므로 **기술적(descriptive) 판정만**)
- noise floor δ = R1(full1024)과 `full_rep`의 eval 차이(50k, 60k 각각). 같은 seed, 같은 분기점이지만 DataLoader와 cuDNN의 비결정성 때문에 생긴다
- 각 팔의 60k eval이 `full_rep`/R1 범위에서 δ 이상 벗어나지 않으면 **"차이를 검출하지 못함"**, 둘 다보다 δ 이상 낮으면 **"손실 신호"**라고만 부른다. 성능이 같다고 주장하지 않는다
- H1b: s20 팔들의 40k eval 손실(full 대비)이 40k 팔들의 60k eval 손실보다 뚜렷하게 크면(δ 이상) "시점 의존 신호"로 기록한다
- 어떤 신호든 **E3(3 seed 이상)에서 확인되기 전에는 결론으로 쓰지 않는다**

## 4. 다음 실험 (P1/P2 결과에 따라 순서 조정)

### E0. 계산 특성 확인 (파일럿 후 자동 실행: `runs/post_pilot.sh`)
- WM 업데이트 시간과 메모리: B ∈ {4, 8, 16, 32}(T=64), T ∈ {16, 32}(B=16), fp32 / TF32 / NCRL fp16
- 상상·AC: 1024(ckpt) fp32 / TF32 / fp16, 256(no ckpt)
- freeze 표 메모리 재측정
- 목적: H3(시간이 B에 무관한가), `hardware_feasibility.md`의 미측정 항목 채우기

### E1. Noise floor와 seed
- P1 이후, 결론에 쓸 팔은 서로 다른 분기 seed 3개(예: 7, 8, 9)로 반복한다
- 가능하면 두 번째 base run(seed 2)의 40k 스냅샷도 만든다(약 6시간). base run 하나에서 나온 결론의 일반성을 확인하기 위함

### E2. 업데이트 필요성 신호 (실험 C 후반부)
- 각 팔에서 epoch마다 **방금 수집한 에피소드**(아직 학습 전)의 WM 오차(recon, KL, open-loop, value_gap)와 리플레이 표본 오차를 기록한다. 현재 파일럿에는 이 로그가 없으므로, 팔별로 저장된 buffer(에피소드 파일 보존)를 사후에 스냅샷과 함께 평가하는 방식으로 우선 대체한다
- 분석: 신호(오차 수준, 감소 추세, value_gap)와 "WM step을 생략했을 때의 return 손실" 사이의 관계. 시점별(20k vs 40k) 비교
- 질문 매핑: (1) 오차가 작으면 생략 가능한가 (2) 커도 업데이트가 효과 없는 경우가 있는가 (3) task 무관 오차의 분리(value_gap vs 픽셀)

### E3. 확인 실험 (가설이 살아남은 경우)
- 과제: QW + Meta-World 2개(Shelf Place처럼 retrieval이 정확한 과제 1개, Reach처럼 혼합된 과제 1개)
- 팔: 원본 / 최선의 고정 빈도 / DUTD 방식(새 에피소드 오차로 빈도 ×1.3 또는 ÷1.3, **DUTD baseline**) / 제안 규칙
- seed 3개 이상. 처음부터 학습(150k frame)은 1회 15–17시간이므로, 분기 기반 비교와 전체 학습 비교를 섞어 쓴다

### E4. 상상 시작점 (H4, 선택)
- 256 무작위 vs 256 선택(value 상위, 불확실성 상위, chunk별 균등 간격) vs 1024
- 계산 절감 상한이 가장 큰 축이다. Gap 1 결과가 약하면 우선순위를 올린다

### E5. Dip / 분포 이동 연결 (사용자 관심 항목)
- P2(20k)와 R0의 초기 구간에서 eval 하락(R0: 20k 489 → 30k 463)이 WM 업데이트 생략으로 커지거나 작아지는지 기록한다
- action padding 이슈(H6)를 MW에서 측정한다(`padded_action_probe.py`의 MW 버전)

## 5. 위험 요소와 대응
- **1 seed 분기의 한계**: 분기점이 하나(R0 seed 1)라서 base run의 특성에 결과가 묶일 수 있다 → E1
- **QW는 쉬운 과제**: 40k 이후 빠르게 900점대로 오른다. WM 업데이트가 덜 필요해 보일 수 있다 → E3에서 MW 확인
- **eval noise**: 10 에피소드로는 약 5% 차이를 구분하기 어렵다. 학습 중 return은 guidance 구간 때문에 매우 출렁인다(5–970) → seed 수로 보완
- **계산 절감 상한**: WM만 줄이면 최대 32–40%. 논문 주장의 크기가 여기서 결정된다

## 6. 재현 명령
```bash
# P1/P2 큐 (이미 실행 중)
cd ~/RL/ncrl_light/runs/branch40k && ./queue_runner.sh            # pending.txt의 팔을 순서대로 실행
# 단일 팔
R=~/RL/ncrl_runs/baseline/logs/finetune/dmc-quadruped-walk/09af53c751cc43f5b4f6d07d6a082f14_1
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -I ~/RL/ncrl_light/tools/branch_selective.py --wm_every 4 -- \
  --base_run $R --snapshot $R/snapshot_40000.full.pt --out wm_every4 --epochs 20
# 결과 요약
python3 ~/RL/ncrl_light/tools/summarize_pilot.py
```
