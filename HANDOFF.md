# 인수인계 (새 채팅은 여기서 시작)

작성 2026-10-25 10:40. 레포: `~/RL/ncrl_light` (비공개 GitHub: https://github.com/YeeeeeHo/ncrl-light, `git push`로 갱신)

## 1. 목표
NCRL(ICLR 2026, 사전학습된 285.7M RSSM world model + retrieval/rehearsal/BC guidance)을 새 task에 fine-tuning할 때의 계산 비용을 줄인다. 특히 **불필요한 WM 업데이트를 식별하고 제거**하는 데 집중한다. 최종 목표는 새 방법을 제안하는 논문이다. 사용자 연구실의 주제는 계산 절감이다.

## 2. 현재 상태
- **분기 파일럿 종료**(2026-10-25 10:09). GPU는 비어 있고, 실행 중인 작업은 없다
- **보류**: 40k WM 1/2, GRU freeze(`runs/branch40k/pending.txt`의 `#HOLD`), 후속 하드웨어 측정(`runs/post_pilot.sh`: 배치·시퀀스·TF32/fp16별 시간·메모리, freeze 표 메모리 재측정)
- **문서**: README(그림 4개), 분석 문서 5종, `results/freq_summary_20261025.md`(최종 비교), `results/status_20261025.md`(전체 정리)

## 3. 핵심 결과 (RTX 4070, Quadruped Walk, seed 1)

**계산 (측정)**
- wall-clock의 99%가 업데이트다
- 업데이트 1회 2.92초 = WM 1.21초(41%) + 상상·actor-critic 1.71초(59%)
- WM scan은 대역폭에 묶여 있다. WM을 통째로 빼도 −32%가 상한이다
- encoder·decoder freeze는 −4~9%, GRU freeze는 WM 시간 −34%

**WM 진단 (forward만)**
- 40k WM의 open-loop 오차: 이미 본 에피소드 10.8 vs 다음 에피소드 73.0 (사전학습 모델 90.3)
- rehearsal 데이터 오차는 유지되고, 다른 embodiment는 잊는다

**12GB 실행**
- 상상 step checkpointing(`probe/ckpt_patch.py`)으로 원본 설정을 결과 그대로 실행한다(차이 0.0). 피크 8.9 GiB

**분기 파일럿 — 40k 분기** (원본 3회 반복)
| 조건 | 50k | 60k | 손실 | 업데이트 시간 |
|---|---:|---:|---:|---:|
| Full | 806 | 882 (846–914) | — | 357 s/epoch |
| Every4 | 882 | 895 | +12 | −23% |
| Every8 | 770 | 683 | −199 | −26% |
| OFF (old) | 759 | 711 | −171 | −31% |

**분기 파일럿 — 20k·60k 분기**
- 20k: Full 707, Every4 510, OFF(strict) 551. 단 Full끼리도 580–707로 흔들려 판단 불가
- 60k: Full 915, OFF(strict) 842(−73). 원본이 이미 정체된 구간이다

**해석 (모두 1 seed)**
- 1/4가 최적점이다. 1/8부터 손실 신호가 있다
- 손실은 +10k가 아니라 +20k에서 늦게 나타난다
- 빈도를 줄여도 peak VRAM은 그대로다(10.29 GiB). OFF만 9.53 GiB
- 학습 중 return은 BC guidance 구간 때문에 판단 지표가 되지 못한다. **eval로 판단한다**
- 구조적 한계: 고정 1/4가 WM 쪽 상한(−31%)의 대부분(−23%)을 이미 얻는다. 적응형 방법의 여지가 작으므로, 고정 빈도가 깨지는 task를 찾거나 상상 시작점 축(59%)과 결합해야 한다

## 4. 다음 단계 (사용자 계획, 아직 시작 안 함. 시작 전에 사용자 확인 필요)
1. **Quadruped Walk 독립 3 seed 검증**: Full vs Every4. 설계를 사용자와 먼저 정한다
   - 처음부터 독립 학습이면 150k frame 1회 약 15시간(100k frame이면 약 10시간)
   - eval 에피소드를 30개로 늘릴 것을 제안함
   - 40k 기준 표준편차가 약 34이므로, seed 3개로는 약 80점 이상의 차이만 검출된다
2. **다른 task에서 Full vs Every4**: Meta-World 1회 약 17시간(150k frame, epoch당 25 업데이트)
3. (선택, 추가 학습 없음) 보존된 팔별 리플레이(`runs/branch40k/<arm>/buffer`)로 "손실이 늦게 나타나는 이유" 분석: 새 에피소드에 대한 WM 오차와 value gap

## 5. 도구 사용법
- **분기**: `tools/branch_selective.py [--wm_every K | --strict_off | --freeze G..] [--save_end_state NAME] [--eval_only] [--snapshot_override P] [--base_run_override P] -- --base_run R --snapshot S --out DIR --epochs 20`. 내부적으로 `probe/branch_ncrl.py`를 수정 없이 실행한다
- **큐**: `runs/branch40k/queue_runner.sh` + `pending.txt`(한 줄 = `name|args`, 대기 중인 줄은 수정 가능). 감시는 `runs/watchdog2.sh`
- **요약**: `python3 tools/summarize_pilot.py`(분기 시점별 표), `tools/compare_freq.py`(빈도 비교 + 그림), `tools/make_figures.py`(README 그림. ncrl_env python 필요)
- **원본 학습**: `probe/run_ncrl.py <hydra overrides>`. 정확한 인자는 `hardware_feasibility.md` §6
- **분기 지점**: base run `~/RL/ncrl_runs/baseline/logs/finetune/dmc-quadruped-walk/09af53c751cc43f5b4f6d07d6a082f14_1/snapshot_{20000,40000}.full.pt`, 60k는 `runs/branch40k/full_to60k/state_60000.full.pt`
- **환경**: conda `ncrl_env`, 실행 시 `MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. GPU 작업은 한 번에 하나만 돌린다(12GB)

## 6. 주의
- **구현 차이**: 40k OFF는 weight decay로 WM이 0.25% 줄어든 "old off"다. 20k·60k OFF는 strict(변화 0.0)다
- **peak VRAM**: 2026-10-24 22:3x 이전에 시작한 팔은 미기록이다
- **경로**: 스크립트에 절대경로(`/home/yeho/RL/...`)가 들어 있다
- **NCRL 코드 버그**: `nets.py`의 `sampe` 오타로 action padding이 적용되지 않는다(QW 영향 +0.47%, MW는 미측정)

## 7. 사용자 선호
- 모든 설명은 한국어로 한다
- 결과를 미리 짐작해서 말하지 않는다. seed 1개 결과는 "차이를 검출하지 못함"까지만 말한다
- 확인할 수 있는 것은 묻지 말고 확인한 뒤 보고한다
- **추가 학습은 자동으로 시작하지 않는다**(명시적 지시가 있을 때만)
- 긴 작업이 끝나거나 문제가 생기면 푸시 알림을 보낸다
- 휴대폰일 때는 파일을 전송하고, 컴퓨터 앞에 있을 때는 경로만 알려 준다
