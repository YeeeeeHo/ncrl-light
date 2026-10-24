# 분기 파일럿 중간 결과 (2026-10-24 20:22, 9개 팔 중 2개 완료)

**한 줄 요약**: 40k frame부터 WM 업데이트를 완전히 멈춰도 20k frame 동안 학습은 계속 진행됐고, 업데이트 시간은 32% 줄었다. 성능 차이는 지표에 따라 방향이 엇갈린다. 같은 설정을 다시 돌렸을 때 생기는 흔들림(noise)이 그 차이만큼 커서, **seed 1개로는 "손해가 있다/없다"를 판단할 수 없다.**

## 설정
- Quadruped Walk. 원본 실행(seed 1)의 40k frame 학습 상태에서 분기, 분기 seed 7, 20 epoch(= +20k frame = 2500 업데이트)
- 모든 팔에서 env step 수, AC 업데이트 수(2500), 상상 시작점(1024)은 같다. **WM 업데이트 수만 다르다**
- eval: 10k frame마다 10 에피소드, 평균 action
- 판정 규칙(결과 전에 `experiment_plan.md` §3에 기록): noise floor δ = |원본 − 원본 재실행|

## 결과

| 팔 | WM 업데이트 | 업데이트 시간/epoch | eval 50k | eval 60k | 학습 에피소드 평균 return (20 ep) |
|---|---:|---:|---:|---:|---:|
| 원본 R1 (기존 실행) | 2500 | 355 s | 740 | 846 | 570 |
| 원본 재실행 `full_rep` | 2500 | 362 s | 861 | 914 | 563 |
| **WM 중단 `wm_off`** | **0** | **247 s (−32%)** | 759 | **711** | **628** |

- **noise floor δ**: 50k에서 121, 60k에서 68. 같은 상태, 같은 seed, 같은 알고리즘인데도 DataLoader와 cuDNN의 비결정성 때문에 eval이 이만큼 달라진다
- **사전 규칙대로 읽으면**:
  - 50k: wm_off(759)가 두 원본(740, 861) 사이에 있다 → 차이를 검출하지 못함
  - 60k: wm_off(711)가 두 원본보다 δ(68) 이상 낮다(−135, −203) → "손실 신호"
- **반대 방향의 증거**: 학습 중 수집한 에피소드의 return(stochastic 정책, guidance 구간 포함)은 wm_off가 오히려 높다. epoch별 짝 차이: wm_off − full_rep = +68 ± 32(SE), wm_off − R1 = +61 ± 24. 19 epoch 중 14 epoch에서 wm_off가 높았다. 같은 비교를 원본 두 실행끼리 하면 −7 ± 34
- **상상 속 가치는 거의 같다**: 60k 시점 critic target이 wm_off 175.45, full_rep 175.47. WM을 멈춰도 actor-critic이 보는 상상 return은 원본과 같은 속도로 올랐다
- **wm_off의 WM 변화량**: 모든 모듈이 정확히 0.25% 줄었다. 이는 reward head 단계에서 NCRL 옵티마이저가 WM 전체에 weight decay(1e-6 × 2500회)를 적용하기 때문이다. 원본 동작을 그대로 둔 결과이며, gradient 업데이트는 0회다

## 해석 (현재 수준)
1. **40k 이후 WM 업데이트가 학습 진행에 필수는 아니었다.** WM을 고정한 채 actor-critic만 학습해도 eval이 574(40k, base run) → 711–759로, 학습 return이 900대까지 올랐다
2. **그러나 손해가 없다고 말할 수는 없다.** 60k eval은 낮았고, 학습 return은 높았다. 두 지표가 엇갈리고, 같은 설정끼리의 차이(δ)가 효과 크기와 비슷하다
3. **계산 절감은 측정한 그대로다**: epoch당 362초 → 247초(−32%). 사전 계산(`computation_analysis.md` §4)과 일치한다
4. 따라서 이 질문은 **seed를 늘려야** 답할 수 있다(`experiment_plan.md` E1). 남은 팔(1/4, 1/2, 1/8, GRU freeze)도 seed 1개로는 같은 한계를 가질 것이다. 그래도 20k 분기와 40k 분기의 비교(초반에 멈추면 더 나빠지는가)는 효과가 크다면 보일 수 있다

## 다음
- 진행 중: `wm_every4`(20:21 시작) → `s20_full` → `s20_wm_off` → `s20_wm_every4` → `wm_every2` → `freeze_gru` → `wm_every8` → 후속 측정. 내일 08시경 종료 예정
- 제안(사용자 확인 필요): 파일럿이 끝나면 `wm_off`와 `full_rep`를 분기 seed 2개씩 추가(8, 9)해 noise floor와 효과를 다시 추정한다. 팔당 1.4–2시간, 총 약 7시간

## 원자료
- `runs/branch40k/{wm_off,full_rep}/branch_log.jsonl`, `update_metrics.jsonl`, `selective_end.json`
- 원본 R1: `~/RL/ncrl_runs/branch_qw40k/full1024/branch_log.jsonl`
- 표 생성: `python3 tools/summarize_pilot.py` → `results/pilot_summary.md`
