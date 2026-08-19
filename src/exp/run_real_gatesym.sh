#!/bin/bash
# Symmetric gate on real data: PTB-XL and HAPT, both stems, 5 seeds x 5 lambdas.
# Five seeds, not three, because the asymmetric baselines this is compared against all
# used [0,1,2,3,4] and the comparison is paired seed by seed.
#
# Why these two. Section 7.1 split SEP into inclusion / allocation / exclusion and
# found that every real-data loss against SFA lives in the exclusion term and nowhere
# else (PTB-XL -0.136 and -0.051, HAPT/Reg -0.007). The symmetric gate is aimed at
# exactly that term, so these are the settings where it can actually be judged.
# Sleep-EDF is deliberately excluded: its optimal lambda is 0 and the smoke test
# showed the symmetric gate helps only at lambda > 0.
#
# The smoke test on synthetic, seed 0, one stem, corrected the hypothesis before this
# run. The original claim was that the symmetric gate would create exclusion on its
# own and make the penalty unnecessary. It does not: at lambda=0 exclusion went
# 0.205 -> 0.192, i.e. nothing. Not rewarding transient content is not the same as
# pushing it out, and nothing in the objective pushes. What it did instead was raise
# exclusion at lambda=1 from 0.689 to 0.869, taking the product from 0.599 to 0.755.
# So the mechanism is amplification of the penalty, not replacement of it.
#
# The prediction under test, written before the run:
#   * exclusion rises at lambda > 0 but NOT at lambda = 0
#   * the gain is largest where the penalty was already doing most of the work, i.e.
#     PTB-XL at its optimum lambda=4
#   * inclusion holds. z_slow now trains on a weighted subset of horizons, and on
#     PTB-XL only 47% of sampled deltas exceed tau, so this is where it could break.
#
# Checkpoints: the 30 gated trainings per dataset are fresh (gate_sym enters the key
# only when set), the 6 ungated controls are reused from the asymmetric runs.
set -u
cd "$(dirname "$0")"
export HGLP_NSEED=5 HGLP_GATE_SYM=1

for ds in ptbxl hapt; do
  for stem in nce+ema l1+ema; do
    out=/tmp/rot_${ds}_sym_${stem%%+*}.log
    echo "=== $(date +%H:%M) start ${ds} gate_sym ${stem} ==="
    HGLP_TAG=_sym python3 rotation.py "$ds" "$stem" > "$out" 2>&1
    echo "    exit=$? -> $out"
  done
done

echo "=== done $(date +%H:%M) ==="
ls -la ../../runs_v2/rotation_{ptbxl,hapt}_*_sym.json 2>/dev/null
