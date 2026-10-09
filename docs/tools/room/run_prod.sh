#!/bin/bash
# production passes for materials 145 50 453 9
D=$(dirname "$(readlink -f "$0")")
J=""
for m in 145 50 453 9; do J="$J swatch:$m::256"; done
for m in 145 50 453 9; do J="$J neural:pillowsL:$m:1024 neural:pillowsR:$m:1024"; done
for m in 453 9; do J="$J neural:sofaL:$m:1024 neural:sofaR:$m:1024 neural:cushions:$m:1024"; done
"$D/run_batch.sh" ${GPU:-1} "$J"
