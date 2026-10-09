#!/bin/bash
# usage: run_xtalk.sh GPU name...   (one process per render: clean GPU state, own timing)
export PYTHONNOUSERSITE=1
R=/home/zla247/anaconda3/envs/fipt-mitsuba/bin/python
X=$(dirname "$(readlink -f "$0")")/xtalk.py
cd /home/zla247/projects/SGHyperMaterials
G=$1; shift
for n in "$@"; do
  CUDA_VISIBLE_DEVICES=$G $R $X render $n 2>&1 | grep -E "^(a|b|c|obj_|lo_)[a-zA-Z0-9_]* \{|Error|error|Traceback"
done
echo DONE $(date +%T)
