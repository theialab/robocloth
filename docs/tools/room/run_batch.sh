#!/bin/bash
# usage: run_batch.sh GPU "stage:obj:mat:spp[:suffix] ..."   (logs one JSON line per pass)
export PYTHONNOUSERSITE=1
R=/home/zla247/anaconda3/envs/fipt-mitsuba/bin/python
P=$(dirname "$(readlink -f "$0")")/passes.py
cd /home/zla247/projects/SGHyperMaterials
for job in $2; do
  IFS=: read -r st o m spp suf <<< "$job"
  echo "== $job $(date +%T)"
  if [ "$st" = swatch ]; then
    CUDA_VISIBLE_DEVICES=$1 $R $P swatch $o --spp $spp --batch 16 2>&1 | grep -E '^swatch|Error|error'
  elif [ "$st" = lo ]; then
    CUDA_VISIBLE_DEVICES=$1 $R $P lo $o $m 2>&1 | grep -E '^\{|Error|error'
  elif [ "$st" = xtalk ]; then
    CUDA_VISIBLE_DEVICES=$1 $R $(dirname "$P")/xtalk.py render $o 2>&1 | grep -E '^c1920|Error|error'
  elif [ "$st" = masks ]; then
    CUDA_VISIBLE_DEVICES=$1 $R $P masks --spp $spp 2>&1 | grep -E '^aov|Error|error'
  else
    CUDA_VISIBLE_DEVICES=$1 $R $P neural $o $m --spp $spp --batch ${BATCH:-32} ${suf:+--suffix $suf} 2>&1 | grep -E '^\{|Error|error'
  fi
done
echo DONE $(date +%T)
