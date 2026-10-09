#!/bin/bash
# lists every r5 pass whose output is missing (as queue lines); exit 1 if any
P5=/media/raid/cloth/output/teaser_video/room_demo/passes_r5
miss=0
[ -f $P5/lo_grey.exr ] || { echo "lo:grey::"; miss=1; }
for m in 453 8 145 226 9 50 367 311 314 370; do
  for r in sofaL pillow59 pillow55 sofaR pillow3 pillow5 cush58 cush2 cush56 cush57 rug curtainL; do
    [ -f $P5/neural_${r}_${m}.exr ] || { echo "neural:$r:$m:1024"; miss=1; }
    [ -f $P5/lo_${r}_${m}.exr ] || { echo "lo:$r:$m:"; miss=1; }
  done
done
for m in 226 370; do [ -f $P5/swatch_${m}.exr ] || { echo "swatch:$m::256"; miss=1; }; done
exit $miss
