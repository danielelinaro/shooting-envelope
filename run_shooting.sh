#!/bin/bash

for T_small in 100 50 40 25 20 16 10 8 6.66666666 5 ; do
    python examples/buck.py shooting-comparison "${T_small}e-6"
done

outdir="BUCK_SIMULATIONS"
mv buck_shooting_*.pkl $outdir
mv buck_shooting_*.pdf $outdir

