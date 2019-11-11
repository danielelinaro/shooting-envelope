ground electrical gnd

parameters PERIOD=2*pi

Tr        tran stop=100*PERIOD uic=1
Env   envelope period=PERIOD autonomous=yes tstop=4k*PERIOD restart=0 ereltol=10m corrector=0.55 \
               annotate=4
Trful     tran tstop=4k*PERIOD uic=1 method=2 order=6 tmax=PERIOD/1000

I1  x    y  u VAND IC=1m
Vu  u  gnd    vsource vsin=200m freq=1m

model VAND nport veriloga="vand.va"
