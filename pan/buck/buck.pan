ground electrical gnd

parameters EIN=20 KP=0.1 KI=10 VREF=10 VDIG=1 FREQ=20k PERIOD=1/FREQ

Tr tran stop=100*PERIOD uic=1 tmax=PERIOD/1k
En envelope period=PERIOD stop=400m restart=no annotate=3 maxh=50

Sw       l   gnd   rm   c   nport \
		  func1=(v(p2) > 1u) ? v(p1) - (EIN+sin(2*pi*50*time)) : v(p1) \
		  func2=i(p2) 
L1       l     w   inductor  l=1m
Rl       w     o   resistor  r=10m
Co       o   gnd   capacitor c=20u
Ro       o   gnd   resistor  r=6

X1       x   gnd   o   vcvs func=v(o)-VREF
Pi       y   gnd   x   gnd  svcvs dcgain=KP numer=[KI,KP] denom=[0,1]
X2       c   gnd   y   vcvs func=limit(v(y),1m,1-10m)
Vp      rm   gnd   vsource t=0 v=0 t=PERIOD-1n v=1 t=PERIOD v=0 period=PERIOD

