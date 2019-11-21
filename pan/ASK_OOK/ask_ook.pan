ground electrical gnd

parameters VDD=1.8 F1=2G F2=1M CR=1f LR=1/(CR*(2*pi*F1)^2)

options rawkeep=true

Eq dc annotate=3 print=true vabstol=1p vreltol=1p ireltol=1p
Tr tran stop=100/F1 trabstol=1a vabstol=0.1p iabstol=0.1p tmax=1m/F1 devvars=yes uic=0
En envelope period=1/F1 tstop=300/F1 trabstol=1a vabstol=0.1p iabstol=0.1p \
            corrector=0.5 tmax=10m/F1 restart=0

Evv   vdd    gnd   vsource vdc=VDD
M1    vdd     g1   d10   MOS
D1    d10    vdd   DIODE
Rd1   d10    l10   resistor  r=10
Cm1   vdd    d10   capacitor c=100f

L1    l10    l20   inductor  l=20n
C1    l20    gnd   capacitor c=0.5n
L2    l20    l30   inductor  l=20n

Rd2   l30    d20   resistor  r=10
M2    d20     g2   gnd MOS
Cm2   d20    gnd   capacitor c=100f

C2    l30    out   capacitor c=1n
Ctl   out    gnd   capacitor c=CR
Ltl   out    gnd   inductor  l=LR
Rl    out    gnd   resistor  r=300

Vg1    g1    d10   vsource v1=0 v2=VDD tr=10n tf=10n width=0.1/F2-10n period=1/F2
Vg2    g2    gnd   vsource v1=0 v2=VDD tr=10p tf=10p width=0.5/F1-10p period=1/F1

model DIO diode

subckt MOS d g s
parameters BETA=0.025 ALPHA=1 KT=2 VT=1
M1   d   s   g   vccs func=0.5*BETA*(KT*(v(g,s)-VT)+ln(exp(KT*(v(g,s)-VT)) + exp(-(KT*(v(g,s)-VT)))))*tanh(ALPHA*v(d,s))
ends

subckt DIODE a c
parameters IS=1u ETA=2 VTEMP=0.026
D1 a c vccs func=IS*(exp(v(a,c) / (ETA*VTEMP)) - 1)
ends

