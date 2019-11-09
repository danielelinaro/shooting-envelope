ground electrical gnd

;parameters PERIOD=20u C=47u L=10u R=5 RS=0 VIN=5 VREF=5 KI=1 VDIG=1
parameters PERIOD=20u C=47u*30 L=10u*2 R=5 RS=0 VIN=5 VREF=5 KI=1 VDIG=1

Tr tran stop=200.0*PERIOD uic=1

;Sh shooting period=100*PERIOD floquet=true restart=0 eigf=true \
;            tmax=PERIOD/1k samples=32k

Env envelope tstop=300*PERIOD period=2*PERIOD restart=0 \
             method=2 order=2 eabstol=1e-6 ereltol=1e-3

Save control begin

    Time = get("Tr.time");
    Vc   = get("Tr.c");
    Il   = get("Tr.l");

    save("mat5","tr.mat","time", Time, "vc", Vc, "il", Il );

    clear Time Vc Il

    Time = get("Env.time");
    Vc   = get("Env.c");
    Il   = get("Env.l");

    save("mat5","env.mat","time", Time, "vc", Vc, "il", Il );

endcontrol

;    clear Time Vc Il
;
;    Time = get("Sh.time");
;    Vc   = get("Sh.c");
;    Il   = get("Sh.l");
;
;    save("mat5","sh.mat","time", Time, "vc", Vc, "il", Il );
;
;    clear Time Vc Il
;
;    Time = get("Sh.eigtime");
;    u1C  = get("Sh.u1@c");
;    u1L  = get("Sh.u1@l");
;
;    save("mat5","eig.mat","time", Time, "u1c", u1C, "u1L", u1L );

; Circuit

;C1    c    gnd   capacitor c=1 ic=9.3124
C1    c    gnd   capacitor c=1 ic=10.154335434351671
Ic    c    gnd   z   l  g vccs func=(v(z) > VDIG/2) ? v(g)/C*v(c) : v(g)/C*v(c) - 1/C*v(l)

Vg    g    gnd   vsource v1=1/(2*R) v2=1/R tr=1u tf=1u width=75*PERIOD period=100*PERIOD

;L1    l    gnd   capacitor c=1 ic=1.2804
L1    l    gnd   capacitor c=1 ic=1.623030961224813
Ll    l    gnd   z   c  vccs func=(v(z) > VDIG/2) ? RS/L*v(l) - VIN/L : 1/L*v(c) + RS/L*v(l) - VIN/L

I1    l    ref   a2d dignet="Boost.l" vt=0
I2    s    gnd   a2d dignet="Boost.s" vt=0
I3    z    gnd   d2a dignet="Boost.z" vl=0 vh=VDIG

Vr  ref    gnd   vsource vdc=VREF/KI
Vs    s    gnd   vsource vsin=1 freq=1/PERIOD

verilog_include "cntrBoost.v"
