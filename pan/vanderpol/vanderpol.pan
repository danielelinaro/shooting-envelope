ground electrical 0

parameter ALPHA=1m
; Analyses

En envelope autonomous=yes period=2*pi uic=yes tstop=50k

Sh shooting period=2*pi autonomous=yes uic=yes method=2 order=6 \
            floquet=yes printmo=yes tmax=100u \
	    eigf=1 eigfnorm=1 samples=200 restart=0

; Circuit

p1    x     0  x    vccs func=-ALPHA*v(x) + ALPHA*v(x)^3
;p1    x     0  x    0  poly n0=0 n1=-ALPHA n2=0 n3=ALPHA d0=1
c1    x     0  capacitor c=1 icon=2m
l1    x     0  inductor  l=1
