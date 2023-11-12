# Copyright (c) 2011, Roger Lew BSD [see LICENSE.txt]
# This software is funded in part by NIH Grant P20 RR016454.


# This is a collection of scripts used to generate C-H comparisons
# for qsturng. As you can probably guess, my R's skills are not all that good.

setwd('D:\\USERS\\roger\\programming\\python\\development\\qsturng')

ps = seq(length=100, from=.5, to=.999)

for (r in c(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
            22,23,24,25,26,27,28,29,30,35,40,50,60,70,80,90,100,200)) {
    for (v in c(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                22,24,26,30,35,40,50,60,90,120,240,480,1e38)) {
        m = qtukey(ps, r, v)
        fname = sprintf('CH_r=%i,v=%.0f.dat',r,v)
        print(fname)
        write(rbind(ps, m),
              file=fname,
              ncolumns=2,
              append=FALSE,
              sep=',')
    }
}

rs = c(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,60,80,100)

for (v in c(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
           17,18,19,20,24,30,40,60,120,1e38)) {
    m = qtukey(0.30, rs, v)
    fname = sprintf('CH_p30.dat')
    print(fname)
    write(rbind(m),
          file=fname,
          ncolumns=26,
          append=TRUE,
          sep=' ')
}

for i in
for (v in c(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
           17,18,19,20,24,30,40,60,120,1e38)) {
    m = qtukey(0.675, rs, v)
    fname = sprintf('CH_p675.dat',r,v)
    print(fname)
    write(rbind(m),
          file=fname,
          ncolumns=26,
          append=TRUE,
          sep=' ')
}

for (v in c(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
           17,18,19,20,24,30,40,60,120,1e38)) {
    m = qtukey(0.75, rs, v)
    fname = sprintf('CH_p75.dat',r,v)
    print(fname)
    write(rbind(m),
          file=fname,
          ncolumns=26,
          append=TRUE,
          sep=' ')
}

for (v in c(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
           17,18,19,20,24,30,40,60,120,1e38)) {
    m = qtukey(0.975, rs, v)
    fname = sprintf('CH_p975.dat')
    print(fname)
    write(rbind(m),
          file=fname,
          ncolumns=26,
          append=TRUE,
          sep=' ')
}

i = 0;
for (i in 0:9999) {
    p = runif(1, .5, .95);
    r = sample(2:100, 1);
    v = runif(1, 2, 1000);
    q = qtukey(p,r,v);
    if (!is.nan(q)) {
        write(c(p,r,v,q),
              file='bootleg.dat',
              ncolumns=4,
              append=TRUE,
              sep=',');
        i = i + 1;
    }
}