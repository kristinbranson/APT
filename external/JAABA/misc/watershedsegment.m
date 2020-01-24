function L = watershedsegment(bw)

D = bwdist(~bw);
D = -D;
D(~bw) = -Inf;
L = watershed(D);