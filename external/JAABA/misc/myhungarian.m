function [a,c] = myhungarian(A)
%MYHUNGARIAN

ndig = 3;

A = round(A*10^ndig);

[a,c] = myhung(A);
c = c *10^(-ndig);
