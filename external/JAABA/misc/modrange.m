% out = modrange(in, lower, upper)
% this function puts the n-d array in into the range [lower,upper).
% examples:
% modrange(100,100,200) returns 100
% modrange(300,100,200) returns 100
% modrange(3*pi/2,-pi,pi) returns -pi/2
% modrange(1,1,101) returns 1
% modrange(100,1,101) returns 1
% modrange(101,1,101) returns 1
% modrange(100,1,101) returns 100
% modrange(0,1,101) returns 100
function out = modrange(in,l,u)
  
  out = mod(in-l,u-l)+l;