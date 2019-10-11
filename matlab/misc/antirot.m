function x1 = antirot(x0,R,c)
assert(size(x0,1)==2);
szassert(R,[2 2]);
szassert(c,[2 1]);

x1 = R.'*(x0-c)+c;