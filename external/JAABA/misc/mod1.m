%     MOD1(x,y) is like MOD(x,y), except that it returns a value between 1
%     and y, instead of between 0 and y-1.

function v = mod1(x,y)

v = mod(x-1,y)+1;