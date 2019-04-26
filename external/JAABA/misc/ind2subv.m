function [v] = ind2subv(siz,ndx)
%IND2SUB Multiple subscripts from linear index.
%   IND2SUBV is used to determine the equivalent subscript values
%   corresponding to a given single index into an array.
%
%   [V] = IND2SUBV(SIZ,IND) returns the array V containing the
%   equivalent subscripts corresponding to the index
%   matrix IND for a matrix of size SIZ.  

siz = double(siz);
nout = length(siz);
v = nan([numel(ndx),nout]);


if nout > 2
    k = cumprod(siz);
    for i = nout:-1:3,
        vi = rem(ndx-1, k(i-1)) + 1;
        vj = (ndx - vi)/k(i-1) + 1;
        v(:,i) = double(vj);
        ndx = vi;
    end
end

if nout >= 2
    vi = rem(ndx-1, siz(1)) + 1;
    v(:,2) = double((ndx - vi)/siz(1) + 1);
    v(:,1) = double(vi);
else 
    v(:,1) = double(ndx);
end

