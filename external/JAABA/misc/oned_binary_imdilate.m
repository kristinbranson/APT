function c = oned_binary_imdilate(a,b)

sza = size(a);
szb = size(b);
if nnz(sza>1) > 1 || nnz(szb>1) > 1,
  error('oned_binary_imdilate is only meant for one-dimensional arrays');
end
nb = length(b);
if mod(nb,2) == 0,
  b = [b,false];
end

c = myconv(a,b,'same',false) > 0;