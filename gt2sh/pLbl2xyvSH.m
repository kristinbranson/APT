function xyv = pLbl2xyvSH(pLbl)
% pLbl: [nx20]
% xyv: [nx5x2x2]. n,npt=5,x/y,ivw

[n,d] = size(pLbl);
assert(d==20);
xyv = nan(n,5,2,2); 
for ivw=1:2
  xy = pLbl(:,[1:5 11:15]+(ivw-1)*5);
  xyv(:,:,:,ivw) = reshape(xy,n,5,2);
end