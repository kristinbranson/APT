function [...
  xyRed,... % [Nxdx2] 
  xyRed47,... % [Nx4x2]
  xy,... % [Nx7x2xRT]
  xy47,... % [Nx4x2xRT]
  djump47,... % [Nx4]. djump47(1,:) is all nan
  djump47av,... % [N]
  xyRepMad,... % [Nx7]. Mad-of-L2norm-of-rep-from-repcentroid
  xyRepMad47Av] ... % [N]  
    = janResults(res)
% Extract Jan results
%
% res: results structure, fields .pTstT, .pTstTRed

[N,RT,D,Tp1] = size(res.pTstT);
assert(isequal(size(res.pTstTRed),[N D Tp1]));
d = D/2;

pTstT = res.pTstT(:,:,:,end);
pTstTRed = res.pTstTRed(:,:,end);
xy = reshape(permute(pTstT,[1 3 2]),N,d,2,RT);
xy47 = xy(:,4:7,:,:);
xyRed = reshape(pTstTRed,N,d,2);
xyRed47 = xyRed(:,4:7,:);

djump47 = nan(N,4);
for i = 2:N
  dxyRed = squeeze(xyRed47(i,:,:)-xyRed47(i-1,:,:)); % 4x2  
  djump47(i,:) = sqrt(sum(dxyRed.^2,2));  
end 
djump47av = mean(djump47,2);

% xy = [N 7 2 RT]
xyCent = mean(xy,4); % [Nx7x2, replicate-centroid]
xyDev = bsxfun(@minus,xy,xyCent); % [Nx7x2xRT], deviation of each iTrl/iRep from rep-centroid
xyDev = squeeze(sqrt(sum(xyDev.^2,3))); % [Nx7XRT], L2norm of each pt/Rep from rep-centroid
xyRepMad = median(xyDev,3); % [Nx7] % median-across-reps
xyRepMad47Av = mean(xyRepMad(:,4:7),2); % [N]
