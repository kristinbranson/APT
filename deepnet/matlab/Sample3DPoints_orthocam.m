function [P,w,idxfront,idxside] = Sample3DPoints_orthocam(xfront,xside,Sfront,Sside,calObj,wfront,wside,K)

nfront = size(xfront,2);
nside = size(xside,2);
w = nan(nfront,nside);
pfront = nan(nfront,nside);
pside = nan(nfront,nside);
P = nan(3,nfront,nside);
for ifront = 1:nfront
  for iside = 1:nside
    [P(:,ifront,iside),~,xside_re,xfront_re] = calObj.stereoTriangulate(...
      xside(:,iside),xfront(:,ifront));
    pfront(ifront,iside) = mvnpdf(xfront_re',xfront(:,ifront)',Sfront(:,:,ifront));
    pside(ifront,iside) = mvnpdf(xside_re',xside(:,iside)',Sside(:,:,iside));
    w(ifront,iside) = pfront(ifront,iside)*pside(ifront,iside)*wfront(ifront)*wside(iside);
  end
end

w = w / sum(w(:));
[~,order] = sort(w(:),1,'descend');

P = P(:,order(1:K));
w = w(order(1:K));
[idxfront,idxside] = ind2sub([nfront,nside],order(1:K));