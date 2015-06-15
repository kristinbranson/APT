function bboxes = InitializeSecondRoundTracking(p,fid,nfids,medfilwidth,windowradius)

if medfilwidth == 0,
  p_med = p;
else
  p_med=medfilt1(p,medfilwidth);
  p_med(1,:) = p(1,:);
end

xi = fid;
yi = fid + nfids;

bboxes=[p_med(:,xi)-windowradius p_med(:,yi)-windowradius 2*windowradius*ones(size(p,1),2)];
