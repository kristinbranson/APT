function [hmnrnc,hmBestTrajPQ,acBestTrajUV,acCtrPQ,totalcost] = ...
                            ChooseBestTrajectory_grid(hmfcn,n,varargin)
% Grid Viterbi
%
% hmfcn: fcn handle with sig: hm = hmfcn(i) for i in 1..n
% n: run from frames 1..n
%
% hmnrnc: [hmnr hmnc] heatmap size
% hmBestTraj: [nx2] p,q indices into heatmaps for best traj
% acBestTraj: [nx2] u,v indices into acwins for best traj
% totalcost: [1] total cost of best traj

% TODO: right now ACs have infs where heatmaps are zero, totally
% prohibiting locating tracks there.
 
[hm11xyOrig,acrad,dx,maxvx,poslambda,dampen] = myparse(varargin,...
  'hm11xyOrig',[1 1],... % Either [1x2] or [nx2]. The upper-left heatmap 
      ...                % pixel hm(1,1) maps to this (x,y) coordinate in 
      ...                % the original/raw movie. CURRENTLY UNUSED
  'hmConsiderRadius',12,... % consider window of heatmap of this size (within this many px of heatmap peak)
  'dx',1,... % spatial stepsize when discretizing consider-window 
  'maxvx',[],... % maximum allowed movement in any single direction
  'poslambda',1/50,... % motion model costs are mulitplied by this fac when adding to app costs
  'dampen',0.25... % velocity damping factor
  );

if isempty(maxvx)
  maxvx = 6*acrad;
else
  assert(maxvx==round(maxvx));
end

if size(hm11xyOrig,1)==1
  hm11xyOrig = repmat(hm11xyOrig,n,1);
end
szassert(hm11xyOrig,[n 2]);

% [priordistfun,poslambda,poslambdafac,dampen,fix] = myparse(varargin,...
%   'priordist',@(x) zeros(size(x,1),1),...  % [K] = priordist([KxD]) returns assumed/prior position cost for t=1
%   'poslambda',[],... % (optional) position costs are multiplied by this scale factor when added to appearance costs. If not supplied, empirically generated via random sampling
%   'poslambdafac',[],... % (optional), fudge/scale factor for empirical generation of poslambda. Used only if poslambda==[]
%   'dampen',.5,... % velocity damping factor. pos(t) is predicted as pos(t-1)+dampen*(pos(t-1)-pos(t-2)). 1=>full extrapolation, 0=>velocity irrelevant
%   'fix',[]);

gv = GridViterbi(maxvx,dx,dampen);

% For each HM we consider a square window of size acsz. We call these "AC
% windows". The chosen state is assumed to live in this window. We attempt 
% to judiciously choose the AC window for each frame.
%
% See GridViterbi.m for coord system notes. (p,q)=(1,1) at frame n maps to 
% (x,y)=hm11xyOrig(n,:) in original movie coords.

acsz = 2*acrad+1;
acCtrPQ = nan(n,2); % (row,col) of center of ac window in its heatmap

% first frame
hm1 = hmfcn(1); 
hmnrnc = size(hm1);
[ac1,pctr1,qctr1] = GridViterbi.hm2ac(hm1,acrad,hm11xyOrig(1,:));
priorc1 = zeros(size(ac1));
acCtrPQ(1,:) = [pctr1 qctr1];

% second frame
hm2 = hmfcn(2);
[ac2,pctr2,qctr2] = GridViterbi.hm2ac(hm2,acrad,hm11xyOrig(2,:));
acCtrPQ(2,:) = [pctr2 qctr2];

% mc2(u2,v2,u1,v1) is the motion cost for transitioning from 
% (u0,v0,t=0)->(u1,v1,t=1)->(u2,v2,t=2) where we assume acwins@t0 @t1 are
% aligned, and u0==u1 and v0==v1
% Right now we consider motion cost in the heatmap (body-centered) frame
hmt1xgv = acCtrPQ(1,2)-acrad:acCtrPQ(1,2)+acrad;
hmt1ygv = acCtrPQ(1,1)-acrad:acCtrPQ(1,1)+acrad;
[hmt1x,hmt1y] = meshgrid(hmt1xgv,hmt1ygv);

hmt2xgv = acCtrPQ(2,2)-acrad:acCtrPQ(2,2)+acrad;
hmt2ygv = acCtrPQ(2,1)-acrad:acCtrPQ(2,1)+acrad;
[hmt2x,hmt2y] = meshgrid(hmt2xgv,hmt2ygv);

hmt1x = reshape(hmt1x,[1 1 acsz acsz]);
hmt1y = reshape(hmt1y,[1 1 acsz acsz]);
l2t1t2 = (hmt1x-hmt2x).^2 + (hmt1y-hmt2y).^2; % assume vel=0 from t0 to t1, so hmt1x is hmt2x_pred, etc
mc2 = poslambda * l2t1t2;
szassert(mc2,[acsz acsz acsz acsz]);
  % mct2(u2,v2,u1,v1) is l2
  % distance/motion cost for transitioning from
  %   (u0,v0,0)->(u1,v1,1)->(u2,v2,2) assuming u0=u1 and v0=v1

szassert(ac1,[acsz acsz]);
szassert(ac2,[acsz acsz]);
costprev = reshape(ac1,[1 1 acsz acsz]) + reshape(priorc1,[1 1 acsz acsz]) ...
         + ac2 + mc2;
szassert(costprev,[acsz acsz acsz acsz]);
% costprev(u2,v2,u1,v1) is the current/running minimum/best total cost that 
% ends at (u1,v1,t-2) and (u2,v2,t-1).
% Here we have initialized costprev for t=3.

acnumel = acsz*acsz;
clsPrevIdx = 'uint16';
assert(acnumel<intmax(clsPrevIdx),'AC window has more than %d elements.',...
  intmax(clsPrevIdx));

prev = zeros(acsz,acsz,acsz,acsz,n,clsPrevIdx);
% prev(ut,vt,utm1,vtm1,t) contains a linear index into [1..acnumel] 
% representing the optimal/best (utm2,vtm2) that leads to 
% (utm1,vtm1,t-1)->(ut,vt,t). prev is only defined for t>=3.
stmp = whos('prev');
fprintf(1,'Optimal path array is a %s with %.3g GB.\n',clsPrevIdx,stmp.bytes/1e9);

for t=3:n
  if mod(t,1)==0
    fprintf('Frame %d / %d\n',t,n);
  end

  hmt = hmfcn(t);
  [act,pctrt,qctrt] = GridViterbi.hm2ac(hmt,acrad,hm11xyOrig(t,:));
  acCtrPQ(t,:) = [pctrt qctrt];

  % costcurr(ut,vt,u2,v2) is best cost ending at (u2,v2,t-1)->(ut,vt,t)
  costcurr = nan(acsz,acsz,acsz,acsz);
  for ut=1:acsz
  for vt=1:acsz
    mctL2 = poslambda * gv.getMotionCostL2(ut,vt,acCtrPQ(t-2:t,:),acrad);
    % mct is [acsz x acsz x acsz x acsz]
    % mct(u2,v2,u1,v1) is motion cost for transitioning from
    % (u1,v1,t-2)->(u2,v2,t-1)->(ut,vt,t);
    
    totcost = costprev + mctL2 + act(ut,vt); % [acsz x acsz x acsz x acsz]
    % totcost(u2,v2,u1,v1) gives total cost of transitioning from 
    % (u1,v1,t-2)->(u2,v2,t-1)->(ut,v2,t)
    totcost = reshape(totcost,[acnumel acnumel]); 
    % totcost(h,g) gives total cost of transition from
    % g~(u1,v1,t-2)->h~(u2,v2,t-1)->(ut,vt,t)
    [mintotcost,pidx] = min(totcost,[],2); % for each h, best cost and best g leading to said cost
    szassert(mintotcost,[acnumel 1]); 
    costcurr(ut,vt,:,:) = reshape(mintotcost,[acsz acsz]);
    prev(ut,vt,:,:,t) = reshape(pidx,[acsz acsz]);
  end
  end

  costprev = costcurr;
  
  
%   % vel and predpos are K x K x D
%   % vel(v,u,:) is the velocity assuming t-2 = u and t-1 = v
%   % predpos(v,u,:) is the position assuming t-2 = u and t-1 = v
%   vel = bsxfun(@minus,reshape(X(:,:,t-1),[K,1,D]),reshape(X(:,:,t-2),[1,K,D]));
%   predpos = bsxfun(@plus, reshape(X(:,:,t-1),[K,1,D]), dampen*vel);
%   
%   costcurr = nan(K,K);
%   for w = 1:K
%     % poscost is K x K, cost of transitioning from (v,u) to w
%     poscost = poslambda * ...
%               sum(bsxfun(@minus, reshape(X(w,:,t),[1,1,D]), predpos).^2, 3);
%     [costcurr(w,:),prev(w,:,t)] = min( appearancecost(t,w) + poscost + costprev, [], 2 );
%   end
% 
%   costprev = costcurr;
end

acBestTrajUV = nan(n,2);
[totalcost,i] = min(costprev(:));
[acBestTrajUV(n,1),acBestTrajUV(n,2),acBestTrajUV(n-1,1),acBestTrajUV(n-1,2)] ...
  = ind2sub([acsz acsz acsz acsz],i);
for t=n-2:-1:1
  idxBestT = prev(acBestTrajUV(t+2,1),acBestTrajUV(t+2,2),...
                  acBestTrajUV(t+1,1),acBestTrajUV(t+1,2),t+2);
  [utmp,vtmp] = ind2sub([acsz acsz],idxBestT);          
  acBestTrajUV(t,:) = [utmp vtmp];
end

hmBestTrajPQ = acCtrPQ-acrad+acBestTrajUV-1;
