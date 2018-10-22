function [hmnrnc,hmidxBestTraj,totalcost] = ...
                            ChooseBestTrajectory_grid(hmfcn,n,varargin)
% Grid Viterbi
%
% hmfcn: fcn handle with sig: hm = hmfcn(i) for i in 1..n
% n: run from frames 1..n
%
% hmnrnc: [hmnr hmnc] heatmap size
% hmidxBestTraj: [nx1] linear indices into heatmaps for best trajectory
% totalcost: [1] total cost of best traj

[hm11xyOrig,acrad,dx,maxvx,poslambda,dampen] = myparse(varargin,...
  'hm11xyOrig',[1 1],... % Either [1x2] or [nx2]. The upper-left heatmap 
      ...                % pixel hm(1,1) maps to this (x,y) coordinate in 
      ...                % the original/raw movie. CURRENTLY UNUSED
  'hmConsiderRadius',30,... % consider window of heatmap of this size (within this many px of heatmap peak)
  'dx',1,... % spatial stepsize when discretizing consider-window 
  'maxvx',30,... % maximum allowed movement in any single direction
  'poslambda',1/50,... % motion model costs are mulitplied by this fac when adding to app costs
  'dampen',0.25... % velocity damping factor
  );

assert(maxvx==round(maxvx));

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

% Clean me up, all this just for l2mmcZeroVel and only used for initialization
% l2Big has size [maxvxsz x maxvxsz] where maxvxsz = 2*maxvx+1.
[l2mmcBig,l2mmcBigx1,l2mmcBigy1,l2mmcBigx2,l2mmcBigy2] = ...
  GridViterbi.precompMMC(maxvx,dx,dampen);
maxvxsz = 2*maxvx+1;
max2vxsz = 4*maxvx+1;
assert(isequal(l2mmcBigx1(1,end),l2mmcBigy1(end,1),maxvxsz));
assert(isequal(l2mmcBigx2(1,end),l2mmcBigy2(end,1),max2vxsz));
% l2Big(l2Step1Mid,l2Step1Mid,:,:) gives the l2 dist for starting at (0,0), 
% staying there at t=2, then moving to (u,v) at t=3
l2Step1Mid = maxvx+1;
l2Step2Mid = 2*maxvx+1;
assert(dx==1,'Currently require dx==1.');
l2mmcZeroVel = squeeze(l2mmcBig(l2Step1Mid,l2Step1Mid,:,:));
szassert(l2mmcZeroVel,[max2vxsz max2vxsz]); 
% l2mmcZeroVel is motion cost of starting at (0,0), staying there, then
% moving to (u,v) on t=3. The origin is at (2*maxvx+1,2*maxvx+1) 

% For each HM we consider a square window of size acsz. We call these "AC
% windows". The chosen state is assumed to live in this window. We attempt 
% to judiciously choose the AC window for each frame.
%
% See GridViterbi.m for coord system notes. (p,q)=(1,1) at frame n maps to 
% (x,y)=hm11xyOrig(n,:) in original movie coords.

acsz = 2*acrad+1;
acCtrPQ = nan(n,2); % (row,col) of center of ac window relative to its heatmap

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
% mc2 assumes zero velocity
% mc2(u2,v2,u1,v1) is motion cost for transitioning from (u1,v1,t=1)->(u2,v2,t=2)
% Right now we consider motion cost in the heatmap (body-centered) frame.
mc2 = nan(acsz,acsz,acsz,acsz);
% Cleanup this init
dp12 = pctr2-pctr1; % ac2 is offset by this many rows relative to ac1 (when considered rel to their heatmap frames)
dq12 = qctr2-qctr1; % etc
for p1=1:acsz
for q1=1:acsz
  % xxx do we want L2 or sqrt(L2)?
  l2Step2R1 = l2Step2Mid-p1+1+dp12;
  l2Step2C1 = l2Step2Mid-q1+1+dq12;  
  l2Step2RIdx = l2Step2R1:l2Step2R1+acsz-1;
  l2Step2CIdx = l2Step2C1:l2Step2C1+acsz-1;
  l2_2rel1 = l2mmcZeroVel(l2Step2RIdx,l2Step2CIdx); % l2 dist from each AC gridpt at t=2, rel to t=1
  mc2(:,:,p1,q1) = poslambda*l2_2rel1;
end
end

szassert(ac1,[acsz acsz]);
szassert(ac2,[acsz acsz]);
costprev = reshape(ac1,[1 1 acsz acsz]) + reshape(priorc1,[1 1 acsz acsz]) ...
         + ac2 + mc2;
szassert(costprev,acsz,acsz,acsz,acsz);

% costprev(u2,v2,u1,v1) is the current/running minimum/best total cost that 
% ends at (u1,v1,t-2) and (u2,v2,t-1).
% Here we have initialized costprev for t=3.

acnumel = acsz*acsz;
clsPrevIdx = 'uint16';
assert(acnumel<intmax(clsPrevIdx),'AC window has more than %d elements.',...
  intmax(clsPrevIdx));

prev = zeros(acsz,acsz,acsz,acsz,n,clsPrevIdx);
stmp = whos('prev');
fprintf(1,'Optimal path array is a %s with %.3g GB.\n',clsPrevIdx,stmp.bytes/1e9);

for t = 3:n
  if mod(t,100)==0
    fprintf('Frame %d / %d\n',t,n);
  end

  hmt = hmfcn(t);
  [act,pctrt,qctrt] = GridViterbi.hm2ac(hmt,acrad,hm11xyOrig(t,:));
  acCtrPQ(t,:) = [pctrt qctrt];

  % costcurr(ut,vt,u2,v2) is best cost ending at (u2,v2,t-1)->(ut,vt,t)
  costcurr = nan(acsz,acsz,acsz,acsz);
  for ut=1:acsz
  for vt=1:acsz
    mctL2 = gv.getMotionCost(ut,vt,acCtrPQ(t-2:t,:),acrad);
    % mct is [acszxacszxacszxacsz]; poslambda*l2 for transitioning from
    % mct(u2,v2,u1,v1) is motion cost for transitioning from
    % (u1,v1,t-2)->(u2,v2,t-1)->(ut,vt,t);
    
    totcost = costprev + mctL2 + act(ut,vt); % [acsz x acsz x acsz x acsz]
    totcost = reshape(totcost,[acnumel acnumel]); 
    % totcost(q,p) gives total cost of transition from p~(u1,v1,t-2)->q~(u2,v2,t-1)
    [mintotcost,pidx] = min(totcost,[],2);
    szassert(mintotcost,[acnumel 1]);
    costcurr(ut,vt,:,:) = mintotcost;
    prev(ut,vt,:,:,t) = pidx;
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



% find the best last state
[totalcost,i] = min(costprev(:));
idx = nan(1,T);
[idx(T),idx(T-1)] = ind2sub([K,K],i);

for t = T-2:-1:1
  idx(t) = prev(idx(t+2),idx(t+1),t+2);
end

Xbest = nan(D,T);
for t = 1:T
  Xbest(:,t) = X(idx(t),:,t);
end