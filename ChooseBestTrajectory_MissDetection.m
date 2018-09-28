function [Xbest,vbest,idx,totalcost,poslambda,misscost] = ...
  ChooseBestTrajectory_MissDetection(X,appearancecost,varargin)
% Select trajectory through CPR-generated replicate clouds 
%
% X: [DxTxK] full CPR tracking results
% appearancecost: [TxK] scalar cost for each shape
%
% Xbest: [DxT] selected replicates representing "best" traj
% vbest: [1xT] whether the object was detected
% idx: [T] replicate indices (indices into 3rd dim of X). Xbest(:,t) is
% equal to X(:,t,idx(t)). idx(t) will be K+1 if not detected
% totalcost: [1] cost of the best trajectory

[priordistfun,poslambda,poslambdafac,dampen,fix,...
  misscost,misscostprctile,misscostfac] = myparse(varargin,...
  'priordist',@(x) zeros(size(x,1),1),...  % [K] = priordist([KxD]) returns assumed/prior position cost for t=1
  'poslambda',[],... % (optional) position costs are multiplied by this scale factor when added to appearance costs. If not supplied, empirically generated via random sampling
  'poslambdafac',[],... % (optional), fudge/scale factor for empirical generation of poslambda. Used only if poslambda==[]
  'dampen',.5,... % velocity damping factor. pos(t) is predicted as pos(t-1)+dampen*(pos(t-1)-pos(t-2)). 1=>full extrapolation, 0=>velocity irrelevant
  'fix',[],...
  'misscost',[],...
  'misscostprctile',99,...
  'misscostfac',3); % no appearance cost in this frame, no motion cost for previous and next frame

[D,T,K] = size(X);
szassert(appearancecost,[T K]);

if ~isempty(fix) && numel(fix)~=T
  error('fix must be a vector of length T');
end

if ~isempty(fix) && ~all(isnan(fix))
  if ~any(isnan(fix))    
    fprintf('All positions fixed, just returning.\n');
    idx = fix; 
    Xbest = nan(D,T);
    for t = 1:T
      Xbest(:,t) = X(:,t,idx(t));
    end
    totalcost = nan;
    poslambda = nan;
    return;    
  end
  
  t0 = find(isnan(fix),1);
  t1 = find(isnan(fix),1,'last');
  if t0>3 || t1<T-2
  
    args = varargin;
    if t0 > 3
      t0 = t0 - 2;
      i = find(strcmp(args(1:2:end),'priordist'));
      if ~isempty(i)
        args{2*i} = @(x) zeros(size(x,1),1);
      end
    else
      t0 = 1;
    end
    t1 = min(t1+2,T);
    
    idx = fix;
    Xbest = nan(D,T);
    for t = find(~isnan(fix(:)'))
      Xbest(:,t) = X(:,t,idx(t));
    end
    
    fix = fix(t0:t1);
    X = X(:,t0:t1,:);
    appearancecost = appearancecost(t0:t1,:);
    
    i = find(strcmp(args(1:2:end),'fix'));
    if ~isempty(i)
      args(2*i-1:2*i) = [];
    end
    
    [Xbest(:,t0:t1),idx(t0:t1),totalcost,poslambda] = ...
            ChooseBestTrajectory_MissDetection(X,appearancecost,args{:},'fix',fix);
          
    return;
  end
end

X = permute(X,[3,1,2]);
szassert(X,[K D T]);

% there are more efficient ways to do this...
if ~isempty(fix)
  for t = find(~isnan(fix(:)'))
    appearancecost(t,[1:fix(t)-1,fix(t)+1:K]) = inf;
  end
end

% costprev(w,v) is the min cost that ends at t = w and t-1 = v

if isempty(poslambda) || isempty(misscost),

  % Estimate poslambda as ratio of (typical variability in appearance cost)
  % to (typical variability in position cost). The total cost at each
  % timepoint t is poslambda*poscost+appearancecost, and this value is 
  % minimized to find the best trajectory. Note the absolute scales of
  % poscost and appearancecost are irrelevant, the idea here is that
  % poslambda is set so that fluctuations in positioncost and 
  % appearancecost carry comparable weight in the minimization.
  
  Ksample = min(K,5);
  count = (T-2) * Ksample^3;
  errs = nan(1,count);
  off = 0;
  minposcost = nan(1,T);
  for t = 3:T
    ws = randsample(K,Ksample);
    v = randsample(K,Ksample);
    u = randsample(K,Ksample);
    
    vel = bsxfun(@minus,reshape(X(v,:,t-1),[Ksample,1,D]),reshape(X(u,:,t-2),[1,Ksample,D]));
    predpos = bsxfun(@plus, reshape(X(v,:,t-1),[Ksample,1,D]), dampen*vel);
    
    for w = ws'
      poscost = sum(bsxfun(@minus, reshape(X(w,:,t),[1,1,D]), predpos).^2, 3);
      errs(off+1:off+numel(poscost)) = poscost;
      off = off + numel(poscost);
      minposcost(t) = min(minposcost(t),min(poscost(:)));
    end
  end
  
  if isempty(poslambda),
    mederr = nanmedian(errs(:));
    mad_pos = nanmedian( abs( errs(:) - mederr) );
    %mad_app = median( abs( appearancecost(~isinf(appearancecost)) - median(appearancecost(~isinf(appearancecost)))) );
    a = appearancecost;
    a(isinf(a)) = nan;
    mad_app = nanmedian( abs( a(:) - nanmedian(a(:)) ) );
    poslambda = mad_app/mad_pos;
    
    if isempty(poslambdafac)
      fprintf('Chose poslambda = %f\n',poslambda);
    else
      poslambda = poslambda*poslambdafac;
      fprintf('Chose poslambda = %f, after scaling by poslambdafac = %f\n',...
        poslambda,poslambdafac);
    end
  end
  %assert(~isnan(poslambda) && ~isinf(poslambda));
end

if isempty(misscost),
  minappearancecost = min(appearancecost,[],2);
  idxgood = ~isnan(minposcost(:)) & ~isnan(minappearancecost(:));
  misscost = prctile(minappearancecost(idxgood)+poslambda*minposcost(idxgood)',misscostprctile)*misscostfac;
end

% add an option of missing detection with high cost
appearancecost(:,end+1) = misscost;

% initialization

% first frame: position cost is from prior
poscost0 = priordistfun(X(:,:,1));
poscost0(end+1) = 0; % add in possibility of skipping detection

assert(isvector(poscost0) && numel(poscost0)==(K+1));

% second frame: position cost assumes zero velocity
% poscost1(w,v) corresponds to w at t=2, v at t=1
ismissing = reshape(any(isnan(X),2),[K,T]);
poscost1 = poslambda * pdist2(X(:,:,2),X(:,:,1),'sqeuclidean');
poscost1(ismissing(:,2),:) = inf;
poscost1(:,ismissing(:,1)) = inf;
poscost1(:,end+1) = 0; % if was missed in previous frame, no position-based cost
poscost1(end+1,:) = 0; % if is missed in this frame, no position-based cost
% is position at previous time idx(1:t-1) <= K. So if we choose w = K+1,
% then we are assuming X is actually X(v,:,1) 

% missX: last position on minimum cost trajectory ending with t-1=v=K+1 and
% t-2=u=K+1
% for t=3, special case, since no previous history, no penalty
missX = nan(1,D);

% [K+1xK+1]. costprev(w,v) is the minimum total cost that ends at w at t-1 and
% v at t-2. (Here we will be starting at t=3.)
% This cost is computed as (assumed/prior cost for t=1)+(appearancecost for
% t=1)+(position cost for transitioning from t=1 to t=2)+(appearance cost
% for t=2)
costprev = bsxfun(@plus, poscost0(:)'+appearancecost(1,:), ...
                         bsxfun(@plus, appearancecost(2,:)', poscost1));

% for tracking back and finding optimal states
% prev(w,v,t) gives replicate index (index into 1..K+1) giving best/chosen 
% replicate u giving minimum/best u->v->w progression over (t-2)->(t-1)->t
prev = nan(K+1,K+1,T);

predpos = nan(K+1,K+1,D);

for t = 3:T
%   if mod(t,100)==0
%     fprintf('Frame %d / %d\n',t,T);
%   end
  
  % vel is K x K x D
  % vel(v,u,:) is the velocity assuming t-2 = u and t-1 = v
  % predpos is K+1 x K+1 x D
  % predpos(v,u,:) is the position assuming t-2 = u and t-1 = v
  vel = bsxfun(@minus,reshape(X(:,:,t-1),[K,1,D]),reshape(X(:,:,t-2),[1,K,D]));
  predpos(:) = nan; 
  predpos(1:K,1:K,:) = bsxfun(@plus, reshape(X(:,:,t-1),[K,1,D]), dampen*vel);
  % if t-1=v=K+1, t-2=u<=K, then predpos is from t-2
  predpos(K+1,1:K,:) = reshape(X(:,:,t-2),[1,K,D]);
  % if t-2=u=K+1, t-1=v<=K, then predpos is from t-1
  predpos(1:K,K+1,:) = reshape(X(:,:,t-1),[K,1,D]);
  % if t-1=v=K+1 and t-2=u=K+1, then predpos is the missX corresponding to
  % the best trajectory ending as such
  predpos(K+1,K+1,:) = reshape(missX,[1,1,D]);
  
  costcurr = nan(K+1,K+1);
  
  % if visible at time t
  for w = 1:K+1,
    % poscost is K+1 x K+1, cost of transitioning from (t-1=v,t-2=u) to w
    if w <= K,
      poscost = poslambda * ...
        sum(bsxfun(@minus, reshape(X(w,:,t),[1,1,D]), predpos).^2, 3);
      poscost(isnan(poscost)) = inf;
      % if missX has not been assigned yet, no penalty
      if any(isnan(missX)),
        poscost(K+1,K+1) = 0;
      end
    else
      poscost(:) = 0;
    end
    % prediction is previous position if
    [costcurr(w,:),prev(w,:,t)] = min( appearancecost(t,w) + poscost + costprev, [], 2 );
  end
  % missX should correspond to best trajectory ending with t-1=v=K+1 --
  % note this is greedy!
  if prev(K+1,K+1,t) == K+1,
    % then keep missX the same, not visible
  else
    missX = X(prev(K+1,K+1,t),:,t-2);
  end
  
  costprev = costcurr;
end

% find the best last state
[totalcost,i] = min(costprev(:));
idx = nan(1,T);
[idx(T),idx(T-1)] = ind2sub([K+1,K+1],i);

for t = T-2:-1:1
  idx(t) = prev(idx(t+2),idx(t+1),t+2);
end

Xbest = nan(D,T);
vbest = true(1,T);
for t = 1:T
  vbest(t) = idx(t) <= K;
  if idx(t) <= K,
    Xbest(:,t) = X(idx(t),:,t);
  end
end

% interpolate for missed detections
[t0s,t1s] = get_interval_ends(vbest==0);
t1s = t1s-1;
for i = 1:numel(t0s),
  
  % intervals that are on the ends
  if t0s(i) == 1,
    if t1s(i) < T,
      for t = 1:t1s(i)-1,
        Xbest(:,t) = Xbest(:,t1s(i)+1);
      end
    else
      warning('All frames assigned to be missed detections');
    end
  elseif t1s(i) == T,
    for t = t0s(i):T,
      Xbest(:,t) = Xbest(:,t0s(i)-1);
    end    
  else
    for d = 1:D,
      xinterp = linspace(Xbest(d,t0s(i)-1),Xbest(d,t1s(i)+1),t1s(i)-t0s(i)+3);
      Xbest(d,t0s(i):t1s(i)) = xinterp(2:end-1);
    end
  end
  
end
%fprintf('Done\n');
