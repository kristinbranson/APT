function [Y,e,eData,eSmth,eCnst] = poseNMS_K( X, S, R, K, varargin )
%% "Merging Pose Estimates Across Space and Time". X.P. Burgos-Artizzu,
% D.Hall, P.Perona, P.Dollar. BMVC 2013, Bristol, UK.
%
% Compute sequence of trajectories explaining observed detections, for a 
% given number of objects K known. (For unknown number of objects see
% poseNMS.m)
%
% USAGE
%  Y = poseNMS_K( X, S, R, K, varargin )
%
% INPUTS
%  X          - [1xT] cell of [ntxp] centers
%                    nt=number of detections in frame t
%                    p=detection/pose parametrization
%  S          - [1xT] cell of [ntx1] scores
%  R          - [1xT] cell of [] or [ntxm] candidate matrices
%                   (Appendix B in paper)
%                    m=number of distinct objects
%                    R says to which object detections nt belong (0/1)
%  K          - number of objects being tracked
%  varargin   - additional params (struct or name/value pairs)
%   .Y          - [] initial solution (must be mxpxT)
%   .norm       - [1] distance normalization
%   .isAng      - [0] length p vector indicating dims that are angles
%   .lambda     - [.1] time weighting (constant position)
%   .lambda2    - [0] weighting of smoothness term
%   .nPhase     - [4] number of optimization phases
%   .window     - [500] window size
%   .symmetric  - [1] (simmetric=1, backward only=0) window
%   .show       - [0] figure to display results in
%   .bnds       - [] parse video into k segments: bnds(i)+1:bnds(i+1)
%   .ms         - [] parse video into k segments with K=ms(i) per segm.
%
% OUTPUTS
%  Y          - [mxpxT] tracking solution
%
% EXAMPLE
%
% See also poseNMS
%
% Copyright 2013 X.P. Burgos-Artizzu, D.Hall, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]

% get/check inputs
dfs= {'Y',[], 'norm',1,'th',10, 'isAng','REQ', 'lambda',.1, 'lambda2',0, ...
  'nPhase',4, 'window',500, 'symmetric',1, 'show',0, 'bnds',[], 'ms',[] };
[Y,norm,~,isAng,lambda2,lambda,nPhase,window,symm,show,bnds,ms] ...
  = getPrmDflt(varargin,dfs,1);
T=length(X); assert(length(S)==T); assert(length(R)==T); assert(norm>0);
for i=1:T, assert(isempty(R{i}) || size(X{i},1)==size(R{i},1)); end
for i=1:T, assert(size(X{i},1)==size(S{i},1)); end

% parse video into k segments and call poseNMS_K() on each
if( ~isempty(bnds) && ~isempty(ms) ), k=length(ms);
  assert(bnds(1)==0 && bnds(end)==T && all(bnds)<=T && length(bnds)==k+1);
  prm1=getPrmDflt(varargin,dfs,1); prm1.show=0; prm1.ms=[]; prm1.bnds=[];
  if(isempty(Y)), Y=cell(1,1,T); end; assert(iscell(Y) && length(Y)==T);
  [e,eData,eSmth,eCnst]=deal(0); Y1=cell(1,k); X1=Y1; S1=Y1; R1=Y1;
  for i=1:k, f0=bnds(i)+1; f1=bnds(i+1); Y1{i}=Y(f0:f1);
    X1{i}=X(f0:f1); S1{i}=S(f0:f1); R1{i}=R(f0:f1); end; clear X Y S R;
  parfor i=1:k, T1=length(Y1{i});
    if(ms(i)==0), Y1{i}=cell(1,1,T1); else
      prm1t=prm1; prm1t.Y=cell2array(Y1{i});
      Y1t = poseNMS_K(X1{i},S1{i},R1{i},ms(i),prm1t);
      Y1{i} = mat2cell2(Y1t,[1 1 T1]);
    end
  end
  Y=cat(3,Y1{:}); return;
end

% initialization: create prm struct
eData=cell2array(S); eData=sum(eData(:)); z=norm*norm;
eSmth=(T-2)*K; eCnst=(T-1)*K; p=length(isAng);
for i=1:T, if(isempty(X{i})), X{i}=zeros(1,p); S{i}=0; end; end
for i=1:T, if(~isempty(R{i})), R{i}=R{i}(:,1:K); end; end
if(isempty(isAng)), isAng=false(1,p); end; %assert(length(isAng)==p);
isAng=logical(isAng); pAng=sum(isAng); p=p+pAng;
prm = struct('eData',eData, 'eSmth',eSmth, 'eCnst',eCnst', 'pAng',pAng, ...
  'lambda',lambda, 'lambda2',lambda2, 'K',K, 'p',p, 'debug',0, 'z',z);

% convert X/Y from standard to extended format
if(pAng>0), [d,ord]=sort(isAng);
  for i=1:T, X{i}=convToExt(X{i}(:,ord),pAng); end
  if(~isempty(Y)), Y=convToExt(Y(:,ord,:),pAng); end
end

% solve for best Y
Y = solve( X, S, R, Y, nPhase, window, symm, prm );

% compute energy
[e,eData,eSmth,eCnst] = energy( X, S, R, Y, prm );

% convert X/Y from extended to standard format
if(pAng>0), [d,ord]=sort(ord);
  for i=1:T, X{i}=convFrExt(X{i},pAng); X{i}=X{i}(:,ord); end
  Y=convFrExt(Y,pAng); Y=Y(:,ord,:);
end

% visualize final results
if( show )
  fprintf('e=%.5f eData=%.5f eSmth=%.5f eCnst=%.5f\n',e,eData,eSmth,eCnst);
  display2D( X, Y, show );
end

end

function display2D( X, Y, show )
% display 2D results Y in 3D (x,y,time)
[K,p,T]=size(Y); if(p~=2), return; end; Y1=permute(Y,[1 3 2]);
figure(show); clf; hold on; zlabel('time'); view(-25,25); cols='grmyck';
for j=1:K, plot3(Y1(j,:,1),Y1(j,:,2),1:T,'-o','Color',cols(j)); end
for t=1:T, X{t}(:,3)=t; end; X=cat(1,X{:}); n=size(X,1); n2=1e5;
if(n>n2), X=X(randSample(n,n2),:); end; plot3(X(:,1),X(:,2),X(:,3),'.b');
end

function Y = solve( X, S, R, Y, nPhase, window, symm, prm )
if(isempty(Y)), Y=init(X,S,R,1,prm); end; [K,p,T]=size(Y);
if(symm && mod(window,2)==1), window=window+1; end; window=min(T,window);
tid=ticStatus('solving for Y',[],5);
for p=1:nPhase, flip=mod(p,2)==0;
  if(flip), X=fliplr(X); S=fliplr(S); R=fliplr(R); Y=flipdim(Y,3); end
  if(symm), seek=1:window/2:(T-window/2-1);
  else  seek=1:window:(T-window-1);
  end
  for t0=seek, t1=min(t0+window,T);
    % init new solution Y1 for region t0:t1
    Y1 = init(X(t0:t1),S(t0:t1),R(t0:t1),1,prm);
    Y1 = refine(X(t0:t1),S(t0:t1),R(t0:t1),Y1,1,0,prm);
    % merge Y1 into the rest of Y
    if(t0>1), Y1=merge(X(1:t1),S(1:t1),R(1:t1),Y(:,:,1:t0-1),Y1,prm); end
    if(t1<T), Y1=merge(X,S,R,Y1,Y(:,:,t1+1:end),prm); end
    % keep if improves energy locally (frames t0-1:t1 may have changed)
    te0=max(t0-3,1); te1=min(t1+2,T);
    e0=energy(X(te0:te1),S(te0:te1),R(te0:te1),Y(:,:,te0:te1),prm);
    e1=energy(X(te0:te1),S(te0:te1),R(te0:te1),Y1(:,:,te0:te1),prm);
    if(prm.debug), ea=energy(X,S,R,Y,prm); eb=energy(X,S,R,Y1,prm);
      assert( abs((ea-eb)-(e0-e1))<1e-10 ); end
    if( e1<e0 ), Y=Y1; end; tocStatus(tid,(p-1+t1/T)/nPhase);
  end
  % finally refine entire solution
  Y = refine(X,S,R,Y,1,0,prm);
  if(flip), X=fliplr(X); S=fliplr(S); R=fliplr(R); Y=flipdim(Y,3); end
end
end

function Y = merge( X, S, R, Y0, Y1, prm )
% merge Y=cat(3,Y0,Y1) after permuting (if possible)
T0=size(Y0,3); T1=size(Y1,3); assert(length(R)==T0+T1); K=prm.K;
p0=1; for t=1:T0, p0=p0 && isempty(R{t}); end
p1=1; for t=T0+1:T0+T1, p1=p1 && isempty(R{t}); end
if((p0 || p1) && K<=6), ord=perms(1:K); else ord=1:K; p0=0; p1=0; end
for i=1:size(ord,1)
  if( ~p0 && ~p1 ), Y=cat(3,Y0,Y1);
  elseif( p0 ), Y=cat(3,Y0(ord(i,:),:,:),Y1);
  elseif( p1 ), Y=cat(3,Y0,Y1(ord(i,:),:,:));
  end; Y(:,:,T0)=refine_t(X{T0},S{T0},R{T0},Y,T0,0,prm);
  e=energy_t(X{T0},S{T0},R{T0},Y,T0,prm);
  if(i==1 || e<eBst), Ybst=Y; eBst=e; end
end; Y=Ybst;
end

function Y = init( X, S, R, nInit, prm )
T=length(X); Y=zeros(prm.K,prm.p,T); tid=ticStatus('initializing Y');
for i=1:nInit
  % build solution starting from random location tr
  p=squeeze(sum(cell2array(S),1)); p=max(p/sum(p),1e-10);
  tr=find(mnrnd(1,p/sum(p)));
  for t=0:T-1, if(t==0), restart=50; else restart=2; end
    if(tr+t<=T), t0=tr; t1=tr+t; td=1; else t0=T; t1=T-t; td=-1; end
    tPrv=t1-td; if(tPrv>=1 && tPrv<=T), Y(:,:,t1)=Y(:,:,tPrv); end
    Y(:,:,t1)=refine_t(X{t1},S{t1},R{t1},Y(:,:,t0:td:t1),t+1,restart,prm);
    tocStatus( tid, ((i-1)+(t+1)/T)/nInit );
  end
  % keep best solution found so far
  if(nInit==1), return; end; e=energy(X,S,R,Y,prm);
  if(i==1 || e<eBst), eBst=e; bestY=Y; end
end; Y=bestY;
end

function Y = refine( X, S, R, Y, phase, restart, prm )
for p=1:phase, T=size(Y,3);
  for t=1:T, if(mod(p,2)), t0=t; else t0=T-t+1; end
    Y(:,:,t0)=refine_t(X{t0},S{t0},R{t0},Y,t0,restart,prm);
  end
end
end

function Yt = refine_t( Xt, St, Rt, Y, t, restart, prm )
% refine Y(:,:,t) by performing max descent (need at most Y(:,:,t-2:t+2))
[K,p,T]=size(Y); Y=Y(:,:,max(1,t-2):min(T,t+2)); T=size(Y,3); t=min(t,3);
pAng=prm.pAng; A=convFrExt(Y,pAng);
if(t>1), Yp=Y(:,:,t-1); end; if(t+1<=T), Yn=Y(:,:,t+1); end
if(t>1), Ap=A(:,:,t-1); end; if(t+1<=T), An=A(:,:,t+1); end
if(t>2), AP=A(:,:,t-2); end; if(t+2<=T), AN=A(:,:,t+2); end
% stuff for refining smoothness term (ySmth,wSmth)
ySmth=zeros(K,p); wSmth=zeros(K,1);
if(t>1+1), ySmth=ySmth+convToExt(2*Ap-AP,pAng)/4; wSmth=wSmth+1/4; end
if(t>1 && t<T), ySmth=ySmth+constrain((Yp+Yn)/2,pAng); wSmth=wSmth+1; end
if(t<T-1), ySmth=ySmth+convToExt(2*An-AN,pAng)/4; wSmth=wSmth+1/4; end
ySmth=prm.lambda*ySmth/prm.eSmth; wSmth=prm.lambda*wSmth/prm.eSmth;
% stuff for refining constant term (yCnst,wCnst)
yCnst=zeros(K,p); wCnst=zeros(K,1);
if(t>1), yCnst=yCnst+Yp; wCnst=wCnst+1; end
if(t<T), yCnst=yCnst+Yn; wCnst=wCnst+1; end
yCnst=prm.lambda2*yCnst/prm.eCnst; wCnst=prm.lambda2*wCnst/prm.eCnst;
% candidates for random initializations
cnd=cell(1,K); if(~isempty(Rt)), for j=1:K, cnd{j}=find(Rt(:,j)); end; end
for j=1:K, if(isempty(cnd{j})), cnd{j}=1:size(Xt,1); end; end; z=prm.z;
for r=0:restart
  % if r==0 use previous Yt else randomly initialize Yt
  if(r==0), Yt=Y(:,:,t); else
    for j=1:K, Yt(j,:)=Xt(randSample(cnd{j},1,1),:); end; end
  for i=1:10
    % stuff for refining data term (yData,wData)
    D=dist(Xt,Yt,Rt,z); A=bsxfun(@le,D,min(D,[],2)); %nxm
    A=bsxfun(@rdivide,A,sum(A,2)); A=A&(D<z); %nxm
    wData=sum(bsxfun(@times,A,St),1)'; %mx1
    yData=zeros(K,p); SX=bsxfun(@times,Xt,St); %SX is nxp
    for j=1:K, yData(j,:)=sum(bsxfun(@times,SX,A(:,j)),1); end
    yData=yData/prm.eData; wData=wData/prm.eData;
    % perform actual refinenement
    yTotal=yData+ySmth+yCnst; wTotal=wData+wSmth+wCnst;
    bad=(wTotal==0); yTotal(bad,:)=Yt(bad,:); wTotal(bad)=1;
    Yt0=Yt; Yt=bsxfun(@rdivide,yTotal,wTotal);
    if(pAng), Yt=constrain(Yt,pAng); end
    if( sum((Yt0-Yt).^2,2)<1e-5 ), break; end
    % optionally check that energy actually decreased
    if( ~prm.debug ), continue; end
    Y(:,:,t)=Yt0; e0=energy_t(Xt,St,Rt,Y,t,prm);
    Y(:,:,t)=Yt; e1=energy_t(Xt,St,Rt,Y,t,prm);
    assert(e0-e1>=-1e-10);
  end
  % keep best solution found so far
  if(restart==0), break; end;
  Y(:,:,t)=Yt; e=energy_t(Xt,St,Rt,Y,t,prm);
  if(r==0 || e<eBst), eBst=e; bstYt=Yt; end
end
if(restart>0), Yt=bstYt; end
end

function [e,eData,eSmth,eCnst] = energy_t( Xt, St, Rt, Y, t, prm )
% compute energy of Yt only (need at most Y(:,:,t-2:t+2))
[K,p,T]=size(Y); Y=Y(:,:,max(1,t-2):min(T,t+2)); T=size(Y,3); t=min(t,3);
lambda=prm.lambda; lambda2=prm.lambda2; eSmth=0; eCnst=0; z=prm.z;
% compute energy based on coverage at frame t (data term)
eData=sum(min(dist(Xt,Y(:,:,t),Rt,z),[],2).*St);
% compute energy based on motion in frames t-1:t+1 (smoothness term)
if(lambda>0 && T>=3), pAng=prm.pAng; p=size(Y,2)-pAng*2;
  Y1=convFrExt(Y,pAng); Y1=Y1(:,:,1:T-2)+Y1(:,:,3:T)-2*Y1(:,:,2:T-1);
  if(p), eSmth=eSmth+sum(sum(sum(Y1(:,1:p,:).^2)))/4; end
  if(pAng), eSmth=eSmth+sum(sum(sum(1-cos(Y1(:,p+1:end,:)))))/2; end
end
% compute energy based on position in frame t-1:t+1 (constant term)
if(lambda2>0 && t>1), eCnst=eCnst+sum(sum((Y(:,:,t)-Y(:,:,t-1)).^2)); end
if(lambda2>0 && t<T), eCnst=eCnst+sum(sum((Y(:,:,t)-Y(:,:,t+1)).^2)); end
% combine energy, normalize
eData=eData/prm.eData; eSmth=eSmth/prm.eSmth; eCnst=eCnst/prm.eCnst;
e = (eData + lambda*eSmth + lambda2*eCnst)/(1 + lambda + lambda2);
end

function [e,eData,eSmth,eCnst] = energy( X, S, R, Y, prm )
% compute overall energy of Y (normalized so typically in [0,1])
[K,p,T]=size(Y); eData=0; eSmth=0; eCnst=0;
lambda=prm.lambda; lambda2=prm.lambda2; z=prm.z;
% compute energy based on coverage (data term)
for t=1:T, eData=eData+sum(min(dist(X{t},Y(:,:,t),R{t},z),[],2).*S{t}); end
% compute energy based on motion (smoothness term)
if(lambda>0 && T>=3), pAng=prm.pAng; p=size(Y,2)-pAng*2;
  Y1=convFrExt(Y,pAng); Y1=Y1(:,:,1:T-2)+Y1(:,:,3:T)-2*Y1(:,:,2:T-1);
  if(p), eSmth=eSmth+sum(sum(sum(Y1(:,1:p,:).^2)))/4; end
  if(pAng), eSmth=eSmth+sum(sum(sum(1-cos(Y1(:,p+1:end,:)))))/2; end
end
% compute energy based on position (constant term)
if(T>=3 && lambda2>0), eCnst=sum(sum(sum((Y(:,:,1:T-2)-Y(:,:,2:T-1)).^2))); end
%if(lambda2>0), eCnst=sum(sum(sum((Y(:,:,1:T-1)-Y(:,:,2:T)).^2))); end
% combine energy, normalize
eData=eData/prm.eData; eSmth=eSmth/prm.eSmth; eCnst=eCnst/prm.eCnst;
e = (eData + lambda*eSmth + lambda2*eCnst)/(1 + lambda + lambda2);
end

function Y = convFrExt( Y, pAng )
% convert from extended format (a,b,x,y) to standard format (a,b,ang)
if(pAng==0), return; end; p0=size(Y,2)-pAng*2;
Y=[Y(:,1:p0,:) atan2(Y(:,p0+pAng+1:end,:),Y(:,p0+1:p0+pAng,:))];
end

function Y = convToExt( Y, pAng )
% convert from standard format (a,b,ang) to extended format (a,b,x,y)
if(pAng==0), return; end; p0=size(Y,2)-pAng;
Y=[Y(:,1:p0,:) cos(Y(:,p0+1:end,:)) sin(Y(:,p0+1:end,:))];
end

function Y = constrain( Y, pAng )
% constrain extended format so angle coords lie on unit circle
if(pAng==0), return; end; p0=size(Y,2)-pAng*2;
K=max(1e-5,sqrt( Y(:,p0+1:p0+pAng,:).^2 + Y(:,p0+pAng+1:end,:).^2 ));
for i=1:(pAng*2), Y(:,i+p0,:)=Y(:,i+p0,:)./K(:,mod(i-1,pAng)+1,:); end
end

function D = dist( X, Y, R, z )
% All pairs thresholded squared euclidean distance.
%  X - [n x p] matrix of n p-dim vectors
%  Y - [K x p] matrix of K p-dim vectors
%  R - [n x K] if given, R(i,j)=0 implies D(i,j)=1
%  z - [1 x 1] max distance
%  D - [n x K] distance matrix
Yt=Y'; XX=sum(X.*X,2); YY=sum(Yt.*Yt,1);
D=bsxfun(@plus,XX,YY)-2*X*Yt; D=D./size(X,2);
D=min(D,z); if(~isempty(R)), D(R==0)=z; end
end
