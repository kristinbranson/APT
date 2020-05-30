function [p,p_t,fail] = rcprTest1( Is, regModel, p, regPrm, ftrPrm, iniData, ...
    verbose, prunePrm)
% Apply robust cascaded shape regressor.
%
% USAGE
%  p = rcprTest1( Is, regModel, p, regPrm, bboxes, verbose, prunePrm)
%
% INPUTS
%  Is       - cell(N,1) input images. Optionally with channels in 3rd dim
%  regModel - learned multi stage shape regressor (see rcprTrain)
%  p        - [NxDxRT1] initial shapes
%  regPrm   - struct with regression parameters (see regTrain)
%  iniData  - [Nx2] or [Nx4] bbounding boxes/initial positions
%  verbose  - [1] show progress or not 
%  prunePrm - [REQ] parameters for smart restarts 
%     .prune     - [0] whether to use or not smart restarts
%     .maxIter   - [2] number of iterations
%     .th        - [.15] threshold used for pruning 
%     .tIni      - [10] iteration from which to prune
%
% OUTPUTS
%  p        - [NxDxRT1] shape returned by multi stage regressor
%  p_t      - [N*RT1xDx(T+1)] shape over time
%  fail     - used only for pruning
%
% EXAMPLE
%
% See also rcprTest, rcprTrain
%
% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our paper if you use the code:
%  Robust face landmark estimation under occlusion, 
%  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%  ICCV'13, Sydney, Australia

% Modified by Allen Lee, Kristin Branson

% Apply each single stage regressor starting from shape p.
model = regModel.model;
%T=regModel.T;
[N,D,RT1] = size(p);
assert(D==model.D);
p = reshape(permute(p,[1 3 2]),[N*RT1,D]);
imgIds = repmat(1:N,[1 RT1]); 
regs = regModel.regs;

%Get prune parameters
maxIter=prunePrm.maxIter;prune=prunePrm.prune;
th=prunePrm.th;tI=prunePrm.tIni;

%Set up data
p_t = zeros(size(p,1),D,regModel.T+1); %[N*RT1xDx(T+1)];
p_t(:,:,1) = p; % AL: looks like this always gets overwritten later
if model.isFace
  assert(size(iniData,3)==1);
  bbs = iniData(imgIds,:,1);
else
  bbs = [];
end
done = 0; % true/false flag
Ntot = 0; % pruning stuff 
k = 0;
N1=N;p1=p;imgIds1=imgIds;pos=1:N;md=zeros(N,RT1,D,maxIter); 
fail = [];
%Iterate while not finished
while ~done
    %Apply cascade
    tStart = clock;
    %If pruning is active, each loop returns the shapes of the examples
    %that passed the smart restart threshold (good) and 
    %those that did not (bad)
    %
    % AL20151204: If not pruning, good1=1:N
    [good1,bad1,p_t1,p1,p2]=cascadeLoop(Is,model,regModel,regPrm,ftrPrm,N1,RT1,...
        p1,imgIds1,regs,tStart,iniData,bbs,verbose,...
        prune,1,th,tI);      
    %Separate into good/bad (smart restarts)
    good = pos(good1);
    mem = ismember(imgIds,good);
    assert(all(mem),'AL no pruning');
    p_t(mem,:,:) = p_t1;
    assert(isequaln(p1,p_t1(:,:,end)),'AL');
    p(mem,:) = p1;
    Ntot = Ntot+length(good); 
    done = Ntot==N;     
    assert(done,'AL no pruning');
    
    if(~done)
        %Keep iterating only on bad
        Is=Is(bad1,:);N1=length(bad1);pos=pos(bad1);
        imgIds1 = repmat(1:N1,[1 RT1]);
        if(model.isFace) 
            iniData=iniData(bad1,:,:);bbs=iniData(imgIds1,:,1); 
            p1=shapeGt('initTest',[],iniData,...
                model,regModel.pStar,regModel.pGtN,RT1);
            p1=reshape(permute(p1,[1 3 2]),[N1*RT1,D]);
        else
          assert(false,'AL: broken codepath initTest args');
%             iniData=iniData(bad1,:,:);
%             p1=shapeGt('initTest',Is,model,...
%                 iniData,regModel.pStar,RT1);
%             p1=reshape(permute(p1,[1 3 2]),[N1*RT1,D]);
        end
        
        md(pos,:,:,k+1)=reshape(p2,[N1,RT1,D]);
        k=k+1;
        %If maxIter has been reached, use median of all 
        if(k>=maxIter)
            RT=RT1*maxIter;
            p1=reshape(permute(md(pos,:,:,:),[1 3 2 4]),[N1,D,RT]);
            %select initialization for each indepently based on distance
            dist=zeros(N1,RT);dist2=zeros(N1,RT);
            for r=1:RT
                aux = permute(shapeGt('dist',model,p1(:,:,:),p1(:,:,r)),[1 3 2]);
                close=zeros(N1,RT);
                close(aux<th)=1;
                dist(:,r)=sum(close,2); 
                try dist2(:,r)=mean(aux,2);
                catch 
                    warning('not enough shapes');size(aux),size(dist2)
                end
            end
            %expand to RT1 different initializations
            p2=zeros(N1,D,RT1);
            for n=1:N1
                ind=find(dist(n,:)>=RT*0.7);%4
                if(length(ind)>=RT1)
                    use=randSample(ind,RT1);
                    p2(n,:,:)=p1(n,:,use);
                else
                    [~,ix]=sort(dist2(n,:));
                    p2(n,:,:)=p1(n,:,ix(1:RT1));
                end
            end
            %Call cascade loop one last time 
            p1=reshape(permute(p2,[1 3 2]),[N1*RT1,D]);tStart=clock;
            [~,~,p_t1,p1,~]=cascadeLoop(Is,model,regModel,regPrm,ftrPrm,N1,RT1,...
                p1,imgIds1,regs,tStart,iniData,bbs,verbose,...
                0,tI,th,tI);
            remain=pos; ind=ismember(imgIds,remain);
            p_t(ind,:,:)=p_t1;p(ind,:)=p1;
            fail = remain;
            done=1;
        end
    end
end
%reconvert p from [N*RT1xD] to [NxDxRT1]
p = permute(reshape(p,[N,RT1,D]),[1 3 2]);
%p_t=permute(reshape(p_t,[N,RT1,D,T+1]),[1 3 2 4]);
end

%Apply full RCPR cascade with check in between if smart restart is enabled
function [good,bad,p_t,p,p2] = cascadeLoop(Is,model,regModel,regPrm,ftrPrm,...
  N,RT1,p,imgIds,regs,tStart,bboxes,bbs,verbose,prune,t0,th,tI)
% p (input): [MxD] initial shapes, absolute coords. M=N*RT1
%
% good,bad,p,p2: only used for pruning
% p_t: [MxDx(T+1)]. shapes over all initialconds/iterations, absolute coords. 
%   p_t(:,:,1)=p, p_t(:,:,T+1) is the final iteration.
%
% p: for no-pruning, should equal p_t(:,:,T+1)

T = regModel.T; % number of regressors/stages  
D = regModel.model.D;
M = N*RT1;  
assert(numel(Is)==N);
assert(isequal(size(p),[M,D])); % initial positions/shapes, absolute coords
assert(numel(imgIds)==M && all(ismember(imgIds,1:N))); % labels rows of p, bbs
assert(numel(regs)==T); % regressor structs
% tStart: time value from clock()
assert(isequal(size(bboxes),[N 2*model.d]));
assert(isequal(size(bbs),[M 2*model.d])); % should equal bboxes(imgIds,:)
% prune: scalar logical
% t0: starting t-index (regressor index)
% th, tI: prune paremeters

p_t = zeros(M,D,T+1); % shapes over all initial conds/iterations, absolute coords
p_t(:,:,1) = p;
good = 1:N; % indices of 'good' images; used only for pruning
bad = []; % only for pruning
p2 = []; % only for pruning
for t = t0:T
  fprintf(1,'cascadeLoop, %d/%d\n',t,T);
  
  %Compute shape-indexed features
  ftrPos = regs(t).ftrPos;
  switch ftrPos.type
    case 'kborig_hack'
      ftrs = shapeGt('ftrsCompKBOrig',model,p,Is,ftrPos,...
          imgIds,regModel.pStar,bboxes,regPrm.occlPrm);
    case {'1lm' '2lm' '2lmdiff'} %{5 6 7 8 9 10 11}
      [ftrs,regPrm.occlD] = shapeGt('ftrsCompDup2',model,p,Is,ftrPos,...
        imgIds,regModel.pStar,bboxes,regPrm.occlPrm);
    case {3 4}
      assert(false,'AL new Is');
      [ftrs,regPrm.occlD] = shapeGt('ftrsCompDup',model,p,Is,ftrPos,...
        imgIds,regModel.pStar,bboxes,regPrm.occlPrm);
    otherwise
      assert(false,'AL new Is');
      [ftrs,regPrm.occlD] = shapeGt('ftrsCompIm',model,p,Is,ftrPos,...
        imgIds,regModel.pStar,bboxes,regPrm.occlPrm);
  end
  %Retrieve learnt regressors
  regt = regs(t).regInfo;
  %Apply regressors
  p1 = shapeGt('projectPose',model,p,bbs); % p1 is normalized
  pDel = regApply(p1,ftrs,regt,regPrm,ftrPrm); % pDel is normalized
  
  if regPrm.USE_AL_CORRECTION
    p = Shape.applyRIDiff(p1,pDel,1,3); % XXXAL HARDCODED HEAD/TAIL
  else
    p = shapeGt('compose',model,pDel,p,bbs); % p (output) is normalized
  end
  p = shapeGt('reprojectPose',model,p,bbs);
  p_t(:,:,t+1) = p;
  %If reached checkpoint, check state of restarts
  if prune && T>tI && t==tI % AL: second cond seems unnecessary
    assert(false,'AL unchecked codepath');
    [p_t,p,good,bad,p2] = checkState(p_t,model,imgIds,N,t,th,RT1);
    if isempty(good)
      return;
    end
    Is = Is(good,:);
    N = length(good);
    imgIds = repmat(1:N,[1 RT1]);
    if model.isFace
      bboxes = bboxes(good,:);
      bbs = bboxes(imgIds,:);
    end
  end
  if (t==1 || mod(t,5)==0) && verbose
    msg = tStatus(tStart,t,T);
    fprintf(['Applying ' msg]);
  end
end
end

function [p_t,p,good,bad,p2]=checkState(p_t,model,imgIds,N,t,th,RT1)
    %Confidence computation=variance between different restarts
    %If output has low variance and low distance, continue (good)
    %ow recurse with new initialization (bad)
    p = permute(p_t(:,:,t+1),[3 2 1]);
    conf = zeros(N,RT1);
    for n = 1:N
        pn=p(:,:,imgIds==n);md=median(pn,3);
        %variance=distance from median of all predictions
        conf(n,:)=shapeGt('dist',model,pn,md);
    end
    dist=mean(conf,2);
    bad=find(dist>th);good=find(dist<=th);
    p2=p_t(ismember(imgIds,bad),:,t+1);
    p_t=p_t(ismember(imgIds,good),:,:);p=p_t(:,:,t+1);
    if(isempty(good)),return; end
end

function msg = tStatus(tStart,t,T)
elptime = etime(clock,tStart);
fracDone = max( t/T, .00001 );
esttime = elptime/fracDone - elptime;
if( elptime/fracDone < 600 )
    elptimeS  = num2str(elptime,'%.1f');
    esttimeS  = num2str(esttime,'%.1f');
    timetypeS = 's';
else
    elptimeS  = num2str(elptime/60,'%.1f');
    esttimeS  = num2str(esttime/60,'%.1f');
    timetypeS = 'm';
end
msg = ['  [elapsed=' elptimeS timetypeS ...
    ' / remaining~=' esttimeS timetypeS ']\n' ];
end
