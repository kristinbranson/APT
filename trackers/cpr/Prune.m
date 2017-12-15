classdef Prune
  methods (Static)
    
    % Pruning function signature
    %
    % [pTrk,score,info] = fcn(pTrkFull,p1,v1,...)
    %
    % pTrkFull: [NxNrepxD] Final shapes over multiple replicates
    %
    % pTrk: [NxD] Selected/pruned shapes
    % score: [Nx1] 'Quality' score, higher is better. Arbitrary units but
    %   should be comparable to other quality scores from same fcn
    % info: Arbitrary diagnostic data, eg for calibration of any input
    %   params    
    
    function [pTrk,score,info] = median(pTrkFull)
      assert(ndims(pTrkFull)==3);

      pTrk = squeeze(median(pTrkFull,2));
      score = squeeze(mad(pTrkFull,1,2));
      score = -mean(score,2); % average mad over all x/y coords, all pts
      info = [];
    end
    
    % 1. maxdensity. Weights given for each landmark by gaussian KDE (nbor 
    %  count). this gives a prob for each replicate, per point. The best overall 
    %  replicate is the one that maximizes the product of per-point probs.
    % 2. Per-point gaussian KDE. (not impl) the final shape doesn't have to be an 
    %  actual shape. For each point, just take the replicate point that maximizes 
    %  the nbor count. So the final shape need not be a real shape.
    % 3. Globalmin. Minimize distance to all other replicates in full replicate 
    % space. Like 1. but we do not go pt-by-pt, the shape is considered as a whole.

    
    function [pTrk,score,info] = maxdensity(pTrkFull,varargin)
      sigma = myparse(varargin,...
        'sigma',5); % sigma in R^2 space
      
      assert(ndims(pTrkFull)==3);
      [N,nRep,D] = size(pTrkFull);
      d = 2;
      npts = D/d;
      
      pTrk = nan(N,D);
      score = nan(N,1);
      info = cell(N,1); % info{i} is [nRepPairs x npts] pairwise dist^2 
                % values between replicates, per pt. Maybe useful for 
                % calibration of sigma
      
      pTrkFullPts = reshape(pTrkFull,[N nRep npts d]);
      for i=1:N
        probAcc = zeros(1,nRep); % probability accumulator
        for ipt=1:npts
          ptmp = squeeze(pTrkFullPts(i,:,ipt,:)); % [nRep x d]
          d2 = pdist(ptmp,'euclidean').^2;

          % The sum-of-guassians here is ~ a "neighbor-counter" where each
          % neighbor within sigma adds a 1 and all others have no effect.
          %
          % squareform(...) is the symmetric matrix of "boltzmann weights"
          % computed from squared-distances in d. small weight => large
          % distance, large weight => small distance
          %
          % sum(...,1) forms a sum-of-weights for the ith replicate. Larger
          % values mean more likely replicate.
          w = sum(squareform(exp( -d2/sigma^2/2 )),1);
          
          % Normalizing gives a 'probability' per replicate (for this
          % landmark/part). (In fact looks like the normalization does not
          % affect the replicate chosen)
          w = w / sum(w);
          
          probAcc = probAcc + log(w); % accumulate by summing over pts/parts. maximizing sum-of-logs is like max-ing prod-of-probs
          
          info{i}(:,ipt) = d2;
        end
        
        [score(i),iRep] = max(probAcc); % most-likely replicate for this trial
        pTrk(i,:) = pTrkFull(i,iRep,:);
      end
    end    
    
    function [pTrk,score,info] = globalmin(pTrkFull,varargin)
      sigma = myparse(varargin,...
        'sigma',5); % sigma in full pose space R^[nptx2]
      
      assert(ndims(pTrkFull)==3);
      [N,~,D] = size(pTrkFull);
      
      pTrk = nan(N,D);
      score = nan(N,1);
      info = cell(N,1);
      for i=1:N
        pTrkI = squeeze(pTrkFull(i,:,:)); % [nRepxD]
        d2 = pdist(pTrkI,'euclidean').^2;
        d2mat = squareform(d2); % [nRepxnRep] dist^2 from rep i to rep j in full pose space
        wmat = exp(-d2mat/2/sigma^2); % [nRepxnRep] Boltzmann weight matrix
        wsum = sum(wmat,1); % [1xnRep], sum of weights (nbor count) for each rep
        
        [score(i),iRep] = max(wsum);
        pTrk(i,:) = pTrkI(iRep,:);
        info{i} = d2(:);
      end
    end
    
    function [pTrk,score,info] = besttraj(pTrkFull,varargin)
      % It is assumed that pTrkFull are from consecutive frames.
      
      [sigma,poslambda,poslambdafac,dampen] = myparse(varargin,...
        'sigma',nan,... % sigma in R^2 space
        'poslambda',[],... % see ChooseBestTrajectory
        'poslambdafac',[],... % see ChooseBestTrajectory
        'dampen',0.5... % etc
        );
      
      assert(ndims(pTrkFull)==3);
      [N,K,D] = size(pTrkFull); % K==nRep
      d = 2;
      npts = D/d;
      
      % Compute appearancecost (AC). See Prune.maxdensity
      % Note, I think the AC needs to be within-iter comparable, but not
      % necessarily between-iter comparable. Some some datasets have seen 
      % that the AC based on the maxdensity maxPr is not anticorrelated 
      % with GT tracking error. Why? One theory is, when tracking is good, 
      % many reps are similar/close together, with the result that 
      % probability gets widely shared among similar reps (for a given pt), 
      % leading to smaller maxPr scores for the best replicate. With
      % less-than-pristine tracking, the reps are spread out, and it is 
      % more likely that one rep stands out and captures a greater share of 
      % pr.

      appcost = nan(N,K);
      pTrkFullPts = reshape(pTrkFull,[N K npts d]);
      for i=1:N
        probAcc = zeros(1,K); % probability accumulator
        for ipt=1:npts          
          ptmp = squeeze(pTrkFullPts(i,:,ipt,:)); % [K x d]
          d2 = pdist(ptmp,'euclidean').^2; % [KxK]
          w = sum(squareform(exp( -d2/sigma^2/2 )),1);
          w = w / sum(w);
          probAcc = probAcc + log(w); % accumulate by summing over pts/parts. maximizing sum-of-logs is like max-ing prod-of-probs
        end
        appcost(i,:) = -probAcc; % higher probAcc => lower appearance cost
      end

      X = permute(pTrkFull,[3 1 2]);
      [Xbest,idx,totalcost,poslambdaused] = ChooseBestTrajectory(X,appcost,...
        'poslambda',poslambda,'poslambdafac',poslambdafac,'dampen',dampen);
      pTrk = Xbest';
      score = nan(N,1);
      info = struct('idx',idx(:),'totalcost',totalcost,...
        'poslambdafac',poslambdafac,'poslambdaused',poslambdaused);
    end
    
    function [pTrk,pruneMD,tblSegments] = applybesttraj2segs(pTrkFull,pTrkMD,...
        besttrajArgs,varargin)
      % Apply Prune.besttraj to segments of consecutive frames in 
      % pTrkFull. pTrkFull need not represent consecutive frames.
      %
      % pTrkFull: [NxKxD]. First dim doesn't have to be consecutive frames.
      % pTrkMD: [N] MFTable labeling rows of pTrkFull.
      % besttrajArgs: cell vec of optional PV pairs passed to Prune.besttraj
      %
      % pTrk: [NxD], reduced tracking for each row of pTrkFull.
      % pruneMD: [N], pruning MD labeling rows of pTrk.
      % tblSegments: [nSeg] table detailing segments of consecutive frames 
      %   where smoothing was applied
      
      smallSegmentWarnThresh = myparse(varargin,...
        'smallSegmentWarnThresh',15 ... % throw warning for segments with 
        ... % fewer than this many frames. Use inf to never throw warning
      );
      
      [N,~,D] = size(pTrkFull);
      assert(istable(pTrkMD) && height(pTrkMD)==N);
      
      tblSegments = Prune.analyzeConsecSegments(pTrkMD);
      nSeg = height(tblSegments);
      fprintf('%d segments found.\n',nSeg);
      
      pTrk = nan(N,D);
      infoSeg = cell(nSeg,1);
      segFrm0 = nan(N,1); % for each row of pTrk, the "segment ID" or first-frame-of-segment analyzed for that row
      poslambdaUsed = nan(N,1); % for each row of pTrk, the poslambdaused
      for iSeg=1:nSeg
        rowSeg = tblSegments(iSeg,:);
        tfSeg = pTrkMD.mov==rowSeg.mov & pTrkMD.iTgt==rowSeg.iTgt & ...
          rowSeg.frm0<=pTrkMD.frm & pTrkMD.frm<=rowSeg.frm1;
        pTrkMDthis = pTrkMD(tfSeg,:);
        assert(isequal(pTrkMDthis.frm,(rowSeg.frm0:rowSeg.frm1)'));
        
        nfrmsSeg = height(pTrkMDthis);
        if nfrmsSeg<smallSegmentWarnThresh
          warningNoTrace('Segment of %d frames encountered. Trajectory-smoothing may perform better on longer stretches of consecutively tracked frames.',nfrmsSeg);
        end
        [pTrk(tfSeg,:),~,infoSeg{iSeg}] = ...
          Prune.besttraj(pTrkFull(tfSeg,:,:),besttrajArgs{:});
        
        segFrm0(tfSeg) = rowSeg.frm0;
        poslambdaUsed(tfSeg) = infoSeg{iSeg}.poslambdaused;
      end
      
      tblSegments.pruneInfo = infoSeg(:);
      pruneMD = table(segFrm0,poslambdaUsed);      
    end
    
    function tblSegments = analyzeConsecSegments(pTrkMD)
      s = struct('mov',cell(0,1),'iTgt',[],'frm0',[],'frm1',[]);
      nMov = max(pTrkMD.mov);
      nTgt = max(pTrkMD.iTgt);
      for iMov=1:nMov
      for iTgt=1:nTgt
        tfThisMovTgt = pTrkMD.mov==iMov & pTrkMD.iTgt==iTgt;
        if nnz(tfThisMovTgt)==0
          continue;
        end
        
        mdThis = pTrkMD(tfThisMovTgt,:);
        assert(issorted(mdThis.frm));
        
        iSegStart = 1; % row of mdThis at start of next seg
        while iSegStart<=height(mdThis)
          row0 = mdThis(iSegStart,:);
          for iSegEnd=iSegStart:height(mdThis)
            row1 = mdThis(iSegEnd,:);
            if row1.frm-row0.frm==iSegEnd-iSegStart
              % OK, segment continues
            else
              % segment is over. On last segment, will not hit this branch but that's OK
              iSegEnd = iSegEnd-1;
              break;
            end
          end
          
          s(end+1,1).mov = iMov; %#ok<AGROW>
          s(end).iTgt = iTgt;
          s(end).frm0 = mdThis.frm(iSegStart);
          s(end).frm1 = mdThis.frm(iSegEnd);
          
          iSegStart = iSegEnd+1;
        end
      end
      end
      
      tblSegments = struct2table(s);      
    end
    
  end  
  
end