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
          d2 = pdist(ptmp,'squaredeuclidean');

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
        d2 = pdist(pTrkI,'squaredeuclidean');
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
      
      [sigma,poslambda] = myparse(varargin,...
        'sigma',[],... % sigma in R^2 space
        'poslambda',[]);
      
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
          d2 = pdist(ptmp,'squaredeuclidean'); % [KxK]
          w = sum(squareform(exp( -d2/sigma^2/2 )),1);
          w = w / sum(w);
          probAcc = probAcc + log(w); % accumulate by summing over pts/parts. maximizing sum-of-logs is like max-ing prod-of-probs
        end
        appcost(i,:) = -probAcc; % higher probAcc => lower appearance cost
      end

      X = permute(pTrkFull,[3 1 2]);
      [Xbest,idx,totalcost,poslambda] = ChooseBestTrajectory(X,appcost,...
        'poslambda',poslambda);
      pTrk = Xbest';
      score = nan(N,1);
      info = idx(:);
    end
  end
end