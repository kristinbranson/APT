classdef PostProcess < handle
  
  properties
    
    algorithm = 'maxdensity';
    nativeformat = 'sample';
    jointsamples = true;
    
    % parameters for KDE
    kde_sigma = 5;
    
    % parameters for Viterbi
    viterbi_poslambda = [];
    viterbi_poslambdafac = [];
    viterbi_dampen = 0.5;
    viterbi_misscost = [];
    
    % sampledata will have be a struct with fields x and w
    % sampledata.x is the location of the samples, and will be a matrix of
    % size N x nRep x npts x nviews x d, corresponding to N frames, nRep
    % replicates, npts landmarks each across nviews views in d dimensions
    % (d should be 2) 
    % sampledata.w is the weight of each sample, computed using KDE.
    % sampledata.w.indep is if we do KDE independently in 2-d for each
    % landmark point, while sampledata.w.joint is if we do KDE in 
    % sampledata.w.indep is N x nRep x npts x nviews and sampledata.w.joint
    % is N x nrep.
    sampledata = [];
    heatmapdata = [];
    postdata = struct;
    
  end
  
  properties (Constant)
    
    algorithms_jointmatters = {'maxdensity','viterbi'};
    algorithms = {'maxdensity','median','viterbi'};
    
  end
    
  methods
    
    function obj = PostProcess()
      
%       assert(mod(nargin,2)==0);
%       for i = 1:2:numel(varargin),
%         obj.(varargin{i}) = varargin{i+1};
%       end
      
    end
    
    function SetSampleData(obj,x)

      % special case if nviews == 1
      [N,nRep,npts,nviews,d] = size(x);
      if nviews > 1 && d == 1,
        d = nviews;
        nviews = 1;
        x = reshape(x,[N,nRep,npts,nviews,d]);
      end
      
      % clear out all postprocessing and heatmap results
      obj.postdata = struct;
      obj.heatmapdata = [];
      
      obj.sampledata = struct;
      obj.sampledata.x = x;
      obj.sampledata.w = struct;
      obj.sampledata.w.indep = [];
      obj.sampledata.w.joint = [];
      obj.nativeformat = 'sample';
      
    end

    function SetHeatmapData(obj,scores,scales)
      
      % clear out all postprocessing and gmm results
      obj.postdata = struct;
      obj.sampledata = [];
      obj.SetJointSamples(false);
      
      obj.heatmapdata = struct;
      
      % scores
      [T,nviews,ny,nx] = size(scores); %#ok<ASGLU>
      % if nviews is 1, this dimension may have been skipped
      if nx == 1,
        [T,ny,nx] = size(scores);
        nviews = 1;
        scores = reshape(scores,[T,nviews,ny,nx]);
      end
      obj.heatmapdata.scores = scores;
      % check scale of scores
      minscore = min(scores(:));
      maxscore = max(scores(:));
      if minscore < 0 || maxscore > 1,
        warning('Heatmap values go outside of 0 and 1, minscore = %f, maxscore = %f',minscore,maxscore);
      end
      
      % scales
      if nargin >= 3,
        obj.heatmapdata.scales = scales;
      else
        % by default don't scale
        obj.heatmapdata.scales = ones(nviews,d);
      end

      % grid
      obj.xgrid = cell(1,nviews);
      obj.ygrid = cell(1,nviews);
      for i = 1:nviews,
        [obj.xgrid{i},obj.ygrid{i}] = meshgrid( (1:nx-1)*obj.heatmapdata.scales(i,1),...
          (1:ny-1)*obj.heatmapdata.scales(i,2) );
      end
      
      obj.nativeformat = 'heatmap';
      
    end
    
    function Heatmap2SampleData(obj)
      
      error('TODO');
      % TODO
      
    end
    
    function SetUNetHeatmapData(obj,scores,scales)
      
      % scores should be nviews x T x ny x nx
      % scales should be nviews x 2
      
      [T,nviews,ny,nx] = size(scores); %#ok<ASGLU>
      if nargin >= 3,
        [nviews2,d] = size(scales);      
        assert(d==2 && nviews==nviews2);
      else
        scales = ones(nviews,2);
      end
      
      % this is probably some python to matlab thing
      hm_miny = 2;
      hm_minx = 2;
      
      minscore = -1;
      maxscore = 1;

      hmd = struct;
      hmd.scores = (scores(:,:,hm_miny:end,hm_minx:end)-minscore)/(maxscore-minscore);
      hmd.scales = scales;
      obj.SetHeatMapData(hmd);

    end
    
    function run(obj)
      
      if ismember(obj.algorithm,PostProcess.algorithms_jointmatters),
        if obj.jointsamples,
          algorithm = [obj.algorithm,'_joint']; %#ok<*PROP>
        else
          algorithm = [obj.algorithm,'_indep'];
        end
      else
        algorithm = obj.algorithm;
      end

      if isfield(obj.postdata,algorithm) && ~isempty(obj.postdata.(algorithm)),
        fprintf('Post-processing for algorithm %s already run.\n',algorithm);
        return;
      end

      
      switch obj.algorithm,
        
        case 'maxdensity',
          obj.RunMaxDensity();
        case 'median',
          obj.RunMedian();
        case 'viterbi',
          obj.RunViterbi();
        otherwise
          error('Not implemented %s',obj.algorithm);
      end
      
    end
    
    function RunMaxDensity(obj)
      
      if strcmp(obj.nativeformat,'sample'),
        obj.RunMaxDensity_SampleData();
      elseif strcmp(obj.nativeformat,'heatmap'),
        obj.RunMaxDensity_HeatmapData();
      else
        error('Not implemented maxdensity %s',obj.nativeformat);
      end
    end
    
    function RunMedian(obj)
      
      if strcmp(obj.nativeformat,'sample'),
        obj.RunMedian_SampleData();
      elseif strcmp(obj.nativeformat,'heatmap'),
        obj.RunMedian_HeatmapData();
      else
        error('Not implemented maxdensity %s',obj.nativeformat);
      end
    end
    
    function SetJointSamples(obj,v)
      
      if strcmp(obj.nativeformat,'heatmap') && v,
        warning('Joint analysis does not make sense for heatmap data');
        return;
      end
      
      if obj.jointsamples == v,
        return;
      end
      obj.jointsamples = v;
      
    end
    
    function SetKDESigma(obj,sigma)
      
      if obj.kde_sigma == sigma,
        return;
      end
      
      obj.kde_sigma = sigma;
      
      % clear out stored results
      if isstruct(obj.sampledata) && ...
          isfield(obj.sampledata,'w') && isstruct(obj.sampledata.w),
        
        if isfield(obj.sampledata.w,'joint'),
          obj.sampledata.w.joint = [];
        end
        if isfield(obj.sampledata.w,'indep'),
          obj.sampledata.w.indep = [];
        end
        
      end
      
      % probably don't need to clear out everything, but let's do it
      % anyways...
      obj.postdata = [];

    end
    
    function SetViterbiParams(obj,varargin)

      ischange = false;
      for i = 1:2:numel(varargin),
        switch(varargin{i}),
          case 'poslambda',
            poslambda = varargin{i+1};
            if numel(obj.viterbi_poslambda) ~= numel(poslambda) || ...
                any(obj.viterbi_poslambda(:) ~= poslambda(:)),
              ischange = true;
              obj.viterbi_poslambda = poslambda;
            end
          case 'poslambdafac',
            poslambdafac = varargin{i+1};
            if numel(obj.viterbi_poslambdafac) ~= numel(poslambdafac) || ...
                any(obj.viterbi_poslambdafac(:) ~= poslambdafac(:)),
              ischange = true;
              obj.viterbi_poslambdafac = poslambdafac;
            end
          case 'dampen',
            dampen = varargin{i+1};
            if numel(obj.viterbi_dampen) ~= numel(dampen) || ...
                any(obj.viterbi_dampen(:) ~= dampen(:)),
              ischange = true;
              obj.viterbi_dampen = dampen;
            end
          case 'misscost',
            misscost = varargin{i+1};
            if numel(obj.viterbi_misscost) ~= numel(misscost) || ...
                any(obj.viterbi_misscost(:) ~= misscost(:)),
              ischange = true;
              obj.viterbi_misscost = misscost;
            end
          otherwise
            error('Unknown Viterbi parameter %s',varargin{i});
        end
      end
      if ischange,
        if isfield(obj.postdata,'viterbi_joint'),
          obj.postdata.viterbi_joint = [];
        end
        if isfield(obj.postdata,'viterbi_indep'),
          obj.postdata.viterbi_indep = [];
        end
      end
    end
    
    function res = NeedKDE(obj)
      
      res = ~isfield(obj.sampledata,'w') || ...
        ( obj.jointsamples && (~isfield(obj.sampledata.w,'joint') || isempty(obj.sampledata.w.joint) ) ) || ...
        ( ~obj.jointsamples && (~isfield(obj.sampledata.w,'indep') || isempty(obj.sampledata.w.indep) ) );

    end
    
    function RunMaxDensity_SampleData(obj)
      
      [N,K,npts,nviews,d] = size(obj.sampledata.x); %#ok<ASGLU>
            
      if obj.NeedKDE(),
        obj.ComputeKDE();
      end

      
      if obj.jointsamples,
        
        w = obj.sampledata.w.joint;
        obj.postdata.maxdensity_joint = struct;
        [obj.postdata.maxdensity_joint.score,k] = max(w,[],2);
        obj.postdata.maxdensity_joint.x = reshape(obj.sampledata.x(:,1,:,:,:),[N,npts,nviews,d]);
        for i = 1:N,
          obj.postdata.maxdensity_joint.x(i,:,:,:) = obj.sampledata.x(i,k(i),:,:,:);
        end
        
      else
        
        % this is different from what maxdensity did
        w = obj.sampledata.w.indep;
        obj.postdata.maxdensity_indep = struct;
        obj.postdata.maxdensity_indep.x = reshape(obj.sampledata.x(:,1,:,:,:),[N,npts,nviews,d]);
        obj.postdata.maxdensity_indep.score = zeros([N,1]);
        for ipt=1:npts,
          for ivw=1:nviews,

            % sampledata.w.indep is N x nRep x npts x nviews
            [scorecurr,k] = max(w(:,:,ipt,ivw),[],2);
            obj.postdata.maxdensity_indep.score = obj.postdata.maxdensity_indep.score + log(scorecurr);
            for i = 1:N,
              obj.postdata.maxdensity_indep.x(i,ipt,ivw,:) = obj.sampledata.x(i,k(i),ipt,ivw,:);
            end
            
          end
        end
      end
      
    end
    
    function RunMedian_SampleData(obj)
      
      [N,K,npts,nviews,d] = size(obj.sampledata.x); %#ok<ASGLU>
      obj.postdata.median.x = reshape(median(obj.sampledata.x,2),[N,npts,nviews,d]);
      obj.postdata.median.score = -mean(reshape(mad(obj.sampledata.x,1,2),[N,npts*nviews*d]),2);
      
    end

    function RunMedian_HeatmapData(obj)
      error('Not implemented');
    end
    
    function RunMaxDensity_HeatmapData(obj)
      
      error('Not implemented');
      
    end
    
    function ComputeKDE(obj)

      starttime = tic;
      
      % TODO: could check to see if running is necessary?
      [N,nRep,npts,nviews,d] = size(obj.sampledata.x);
      D = npts*nviews*d;
      
      if obj.jointsamples,
        
        fprintf('Computing KDE, joint\n');
          
        x = reshape(permute(obj.sampledata.x,[3,4,5,6,2,1]),[D,1,nRep,N]);
        d2 = reshape(sum( (x-reshape(x,[D,nRep,1,N])).^2,1 ),[nRep,nRep,N]);
        w= sum(exp( -d2/obj.kde_sigma^2/2 ),1)-1; % subtract 1 because we have included this replicate
        w = w ./ sum(w,2);
        obj.sampledata.w.joint = permute(w,[3,2,1]);
        
        
%         for i=1:N,
%           x = reshape(obj.sampledata.x(i,:,:,:,:),[nRep,D]);
%           d2 = pdist(x,'euclidean').^2;
%           w = sum(squareform(exp(-d2/2/obj.kde_sigma^2)),1); % 1 x nRep
%           % note that globalmin was not doing this normalization when I
%           % think it should
%           w = w / sum(w);
%           obj.sampledata.w.joint(i,:) = w;
%         end

      else
        
        fprintf('Computing KDE, indep\n');
        d2 = sum( (reshape(obj.sampledata.x,[N,1,nRep,npts,nviews,d])-reshape(obj.sampledata.x,[N,nRep,1,npts,nviews,d])).^2, 6 );
        w= sum(exp( -d2/obj.kde_sigma^2/2 ),2)-1; % subtract 1 because we have included this replicate
        w = w ./ sum(w,3);
        obj.sampledata.w.indep = reshape(w,[N,nRep,npts,nviews]);
        
%         for i=1:N,
%           for ipt=1:npts,
%             for ivw = 1:nviews,
%               ptmp = reshape(obj.sampledata.x(i,:,ipt,ivw,:),[nRep,d]);
%               d2 = pdist(ptmp,'euclidean').^2;
%               w = sum(squareform(exp( -d2/obj.kde_sigma^2/2 )),1);
%               w = w / sum(w);
%               obj.sampledata.w.indep(i,:,ipt,ivw) = w;
%             end
%           end
%         end
      end
      
      fprintf('Done computing KDE, time = %f\n',toc(starttime));
      
    end
    
    function RunViterbi(obj)
      % It is assumed that pTrkFull are from consecutive frames.

      if isempty(obj.sampledata),
        obj.ConvertToSampleData();
      end
      if obj.NeedKDE(),
        obj.ComputeKDE();
      end

      [N,K,npts,nviews,d] = size(obj.sampledata.x); 
      D = npts*nviews*d;
      
      if obj.jointsamples,
        
        appcost = -log(obj.sampledata.w.joint);
        X = permute(reshape(obj.sampledata.x,[N,K,D]),[3 1 2]);
        obj.postdata.viterbi_joint = struct;
        obj.postdata.viterbi_joint.x = [];
        obj.postdata.viterbi_joint.score = [];
        obj.postdata.viterbi_joint.isvisible = [];
        if isinf(obj.viterbi_misscost),
          [Xbest,idx,totalcost,poslambdaused] = ChooseBestTrajectory(X,appcost,...
            'poslambda',obj.viterbi_poslambda,...
            'poslambdafac',obj.viterbi_poslambdafac,...
            'dampen',obj.viterbi_dampen);
          obj.postdata.viterbi_joint.isvisible = true(N,1);
        else
          [Xbest,isvisiblebest,idxcurr,totalcost,poslambdacurr,misscostcurr] = ...
            ChooseBestTrajectory_MissDetection(X,appcost,...
            'poslambda',obj.viterbi_poslambda,...
            'poslambdafac',obj.viterbi_poslambdafac,...
            'dampen',obj.viterbi_dampen,...
            'misscost',obj.viterbi_misscost);
          obj.postdata.viterbi_joint.isvisible = isvisiblebest;
        end
        obj.postdata.viterbi_joint.x = reshape(Xbest',[N,npts,nviews,d]);
        obj.postdata.viterbi_joint.score = totalcost;

      else

        obj.postdata.viterbi_indep = struct;
        obj.postdata.viterbi_indep.x = reshape(obj.sampledata.x(:,1,:),[N,npts,nviews,d]);
        obj.postdata.viterbi_indep.score = nan(npts,nviews);
        obj.postdata.viterbi_indep.isvisible = true(N,npts,nviews);
        
        for ipt = 1:npts,
          for ivw = 1:nviews,
            % N x nRep x npts x nviews
            appcost = -log(obj.sampledata.w.indep(:,:,ipt,ivw));
            X = permute(reshape(obj.sampledata.x(:,:,ipt,ivw,:),[N,K,2]),[3 1 2]);
            if isinf(obj.viterbi_misscost),
              [Xbest,idx,totalcost,poslambdaused] = ChooseBestTrajectory(X,appcost,...
                'poslambda',obj.viterbi_poslambda,...
                'poslambdafac',obj.viterbi_poslambdafac,...
                'dampen',obj.viterbi_dampen);
            else
              [Xbest,isvisiblebest,idxcurr,totalcost,poslambdacurr,misscostcurr] = ...
                ChooseBestTrajectory_MissDetection(X,appcost,...
                'poslambda',obj.viterbi_poslambda,...
                'poslambdafac',obj.viterbi_poslambdafac,...
                'dampen',obj.viterbi_dampen,...
                'misscost',obj.viterbi_misscost);
              obj.postdata.viterbi_indep.isvisible(:,ipt,ivw) = isvisiblebest;
            end
            obj.postdata.viterbi_indep.x(:,ipt,ivw,:) = Xbest';
            obj.postdata.viterbi_indep.score(ipt,ivw) = totalcost;
          end
        end
      end

    end
    
    function ClearComputedResults(obj)
      
      obj.postdata = struct;
      if isstruct(obj.sampledata) && isfield(obj.sampledata,'w'),
        obj.sampledata.w = struct;
      end
      if ~strcmp(obj.nativeformat,'heatmap'),
        obj.heatmapdata = [];
      end
      if ~strcmp(obj.nativeformat,'sample'),
        obj.sampledata = [];
      end

    end
    
  end
  
  methods (Static)
    
    
  end
end