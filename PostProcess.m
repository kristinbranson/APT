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
    
    % parameters for gmm fitting from heatmap
    % r_nonmax: radius of region considered in non-maximal suppression
    % thresh_perc: threshold for non-maximal suppression
    % nmixpermax: number of mixture components per local maximum found with
    % non-maximal suppression
    % minpxpermix: minimum number of pixels to have per mixture component
    gmmfitheatmap_params = struct('r_nonmax',4,'thresh_perc',99.95,...
      'nmixpermax',5,'minpxpermix',4);
    
    % sampledata will have be a struct with fields x and w
    % sampledata.x is the location of the samples, and will be a matrix of
    % size N x nRep x npts x d, corresponding to N frames, nRep
    % replicates, npts landmarks in d dimensions
    % sampledata.w is the weight of each sample, computed using KDE.
    % sampledata.w.indep is if we do KDE independently in 2-d for each
    % landmark point, while sampledata.w.joint is if we do KDE in 
    % sampledata.w.indep is N x nRep x npts x nviews and sampledata.w.joint
    % is N x nrep.
    sampledata = [];
    heatmapdata = [];
    gmmdata = [];
    postdata = struct;
    
    caldata = [];
    reconstructfun = [];
    reconstructfun_from_caldata = false;
    usegeometricerror = true;
    
    sample_viewsindependent = true;
    heatmap_viewsindependent = true;
    
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
    
    function PropagateDataReset(obj,stepname)

      if ismember(stepname,{'reconstruct','sampledata',...
          'heatmapdata'}),
        obj.ClearComputedResults();
      elseif ismember(stepname,{'viterbi'}),
        
        if isfield(obj.postdata,'viterbi_joint'),
          obj.postdata.viterbi_joint = [];
        end
        if isfield(obj.postdata,'viterbi_indep'),
          obj.postdata.viterbi_indep = [];
        end
        
      end
      
      if ismember(stepname,{'sampledata'}),
        obj.heatmapdata = [];
      end
      
      if ismember(stepname,{'heatmapdata'}),
        obj.sampledata = [];
      end
      
      if ismember(stepname,{'heatmapdata','sampledata','reconstruct'}),
        obj.gmmdata = [];
      end
      
    end
    
    function [X,x_re] = ReconstructSampleMultiView(obj)
      
      if ~isfield(obj.sampledata,'x_perview') || isempty(obj.sampledata.x_perview),
        error('Need to set sampledata.x_perview');
      end
      [N,nRep,npts,nviews,d] = size(obj.sampledata.x_perview);
      
      % X is d x N*nRep*npts
      [X,x_re] = obj.reconstructfun(permute(reshape(obj.sampledata.x_perview,[N*nRep*npts,nviews,d]),[2,3,1]));
      d = size(X,1);
      X = reshape(permute(X,[2,1]),[N,nRep,npts,d]);
      obj.sampledata.x = X;
      obj.sampledata.x_re_perview = x_re;
      
      obj.PropagateDataReset('reconstruct');
      
    end
    
    function SetCalibrationData(obj,caldata)
      
      obj.caldata = caldata;
      obj.SetReconstructFunFromCalData();
      
    end
    
    function SetReconstructFunFromCalData(obj)
      
      obj.SetReconstructFun(get_reconstruct_fcn(obj.caldata,obj.usegeometricerror),true);
      
    end
    
    function SetUseGeometricError(obj,value)
      obj.usegeometricerror = value;
      if obj.reconstructfun_from_caldata && ~isempty(obj.caldata),
        obj.SetReconstructFunFromCalData();
      end
    end
    
    function SetReconstructFun(obj,reconstructfun,reconstructfun_from_caldata)
      
      if nargin < 3,
        obj.reconstructfun_from_caldata = false;
      else
        obj.reconstructfun_from_caldata = reconstructfun_from_caldata;
      end

      obj.reconstructfun = reconstructfun;
      obj.PropagateDataReset('reconstruct');
      
    end
    
    function SetSampleViewsIndependent(obj,v)
      
      if v == obj.sample_viewsindependent,
        return;
      end
      obj.sample_viewsindependent = v;
      if strcmp(obj.nativeformat,'sample') && isfield(obj.sampledata,'x_in'),
        obj.SetSampleData(obj.x_in);
      end
      
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
      obj.PropagateDataReset('sampledata');      
      obj.sampledata = struct;
      
      if nviews > 1,
        
        if isempty(obj.reconstructfun),
          error('Reconstruction function needed');
        end

        obj.sampledata.x_in = x;
        if obj.sample_viewsindependent,
          
          grididx = cell(1,nviews);
          [grididx{:}] = deal(1:nRep);
          [grididx{:}] = ndgrid(grididx{:});
          obj.sampledata.x_perview = repmat(x(:,1,:,:,:),[1,nRep^nviews,1,1,1]);
          for viewi = 1:nviews,
            obj.sampledata.x_perview(:,:,:,viewi,:) = x(:,grididx{viewi},:,viewi,:);
          end
          nRep = nRep^nviews;
          
        else
          obj.sampledata.x_perview = x;
        end
        obj.ReconstructSampleMultiView();

      else

        obj.sampledata.x_in = [];
        obj.sampledata.x = reshape(x,[N,nRep,npts,d]);
        obj.sampledata.x_perview = [];
        obj.sampledata.x_re_perview = [];
        
      end
      obj.sampledata.w = struct;
      obj.sampledata.w.indep = [];
      obj.sampledata.w.joint = [];
      obj.nativeformat = 'sample';
      
    end

    function SetHeatmapData(obj,readscorefuns,nframes,scales)
      
      % clear out all postprocessing and gmm results
      obj.PropagateDataReset('heatmapdata');
      obj.SetJointSamples(false);
      
      obj.heatmapdata = struct;
      [npts,nviews] = size(readscorefuns);
      
      nys = nan(1,nviews);
      nxs = nan(1,nviews);
      for viewi = 1:nviews,
        scores1 = readscorefuns{1,viewi}(1);
        [nys(viewi),nxs(viewi)] = size(scores1);
      end
      
      obj.heatmapdata.readscorefuns = readscorefuns;
      obj.heatmapdata.nframes = nframes;
      obj.heatmapdata.npts = npts;
      obj.heatmapdata.nviews = nviews;
      obj.heatmapdata.nys = nys;
      obj.heatmapdata.nxs = nxs;
      
      % scales
      if nargin >= 3,
        obj.heatmapdata.scales = scales;
      else
        % by default don't scale
        obj.heatmapdata.scales = ones(nviews,d);
      end

      % grid
      obj.heatmapdata.grid = cell(2,nviews);
      for i = 1:nviews,
        [obj.heatmapdata.grid{1,i},obj.heatmapdata.grid{2,i}] = meshgrid( (1:nxs(i))*obj.heatmapdata.scales(i,1),...
          (1:nys(i))*obj.heatmapdata.scales(i,2) );
      end
      
      obj.nativeformat = 'heatmap';
      
    end
    
    function Heatmap2SampleData(obj)
      
      obj.gmmdata = repmat(struct('mu',zeros(0,2),'S',zeros(2,2,0),'w',[]),...
        [obj.heatmapdata.nframes,obj.heatmapdata.npts,obj.heatmapdata.nviews]);
      obj.sampledata = [];
      
      gmmparams = struct2paramscell(obj.gmmfitheatmap_params);
      
      for pti = 1:obj.heatmapdata.npts,
        for t = 1:obj.heatmapdata.nframes,
          for viewi = 1:obj.heatmapdata.nviews,
          
            scores = obj.heatmapdata.readscorefuns{pti,viewi}(t);
            [obj.gmmdata(t,pti,viewi).mu,obj.gmmdata(t,pti,viewi).w,...
              obj.gmmdata(t,pti,viewi).S] = ...
              GMMFitHeatmapData(scores,gmmparams{:});
          end
            
          [Xsample,Wsample,x_re_sample,x_sample] = SampleGMMCandidatePoints(reshape(obj.gmmdata(t,pti,:),[1,obj.heatmapdata.nviews]),varargin);
          nsamplescurr = numel(Wsample);
          obj.sampledata.x(t,1:nsamplescurr,pti,viewi,:) = Xsample;
          obj.sampledata.w.indep = Wsample;
          if obj.heatmapdata.nviews > 1,
            obj.sampledata.x_re_perview(t,1:nsamplescurr,pti,:,:) = x_re_sample;
            obj.sampledata.x_perview(t,1:nsamplescurr,pti,:,:) = x_sample;
          else
            obj.sampledata.x_perview = [];
            obj.sampledata.x_re_perview = [];
          end
        end
        
      end
      
    end
    
%     function SetUNetHeatmapData(obj,scores,scales)
%       
%       % scores should be nviews x T x ny x nx
%       % scales should be nviews x 2
%       
%       [T,nviews,ny,nx] = size(scores); %#ok<ASGLU>
%       if nargin >= 3,
%         [nviews2,d] = size(scales);      
%         assert(d==2 && nviews==nviews2);
%       else
%         scales = ones(nviews,2);
%       end
%       
%       % this is probably some python to matlab thing
%       hm_miny = 2;
%       hm_minx = 2;
%       
%       minscore = -1;
%       maxscore = 1;
% 
%       hmd = struct;
%       hmd.scores = (scores(:,:,hm_miny:end,hm_minx:end)-minscore)/(maxscore-minscore);
%       hmd.scales = scales;
%       obj.SetHeatMapData(hmd);
% 
%     end
    
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
      
      obj.PropagateDataReset('sampledata');

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
        obj.PropagateDataReset('viterbi');
      end
    end
    
    function res = NeedKDE(obj)
      
      res = ~isfield(obj.sampledata,'w') || ...
        ( obj.jointsamples && (~isfield(obj.sampledata.w,'joint') || isempty(obj.sampledata.w.joint) ) ) || ...
        ( ~obj.jointsamples && (~isfield(obj.sampledata.w,'indep') || isempty(obj.sampledata.w.indep) ) );

    end
    
    function RunMaxDensity_SampleData(obj)
      
      [N,K,npts,d] = size(obj.sampledata.x); %#ok<ASGLU>
            
      if obj.NeedKDE(),
        obj.ComputeKDE();
      end

      
      if obj.jointsamples,
        
        w = obj.sampledata.w.joint;
        obj.postdata.maxdensity_joint = struct;
        [obj.postdata.maxdensity_joint.score,k] = max(w,[],2);
        obj.postdata.maxdensity_joint.x = reshape(obj.sampledata.x(:,1,:,:),[N,npts,d]);
        for i = 1:N,
          obj.postdata.maxdensity_joint.x(i,:,:) = obj.sampledata.x(i,k(i),:,:);
        end
        
      else
        
        % this is different from what maxdensity did
        w = obj.sampledata.w.indep;
        obj.postdata.maxdensity_indep = struct;
        obj.postdata.maxdensity_indep.x = reshape(obj.sampledata.x(:,1,:,:),[N,npts,d]);
        obj.postdata.maxdensity_indep.score = zeros([N,1]);
        for ipt=1:npts,

          % sampledata.w.indep is N x nRep x npts
          [scorecurr,k] = max(w(:,:,ipt),[],2);
          obj.postdata.maxdensity_indep.score = obj.postdata.maxdensity_indep.score + log(scorecurr);
          for i = 1:N,
            obj.postdata.maxdensity_indep.x(i,ipt,:) = obj.sampledata.x(i,k(i),ipt,:);
          end

        end
      end
      
    end
    
    function RunMedian_SampleData(obj)
      
      [N,K,npts,d] = size(obj.sampledata.x); %#ok<ASGLU>
      % should we use the weights?
      if strcmp(obj.nativeformat,'sample') || obj.NeedKDE(),
        obj.postdata.median.x = reshape(median(obj.sampledata.x,2),[N,npts,d]);
        obj.postdata.median.score = -mean(reshape(mad(obj.sampledata.x,1,2),[N,npts*d]),2);
      else
        if obj.jointsamples,
          w = repmat(obj.sampledata.w.joint,[1,1,npts,d]);
        else
          w = repmat(obj.sampledata.w.indep,[1,1,1,d]);
        end
        obj.postdata.median.x = reshape(weighted_prctile(obj.sampledata.x,50,w,2),[N,npts,d]);
        obj.postdata.median.score = -mean(reshape(weighted_prctile(abs(obj.sampledata.x - reshape(obj.postdata.median.x,[N,1,npts,d])),50,w,2),[N,npts*d]),2);
      end
      
    end

    function RunMedian_HeatmapData(obj)
      
      d = 2;
      if obj.heatmapdata.nviews == 1,
        viewi = 1;
        obj.postdata.median.x = nan([obj.heatmapdata.nframes,obj.heatmapdata.npts,d]);
        for t = 1:obj.heatmapdata.nframes,
          for pti = 1:npts,
            scores = obj.heatmapdata.readscorefuns{pti,viewi}(t);
            for di = 1:d,
              obj.postdata.median.x(t,pti,di) = weighted_prctile(obj.heatmapdata.grid{di,viewi}(:),50,scores(:));
            end
          end
        end
      else
        
        if obj.heatmap_viewsindependent,
          x = nan([d,nviews,obj.heatmapdata.nframes,obj.heatmapdata.npts]);
          madx = nan([obj.heatmapdata.nframes,obj.heatmapdata.npts,obj.heatmapdata.nviews,d]);
          for t = 1:obj.heatmapdata.nframes,
            for pti = 1:npts,
              for viewi = 1:nviews,
                scores = obj.heatmapdata.readscorefuns{pti,viewi}(t);
                for di = 1:d,
                  x(di,viewi,t,pti) = weighted_prctile(obj.heatmapdata.grid{di,viewi}(:),50,scores(:));
                  madx(t,pti,viewi,di) = weighted_prctile(abs(obj.heatmapdata.grid{di,viewi}(:)-x(di,viewi,t,pti)),50,scores(:));
                end
              end
            end
          end
          obj.postdata.median.x = obj.reconstructfun(x);
          obj.postdata.median.score = -mean(reshape(madx,[obj.heatmapdata.nframes,obj.heatmapdata.npts*obj.heatmapdata.nviews*d]),2);
          
        else
          error('Not implemented');
        end
      end
      
    end
    
    function RunMaxDensity_HeatmapData(obj)
      
      obj.postdata.maxdensity_indep = struct;
      obj.postdata.maxdensity_indep.score = zeros([obj.heatmapdata.nframes,1]);

      d = 2;
      x = nan([obj.heatmapdata.nframes,obj.heatmapdata.npts,obj.heatmapdata.nviews,d]);
      for t = 1:obj.heatmapdata.nframes,
        maxv = nan(obj.heatmapdata.npts,obj.heatmapdata.nviews);
        for pti = 1:obj.heatmapdata.npts,
          for viewi = 1:obj.heatmapdata.nviews,
            scores = obj.heatmapdata.readscorefuns{pti,viewi}(t);
            [maxv(pti,viewi),idx] = max(scores(:));
            for di = 1:d,
              x(t,pti,viewi,di) = obj.heatmapdata.grid{di,viewi}(idx);
            end
          end
        end
        obj.postdata.maxdensity_indep.score(t) = mean(maxv(:));
      end
      
      if obj.heatmapdata.nviews > 1,
        obj.postdata.maxdensity_indep.x_perview = x;
        x = permute(reshape(x,[obj.heatmapdata.nframes*obj.heatmapdata.npts,obj.heatmapdata.nviews,2]),[3,2,1]);
        obj.postdata.maxdensity_indep.x = obj.reconstructfun(x)';
        d = size(obj.postdata.maxdensity_indep.x,1);
        obj.postdata.maxdensity_indep.x = reshape(obj.postdata.maxdensity_indep.x,[obj.heatmapdata.nframes,obj.heatmapdata.npts,d]);
      else
        obj.postdata.maxdensity_indep.x = reshape(x,[obj.heatmapdata.nframes,obj.heatmapdata.npts,d]);
      end
      
    end
    
    function ComputeKDE(obj)

      starttime = tic;
      
      % TODO: could check to see if running is necessary?
      [N,nRep,npts,d] = size(obj.sampledata.x);
      D = npts*d;
      
      if obj.jointsamples,
        
        fprintf('Computing KDE, joint\n');
          
        x = reshape(permute(obj.sampledata.x,[3,4,5,2,1]),[D,1,nRep,N]);
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
        d2 = sum( (reshape(obj.sampledata.x,[N,1,nRep,npts,d])-reshape(obj.sampledata.x,[N,nRep,1,npts,d])).^2, 5 );
        w= sum(exp( -d2/obj.kde_sigma^2/2 ),2)-1; % subtract 1 because we have included this replicate
        w = w ./ sum(w,3);
        obj.sampledata.w.indep = reshape(w,[N,nRep,npts]);
        
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

      if strcmp(obj.nativeformat,'sample'),
        if obj.NeedKDE(),
          obj.ComputeKDE();
        end
      elseif strcmp(obj.nativeformat,'heatmap'),
        if isempty(obj.sampledata),
          obj.Heatmap2SampleData();
        end
      end

      [N,K,npts,d] = size(obj.sampledata.x); 
      D = npts*d;
      
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
        obj.postdata.viterbi_joint.x = reshape(Xbest',[N,npts,d]);
        obj.postdata.viterbi_joint.score = totalcost;

      else

        obj.postdata.viterbi_indep = struct;
        obj.postdata.viterbi_indep.x = reshape(obj.sampledata.x(:,1,:),[N,npts,d]);
        obj.postdata.viterbi_indep.score = nan(npts,1);
        obj.postdata.viterbi_indep.isvisible = true(N,npts);
        
        for ipt = 1:npts,
          % N x nRep x npts
          appcost = -log(obj.sampledata.w.indep(:,:,ipt));
          X = permute(reshape(obj.sampledata.x(:,:,ipt,:),[N,K,2]),[3 1 2]);
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
            obj.postdata.viterbi_indep.isvisible(:,ipt) = isvisiblebest;
          end
          obj.postdata.viterbi_indep.x(:,ipt,:) = Xbest';
          obj.postdata.viterbi_indep.score(ipt) = totalcost;
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
    
    
    function [mu,w,S,threshcurr] = GMMFitHeatmapData(scores,varargin)
      
      % radius of region considered in non-maximal suppression
      r_nonmax = 4;
      % threshold for non-maximal suppression
      thresh_perc = 99.95;
      % number of mixture components per local maximum found with non-maximal
      % suppression
      nmixpermax = 5;
      % minimum number of pixels to have per mixture component
      minpxpermix = 4;

      [r_nonmax,thresh_perc,thresh,xgrid,ygrid,scalex,scaley,...
        nmixpermax,minpxpermix] = ...
        myparse(varargin,...
        'r_nonmax',r_nonmax,'thresh_perc',thresh_perc,'thresh',[],...
        'xgrid',[],'ygrid',[],'scalex',1,'scaley',1,...
        'nmixpermax',nmixpermax,'minpxpermix',minpxpermix);
      
      % one landmark, one view
      [ny,nx] = size(scores);
      
      if isempty(xgrid),
        [xgrid,ygrid] = meshgrid( (1:nx-1)*scalex,(1:ny-1)*scaley );
      end

      if isempty(thresh),
        tscores = scores(r_nonmax+1:end-r_nonmax,r_nonmax+1:end-r_nonmax);
        threshcurr = prctile(tscores(:),thresh_perc);
      end
      
      tscorescurr = scores;
      minscores = min(scores(:));
      
      % mayank is worried about the boundary for some reason
      tscorescurr(1:r_nonmax,:) = minscores;
      tscorescurr(end-r_nonmax+1:end,:) = minscores;
      tscorescurr(:,1:r_nonmax) = minscores;
      tscorescurr(:,end-r_nonmax+1:end) = minscores;
            
      % threshold and non-maximum suppression
      [r,c] = nonmaxsuppts(tscorescurr,r_nonmax,threshcurr);
      r = r*scaley;
      c = c*scalex;
      idxcurr = scores >= threshcurr;
            
      % set number of centers based on number of maxima found
      k0 = numel(r);
      k = k0+min(numel(r)*nmixpermax,floor(nnz(idxcurr)/minpxpermix));
      
      % initialize clustering
      start = nan(k,2);
      start(1:k0,:) = [c(:),r(:)];
      X = [xgrid(idxcurr),ygrid(idxcurr)];
      d = min(dist2(X,start(1:k0,:)),[],2);
      for j = k0+1:k,
        [~,maxj] = max(d);
        start(j,:) = X(maxj,:);
        d = min(d,dist2(X,start(j,:)));
      end
      
      % gmm fitting
      [mu,S,~,post] = mygmm(X,k,...
        'Start',start,...
        'weights',scores(idxcurr)-threshcurr);
      w = sum(bsxfun(@times,scores(idxcurr),post),1)';
      w(w<0.1) = 0.1;
            
      nanmu = any(isnan(mu),2);
      mu = mu(~nanmu,:);
      w = w(~nanmu);
      S = S(:,:,~nanmu);

    end
    
    function [Xsample,Wsample,x_re_sample,x_sample] = SampleGMMCandidatePoints(gmmdata,varargin)
      
      x_re_sample = [];
      x_sample = [];
      
      nsamples = 50;
      nsamples_neighbor = 0;
      discountneighbor = .05;
      [nsamples,nsamples_neighbor,discountneighbor,gmmdata_prev,gmmdata_next,...
        reconstructfun] = ...
        myparse(varargin,'nsamples',nsamples,...
        'nsamples_neighbor',nsamples_neighbor,'discountneighbor',discountneighbor,...
        'gmmdata_prev',[],'gmmdata_next',[],'reconstructfun',[]);
      
      nviews = numel(gmmdata);
      if nviews > 1,
        d = 3;
      else
        d = 2;
      end
      Kperview = nan(1,nviews);
      Kperview_prev = zeros(1,nviews);
      Kperview_next = zeros(1,nviews);
      for viewi = 1:nviews,
        Kperview(viewi) = numel(gmmdata(viewi).w);
      end
      maxnsamples = prod(Kperview);
      
      if ~isempty(gmmdata_prev),
        for viewi = 1:nviews,
          Kperview_prev(viewi) = numel(gmmdata_prev(viewi).w);
        end
      end
      if ~isempty(gmmdata_next),
        for viewi = 1:nviews,
          Kperview_next(viewi) = numel(gmmdata_next(viewi).w);
        end
      end
      % can choose from current and previous frames, but at least one view
      % must be from previous frame
      maxnsamples_prev = prod(Kperview+Kperview_prev)-maxnsamples;
      maxnsamples_next = prod(Kperview+Kperview_next)-maxnsamples;
      
      nsamples_next = min(nsamples_neighbor,maxnsamples_next);
      nsamples_prev = min(nsamples_neighbor,maxnsamples_prev);
      nsamples_curr = min(nsamples+2*nsamples_neighbor-nsamples_next-nsamples_prev,maxnsamples);

      % compute score for each possible sample from the current frame
      if nviews > 1,
        
        X = nan(d,maxnsamples);
        x = nan(2,nviews,maxnsamples);
        x_re = nan(2,nviews,maxnsamples);
        p_re = nan(nviews,maxnsamples);
        w = nan(nviews,maxnsamples);
        for i = 1:maxnsamples,
          sub = ind2subv(Kperview,i);
          Scurr = nan(2,2,nviews);
          for viewi = 1:nviews,
            x(:,viewi,i) = gmmdata(viewi).mu(:,sub(viewi));
            Scurr(:,:,viewi) = gmmdata(viewi).S(:,:,sub(viewi));
            w(viewi,i) = gmmdata(viewi).w(sub(viewi));
          end
          [X(:,i),x_re(:,:,i)] = reconstructfun(x(:,:,i),Scurr);
          for viewi = 1:nviews,
            p_re(viewi,i) = mvnpdf(x_re(:,viewi,i)',x(:,viewi,i)',Scurr(:,:,viewi));
          end
        end

        W = prod(p_re,1).*prod(w,1);
        
      else
        
        X = gmmdata.mu;
        W = gmmdata.w;
        
      end
      
      % select based on weights
      W = W / sum(W);
      [~,order] = sort(W(:),1,'descend');
      Xsample = X(:,order(1:nsamples_curr));
      Wsample = W(order(1:nsamples_curr));
      if nviews > 1,
        x_sample = x(:,:,order(1:nsamples_curr));
        x_re_sample = x_re(:,:,order(1:nsamples_curr));
      end
      
      % select samples using detections from previous frame
      if nsamples_prev > 0,
        
        % compute score for each possible sample including one sample from
        % the previous view

        if nviews > 1,
        
          X = nan(d,maxnsamples_prev);
          x = nan(2,nviews,maxnsamples_prev);
          x_re = nan(2,nviews,maxnsamples_prev);
          p_re = nan(nviews,maxnsamples_prev);
          w = nan(nviews,maxnsamples_prev);
          Kperview_prevcurr = Kpreview_prev + Kperview_curr;
          i = 0;
          for ii = 1:prod(Kperview+Kperview_prev),
            sub = ind2subv(Kperview_prevcurr,ii);
            if ~any(sub <= Kperview_prev),
              continue;
            end
            i = i + 1;
            Scurr = nan(2,2,nviews);
            for viewi = 1:nviews,
              if sub(viewi) > Kperview_prev(viewi),
                subcurr = sub(viewi)-Kperview_prev(viewi);
                x(:,viewi,i) = gmmdata(viewi).mu(:,subcurr);
                Scurr(:,:,viewi) = gmmdata(viewi).S(:,:,subcurr);
                w(viewi,i) = gmmdata(viewi).w(subcurr);
              else
                x(:,viewi,i) = gmmdata_prev(viewi).mu(:,sub(viewi));
                Scurr(:,:,viewi) = gmmdata(viewi).S(:,:,sub(viewi));
                w(viewi,i) = gmmdata(viewi).w(sub(viewi)) * discountneighbor;
              end
            end
            [X(:,i),x_re(:,:,i)] = reconstructfun(x(:,:,i),Scurr);
            for viewi = 1:nviews,
              p_re(viewi,i) = mvnpdf(x_re(:,viewi,i)',x(:,viewi,i)',Scurr(:,:,viewi));
            end
          end

          W = prod(p_re,1).*prod(w,1);
        
        else
        
          X = gmmdata_prev.mu;
          W = gmmdata_prev.w;
          
        end
      
        % select based on weights
        W = W / sum(W);
        [~,order] = sort(W(:),1,'descend');
        Xsample_prev = X(:,order(1:nsamples_curr));
        Wsample_prev = W(order(1:nsamples_curr));
        if nviews > 1,
          x_sample_prev = x(:,:,order(1:nsamples_prev));
          x_re_sample_prev = x_re(:,:,order(1:nsamples_prev));
        end
      
        Xsample = cat(2,Xsample,Xsample_prev);
        Wsample = cat(2,Wsample,Wsample_prev);
        if nviews > 1,
          x_re_sample = cat(3,x_re_sample,x_re_sample_prev);
          x_sample = cat(3,x_sample,x_sample_prev);
        end
        
      end
      
      % select samples using detections from next frame
      if nsamples_next > 0,
        
        % compute score for each possible sample including one sample from
        % the previous view

        if nviews > 1,
        
          X = nan(d,maxnsamples_next);
          x = nan(2,nviews,maxnsamples_next);
          x_re = nan(2,nviews,maxnsamples_next);
          p_re = nan(nviews,maxnsamples_next);
          w = nan(nviews,maxnsamples_next);
          Kperview_nextcurr = Kpreview_next + Kperview_curr;
          i = 0;
          for ii = 1:prod(Kperview+Kperview_next),
            sub = ind2subv(Kperview_nextcurr,ii);
            if ~any(sub <= Kperview_next),
              continue;
            end
            i = i + 1;
            Scurr = nan(2,2,nviews);
            for viewi = 1:nviews,
              if sub(viewi) > Kperview_next(viewi),
                subcurr = sub(viewi)-Kperview_next(viewi);
                x(:,viewi,i) = gmmdata(viewi).mu(:,subcurr);
                Scurr(:,:,viewi) = gmmdata(viewi).S(:,:,subcurr);
                w(viewi,i) = gmmdata(viewi).w(subcurr);
              else
                x(:,viewi,i) = gmmdata_next(viewi).mu(:,sub(viewi));
                Scurr(:,:,viewi) = gmmdata(viewi).S(:,:,sub(viewi));
                w(viewi,i) = gmmdata(viewi).w(sub(viewi)) * discountneighbor;
              end
            end
            [X(:,i),x_re(:,:,i)] = reconstructfun(x(:,:,i),Scurr);
            for viewi = 1:nviews,
              p_re(viewi,i) = mvnpdf(x_re(:,viewi,i)',x(:,viewi,i)',Scurr(:,:,viewi));
            end
          end

          W = prod(p_re,1).*prod(w,1);
        
        else
        
          X = gmmdata_next.mu;
          W = gmmdata_next.w;
          
        end
      
        % select based on weights
        W = W / sum(W);
        [~,order] = sort(W(:),1,'descend');
        Xsample_next = X(:,order(1:nsamples_curr));
        Wsample_next = W(order(1:nsamples_curr));
        if nviews > 1,
          x_sample_next = x(:,:,order(1:nsamples_next));
          x_re_sample_next = x_re(:,:,order(1:nsamples_next));
        end
      
        Xsample = cat(2,Xsample,Xsample_next);
        Wsample = cat(2,Wsample,Wsample_next);
        if nviews > 1,
          x_re_sample = cat(3,x_re_sample,x_re_sample_next);
          x_sample = cat(3,x_sample,x_sample_next);
        end
        
      end
      
    end
    
  end
end