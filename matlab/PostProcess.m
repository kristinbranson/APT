classdef PostProcess < handle
   
  properties
    pts2run = []; % row vec into 1..npts, used when pts independent. use at own risk
  end
  properties (GetAccess=public,SetAccess=public)
    
    algorithm = 'maxdensity';
    jointsamples = true;
    
    ncores = [];
    
    % parameters for KDE
    kde_sigma = 5;
    
    % parameters for Viterbi
    viterbi_poslambda = [];
    viterbi_poslambdafac = [];
    viterbi_dampen = 0.5;
    viterbi_misscost = [];
    viterbi_grid_acradius = [];
    
    % obsolete
    % parameters for gmm fitting from heatmap
    % r_nonmax: radius of region considered in non-maximal suppression
    % thresh_perc: threshold for non-maximal suppression
    % nmixpermax: number of mixture components per local maximum found with
    % non-maximal suppression
    % minpxpermix: minimum number of pixels to have per mixture component
%     gmmfitheatmap_params = struct('r_nonmax',4,'thresh_perc',99.95,...
%       'nmixpermax',5,'minpxpermix',4);
    
    % parameters for sampling from a GMM
%     gmmsample_params = struct('nsamples',50,'nsamples_neighbor',0,'discount_neighbor',.05);
    
    % sampledata will have be a struct with fields x and w
    % sampledata.x is the location of the samples, and will be a matrix of
    % size N x nRep x npts x d, corresponding to N frames, nRep
    % replicates, npts landmarks in d dimensions
    % sampledata.w is the weight of each sample, computed using KDE.
    % sampledata.w.indep is if we do KDE independently in 2-d for each
    % landmark point, while sampledata.w.joint is if we do KDE in 
    % sampledata.w.indep is N x nRep x npts x nviews and sampledata.w.joint
    % is N x nrep.
    
    caldata = [];
    sigma_reconstruct = [];
    usegeometricerror = true;
    
    sample_viewsindependent = true;
    sample_pointsindependent = false;
    nsamples_viewsindependent = [];
    heatmap_viewsindependent = true;
    heatmap_pointsindependent = true;
    heatmap_nsamples = 25;
    heatmap_sample_algorithm = 'localmaxima'; % either 'localmaxima' or 'gmm'
    heatmap_sample_thresh_prctile = 99.5; % used for 'localmaxima'
    heatmap_sample_r_nonmax = 2; % used for 'localmaxima'
    heatmap_lowthresh = 0;
    heatmap_highthresh = 1;
    
    preloadMovies = false;
    movFiles = {};
    % currently only has .frm field labeling N frames of native data, ie
    % tblMFT.frm(n) gives the true movie frame for native-index n
    tblMFT = []; 
    % [nview] struct array of "cropped trx". See SetHeatmapData. nview>1 
    % unsupported. Only 'content' fields .x, .y etc are used; metadata
    % fields like .nframes, .firstframe etc in general will be incorrect
    % since this is a cropped trx
    trx = []; 

    % [nviews x 2]. heatmap_origin(iview,:) is the [x y] or [c r] in the
    % heatmap coord system that gets mapped to trx.x,trx,y in the
    % real/movie coord system
    heatmap_origin = nan;
    
    % Either 'sample' or 'heatmap'
    nativeformat = 'sample';
    
    sampledata = [];
    kdedata = [];
    heatmapdata = [];
    %gmmdata = [];
    postdata = struct;
    
    nviews = [];
    % Number of frames of native data available. Currently assumed to be a 
    % "block" of consecutive frames; the block need not start at
    % movieframe=1.
    N = []; 
    d = [];
    npts = [];
    
    reconstructfun = [];
    projectfun = [];
    reconstructfun_from_caldata = false;
    sigma_reconstruct_used = [];
    
    readframes = {};
    
    frames = []; % Either [], or [nframes] native samples to consider
    radius_frames = 100; 
    dilated_frames = []; % Either [], or [N] logical vector labeling native samples to consider. Generated from .frames via dilation
    
  end
  
  properties (Dependent, Access = protected)
    
    ismultiview;
    isframes;
    
  end
  
  properties (Constant)
    
    algorithms_jointmatters = {'maxdensity','viterbi','viterbi_miss'};
    algorithms = {'maxdensity','median','viterbi'};
    
  end
    
  methods
    
    function obj = PostProcess()
      
      obj.ncores = feature('numcores');
      if obj.ncores == 1,
        obj.ncores = 0;
      end
      
%       assert(mod(nargin,2)==0);
%       for i = 1:2:numel(varargin),
%         obj.(varargin{i}) = varargin{i+1};
%       end
      
    end
    
    function val = get.ismultiview(obj)
      
      val = ~isempty(obj.nviews) && obj.nviews > 1;
      
    end    
    
    function val = get.isframes(obj)
      
      val = ~isempty(obj.frames);
      
    end
    
    function SetNCores(obj,val)
      obj.ncores = val;
    end    
    
    function SetKDESigma(obj,val)
      
      if val == obj.kde_sigma,
        return;
      end
      obj.PropagateDataReset('kde');
      obj.kde_sigma = val;
      
    end
    
    function SetHeatmapLowThresh(obj,val)
      
      obj.PropagateDataReset('postheatmap');
      obj.heatmap_lowthresh = val;      
      
    end    
    
    function SetHeatmapHighThresh(obj,val)
      
      obj.PropagateDataReset('postheatmap');
      obj.heatmap_highthresh = val;      
      
    end
    
    function SetHeatmapNumSamples(obj,val)
      obj.PropagateDataReset('heatmap_nsamples');
      obj.heatmap_nsamples = val;        
    end
    
    function SetHeatmapSampleAlg(obj,val)
      if ~any(strcmp(val,{'gmm' 'localmaxima'}))
        error('Invalid heatmap sample algorithm.');
      end
      
      obj.PropagateDataReset('heatmap_sample_algorithm');
      obj.heatmap_sample_algorithm = val;        
    end
    
    function SetAlgorithm(obj,val)
      
      obj.algorithm = val;
      
    end    
    
    function SetSigmaReconstruct(obj,val)
      
      if isempty(obj.sigma_reconstruct) && isempty(val),
        return;
      elseif ~isempty(obj.sigma_reconstruct) && ~isempty(val) && obj.sigma_reconstruct == val,
        return;
      elseif isempty(obj.sigma_reconstruct) && ~isempty(val) && ~isempty(obj.sigma_reconstruct_used) && val == obj.sigma_reconstruct_used,
        return;
      end
      if obj.ismultiview,
        obj.PropagateDataReset('processed_sample');
      end
      obj.sigma_reconstruct = val;
      
    end
    
    function SetUseGeometricError(obj,val)
      
      if obj.usegeometricerror == val,
        return;
      end
      if obj.ismultiview,
        obj.PropagateDataReset('processed_sample')
      end
      obj.usegeometricerror = val;
      
    end
    
    function PropagateDataReset(obj,stepname)

      switch stepname,
        
        case 'postheatmap',
          if strcmp(obj.nativeformat,'heatmap'),
            
            obj.PropagateDataReset('median');
            obj.PropagateDataReset('maxdensity');
            obj.PropagateDataReset('sampledata');
            
          end
          
        case 'heatmapdata',
          
          obj.heatmapdata = [];
          obj.PropagateDataReset('median');
          obj.PropagateDataReset('maxdensity');
          obj.PropagateDataReset('sampledata');
          
        case 'sampledata',
          
          obj.sampledata = [];
          obj.PropagateDataReset('kde');
        
        case 'kde',
          
          if ~isempty(obj.kdedata),
          
            obj.kdedata = [];
            if obj.ismultiview || strcmp(obj.nativeformat,'sample'),
              obj.PropagateDataReset('median');
              obj.PropagateDataReset('maxdensity');
            end
            obj.PropagateDataReset('viterbi');
          end
          
        case {'median','maxdensity','viterbi'},
          
          fns = fieldnames(obj.postdata);
          doremove = ~cellfun(@isempty,regexp(fns,['^',stepname],'once'));
          if any(doremove),
            obj.postdata = rmfield(obj.postdata,fns(doremove));
          end

        case {'heatmap_nsamples','heatmap_sample_algorithm','heatmap_sample_thresh_prctile','heatmap_sample_r_nonmax'},

          obj.PropagateDataReset('sampledata');
          
        case 'processed_sample',
          
          obj.sampledata.x_perview = [];
          obj.sampledata.x = [];
          obj.sampledata.w = [];
          obj.sampledata.x_re_perview = [];

          obj.PropagateDataReset('kde');
          
        case 'trx',
          
          obj.PropagateDataReset('postheatmap');
          obj.PropagateDataReset('processed_sample');
          
        otherwise
          error('Not implemented');
      end
      
%       if ismember(stepname,{'reconstruct','sampledata',...
%           'heatmapdata'}),
%         obj.ClearComputedResults();
%       elseif ismember(stepname,{'viterbi'}),
%         
%         if isfield(obj.postdata,'viterbi_joint'),
%           obj.postdata.viterbi_joint = [];
%         end
%         if isfield(obj.postdata,'viterbi_indep'),
%           obj.postdata.viterbi_indep = [];
%         end
%         
%       end
%       
%       if ismember(stepname,{'sampledata'}),
%         obj.heatmapdata = [];
%       end
%       
%       if ismember(stepname,{'heatmapdata'}),
%         obj.sampledata = [];
%       end
%       
%       if ismember(stepname,{'heatmapdata','sampledata','reconstruct'}),
%         obj.gmmdata = [];
%       end
      
    end
    
    function val = NeedPostData(obj)
      
      algorithm = obj.GetPostDataAlgName();
      val = ~isfield(obj.postdata,algorithm) || ...
        isempty(obj.postdata.(algorithm)) || ...
        ~isfield(obj.postdata.(algorithm),'x') || ...
        isempty(obj.postdata.(algorithm).x);
      
    end
    
    function postdata = GetAllPostData(obj)
      
      if obj.NeedPostData(),
        obj.run();
      end
      
      postdata = obj.postdata;
      
    end
    
    function res = GetPostData(obj)
      if obj.NeedPostData(),
        obj.run();
      end
      res = obj.postdata.(obj.algorithm);
      
    end
    
    function Independent2JointSamples(obj)
      
      % fit GMM so that total number of samples remains the same
      nRep = size(obj.sampledata.x_in,2);
      if isempty(obj.nsamples_viewsindependent),
        nsamples_total = nRep;
      else
        nsamples_total = obj.nsamples_viewsindependent;
      end
      nsamples_perview = max(1,round(nsamples_total.^(1/obj.nviews)));
      nsamples_total = nsamples_perview^obj.nviews;

      % mu is nsamples_perview x d x N x npts x nviews
      % S is d x d x nsamples_perview x N x npts x nviews
      % prior is nsamples_perview x N x npts x nviews if independent,
      % nsamples_perview x N x nviews if joint
      jointpoints = ~obj.sample_pointsindependent;
      if jointpoints,
        w_npts = 1;
      else
        w_npts = obj.npts;
      end
      
      if obj.isframes,

        %[obj.N,nsamples,obj.npts,obj.nviews,d_in] = size(obj.sampledata.x_in); %#ok<ASGLU>
        [mu,prior,S] = PostProcess.GMMFitSamples(obj.sampledata.x_in(obj.dilated_frames,:,:,:,:),nsamples_perview,'jointpoints',jointpoints);
        Nframes = nnz(obj.dilated_frames);
        
      else
        
        [mu,prior,S] = PostProcess.GMMFitSamples(obj.sampledata.x_in,nsamples_perview,'jointpoints',jointpoints);
        Nframes = obj.N;
        
      end
      
      %[mu,prior,S] = PostProcess.GMMFitSamples(obj.sampledata.x_in,nsamples_perview,'jointpoints',jointpoints);
      
      grididx = cell(1,obj.nviews);
      [grididx{:}] = deal(1:nsamples_perview);
      [grididx{:}] = ndgrid(grididx{:});

      d_in = size(obj.sampledata.x_in,5);
      xrep = nan([Nframes,nsamples_total,obj.npts,obj.nviews,d_in]);
      wrep = ones([nsamples_total,Nframes,w_npts]);
      Srep = nan([d_in,d_in,obj.nviews,nsamples_total,Nframes,obj.npts]);
 
      for viewi = 1:obj.nviews,
        xrep(:,:,:,viewi,:) = permute(mu(grididx{viewi},:,:,:,viewi),[3,1,4,5,2]);
        Srep(:,:,viewi,:,:,:) = S(:,:,grididx{viewi},:,:,viewi);
        if jointpoints,
          wrep = prior(grididx{viewi},:,viewi).*wrep;
        else
          wrep = prior(grididx{viewi},:,:,viewi).*wrep;
        end
      end

      Ntotal = nsamples_total*Nframes*obj.npts;
      % x_re is d_in x nviews x (nsamples_total*N*npts)
      [X,x_re] = obj.reconstructfun(reshape(permute(xrep,[5,4,1,2,3]),[d_in,obj.nviews,Ntotal]),...
        'S',reshape(Srep,[d_in,d_in,obj.nviews,Ntotal]));

      if isempty(obj.sigma_reconstruct),
        err = abs(reshape(x_re,[d_in,obj.nviews*Ntotal])-reshape(permute(xrep,[5,4,1,2,3]),[d_in,obj.nviews*Ntotal]));
        sigma_reconstruct = PostProcess.EstimateReconstructionSigma(permute(err,[4,1,2,3,5]));
      else
        sigma_reconstruct = obj.sigma_reconstruct;
      end
      obj.sigma_reconstruct_used = sigma_reconstruct;
      Sigma_reconstruct = eye(2)*sigma_reconstruct^2;
      
      % xrep is N x nsamples_total x npts x nviews x d_in
      % permuting gives us d_in x nviews x N x nsamples_total x npts
      p_re = mvnpdf_2d(reshape(x_re,[d_in,obj.nviews*Ntotal]),reshape(permute(xrep,[5,4,1,2,3]),[d_in,obj.nviews*Ntotal]),Sigma_reconstruct+reshape(permute(Srep,[1,2,3,5,4,6]),[d_in,d_in,obj.nviews*Ntotal]));
      % p_re is 1 x nviews * Ntotal
      p_re = reshape(p_re,[obj.nviews,Ntotal]);
      p_re = reshape(prod(p_re,1),[Nframes,nsamples_total,obj.npts]);
      % p_re is now N x nsamples_total x npts
      p_re = permute(p_re,[2,1,3]);
      if jointpoints,
        p_re = prod(p_re,3);
      end
      % p_re is now nsamples_total x N
      %z_re = sum(p_re,1);
      %p_re = p_re ./ z_re;
      wreptotal = wrep.*p_re;
      z = sum(wreptotal,1);
      z(z==0) = 1;
      wreptotal = wreptotal ./ z;
      
      obj.sampledata.tmp_independent2joint = struct;
      if obj.isframes,
        
        obj.sampledata.tmp_independent2joint.p_re = nan([nsamples_total,obj.N]);
        obj.sampledata.tmp_independent2join.p_re(:,obj.dilated_frames) = p_re;
        obj.sampledata.tmp_independent2joint.pgau = nan([nsamples_total,obj.N]);
        obj.sampledata.tmp_independent2join.pgau(:,obj.dilated_frames) = pgau;
        
      else
        obj.sampledata.tmp_independent2joint.p_re = p_re;
        obj.sampledata.tmp_independent2joint.p_gau = wrep;
      end

      % X should be d_out x Ntotal
      % x_re should be d_in x nviews x Ntotal
      d_out = size(X,1);
      % want X to be N x samples x npts x d_out
      X = reshape(X,[d_out,Nframes,nsamples_total,obj.npts]);
      X = permute(X,[2,3,4,1]);
      % and x_re to be N x nsamples x npts x nview x d_in
      x_re = reshape(x_re,[d_in,obj.nviews,Nframes,nsamples_total,obj.npts]);
      x_re = permute(x_re,[3,4,5,2,1]);
      
        
%         % sanity check, one data point
%         pti = 4;
%         n = 87;
%         samplei = [2,4];
%         
%         % make sure xrep matches
%         tmpx = nan(d_in,obj.nviews);
%         for viewi = 1:obj.nviews,
%           tmpx(:,viewi) = squeeze(mu(samplei(viewi),:,n,pti,viewi));
%         end
% 
%         sampleirep = sub2indv(zeros(1,obj.nviews)+nsamples_perview,samplei);
%         tmpxrep = squeeze(xrep(n,sampleirep,pti,:,:))';
%         assert(all(tmpx(:) == tmpxrep(:)));
%         
%         % make sure prior matches
%         tmpprior = nan(1,obj.nviews);
%         for viewi = 1:obj.nviews,
%           tmpprior(viewi) = prior(samplei(viewi),n,pti,viewi);
%         end        
%         assert(wrep(sampleirep,n,pti)==prod(tmpprior));
%         
%         % make sure X matches
%         [tmpX,tmpx_re] = obj.reconstructfun(tmpxrep);
%         assert(all(squeeze(X(n,sampleirep,pti,:))==tmpX));
%         
%         % make sure x_re matches
%         assert(all(all(tmpx_re == squeeze(x_re(n,sampleirep,pti,:,:))')));
%         
%         % make sure Srep matches
%         for viewi = 1:obj.nviews,
%           assert(all(all(S(:,:,samplei(viewi),n,pti,viewi) == Srep(:,:,viewi,sampleirep,n,pti))));
%         end
% 
%         % mvnpdf_2d(reshape(x_re,[d_in,obj.nviews*Ntotal]),reshape(permute(xrep,[5,4,1,2,3]),[d_in,obj.nviews*Ntotal]),Sigma_reconstruct+reshape(Srep,[d_in,d_in,obj.nviews*Ntotal]));
%         tmp_p_re = nan(1,obj.nviews);
%         for viewi = 1:obj.nviews,
%           tmp_p_re(viewi) = mvnpdf_2d(tmpx_re(:,viewi),tmpx(:,viewi),Sigma_reconstruct+S(:,:,samplei(viewi),n,pti,viewi));
%         end
%         assert(prod(tmp_p_re)==p_re(sampleirep,n,pti));
%         
%         tmpwreptotal = prod(tmpprior)*prod(tmp_p_re);
%         assert(tmpwreptotal==wreptotal(sampleirep,n,pti).*z(1,n,pti));
%         
      
      if obj.isframes,
        
        obj.sampledata.x_perview = nan([obj.N,nsamples_total,obj.npts,obj.nviews,d_in]);
        obj.sampledata.x_perview(obj.dilated_frames,:,:,:,:) = xrep;
        obj.sampledata.x = nan([obj.N,nsamples_total,obj.npts,d_out]);
        obj.sampledata.x(obj.dilated_frames,:,:,:) = X;
        obj.sampledata.w = nan([obj.N,nsamples_total,w_npts]);
        obj.sampledata.w(obj.dilated_frames,:,:) = permute(wreptotal,[2,1,3]);
        obj.sampledata.x_re_perview = nan([obj.N,nsamples_total,obj.npts,obj.nviews,d_in]);
        obj.sampledata.x_re_perview(obj.dilated_frames,:,:,:,:) = x_re;
         
      else
        obj.sampledata.x_perview = xrep;
        obj.sampledata.x = X;
        obj.sampledata.w = permute(wreptotal,[2,1,3]); % w is N x maxnsamples x w_npts
        obj.sampledata.x_re_perview = x_re;
      end
      
    end

    function Heatmap2JointSamples(obj)
      
      if isempty(obj.nsamples_viewsindependent),
        nsamples_total = obj.heatmap_nsamples;
      else
        nsamples_total = obj.nsamples_viewsindependent;
      end
      nsamples_perview = max(1,round(nsamples_total^(1/obj.nviews)));
      obj.Heatmap2SampleData('nsamples_perview',nsamples_perview);
      
    end

    
    
    function [X,x_re] = ReconstructSampleMultiView(obj)
      
      if ~isfield(obj.sampledata,'x_perview') || isempty(obj.sampledata.x_perview),
        error('Need to set sampledata.x_perview');
      end
      [N,nRep,npts,nviews,d_in] = size(obj.sampledata.x_perview); %#ok<ASGLU>
      
      if obj.isframes,
        % these branches/options appear switched
        Nframes = N;
      else
        Nframes = nnz(obj.dilated_frames);
      end
      
      % X is d x N*nRep*npts
      Ntotal = Nframes*nRep*obj.npts;
      if obj.isframes,
        [X,x_re] = obj.reconstructfun(reshape(permute(obj.sampledata.x_perview(obj.dilated_frames,:,:,:,:),[5,4,1,2,3]),[d_in,obj.nviews,Ntotal]));
      else
        [X,x_re] = obj.reconstructfun(reshape(permute(obj.sampledata.x_perview,[5,4,1,2,3]),[d_in,obj.nviews,Ntotal]));
      end
%       [X,x_re] = obj.reconstructfun(reshape(permute(xrep,[5,4,1,2,3]),[d_in,obj.nviews,Ntotal]),...
%         'S',reshape(Srep,[d_in,d_in,obj.nviews,Ntotal]));
      
      d_out = size(X,1);
      X = reshape(permute(X,[2,1]),[Nframes,nRep,obj.npts,d_out]);
      % x_re is d_in x nviews x Ntotal
      % and want it to be N x nsamples x npts x nview x d_in
      x_re = reshape(x_re,[d_in,obj.nviews,Nframes,nRep,obj.npts]);
      x_re = permute(x_re,[3,4,5,2,1]);
      
      % compute reprojection error
      err = abs(x_re - obj.sampledata.x_perview);
      
      if isempty(obj.sigma_reconstruct),
        median_err = median(err(:));
        sigma_reconstruct = median_err*1.4836;
      else
        sigma_reconstruct = obj.sigma_reconstruct;
      end
      obj.sigma_reconstruct_used = sigma_reconstruct;
      % err is N x nsamples x npts x nviews x d_in
      p_re = prod(reshape(normpdf(err(:)/sigma_reconstruct),[Nframes,nRep,obj.npts*obj.nviews*d_in]),3);
      % p_re is N x nsamples
      p_re = p_re ./ sum(p_re,2);

      if obj.isframes,
        obj.sampledata.x = nan([obj.N,nRep,obj.npts,d_out]);
        obj.sampledata.x(obj.dilated_frames,:,:,:) = X;
        obj.sampledata.w = nan([obj.N,nRep]);
        obj.sampledata.w(obj.dilated_frames,:) = p_re;
        obj.sampledata.x_re_perview = nan([obj.N,nRep,obj.npts,obj.nviews,d_in]);
        obj.sampledata.x_re_perview(obj.dilated_frames,:,:,:,:) = x_re;
      else
        obj.sampledata.x = X;
        obj.sampledata.x_re_perview = x_re;
        obj.sampledata.w = p_re;
      end
      %obj.PropagateDataReset('reconstruct');
      
    end
    
    function SetCalibrationData(obj,caldata)
      
      if obj.ismultiview,
        obj.PropagateDataReset('processed_sample');
      end
      obj.caldata = caldata;
      obj.SetReconstructFunFromCalData();
      
    end
    
    function SetMovieFiles(obj,movfiles,preload)
      
      assert(numel(movfiles) == obj.nviews);
      if nargin >= 3,
        obj.preloadMovies = preload;
      end
      obj.movFiles = movfiles;
      
      obj.readframes = cell(1,obj.nviews);
      for i = 1:obj.nviews,
        obj.readframes{i} = get_readframe_fcn(obj.movFiles{i},'preload',obj.preloadMovies);
      end
      
    end
    
    function v = CanReadMovieFrames(obj)
      
      v = ~isempty(obj.movFiles) && ~isempty(obj.readframes) && ~isempty(obj.tblMFT);
      
    end
    
    function SetMFTInfo(obj,tblMFT)
      
      n = size(tblMFT,1);
      assert(obj.N == n);
      obj.tblMFT = tblMFT;
      
    end
    
    function SetReconstructFunFromCalData(obj)
      
      obj.SetReconstructFun(get_reconstruct_fcn(obj.caldata,obj.usegeometricerror),true);
      obj.SetProjectFun(get_project_fcn(obj.caldata));
      
    end
        
    function SetReconstructFun(obj,reconstructfun,reconstructfun_from_caldata)
      
      if nargin < 3,
        obj.reconstructfun_from_caldata = false;
      else
        obj.reconstructfun_from_caldata = reconstructfun_from_caldata;
      end

      obj.reconstructfun = reconstructfun;
      obj.PropagateDataReset('processed_sample');
      
    end
    
    function SetProjectFun(obj,projectfun)

      obj.projectfun = projectfun;
      
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
    
    function InitializeSampleData(obj)
      
      obj.sampledata = struct;
      obj.sampledata.x_in = [];
      obj.sampledata.w_in = [];
      obj.sampledata.x = [];
      obj.sampledata.x_perview = [];
      obj.sampledata.x_re_perview = [];
      obj.sampledata.w = [];
      
    end
    
    function SetDataSize(obj)
      
      switch obj.nativeformat,
        
        case 'sample',
          [obj.N,nsamples,obj.npts,obj.nviews,d_in] = size(obj.sampledata.x_in); %#ok<ASGLU>
          
        case 'heatmap',
          
          obj.N = obj.heatmapdata.N;
          [obj.npts,obj.nviews] = size(obj.heatmapdata.readscorefuns);
        
      end
      
    end

    
    function SetSampleData(obj,x,viewsindependent,pointsindependent,frames)

      [N,nRep,npts,nviews,d] = size(x);
            
      frm = (1:N)';
      obj.tblMFT = table(frm);

      % special case if nviews == 1
      if nviews > 1 && d == 1, %#ok<*PROPLC>
        d = nviews;
        nviews = 1;
        x = reshape(x,[N,nRep,npts,nviews,d]);
      end
            
      if nargin >= 3 && ~isempty(viewsindependent),
        obj.sample_viewsindependent = viewsindependent;
      end
      if nargin >= 4 && ~isempty(viewsindependent),
        obj.sample_pointsindependent = pointsindependent;
      end
      
      if nviews == 1,
        datatype = 'singleview';
      elseif obj.sample_viewsindependent,
        datatype = 'multiview_independent';
      else
        datatype = 'multiview_joint';
      end
            
      % clear out all postprocessing and heatmap results
      obj.PropagateDataReset('sampledata');
      
      obj.InitializeSampleData();
      obj.nativeformat = 'sample';
      obj.sampledata.x_in = x;
      
      obj.SetDataSize();
      
      if nargin >= 5 && ~isempty(frames),
        obj.frames = frames;
      end
      
      switch datatype,
        
        case 'singleview',

          obj.sampledata.x_perview = x;
          obj.sampledata.x = reshape(x,[N,nRep,npts,d]);
          
        case 'multiview_joint',

          if isempty(obj.reconstructfun),
            error('Reconstruction function needed');
          end
          
          obj.sampledata.x_perview = x;
          obj.ReconstructSampleMultiView();
          
        case 'multiview_independent',
          
          obj.Independent2JointSamples();
          
      end 
      
    end
    
    function SetFrames(obj,frames)
      
      if isempty(frames),
        obj.frames = [];
        obj.dilated_frames = [];
      else
        obj.frames = frames;
        obj.dilated_frames = false(1,obj.N);
        obj.dilated_frames(obj.frames) = true;
        se = strel(ones(1,2*obj.radius_frames+1));
        obj.dilated_frames = imdilate(obj.dilated_frames,se);
      end
      
      obj.PropagateDataReset('postheatmap');
      %warning('TODO: propagate changes forward');
      
    end
    
    function [hfig,wreptotal] = PlotSampleScores(obj,varargin)

      [hfig,plotkde] = myparse(varargin,'hfig',[],'plotkde',false);
      
      isviewsindep = ~plotkde && obj.sample_viewsindependent && obj.ismultiview;
      
      if isempty(hfig),
        hfig = figure; 
      else
        figure(hfig);
      end
      
      if isviewsindep,
        p_re = obj.sampledata.tmp_independent2joint.p_re;
        wrep = obj.sampledata.tmp_independent2joint.p_gau;
        w_npts = size(wrep,3);
        nwts = 3;
      else
        nwts = 1;
        if plotkde,
          if obj.NeedKDE()
            obj.ComputeKDE();
          end
          if obj.jointsamples,
            wreptotal = obj.kdedata.joint;
          else
            wreptotal = obj.kdedata.indep;
          end
          w_npts = size(wreptotal,3);
        else
          w_npts = 1;
        end
      end
      if ~plotkde,
        wreptotal = permute(obj.sampledata.w,[2,1,3]);
      end
      
      clf;
      
      hax = createsubplots(nwts,w_npts,.05);
      hax = reshape(hax,[nwts,w_npts]);

      
      [wreptotal,order] = sort(wreptotal,1,'descend');
      
      if isviewsindep,
        for pti = 1:w_npts,
          for ni = 1:obj.N,
            p_re(:,ni,pti) = p_re(order(:,ni,pti),ni,pti);
            wrep(:,ni,pti) = wrep(order(:,ni,pti),ni,pti);
          end
        end
      end

      for pti = 1:w_npts,
      
        if isviewsindep,
          imagesc(log(p_re(:,:,pti)),'Parent',hax(1,pti));
          title(hax(1,pti),'Log reconstruction likelihood')
          imagesc(log(wrep(:,:,pti)),'Parent',hax(2,pti))
          title(hax(2,pti),'Log Gaussian prior');
        end
        imagesc(log(wreptotal(:,:,pti)),'Parent',hax(nwts,pti))
        title(hax(nwts,pti),'Log total likelihood');
        for i = 1:nwts,
          colorbar('peer',hax(i,pti));
        end

      end
      colormap jet;
      
    end
    
    function hfig = PlotReconstructionSamples(obj,n,varargin)
      
      [hfig,scoresigma,p,plotkde] = myparse(varargin,...
        'hfig',[],'scoresigma',ones(1,obj.nviews),'p',2.5,'plotkde',false);
      
      if plotkde && obj.NeedKDE(),
        obj.ComputeKDE();
      end
      
      if isempty(hfig),
        hfig = figure;
      else
        figure(hfig);
      end
      
      clf;
      minx = nan(1,obj.nviews);
      maxx = nan(1,obj.nviews);
      miny = nan(1,obj.nviews);
      maxy = nan(1,obj.nviews);
      for viewi = 1:obj.nviews,
        tmpx = obj.sampledata.x_perview(:,:,:,viewi,1);
        tmp = prctile(tmpx(:),[p,100-p]);
        minx(viewi) = tmp(1);
        maxx(viewi) = tmp(2);
        tmpy = obj.sampledata.x_perview(:,:,:,viewi,2);
        tmp = prctile(tmpy(:),[p,100-p]);
        miny(viewi) = tmp(1);
        maxy(viewi) = tmp(2);
      end

      ismovie = obj.CanReadMovieFrames();
%       if ismovie,
%         minx = ones(1,obj.nviews);
%         miny = ones(1,obj.nviews);
%         maxx = ones(1,obj.nviews);
%         maxy = ones(1,obj.nviews);
%         for viewi = 1:obj.nviews,
%           sz = size(obj.readframes{viewi}(1));
%           maxx(viewi) = sz(2);
%           maxy(viewi) = sz(1);
%         end
%       else
        minx = floor(minx-2*scoresigma);
        maxx = ceil(maxx+2*scoresigma);
        miny = floor(miny-2*scoresigma);
        maxy = ceil(maxy+2*scoresigma);
%       end
      
      nsamples_total = size(obj.sampledata.x,2);

      if plotkde,
        if obj.jointsamples,
          w = obj.kdedata.joint;
        else
          w = obj.kdedata.indep;
        end
      else
        w = obj.sampledata.w;
      end
      w_npts = size(w,3);
          
      
      colors = lines(obj.npts);
      for viewi = 1:obj.nviews,
        subplot(1,obj.nviews+1,viewi);
        hold off;
        [gridx,gridy] = meshgrid(minx(viewi):maxx(viewi),miny(viewi):maxy(viewi));
        scoreim = zeros([maxy(viewi)-miny(viewi)+1,maxx(viewi)-minx(viewi)+1,3]);
        for pti = 1:obj.npts,
%           if w_npts == 1,
            w_pti = 1;
%           else
%             w_pti = 1;
%           end
          score = zeros(maxy(viewi)-miny(viewi)+1,maxx(viewi)-minx(viewi)+1);
                    
          for samplei = 1:nsamples_total,
            score = score + reshape(w(n,samplei,w_pti)*mvnpdf([gridx(:),gridy(:)],squeeze(obj.sampledata.x_perview(n,samplei,pti,viewi,:))',eye(2)*scoresigma(viewi)^2),size(score));
          end
          score = score / max(score(:));
          scoreim = scoreim + score.*reshape(colors(pti,:),[1,1,3]);
          %scatter(obj.sampledata.x_perview(n,:,pti,viewi,1),obj.sampledata.x_perview(n,:,pti,viewi,2),wrep(:,n,pti)/max(wrep(:,n,pti))*100,'o');
        end
        
        if ismovie,
          
          frameim = obj.readframes{viewi}(obj.tblMFT.frm(n));
          if isa(frameim,'uint8'),
            frameim = double(frameim)/255;
          elseif isa(frameim,'uint16'),
            frameim = double(frameim)/(2^16-1);
          end
          if size(frameim,3) ~= 3,
            frameim = repmat(frameim,[1,1,3]);
          end
          frameim = padgrab(frameim,0,miny(viewi),maxy(viewi),minx(viewi),maxx(viewi),1,3);
          
          annim = min(frameim*1+scoreim,1);
          image([minx(viewi),maxx(viewi)],[miny(viewi),maxy(viewi)],annim);
          
        else
        
          image([minx(viewi),maxx(viewi)],[miny(viewi),maxy(viewi)],min(scoreim,1));

        end
        axis image;
        title(sprintf('View %d',viewi));
      end
      subplot(1,obj.nviews+1,obj.nviews+1);
      for pti = 1:obj.npts,
%         if w_npts == 1,
%           w_pti = 1;
%         else
          w_pti = 1;
%         end

        doplot = w(n,:,w_pti) > 0;
        h = scatter3(obj.sampledata.x(n,doplot,pti,1),obj.sampledata.x(n,doplot,pti,2),obj.sampledata.x(n,doplot,pti,3),w(n,doplot,w_pti)*100,'o');
        set(h,'CData',colors(pti,:));
        hold on;
      end
      axis equal;
      title('3d reconstruction');
        
    end
    
    
    function hfig = PlotSampleDistribution(obj,n,varargin)
      
      [hfig,scoresigma,p,plotkde] = myparse(varargin,'hfig',[],'scoresigma',ones(1,obj.nviews),'p',2.5,'plotkde',false);
          
      if plotkde && obj.NeedKDE(),
        obj.ComputeKDE();
      end
      
      if isempty(hfig),
        hfig = figure;
      else
        figure(hfig);
      end
      
      clf;
      nrplot = 1;
      if strcmp(obj.nativeformat,'heatmap'),
        nrplot = 2;
      end
      hax = createsubplots(nrplot,obj.nviews,.05);
      hax = reshape(hax,[nrplot,obj.nviews]);
      
      minx = nan(1,obj.nviews);
      maxx = nan(1,obj.nviews);
      miny = nan(1,obj.nviews);
      maxy = nan(1,obj.nviews);
      for viewi = 1:obj.nviews,
        tmpx = obj.sampledata.x_perview(:,:,:,viewi,1);
        tmp = prctile(tmpx(:),[p,100-p]);
        minx(viewi) = tmp(1);
        maxx(viewi) = tmp(2);
        tmpy = obj.sampledata.x_perview(:,:,:,viewi,2);
        tmp = prctile(tmpy(:),[p,100-p]);
        miny(viewi) = tmp(1);
        maxy(viewi) = tmp(2);
      end
      
      ismovie = obj.CanReadMovieFrames();
%       if ismovie,
%         minx = ones(1,obj.nviews);
%         miny = ones(1,obj.nviews);
%         maxx = ones(1,obj.nviews);
%         maxy = ones(1,obj.nviews);
%         for viewi = 1:obj.nviews,
%           sz = size(obj.readframes{viewi}(1));
%           maxx(viewi) = sz(2);
%           maxy(viewi) = sz(1);
%         end
%       else
        minx = floor(minx-2*scoresigma);
        maxx = ceil(maxx+2*scoresigma);
        miny = floor(miny-2*scoresigma);
        maxy = ceil(maxy+2*scoresigma);
%       end
      
      nsamples_total = size(obj.sampledata.x,2);

      if plotkde,
        if obj.jointsamples,
          w = obj.kdedata.joint(n,:,:);
        else
          w = obj.kdedata.indep(n,:,:);
        end
      else
        if isfield(obj.sampledata,'w') && ~isempty(obj.sampledata.w),
          w = obj.sampledata.w(n,:,:);
        else
          w = ones([1,nsamples_total])/nsamples_total;
        end
      end
      w_npts = size(w,3);
          
      
      colors = lines(obj.npts);
      for viewi = 1:obj.nviews,
        
        for axi = 1:nrplot,
          axes(hax(axi,viewi)); %#ok<LAXES>
          hold off;
          
          if axi == 1,
            
            if obj.IsTrx(),
              [gridx,gridy] = meshgrid(1:obj.heatmapdata.nxs(viewi),1:obj.heatmapdata.nys(viewi));
            else
              [gridx,gridy] = meshgrid(minx(viewi):maxx(viewi),miny(viewi):maxy(viewi));
            end
            scoreim = 0;
            for pti = 1:obj.npts,
              if w_npts == 1,
                w_pti = 1;
              else
                w_pti = pti;
              end
              
              if obj.IsTrx(),
                score = zeros(obj.heatmapdata.nys(viewi),obj.heatmapdata.nys(viewi));
              else
                score = zeros(maxy(viewi)-miny(viewi)+1,maxx(viewi)-minx(viewi)+1);
              end
              
              idxgood = find(all(~isnan(obj.sampledata.x_perview(n,:,pti,viewi,:)),5));
              
              if obj.IsTrx(),
                mu = obj.TransformByTrx(...
                  obj.sampledata.x_perview(n,idxgood,pti,viewi,:),...
                  obj.trx(viewi),obj.heatmap_origin(viewi,:),n);
              else
                mu = obj.sampledata.x_perview(n,idxgood,pti,viewi,:);
              end
              
              for sampleii = 1:numel(idxgood),
                samplei = idxgood(sampleii);
                score = score + reshape(w(1,samplei,w_pti)*mvnpdf([gridx(:),gridy(:)],squeeze(mu(1,samplei,1,1,:))',eye(2)*scoresigma(viewi)^2),size(score));
              end
              score = score / max(score(:));
              scoreim = scoreim + score.*reshape(colors(pti,:),[1,1,3]);
              %scatter(obj.sampledata.x_perview(n,:,pti,viewi,1),obj.sampledata.x_perview(n,:,pti,viewi,2),wrep(:,n,pti)/max(wrep(:,n,pti))*100,'o');
            end
            
          else
            
            if obj.IsTrx(),
              scoreim = zeros([obj.heatmapdata.nys(viewi),obj.heatmapdata.nys(viewi),3]);
            else
              scoreim = zeros([maxy(viewi)-miny(viewi)+1,maxx(viewi)-minx(viewi)+1,3]);
            end
              
            for pti = 1:obj.npts,
              score = obj.ReadHeatmapScore(pti,viewi,n);

              if ~obj.IsTrx(),
                score = score(miny(viewi):maxy(viewi),minx(viewi):maxx(viewi));
              end
              score = score / max(score(:));
              scoreim = scoreim + score.*reshape(colors(pti,:),[1,1,3]);
            end
            
          end
          
          if ismovie,
            
            frameim = obj.readframes{viewi}(obj.tblMFT.frm(n));
            if isa(frameim,'uint8'),
              frameim = double(frameim)/255;
            elseif isa(frameim,'uint16'),
              frameim = double(frameim)/(2^16-1);
            end
            if size(frameim,3) ~= 3,
              frameim = repmat(frameim,[1,1,3]);
            end
            if obj.IsTrx(),
              frameim = CropImAroundTrx(frameim,obj.trx(viewi).x(n),obj.trx(viewi).y(n),obj.trx(viewi).theta(n),obj.heatmap_origin(viewi,1),obj.heatmap_origin(viewi,2));
              frameim = frameim(1:obj.heatmapdata.nys(viewi),1:obj.heatmapdata.nxs(viewi),:);
              xlim = [1,obj.heatmapdata.nxs(viewi)];
              ylim = [1,obj.heatmapdata.nys(viewi)];
            else
              frameim = padgrab(frameim,0,miny(viewi),maxy(viewi),minx(viewi),maxx(viewi),1,3);
              xlim = [minx(viewi),maxx(viewi)];
              ylim = [miny(viewi),maxy(viewi)];
            end
            
            annim = min(frameim*.5+scoreim,1);
            image(xlim,ylim,annim);
            
          else
            
            image([minx(viewi),maxx(viewi)],[miny(viewi),maxy(viewi)],min(scoreim,1));
            
          end
          axis image;
          if axi == 1,
            title(sprintf('View %d, samples',viewi));
          else
            title(sprintf('View %d, heatmap',viewi));
          end
%           hold on;
%           for pti = 1:obj.npts,
%             h = scatter(obj.sampledata.x_perview(n,:,pti,viewi,1),obj.sampledata.x_perview(n,:,pti,viewi,2),obj.sampledata.w(n,:,pti,viewi)/max(obj.sampledata.w(n,:,pti,viewi))*100,'+');
%             set(h,'CData',colors(pti,:)*.75,'LineWidth',2);
%           end

          
        end
      end
    end
    
    function hfig = PlotReprojectionSamples(obj,n,varargin)
      
      [hfig,minalpha,plotsamplenumber] = myparse(varargin,'hfig',[],'minalpha',.1,'plotsamplenumber',false);
      
      if isempty(hfig),
        hfig = figure;
      else
        figure(hfig);
      end
      
      assert(obj.nviews > 1);
      
      clf;
      colors = lines(obj.npts);
      
      minw = squeeze(min(obj.sampledata.w(n,:,:),[],2));
      maxw = squeeze(max(obj.sampledata.w(n,:,:),[],2));
     
      nsamples_total = size(obj.sampledata.x,2);

      ismovie = obj.CanReadMovieFrames();
      
      for viewi = 1:obj.nviews,
        hax = subplot(1,obj.nviews+1,viewi);
        hold off;
        if ismovie,
          frameim = obj.readframes{viewi}(obj.tblMFT.frm(n));
          image(frameim); axis image; hold on;
        end

        for pti = 1:obj.npts,
          if obj.sample_pointsindependent,
            w_pti = pti;
          else
            w_pti = 1;
          end
          for samplei = 1:nsamples_total,
            w = obj.sampledata.w(n,samplei,w_pti);
            alpha = (w-minw(w_pti))/(maxw(w_pti)-minw(w_pti))*(1-minalpha)+minalpha;
            h = scatter(obj.sampledata.x_perview(n,samplei,pti,viewi,1),obj.sampledata.x_perview(n,samplei,pti,viewi,2),'o');
            set(h,'CData',colors(pti,:),'MarkerFaceColor',colors(pti,:),'MarkerEdgeColor','none','MarkerFaceAlpha',alpha);
            if samplei == 1 && pti == 1,
              hold(hax,'on');
            end
%             h = scatter(obj.sampledata.x_re_perview(n,samplei,pti,viewi,1),obj.sampledata.x_re_perview(n,samplei,pti,viewi,2),'s');
            set(h,'CData',colors(pti,:),'MarkerFaceColor',colors(pti,:),'MarkerEdgeColor','none','MarkerFaceAlpha',alpha);
            patch([obj.sampledata.x_re_perview(n,samplei,pti,viewi,1),obj.sampledata.x_perview(n,samplei,pti,viewi,1)],...
              [obj.sampledata.x_re_perview(n,samplei,pti,viewi,2),obj.sampledata.x_perview(n,samplei,pti,viewi,2)],[0,0,0],'EdgeColor',colors(pti,:),'EdgeAlpha',alpha);
            if plotsamplenumber,
              text(obj.sampledata.x_perview(n,samplei,pti,viewi,1),obj.sampledata.x_perview(n,samplei,pti,viewi,2),num2str(samplei),'Color',colors(pti,:)*.7,...
                'HorizontalAlignment','center','VerticalAlignment','middle');
            end
          end
        end
        axis equal;
        axisalmosttight;
        title(sprintf('View %d',viewi));
      end
      subplot(1,obj.nviews+1,obj.nviews+1);
      for pti = 1:obj.npts,
        if obj.sample_pointsindependent,
          w_pti = pti;
        else
          w_pti = 1;
        end

        for samplei = 1:nsamples_total,
          h = scatter3(obj.sampledata.x(n,samplei,pti,1),obj.sampledata.x(n,samplei,pti,2),obj.sampledata.x(n,samplei,pti,3),'o');
          if pti == 1 && samplei == 1,
            hold on;
          end
          w = obj.sampledata.w(n,samplei,w_pti);
          set(h,'CData',colors(pti,:),'MarkerFaceColor',colors(pti,:),'MarkerEdgeColor','none','MarkerFaceAlpha',(w-minw(w_pti))/(maxw(w_pti)-minw(w_pti))*(1-minalpha)+minalpha);
        end
      end
      axis equal;
      title('3d reconstruction');
        
    end
    
    function hfig = PlotReprojection(obj,n,varargin)
      
      [hfig] = myparse(varargin,'hfig',[]);
      
      if isempty(hfig),
        hfig = figure;
      else
        figure(hfig);
      end
      
      clf;
      colors = lines(obj.npts);
      
      ismovie = obj.CanReadMovieFrames();
      alg = obj.GetPostDataAlgName();
      
      hax = nan(1,obj.nviews);
      for viewi = 1:obj.nviews,
        hax(viewi) = subplot(1,obj.nviews+1,viewi);
        hold off;
        if ismovie,
          frameim = obj.readframes{viewi}(obj.tblMFT.frm(n));
          image(frameim); axis image; hold on;
        end
        title(sprintf('%s, view %d',alg,viewi),'Interpreter','none');
      end
      
      for pti = 1:obj.npts,
          
        x = squeeze(obj.postdata.(alg).x(n,pti,:));
        x_re = obj.projectfun(x);
        if size(obj.postdata.(alg).sampleidx,2) == 1,
          k = squeeze(obj.postdata.(alg).sampleidx(n,1,:));
        else
          k = squeeze(obj.postdata.(alg).sampleidx(n,pti,:));
        end
        x_perview = nan([obj.nviews,size(obj.sampledata.x_perview,5),numel(k)]);
        for ki = 1:numel(k),
          x_perview(:,:,ki) = obj.sampledata.x_perview(n,k(ki),pti,:,:);
        end
        for viewi = 1:obj.nviews,
          h1 = plot(hax(viewi),x_re(1,viewi),x_re(2,viewi),'+','Color',colors(pti,:));
          h2 = plot(hax(viewi),squeeze(x_perview(viewi,1,:)),squeeze(x_perview(viewi,2,:)),'o','Color',colors(pti,:));
          plot(hax(viewi),[squeeze(x_perview(viewi,1,:)),repmat(x_re(1,viewi),[numel(k),1])]',...
            [squeeze(x_perview(viewi,2,:)),repmat(x_re(2,viewi),[numel(k),1])]','-','Color',colors(pti,:));
        end
      end
      legend([h1,h2],{'Reprojection','Sample'});
      
      if ~ismovie,
        for viewi = 1:obj.nviews,
          axis(hax(viewi));
          axis equal;
          axisalmosttight;
        end
      end
      
      subplot(1,obj.nviews+1,obj.nviews+1);
      for pti = 1:obj.npts,
        plot3(obj.postdata.(alg).x(n,pti,1),obj.postdata.(alg).x(n,pti,2),obj.postdata.(alg).x(n,pti,3),'+','Color',colors(pti,:));
        if pti == 1,
          hold on;
        end
      end
      axis equal;
      axisalmosttight;
      grid on;
      title('3d reconstruction');
      xlabel('x');
      ylabel('y');
      zlabel('z');
        
    end
    
    function hfig = PlotTimepoint(obj,n,varargin)
      
      [hfig,markerparams,plottext] = myparse(varargin,'hfig',[],'markerparams',{},'plottext',false);
      
      if isempty(hfig),
        hfig = figure;
      else
        figure(hfig);
      end
      
      clf;
      colors = lines(obj.npts);
      
      ismovie = obj.CanReadMovieFrames();
      alg = obj.GetPostDataAlgName();
      
      assert(obj.nviews==1);

      viewi = 1;
      if ismovie,
        frameim = obj.readframes{viewi}(obj.tblMFT.frm(n));
        if size(frameim,3) == 1,
          frameim = repmat(frameim,[1,1,3]);
        end
        image(frameim); axis image; 
      end
      hold on;
      title(sprintf('%s, view %d, t = %d',alg,viewi,n),'Interpreter','none');
      
      for pti = 1:obj.npts,
        plot(obj.postdata.(alg).x(n,pti,1),obj.postdata.(alg).x(n,pti,2),'+','Color',colors(pti,:),markerparams{:});
        if plottext,
          text(obj.postdata.(alg).x(n,pti,1),obj.postdata.(alg).x(n,pti,2),num2str(pti),'Color',colors(pti,:),'HorizontalAlignment','center','VerticalAlignment','middle');
        end
      end

      if ~ismovie,
        axis equal;
        axisalmosttight;
      end
        
    end
    
    function SetHeatmapData(obj,readscorefuns,N,scales,trx,frames)
      %
      %
      % readscorefuns: [npts x nviews] cell array of fcn handles with sig
      %     hm = fcn(n). 
      %   n here is the native sample index, ie n==1 in general does not
      %   correspond to movie frame 1, ie fcn has a frame offset baked-in.\
      %
      % N: The number of native samples that are safe to read. Fcns in
      %   readscorefuns will be called with indices 1..n.
      %
      % scales: sets discretization gridsize for heatmaps
      %
      % trx: "cropped trx" with data vectors cropped to native samples, ie
      %   trx.x(n) gives the trx for native sample n.
      %
      % frames: [nframes] vector of indices into native samples for frames
      %   to consider
      
      obj.nativeformat = 'heatmap';
      
      % clear out all postprocessing and gmm results
      obj.PropagateDataReset('heatmapdata');
      obj.SetJointSamples(false);
      
      obj.heatmapdata = struct;
      [npts,nviews] = size(readscorefuns); %#ok<ASGLU>
      d = 2;
      
%       obj.tblMFT = struct;
%       obj.tblMFT.frm = (1:N)';

      obj.heatmapdata.readscorefuns = readscorefuns;

      if nargin >= 5,
        obj.trx = trx;
        %firstframe = trx.firstframe;
      else
        obj.trx = [];
      end
      if ~isempty(obj.trx) && nviews > 1,
        error('Not implemented');
        %firstframe = 1;
      end
            
      nys = nan(1,nviews);
      nxs = nan(1,nviews);
      for viewi = 1:nviews,
        scores1 = obj.ReadHeatmapScore(1,viewi,1);
        [nys(viewi),nxs(viewi)] = size(scores1);
      end
      
      obj.heatmapdata.N = N;
      obj.heatmapdata.nys = nys;
      obj.heatmapdata.nxs = nxs;
      
      % scales
      if nargin >= 3,
        obj.heatmapdata.scales = scales;
      else
        % by default don't scale
        obj.heatmapdata.scales = ones(nviews,d);
      end
      
      if obj.IsTrx(),
        obj.heatmap_origin = cat(2,ceil((obj.heatmapdata.nxs(:)+1)/2),...
                                   ceil((obj.heatmapdata.nys(:)+1)/2));
      end
      
      % grid
      obj.heatmapdata.grid = cell(1,nviews);
      for i = 1:nviews,
        [xgrid,ygrid] = meshgrid( (1:nxs(i))*obj.heatmapdata.scales(i,1),...
          (1:nys(i))*obj.heatmapdata.scales(i,2) );
         obj.heatmapdata.grid{i} = cat(2,xgrid(:),ygrid(:));
      end
      
      obj.SetDataSize();
      
      obj.heatmap_viewsindependent = true;
      obj.heatmap_pointsindependent = true;
      
      if nviews == 1,
        datatype = 'singleview';
      else
        datatype = 'multiview_independent';
      end
      
      if nargin >= 5 && ~isempty(frames),
        obj.SetFrames(frames);
      end

                  
      switch datatype,
        
        case 'singleview',

          % won't create sampledata right away, in case we don't need it
          
        case 'multiview_independent',
          
          obj.Heatmap2JointSamples();
          
      end 
      
      
    end
    
    function hm = ReadHeatmapScore(obj,pti,viewi,n) % obj const
    
      hm = obj.heatmapdata.readscorefuns{pti,viewi}(n);
      hm = min(max( (hm-obj.heatmap_lowthresh)/(obj.heatmap_highthresh-obj.heatmap_lowthresh), 0), 1);
      
    end

    function [hms,idxs] = ReadHeatmapScores(obj,pti,viewi,ns,varargin)
      
      [issparse] = myparse(varargin,'issparse',false);
      
      hms = cell(size(ns));
      idxs = cell(size(ns));

      readscorefun = obj.heatmapdata.readscorefuns{pti,viewi};
      lowthresh = obj.heatmap_lowthresh;
      highthresh = obj.heatmap_highthresh;
      parfor(i = 1:numel(ns),obj.ncores)
        hm = feval(readscorefun,ns(i));
        hm = min(max( (hm-lowthresh)/(highthresh-lowthresh), 0), 1);
        if issparse,
          idxcurr = hm>0;
          idxs{i} = idxcurr;
          hm = hm(idxcurr);
        end
        hms{i} = hm;
      end
      
    end

    
    function v = IsTrx(obj)
      v = ~isempty(obj.trx);
    end
    
    function n = GetPostDataAlgName(obj,n,jointsamples)
      
      if nargin < 2,
        n = obj.algorithm;
      end
      if strcmp(n,'viterbi'),
        if ~isinf(obj.viterbi_misscost),
          n = [n,'_miss'];
        end
      end
      if ismember(n,PostProcess.algorithms_jointmatters),
        if nargin < 3,
          jointsamples = obj.jointsamples;
        end
        if jointsamples,
          n = [n,'_joint'];
        else 
          n = [n,'_indep'];
        end
      end
      
    end
    
    function Heatmap2SampleData(obj,varargin)
      [nsamples_perview,pts2run] = myparse(varargin,...
        'nsamples_perview',obj.heatmap_nsamples,...
        'pts2run',obj.pts2run...
        );
      if isempty(obj.pts2run)
        pts2run = 1:obj.npts;
      end
      
      d_in = 2;

      isS = ismember(obj.heatmap_sample_algorithm,{'gmm','localmaxima'}); % always true?
      
      if obj.isframes,
        Nframes = nnz(obj.dilated_frames);
        frameidx = find(obj.dilated_frames);
      else
        Nframes = obj.N;
      end
      
      % mu is nsamples_perview x d x N x npts x nviews
      % S is d x d x nsamples_perview x N x npts x nviews
      % prior is nsamples_perview x N x npts x nviews
      mu = nan([nsamples_perview,d_in,Nframes,obj.npts,obj.nviews]);
      if isS,
        S = nan([d_in,d_in,nsamples_perview,Nframes,obj.npts,obj.nviews]);
      end
      prior = nan([nsamples_perview,Nframes,obj.npts,obj.nviews]);
      totalweight = nan([Nframes,obj.npts,obj.nviews]);
      
      chunksize = 24;
      
      %tic;
      
      for viewi = 1:obj.nviews,
        for pti = pts2run(:)',
          fprintf(1,'Heatmap2SampData pt %d\n',pti);
          for n0 = 1:chunksize:Nframes,
            fprintf(1,'Heatmap2SampData n0 %d\n',n0);

            n1 = min(Nframes,n0+chunksize-1);
            ncurr = n1-n0+1;
            %hms = cell(1,ncurr);
            
            if obj.isframes,
              nscurr = frameidx(n0:n0+ncurr-1);
            else
              nscurr = n0:n0+ncurr-1;
            end
            
            switch obj.heatmap_sample_algorithm,
              case 'gmm',
                %xs = cell(1,ncurr);
                
                [hms,xs] = obj.ReadHeatmapScores(pti,viewi,nscurr,'issparse',true);
                totalweight(n0:n1,pti,viewi) = cellfun(@(x) sum(x(:)),hms);
                for ni = 1:numel(hms),
                  %n = n0 + ni - 1;
                  %hm = obj.ReadHeatmapScore(pti,viewi,n);
                  %idxcurr = hm > 0;
                  %hms{ni} = hm(idxcurr);
                  %xs{ni} = obj.heatmapdata.grid{viewi}(idxcurr,:);
                  xs{ni} = obj.heatmapdata.grid{viewi}(xs{ni},:);
                end
                mucurr = nan([nsamples_perview,d_in,ncurr]);
                Scurr = nan([d_in,d_in,nsamples_perview,ncurr]);
                priorcurr = nan([nsamples_perview,ncurr]);
                
                parfor(ni = 1:ncurr,obj.ncores),
                %for ni=1:ncurr
                  disp(ni);
                  x = reshape(xs{ni},[1,size(xs{ni},1),1,1,2]);
                  [mucurr(:,:,ni),priorcurr(:,ni),Scurr(:,:,:,ni)] = ...
                    PostProcess.GMMFitSamples(x,nsamples_perview,'weights',hms{ni}','jointpoints',false);
                end
                mu(:,:,n0:n1,pti,viewi) = mucurr;
                S(:,:,:,n0:n1,pti,viewi) = Scurr;
                prior(:,n0:n1,pti,viewi) = priorcurr;
              case 'localmaxima',

                [hms] = obj.ReadHeatmapScores(pti,viewi,nscurr);
%                 for ni = 1:ncurr,
%                   n = n0 + ni - 1;
%                   hms{ni} = obj.ReadHeatmapScore(pti,viewi,n);
%                 end
                mucurr = nan([nsamples_perview,d_in,ncurr]);
                priorcurr = nan([nsamples_perview,ncurr]);
                Scurr = nan([d_in,d_in,nsamples_perview,ncurr]);
                totalweightcurr = nan([1,ncurr]);
                
                params = {...
                  'thresh_prctile',obj.heatmap_sample_thresh_prctile,...
                  'r_nonmax',obj.heatmap_sample_r_nonmax,...
                  'grid',obj.heatmapdata.grid{viewi}};
                
                parfor(ni = 1:ncurr,obj.ncores),
                  [mucurr(:,:,ni),priorcurr(:,ni),Scurr(:,:,:,ni),~,totalweightcurr(ni)] = ...
                    PostProcess.LocalMaximaSamples(hms{ni},nsamples_perview,params{:}); %#ok<PFBNS>
                end
                mu(:,:,n0:n1,pti,viewi) = mucurr;
                prior(:,n0:n1,pti,viewi) = priorcurr;
                S(:,:,:,n0:n1,pti,viewi) = Scurr;
                totalweight(n0:n1,pti,viewi) = totalweightcurr;
                
            end
          end
        end
      end
      
      obj.InitializeSampleData();
      
      if obj.nviews == 1,

        % [obj.N,nsamples,obj.npts,obj.nviews,d_in] = size(obj.sampledata.x_in); 

        if obj.isframes,
          obj.sampledata.x_in = nan([obj.N,nsamples_perview,obj.npts,obj.nviews,d_in]);
          obj.sampledata.x_in(obj.dilated_frames,:,:,:,:) = permute(mu,[3,1,4,5,2]);
          obj.sampledata.w_in = nan([obj.N,nsamples_perview,obj.npts,obj.nviews]);
          obj.sampledata.w_in(obj.dilated_frames,:,:,:) = permute(prior,[2,1,3,4]);
          obj.sampledata.z_in = nan([obj.N,obj.npts,obj.nviews]);
          obj.sampledata.z_in(obj.dilated_frames,:,:) = totalweight;

        else
          obj.sampledata.x_in = permute(mu,[3,1,4,5,2]);
          obj.sampledata.w_in = permute(prior,[2,1,3,4]);
          obj.sampledata.z_in = totalweight;
        end
        obj.sampledata.x_perview = obj.sampledata.x_in;
        obj.sampledata.x = obj.sampledata.x_in;
        obj.sampledata.w = obj.sampledata.w_in;
        obj.sampledata.z = obj.sampledata.z_in;

        if obj.IsTrx(),
          viewi = 1;
          obj.sampledata.x_in_trx = obj.sampledata.x_in;
          obj.sampledata.x_in = PostProcess.UntransformByTrx(...
            obj.sampledata.x_in,obj.trx(viewi),obj.heatmap_origin(viewi,:));
          obj.sampledata.x_perview = obj.sampledata.x_in;
          obj.sampledata.x = permute(obj.sampledata.x_in,[1,2,3,5,4]);
        end
        
        % TODO .IsCrop()?
        
      else
      
        % mu is currently nsamples x d x N x npts x nviews
        % make it d x nviews x nsamples x N x npts
        mu = permute(mu,[2,5,1,3,4]);
        % S is d x d x nsamples x N x npts x nviews
        % make is d x d x nviews x nsamples x N x npts
        if isS,
          S = permute(S,[1,2,6,3,4,5]);
        end
        % prior is nsamples x N x npts x nviews
        % make it nviews x nsamples x N x npts
        prior = permute(prior,[4,1,2,3]);
        
        grididx = cell(1,obj.nviews);
        [grididx{:}] = deal(1:nsamples_perview);
        [grididx{:}] = ndgrid(grididx{:});
        
        nsamples_total = nsamples_perview^obj.nviews;
        
        xrep = nan([d_in,obj.nviews,nsamples_total,Nframes,obj.npts]);
        wrep = ones([nsamples_total,Nframes,obj.npts]);
        if isS,
          Srep = nan([d_in,d_in,obj.nviews,nsamples_total,Nframes,obj.npts]);
        end
        %xrep = nan([obj.N,nsamples_total,obj.npts,obj.nviews,d_in]);
        %wrep = ones([nsamples_total,obj.N,obj.npts]);
        %if isS,
        %  Srep = nan([d_in,d_in,obj.nviews,nsamples_total,obj.N,obj.npts]);
        %end        
        
        for viewi = 1:obj.nviews,
          xrep(:,viewi,:,:,:) = mu(:,viewi,grididx{viewi},:,:);
          if isS,
            Srep(:,:,viewi,:,:,:) = S(:,:,viewi,grididx{viewi},:,:);
          end
          wrep = permute(prior(viewi,grididx{viewi},:,:),[2,3,4,1]).*wrep;
        end
        
        isrealsample = wrep > 0;
        Ntotal = nsamples_total*Nframes*obj.npts;
        xrep = reshape(xrep,[d_in,obj.nviews,Ntotal]);
        xrep = xrep(:,:,isrealsample);
        if isS,
          Srep = reshape(Srep,[d_in,d_in,obj.nviews,Ntotal]);
          Srep = Srep(:,:,:,isrealsample);
        end
        wrep = wrep(isrealsample);
        
%         sampleidx = repmat((1:nsamples_total)',[1,obj.N,obj.npts]);
%         timepoint = repmat(1:obj.N,[nsamples_total,1,obj.npts]);
%         pt = repmat(reshape(1:obj.npts,[1,1,obj.npts]),[nsamples_total,obj.N,1]);
%         sampleidx = sampleidx(isrealsample);
%         timepoint = timepoint(isrealsample);
%         pt = pt(isrealsample);

        Ntotal1 = nnz(isrealsample);
        % x_re is d_in x nviews x (nsamples_total*N*npts)
        % input uv is [d,nviews,N]
        % S is [d,d,nviews,N]
        if isS,
          [X,x_re] = obj.reconstructfun(xrep,'S',Srep);
%           [X,x_re] = obj.reconstructfun(reshape(permute(xrep,[5,4,1,2,3]),[d_in,obj.nviews,Ntotal]),...
%             'S',reshape(Srep,[d_in,d_in,obj.nviews,Ntotal]));
        else
          [X,x_re] = obj.reconstructfun(xrep);
%           [X,x_re] = obj.reconstructfun(reshape(permute(xrep,[5,4,1,2,3]),[d_in,obj.nviews,Ntotal]));
        end

        % reprojection delta
        dre = x_re-xrep;
        
        if isempty(obj.sigma_reconstruct),
          err = nan([d_in,obj.nviews,Ntotal]);
          err(:,:,isrealsample) = abs(dre);
          err = reshape(err,[d_in,obj.nviews,nsamples_total,Nframes,obj.npts]);
          
          %err = abs(reshape(x_re,[d_in,obj.nviews*Ntotal])-reshape(permute(xrep,[5,4,1,2,3]),[d_in,obj.nviews*Ntotal]));
          sigma_reconstruct = zeros([d_in,d_in,obj.nviews]);
          for di = 1:d_in,
            for viewi = 1:obj.nviews,
              sigma_reconstruct(di,di,viewi) = PostProcess.EstimateReconstructionSigma(permute(err(di,viewi,:,:,:),[3,4,5,1,2]));
            end
          end
        else
          sigma_reconstruct = obj.sigma_reconstruct;
        end
        obj.sigma_reconstruct_used = sigma_reconstruct;
      
        % xrep is N x nsamples_total x npts x nviews x d_in
        % permuting gives us d_in x nviews x N x nsamples_total x npts
        if isS,
          p_re = mvnpdf_2d(reshape(dre,[d_in,obj.nviews*Ntotal1]),0,reshape(sigma_reconstruct+Srep,[d_in,d_in,obj.nviews*Ntotal1]));
        else
          p_re = mvnpdf_2d(reshape(dre,[d_in,obj.nviews*Ntotal1]),0,repmat(sigma_reconstruct,[1,1,Ntotal1]));
        end
        
        % p_re is 1 x nviews * Ntotal1
        p_re = prod(reshape(p_re,[obj.nviews,Ntotal1]),1)';
        wreptotal1 = p_re.*wrep;
        wreptotal = zeros([nsamples_total,Nframes,obj.npts]);
        wreptotal(isrealsample) = wreptotal1;
        
        z = sum(wreptotal,1);
        totalweight_3d = z;
        z(z==0) = 1;
        wreptotal = wreptotal ./ z;
%          p_re(:,ni,pti) = p_re(order(:,ni,pti),ni,pti);
%             wrep(:,ni,pti) = wrep(order(:,ni,pti),ni,pti);
            
        obj.sampledata.tmp_independent2joint = struct;
        tmp_p_re = zeros([nsamples_total,Nframes,obj.npts]);
        tmp_p_re(isrealsample) = p_re;
        tmp_p_gau = zeros([nsamples_total,obj.N,obj.npts]);
        tmp_p_gau(isrealsample) = wrep;
        if obj.isframes,
          obj.sampledata.tmp_independent2joint.p_re = nan([nsamples_total,obj.N,obj.npts]);
          obj.sampledata.tmp_independent2joint.p_re(:,obj.dilated_frames,:,:) = tmp_p_re;
          obj.sampledata.tmp_independent2joint.p_gau = nan([nsamples_total,obj.N,obj.npts]);
          obj.sampledata.tmp_independent2joint.p_gau(:,obj.dilated_frames,:,:) = tmp_p_gau;
        else
          obj.sampledata.tmp_independent2joint.p_re = tmp_p_re;
          obj.sampledata.tmp_independent2joint.p_gau = tmp_p_gau;
        end

        % X should be d_out x Ntotal
        % x_re should be d_in x nviews x Ntotal
        d_out = size(X,1);
        
        % mu is d x nviews x nsamples x N x npts
        % S is d x d x nviews x nsamples x N x npts
        % prior is nviews x nsamples x N x npts

        if obj.isframes,
          obj.sampledata.x_in = nan([obj.N,nsamples,obj.npts,obj.nviews,d_in]);
          obj.sampledata.x_in(obj.dilated_frames,:,:,:,:) = permute(mu,[4,3,5,2,1]);
          obj.sampledata.w_in = nan([obj.N,nsamples,obj.npts,obj.nviews]);
          obj.sampledata.w_in(obj.dilated_frames,:,:,:) = permute(prior,[3,2,4,1]);
          obj.sampledata.z_in = nan([obj.npts,obj.nviews]);
          obj.sampledata.z_in(obj.dilated_frames,:,:) = totalweight;
        else
          % want x_in to be [obj.N,nsamples,obj.npts,obj.nviews,d_in]
          obj.sampledata.x_in = permute(mu,[4,3,5,2,1]);
          obj.sampledata.w_in = permute(prior,[3,2,4,1]);
          obj.sampledata.z_in = totalweight;
        end
        
        % wan x_perview to be [N,nRep,npts,nviews,d_in]
        tmp_x_perview = nan([d_in,obj.nviews,Ntotal]);
        tmp_x_perview(:,:,isrealsample) = xrep;
        tmp_x_perview = reshape(tmp_x_perview,[d_in,obj.nviews,nsamples_total,Nframes,obj.npts]);
        tmp_x_perview = permute(tmp_x_perview,[4,3,5,2,1]);

        % want X to be [N,K,npts,d]
        tmp_x = nan([d_out,Ntotal]);
        tmp_x(:,isrealsample) = X;
        tmp_x = reshape(tmp_x,[d_out,nsamples_total,Nframes,obj.npts]);
        tmp_x = permute(tmp_x,[3,2,4,1]);

        tmp_x_re_perview = nan([d_in,obj.nviews,Ntotal]);
        tmp_x_re_perview(:,:,isrealsample) = x_re;
        tmp_x_re_perview = reshape(tmp_x_re_perview,[d_in,obj.nviews,nsamples_total,obj.N,obj.npts]);
        tmp_x_re_perview = permute(tmp_x_re_perview,[4,3,5,2,1]);
        
        if obj.isframes,
          obj.sampledata.x_perview = nan([obj.N,nsamples_total,obj.npts,obj.nviews,d_in]);
          obj.sampledata.x_perview(obj.dilated_frames,:,:,:,:) = tmp_x_perview;
          
          obj.sampledata.x = nan([obj.N,nsamples_total,obj.npts,d_out]);
          obj.sampledata.x(obj.dilated_frames,:,:,:) = tmp_x;
          
          obj.sampledata.w = nan([obj.N,nsamples_total,obj.npts]);
          obj.sampledata.w(obj.dilated_frames,:,:) = permute(wreptotal,[2,1,3]); % w is N x maxnsamples x npts

          obj.sampledata.z = nan([obj.N,obj.npts]);
          obj.sampledata.z(obj.dilated_frames,:) = totalweight_3d.*prod(totalweight,3); % w is N x maxnsamples x npts
          
          obj.sampledata.x_re_perview = nan([obj.N,nsamples_total,obj.npts.obj.nviews,d_in]);
          obj.sampledata.x_re_perview(obj.dilated_frames,:,:,:,:) = tmp_x_re_perview;
          
        else
          
          obj.sampledata.x_perview = tmp_x_perview;
          obj.sampledata.x = tmp_x;
          obj.sampledata.w = permute(wreptotal,[2,1,3]); % w is N x maxnsamples x npts
          obj.sampledata.z = totalweight_3d.*prod(totalweight,3); % w is N x maxnsamples x npts
          obj.sampledata.x_re_perview = tmp_x_re_perview;
          
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
    
    function run(obj,varargin)
      
      force = myparse(varargin,'force',false);

      algorithm = obj.GetPostDataAlgName();
      
      if ~force && isfield(obj.postdata,algorithm) && ~isempty(obj.postdata.(algorithm)),
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
        case 'viterbi_grid',
          obj.RunViterbiGrid();
        otherwise
          error('Not implemented %s',obj.algorithm);
      end
      
    end
    
    function RunMaxDensity(obj)
      
      if strcmp(obj.nativeformat,'sample') || ...
          strcmp(obj.nativeformat,'heatmap') && obj.nviews > 1,
        obj.RunMaxDensity_SampleData();
      elseif strcmp(obj.nativeformat,'heatmap'),
        obj.RunMaxDensity_SingleHeatmapData();
      else
        error('Not implemented maxdensity %s',obj.nativeformat);
      end
    end
    
    function RunMedian(obj)
      
      if strcmp(obj.nativeformat,'sample') || ...
          strcmp(obj.nativeformat,'heatmap') && obj.nviews > 1,
        obj.RunMedian_SampleData();
      elseif strcmp(obj.nativeformat,'heatmap'),
        obj.RunMedian_SingleHeatmapData();
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
              ischange = isempty(obj.viterbi_poslambda);
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
          case 'grid_acradius',
            acrad = varargin{i+1};
            if numel(obj.viterbi_grid_acradius) ~= numel(acrad) || ...
                any(obj.viterbi_grid_acradius(:) ~= acrad(:)),
              ischange = true;
              obj.viterbi_grid_acradius = acrad;
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
      
      res = isempty(obj.kdedata) || ...
        ( obj.jointsamples && (~isfield(obj.kdedata,'joint') || isempty(obj.kdedata.joint) ) ) || ...
        ( ~obj.jointsamples && (~isfield(obj.kdedata,'indep') || isempty(obj.kdedata.indep) ) );

    end
    
    function RunMaxDensity_SampleData(obj)
      
      [N,K,npts,d] = size(obj.sampledata.x); %#ok<ASGLU>
            
      if obj.NeedKDE(),
        obj.ComputeKDE();
      end

      if obj.isframes,
        Nframes = nnz(obj.dilated_frames);
        frameidx = find(obj.dilated_frames);
      else
        Nframes = N;
        frameidx = 1:N;
      end
      
      if obj.jointsamples,
        
        w = obj.kdedata.joint(frameidx,:);
        obj.postdata.maxdensity_joint = struct;
        obj.postdata.maxdensity_joint.score = nan([N,1]);
        [obj.postdata.maxdensity_joint.score(frameidx),k] = max(w,[],2);
        obj.postdata.maxdensity_joint.x = reshape(obj.sampledata.x(:,1,:,:),[N,npts,d]);
        for i = 1:Nframes,
          obj.postdata.maxdensity_joint.x(frameidx(i),:,:) = obj.sampledata.x(frameidx(i),k(i),:,:);
        end
        obj.postdata.maxdensity_joint.sampleidx = nan([N,1]);
        obj.postdata.maxdensity_joint.sampleidx(frameidx) = k;
        
      else
        
        % this is different from what maxdensity did
        w = obj.kdedata.indep(frameidx,:,:);
        obj.postdata.maxdensity_indep = struct;
        obj.postdata.maxdensity_indep.x = reshape(obj.sampledata.x(:,1,:,:),[N,npts,d]);
        obj.postdata.maxdensity_indep.score = nan([N,npts]);
        obj.postdata.maxdensity_indep.sampleidx = nan([N,npts]);
        for ipt=1:npts,

          % sampledata.w.indep is N x nRep x npts
          [scorecurr,k] = max(w(:,:,ipt),[],2);
          obj.postdata.maxdensity_indep.score(frameidx,ipt) = log(scorecurr);
          obj.postdata.maxdensity_indep.sampleidx(frameidx,ipt) = k;
          for i = 1:Nframes,
            obj.postdata.maxdensity_indep.x(frameidx(i),ipt,:) = obj.sampledata.x(frameidx(i),k(i),ipt,:);
          end

        end
      end
      
    end
    
    function RunMedian_SampleData(obj)
      
      [N,K,npts,d] = size(obj.sampledata.x); 
      
      if obj.isframes,
        Nframes = nnz(obj.dilated_frames);
        frameidx = find(obj.dilated_frames);
      else
        Nframes = N;
        frameidx = 1:N;
      end

      % should we use the weights?
      if isfield(obj.sampledata,'w') && ~isempty(obj.sampledata.w),
        isjoint = size(obj.sampledata.w,3)==1;
        if isjoint,
          w = repmat(obj.sampledata.w(frameidx,:),[1,1,obj.npts,d]);
        else
          w = repmat(obj.sampledata.w(frameidx,:,:),[1,1,1,d]);
        end
      else
        w = ones([Nframes,K,npts,d]);
      end
      [medianx,sampleidx] = weighted_prctile(obj.sampledata.x(frameidx,:,:,:),50,w,2);
      
      mad = weighted_prctile(abs(obj.sampledata.x(frameidx,:,:,:) - medianx),50,w,2);
      score = -mean(reshape(mad,[Nframes,npts*d]),2);
      medianx = reshape(medianx,[Nframes,npts,d]);
      sampleidx = reshape(sampleidx,[Nframes,npts,d]);

      if obj.isframes,
        obj.postdata.median.x = nan([N,npts,d]);
        obj.postdata.median.x(frameidx,:,:) = medianx;
        obj.postdata.median.sampleidx = nan([N,npts]);
        obj.postdata.median.sampleidx(frameidx,:) = sampleidx;
        obj.postdata.median.score = nan([N,1]);
        obj.postdata.median.score(frameidx) = score;

      else
        obj.postdata.median.x = medianx;
        obj.postdata.median.sampleidx = sampleidx;
        obj.postdata.median.score = score;
      
      end
    end

    function RunMedian_SingleHeatmapData(obj)
      
      d = 2;
      viewi = 1;
      if obj.isframes,
        Nframes = nnz(obj.dilated_frames);
        frameidx = find(obj.dilated_frames);
      else
        Nframes = obj.N;
        frameidx = 1:obj.N;
      end
      
%       chunksize = round(2^25 / (obj.heatmapdata.nys*obj.heatmapdata.nxs));
      
%       obj.postdata.median.x = nan([obj.N,obj.npts,d]);
%       obj.postdata.median.score = nan([obj.N,obj.npts,d]);
%       obj.postdata.median.sampleidx = nan([obj.N,obj.npts,d]);
      x = nan([Nframes,obj.npts,d]);
      score = nan([Nframes,obj.npts,d]);
      sampleidx = nan([Nframes,obj.npts,d]);
      grid = obj.heatmapdata.grid{viewi};
      
      npts = obj.npts;
      for pti = 1:npts,
        readscorefun = @(i) obj.ReadHeatmapScore(pti,viewi,i);
        parfor(ti = 1:Nframes,obj.ncores),
          
          %for t0 = 1:chunksize:obj.N,
          %           t1 = min(obj.N,t0+chunksize-1);
          %           [hms,xs] = obj.ReadHeatmapScores(pti,viewi,t0:t1,'issparse',true);
          %           for t = t0:t1,
          %             w = hms{t-t0+1};
          %             idxcurr = xs{t-t0+1};
          scorescurr = feval(readscorefun,frameidx(ti));
          idxcurr = find(scorescurr>0);
          w = scorescurr(idxcurr);
          for di = 1:d,
            [x(ti,pti,di),medidx] = ...
              weighted_prctile(grid(idxcurr,di),50,w); %#ok<PFBNS>
            sampleidx(ti,pti,di) = idxcurr(medidx);
            mad = weighted_prctile(abs(grid(idxcurr,di)-x(ti,pti,di)),50,w);
            score(ti,pti,d) = -mad;
          end
        end
      end
      
      if obj.isframes,
        
        obj.postdata.median.x = nan([obj.N,npts,d]);
        obj.postdata.median.x(frameidx,:,:) = x;
        obj.postdata.median.sampleidx = nan([obj.N,npts,d]);
        obj.postdata.median.sampleidx(frameidx,:,:) = sampleidx;
        obj.postdata.median.score = nan([obj.N,npts,d]);
        obj.postdata.median.score(frameidx,:,:) = score;
        
      else        
        obj.postdata.median.x = x;
        obj.postdata.median.score = score;
        obj.postdata.median.sampleidx = sampleidx;
      end
        
      if obj.IsTrx(),
        viewi = 1;
        obj.postdata.median.x_trx = obj.postdata.median.x;
        obj.postdata.median.x = PostProcess.UntransformByTrx(...
          obj.postdata.median.x,obj.trx(viewi),obj.heatmap_origin(viewi,:));
      end
      
      % .IsCrop()?
    end
    
    function RunMaxDensity_SingleHeatmapData(obj)

      d = 2;
      viewi = 1;
      
      if obj.isframes,
        Nframes = nnz(obj.dilated_frames);
        frameidx = find(obj.dilated_frames);
      else
        Nframes = obj.N;
        frameidx = 1:obj.N;
      end
      
      obj.postdata.maxdensity_indep = struct;
      x = nan([Nframes,obj.npts,d]);
      score = zeros([Nframes,obj.npts]);
      sampleidx = nan([Nframes,obj.npts]);

      grid = obj.heatmapdata.grid{viewi};
      npts = obj.npts;
      for pti = 1:npts,
        readscorefun = @(i) obj.ReadHeatmapScore(pti,viewi,i);
        
        %[~,parfor_progress_filename,parfor_progress_success] = parfor_progress_kb(obj.N);
        fprintf('Max density, pt %d / %d\n',pti,npts);
        parfor(ti = 1:Nframes,obj.ncores),
          scores = feval(readscorefun,frameidx(ti));
          %       for pti = 1:obj.npts,
          %
          %         for t0 = 1:chunksize:obj.N,
          %           t1 = min(obj.N,t0+chunksize-1);
          %           [hms] = obj.ReadHeatmapScores(pti,viewi,t0:t1);
          %           for t = t0:t1,
          %             scores = hms{t-t0+1};
          %scores = obj.ReadHeatmapScore(pti,viewi,t);
          [score(ti,pti),idx] = max(scores(:));
          x(ti,pti,:) = grid(idx,:); %#ok<PFBNS>
          sampleidx(ti,pti) = idx;
%           if parfor_progress_success,
%             parfor_progress_kb(parfor_progress_filename);
%           end
        end
%         fprintf('Max density, pt %d / %d done.\n',pti,npts);
%         if parfor_progress_success,
%           parfor_progress_kb(0,parfor_progress_filename);
%         end

      end
      
      
      if obj.isframes,
        
        obj.postdata.maxdensity_indep.x = nan([obj.N,npts,d]);
        obj.postdata.maxdensity_indep.x(frameidx,:,:) = x;
        obj.postdata.maxdensity_indep.sampleidx = nan([obj.N,npts]);
        obj.postdata.maxdensity_indep.sampleidx(frameidx,:) = sampleidx;
        obj.postdata.maxdensity_indep.score = nan([obj.N,npts]);
        obj.postdata.maxdensity_indep.score(frameidx,:) = score;
        
      else
        
        obj.postdata.maxdensity_indep.x = x; % [N x npts x d]
        obj.postdata.maxdensity_indep.score = score; % [N x npts] heatmap score
        obj.postdata.maxdensity_indep.sampleidx = sampleidx; % [N x npts] linear index into heatmap/grid for x
        
      end
      if obj.IsTrx(),
        viewi = 1;
        obj.postdata.maxdensity_indep.x_trx = obj.postdata.maxdensity_indep.x;
        obj.postdata.maxdensity_indep.x = PostProcess.UntransformByTrx(...
          obj.postdata.maxdensity_indep.x,obj.trx(viewi),obj.heatmap_origin(viewi,:));  
      end
      
      % .IsCrop()?
      
    end
    
    function EstimateKDESigma(obj,p,k)
      % Estimate/Set KDE sigma from .sampledata
      
      if nargin < 2,
        p = 90;
      end
      if nargin < 3,
        k = 1;
      end
      
      if obj.isframes,
        Nframes = nnz(obj.dilated_frames);
        frameidx = find(obj.dilated_frames);
      else
        Nframes = obj.N;
        frameidx = 1:obj.N;
      end
      
      d = size(obj.sampledata.x,4);
      nsamples = size(obj.sampledata.x,2);
      D = abs(reshape(obj.sampledata.x(frameidx,:,:,:) ,[Nframes,nsamples,1,obj.npts,d]) - ...
        reshape(obj.sampledata.x(frameidx,:,:,:),[Nframes,1,nsamples,obj.npts,d]));
      sortD = sort(D,2);
      sortD = sortD(:,2:end,:,:,:);
      kde_sigma = prctile(reshape(sortD(:,k,:,:,:),[Nframes*nsamples*obj.npts*d,1]),p);
      obj.SetKDESigma(kde_sigma);
      
    end
    
    function ComputeKDE(obj,varargin)
      pts2run = myparse(varargin,...
        'pts2run',obj.pts2run...
        );
      if isempty(obj.pts2run)
        obj.pts2run = 1:obj.npts;
      end
      npts2run = numel(pts2run);

      starttime = tic;

      [N,nRep,npts,d] = size(obj.sampledata.x);
      D = obj.npts*d;
      
      if obj.isframes,
        Nframes = nnz(obj.dilated_frames);
        frameidx = find(obj.dilated_frames);
      else
        Nframes = obj.N;
        frameidx = 1:obj.N;
      end
      
      isbadx = any(isnan(obj.sampledata.x(frameidx,:,:,:)),4);
      isbadx2 = all(isnan(obj.sampledata.x(frameidx,:,:,:)),4);
      if isfield(obj.sampledata,'w'),
        w = obj.sampledata.w(frameidx,:,pts2run);
        assert(all(w(isbadx(:,:,pts2run))==0));
      end
      assert(all(isbadx(:)==isbadx2(:)));
      
      
      if obj.jointsamples,
        
        fprintf('Computing KDE, joint\n');

        x = reshape(permute(obj.sampledata.x(frameidx,:,:,:),[3,4,5,2,1]),[D,1,nRep,Nframes]);
        % these have weight 0
        x(isnan(x)) = 0;
        
        if isfield(obj.sampledata,'w') && ~isempty(obj.sampledata.w),
          % w is N x nRep 
          % w0 should be nRep x 1 x N
          w0 = reshape(prod(obj.sampledata.w(frameidx,:,:),3)',[nRep,1,Nframes]);
        else
          w0 = 1;
        end
        c = sqrt(2*pi*obj.kde_sigma^(2*D));
        d2 = reshape(sum( (x-reshape(x,[D,nRep,1,Nframes])).^2,1 ),[nRep,nRep,Nframes]);
        w = sum(w0.*exp( -d2/obj.kde_sigma^2/2 ),1)/c;
        %w = w ./ sum(w,2);
        assert(~any(isnan(w(:))));

        if obj.isframes,
          obj.kdedata.joint = nan([obj.N,nRep]);
          obj.kdedata.joint(frameidx,:) = permute(w,[3,2,1]);
        else
          obj.kdedata.joint = permute(w,[3,2,1]);
        end
              
        
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
        
        w_npts = size(obj.sampledata.w,3);
        assert(w_npts==obj.npts); % AL: not sure when this is not true
        if isfield(obj.sampledata,'w') && ~isempty(obj.sampledata.w),
          % w is N x nRep x npts
          w0 = reshape(obj.sampledata.w(frameidx,:,pts2run),[Nframes,nRep,1,npts2run]);
        else
          w0 = 1;
        end
        
        fprintf('Computing KDE, indep\n');
        c = sqrt(2*pi*obj.kde_sigma^(2*d));
        d2 = sum( ...
          (reshape(obj.sampledata.x(frameidx,:,pts2run,:),[Nframes,1,nRep,npts2run,d])-...
           reshape(obj.sampledata.x(frameidx,:,pts2run,:),[Nframes,nRep,1,npts2run,d])).^2, 5 );
        % d2 is [Nframes x nRep x nRep x npts2run]
        
        % these have weight 0
        d2(isnan(d2)) = inf;
        w = sum(w0.*exp( -d2/obj.kde_sigma^2/2 ),2)/c; % [Nframes x 1 x nRep x npts2run]
        %w = w ./ sum(w,3);
        assert(~any(isnan(w(:))));
        obj.kdedata.indep = nan([N,nRep,npts]);
        if obj.isframes,
          obj.kdedata.indep(frameidx,:,pts2run) = reshape(w,[Nframes,nRep,npts2run]);
        else
          obj.kdedata.indep(:,:,pts2run) = reshape(w,[N,nRep,npts2run]);
        end
        
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
    
    function RunViterbi(obj,varargin)
      % It is assumed that pTrkFull are from consecutive frames.

      pts2run = myparse(varargin,...
        'pts2run',obj.pts2run...
        );
      if isempty(pts2run)
        pts2run = 1:obj.npts;
      end
      
      if strcmp(obj.nativeformat,'heatmap') && obj.nviews == 1 && isempty(obj.sampledata),
        obj.Heatmap2SampleData();
      end
      
      if obj.NeedKDE(),
        obj.ComputeKDE();
      end

      [N,K,npts,d] = size(obj.sampledata.x); 
      D = npts*d;
      
      algorithm = obj.GetPostDataAlgName(); %#ok<*PROP>
      
      if obj.isframes,
        
        [t0s,t1s] = get_interval_ends(obj.dilated_frames);
        t1s = t1s-1;
        
      else
        
        t0s = 1;
        t1s = N;
        
      end
            
      if obj.jointsamples,

        obj.postdata.(algorithm) = struct;
        obj.postdata.(algorithm).x = [];
        obj.postdata.(algorithm).score = [];
        obj.postdata.(algorithm).isvisible = [];
        obj.postdata.(algorithm).sampleidx = [];

        Xbest = nan([N,D]);
        isvisiblebest = nan([N,1]);
        idxcurr = nan([N,1]);
        totalcost = 0;

        
        for i = 1:numel(t0s),
          t0 = t0s(i);
          t1 = t1s(i);
          
          if isfield(obj.sampledata,'z'),
            z = obj.sampledata.z(t0:t1);
          else
            z = 1;
          end
          
          appcost = -log(obj.kdedata.joint(t0:t1).*z);
          X = permute(reshape(obj.sampledata.x(t0:t1,:,:,:),[t1-t0+1,K,D]),[3 1 2]);
          
          [Xbest(t0:t1,:),isvisiblebest(t0:t1),idxcurr(t0:t1),totalcost1,poslambdaused,misscostused] = ...
            obj.RunViterbiHelper(appcost,X); %#ok<ASGLU>
          totalcost = totalcost + totalcost1;
        end
          
        obj.postdata.(algorithm).x = reshape(Xbest',[N,npts,d]);
        obj.postdata.(algorithm).score = totalcost;
        obj.postdata.(algorithm).sampleidx = reshape(idxcurr,[N,1]);
        obj.postdata.(algorithm).isvisible = isvisiblebest;
          
      else

        obj.postdata.(algorithm) = struct;
        obj.postdata.(algorithm).x = reshape(obj.sampledata.x(:,1,:),[N,npts,d]);
        obj.postdata.(algorithm).score = zeros(npts,1);
        obj.postdata.(algorithm).isvisible = true(N,npts);
        obj.postdata.(algorithm).sampleidx = nan(N,npts);
        
        for i = 1:numel(t0s),
          t0 = t0s(i);
          t1 = t1s(i);
          Ncurr = t1-t0+1;
          
          for ipt = pts2run
            
            if isfield(obj.sampledata,'z'),
              z = obj.sampledata.z(t0:t1,ipt);
            else
              z = 1;
            end
            
            % N x nRep x npts
            appcost = -log(obj.kdedata.indep(t0:t1,:,ipt).*z);
            X = permute(reshape(obj.sampledata.x(t0:t1,:,ipt,:),[Ncurr,K,d]),[3 1 2]);
            [Xbest,isvisiblebest,idxcurr,totalcost,poslambdaused,misscostused] = ...
              obj.RunViterbiHelper(appcost,X); %#ok<ASGLU>
            obj.postdata.(algorithm).x(t0:t1,ipt,:) = Xbest';
            obj.postdata.(algorithm).score(ipt) = obj.postdata.(algorithm).score(ipt)+totalcost;
            obj.postdata.(algorithm).sampleidx(t0:t1,ipt) = idxcurr;
            obj.postdata.(algorithm).isvisible(t0:t1,ipt) = isvisiblebest;
          end
        end

      end
      
    end
    
    
    function [Xbest,isvisiblebest,idxcurr,totalcost,poslambdaused,misscostused] = ...
        RunViterbiHelper(obj,appcost,X)
      
      [d,N,K] = size(X);
      Kreal = find(any(~isinf(appcost),1),1,'last');
      if Kreal < K,
        X = X(:,:,1:Kreal);
        appcost = appcost(:,1:Kreal);
      end
      
      if isinf(obj.viterbi_misscost),
        [Xbest,idxcurr,totalcost,poslambdaused] = ChooseBestTrajectory(X,appcost,...
          'poslambda',obj.viterbi_poslambda,...
          'poslambdafac',obj.viterbi_poslambdafac,...
          'dampen',obj.viterbi_dampen); 
        isvisiblebest = true(N,1);
        misscostused = nan;
      else
        [Xbest,isvisiblebest,idxcurr,totalcost,poslambdaused,misscostused] = ...
          ChooseBestTrajectory_MissDetection(X,appcost,...
          'poslambda',obj.viterbi_poslambda,...
          'poslambdafac',obj.viterbi_poslambdafac,...
          'dampen',obj.viterbi_dampen,...
          'misscost',obj.viterbi_misscost); 
      end
      
      
    end
    
    function RunViterbiGrid(obj,varargin)
      % Single-view heatmap viterbi
      
      pts2run = myparse(varargin,...
        'pts2run',obj.pts2run...
        );
      if isempty(pts2run)
        pts2run = 1:obj.npts;
      end
      
      assert( strcmp(obj.nativeformat,'heatmap') );
      assert( obj.nviews==1 );
      assert( ~obj.isframes );
      
      assert(isrow(pts2run));
      d = 2;
      xy = nan(obj.N,obj.npts,d);
      xyAC = nan(obj.N,obj.npts,d);
      acCtrXY = nan(obj.N,obj.npts,d);
      score = nan(obj.npts,1);
      for ipt=pts2run
        fprintf('RunViterbiGrid pt %d\n',ipt);
        iview = 1;
        hmfcn = @(n)obj.ReadHeatmapScore(ipt,iview,n);
        [hmnrnc,hmBestTrajPQ,acBestTrajUV,acCtrPQ,totalcost] = ...
          ChooseBestTrajectory_grid(hmfcn,obj.N,...
          'hmConsiderRadius',obj.viterbi_grid_acradius,...
          'poslambda',obj.viterbi_poslambda,...
          'dampen',obj.viterbi_dampen...
          );
        
        xy(:,ipt,:) = hmBestTrajPQ(:,[2 1]);
        xyAC(:,ipt,:) = acBestTrajUV(:,[2 1]);
        acCtrXY(:,ipt,:) = acCtrPQ(:,[2 1]);
        score(ipt) = totalcost;
      end
      
      alg = 'viterbi_grid';
      obj.postdata.(alg) = struct;
      obj.postdata.(alg).x = xy;
      obj.postdata.(alg).xAC = xyAC;
      obj.postdata.(alg).acCtrXY = acCtrXY;
      obj.postdata.(alg).score = score;
    end
  
%     function ClearComputedResults(obj)
%       
%       obj.postdata = struct;
%       if isstruct(obj.sampledata) && isfield(obj.sampledata,'w'),
%         obj.sampledata.w = [];
%       end
%       if ~strcmp(obj.nativeformat,'heatmap'),
%         obj.heatmapdata = [];
%       end
%       if ~strcmp(obj.nativeformat,'sample'),
%         obj.sampledata = [];
%       end
% 
%     end
    
  end
  
  methods (Static)
    
    function [mu,w,S,kreal,totalweight] = LocalMaximaSamples(hm,k,varargin)
      % Find samples (candidate-points) from local maxima in heatmap
      %
      % hm: 2D heatmap
      % k: number of samples/candidates desired
      %
      % mu: [kx2] (x,y) or equivalently (col,row) sample means/locs
      % w: [kx1] "likelihood" weights. sum(w) guaranteed to equal 1 unless kreal==0.
      % S: [2x2xk]. cov matrices 
      % kreal: number of actual candidates found. candidates (kreal+1)..k
      %   are nans/invalid
      %
      % Note that kreal can be less than k, or even 0.
      
      [thresh,thresh_prctile,r_nonmax,grid] = ...
        myparse(varargin,...
        'thresh',[],... % threshold for nonmax supp. raw value, same units as hm.
        'thresh_prctile',99.5,... % used if thresh is not supplied; thresh set to this prctile of hm 
        'r_nonmax',4,... % radius for nonmax supp
        'grid',[]... % [nim x 2] (x,y) coords for hm. grid(i,:) labels hm(i)
        );
      
      if ~isempty(grid)
        nim = numel(hm);
        szassert(grid,[nim 2]);
      end

      if isempty(thresh),
        thresh = prctile(hm(:),thresh_prctile);
      end
      
      [r,c] = nonmaxsuppts(hm,r_nonmax,thresh);
      idx = sub2ind(size(hm),r,c);
      k0 = numel(r);
      if k0 > k,
        wcurr = hm(idx);
        [~,order] = sort(wcurr(:),1,'descend');
        order = order(1:k);
        r = r(order);
        c = c(order);
        idx = idx(order);
      end      
      kreal = min(k0,k);
      
      mu = nan(k,2);
      w = zeros(k,1);
      S = nan(2,2,k);
      
      if isempty(grid),
        mu(1:kreal,:) = [c(:),r(:)];
      else
        mu(1:kreal,:) = grid(idx,:);
      end

      idxnz = hm>0;
      if isempty(grid),
        [rnz,cnz] = ind2sub(size(hm),find(idxnz)); % or just find(idxnz)        
      else
        cnz = grid(idxnz,1);
        rnz = grid(idxnz,2);
      end
      hmnz = hm(idxnz);
      totalweight = sum(hmnz);

      % fill els of w, S
      if k0 > 1,
        ctrim = false(size(hm));
        ctrim(idx) = 1;
        [~,l] = bwdist(ctrim);
        
        %w = nan(kreal,1);
        for ki = 1:kreal,
          idxnzcurr = l(idxnz)==idx(ki);
          w(ki) = sum(hmnz(idxnzcurr));
          [~,S(:,:,ki)] = weighted_mean_cov([cnz(idxnzcurr),rnz(idxnzcurr)],hmnz(idxnzcurr));
        end
        w = w / sum(w);
        
%         [sortedidx,order] = sort(idx);
%         [~,unorder] = sort(order);
%         w = myhist(l(idxnz),sortedidx,'weights',hm(idxnz));
%         w = w(unorder);
%         w = w / sum(w);
%         w = w(:);        
      elseif k0 == 1
        assert(kreal==1);
        w(1) = 1;
        [~,S(:,:,1)] = weighted_mean_cov([cnz,rnz],hmnz);
      else
        % k0==0; w will be all zeros
      end     
    end   
    
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
        nmixpermax,minpxpermix,singlecomp] = ...
        myparse(varargin,...
        'r_nonmax',r_nonmax,'thresh_perc',thresh_perc,'thresh',[],...
        'xgrid',[],'ygrid',[],'scalex',1,'scaley',1,...
        'nmixpermax',nmixpermax,'minpxpermix',minpxpermix,'singlecomp',false);
      
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

      idxcurr = scores >= threshcurr;
      X = [xgrid(idxcurr),ygrid(idxcurr)];

      if singlecomp
        [~,idx] = max(scores(:));
        [r,c] = ind2sub(size(scores),idx);
        r = r*scaley;
        c = c*scalex;
        k = 1;
        start = [c r];
      else
        % threshold and non-maximum suppression
        [r,c] = nonmaxsuppts(tscorescurr,r_nonmax,threshcurr);
        r = r*scaley;
        c = c*scalex;

        % set number of centers based on number of maxima found
        k0 = numel(r);
        k = k0+min(numel(r)*nmixpermax,floor(nnz(idxcurr)/minpxpermix));

        % initialize clustering
        start = nan(k,2);
        start(1:k0,:) = [c(:),r(:)];
        d = min(dist2(X,start(1:k0,:)),[],2);
        for j = k0+1:k,
          [~,maxj] = max(d);
          start(j,:) = X(maxj,:);
          d = min(d,dist2(X,start(j,:)));
        end
      end
      
      % gmm fitting
      [mu,S,~,post] = mygmm(X,k,...
        'Start',start,...
        'weights',scores(idxcurr)-threshcurr);
      
      w = sum(post,1)';
      fprintf(2,'XXX\n');
      %w = sum(bsxfun(@times,scores(idxcurr),post),1)';
      %w(w<0.1) = 0.1;
            
      nanmu = any(isnan(mu),2);
      mu = mu(~nanmu,:);
      w = w(~nanmu);
      S = S(:,:,~nanmu);
    end
        
    function [mu,w,S] = GMMFitSamples(X,k,varargin)
      
      [N,nRep,npts,nviews,d] = size(X);
      % X is N x nRep x npts x nviews x d
      [weights,jointpoints] = myparse(varargin,...
        'weights',[],... % [N x nRep x npts x nviews], npts==1 for joint
        'jointpoints',false);

      if jointpoints,
        X = permute(X,[1,2,4,5,3]);
        % now X is N x nRep x nviews x d x npts
        % D is d * npts
        d0 = d;
        npts0 = npts;
        npts = 1;
        d = npts0*d0;
        X = reshape(X,[N,nRep,npts,nviews,d]);
        % now X is N x nRep x 1 x nviews x (d0*npts0)
        if ~isempty(weights),
          weights = prod(weights,3);
        end
        
      end
      
      mu = nan([k,d,N,npts,nviews]);
      S = nan([d,d,k,N,npts,nviews]);
      w = nan([k,N,npts,nviews]);
      
      for viewi = 1:nviews,
        for pti = 1:npts,
          for n = 1:N,
            % X is N x nRep x npts x nviews x d so 
            % Xcurr is 1 x nRep x 1 x 1 x d
            Xcurr = reshape(X(n,:,pti,viewi,:),[nRep,d]);
            if isempty(weights),
              weights_curr = [];
            else
              weights_curr = reshape(weights(n,:,pti,viewi),[nRep,1]);
            end
            % mu output is k x d
            % S output is d x d x k
            % post is nRep x k
            [mu(:,:,n,pti,viewi),S(:,:,:,n,pti,viewi),tmp,post] = ...
              mygmm(Xcurr,k,...
              'weights',weights_curr);      
            if isempty(weights),
              w(:,n,pti,viewi) = sum(post,1)';
            else
              w(:,n,pti,viewi) = sum(post,1)';              
              fprintf(2,'XXX\n');
              %w(:,n,pti,viewi) = sum(bsxfun(@times,weights_curr,post),1)';
            end
            w(:,n,pti,viewi) = w(:,n,pti,viewi) / sum(w(:,n,pti,viewi));
          end
        end
      end
      
      if jointpoints,
        % mu0 is k x (d0*npts0) x N x 1 x nviews
        mu0 = mu;
        mu = permute(reshape(mu0,[k,d0,npts0,N,1,nviews]),[1,2,4,3,6,5]);
        % after reshape, mu is k x d0 x npts0 x N x 1 x nviews
        % after permute, mu is k x d0 x N x npts0 x nviews x 1
        S0 = S;
        S = nan([d0,d0,k,N,npts0,nviews]);
        for pti = 1:npts0,
          S(:,:,:,:,pti,:) = S0( d0*(pti-1)+1:d0*pti, d0*(pti-1)+1:d0*pti, :,:,1,: );
        end
        w = reshape(w,[k,N,nviews]);
      end

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
    
    function sigma_reconstruct = EstimateReconstructionSigma(err)
      % err should be nsamples x n
      minerr = min(err,[],1);
      median_min_err = nanmedian(minerr(:));
      sigma_reconstruct = median_min_err*1.4836;
    end

    % xout1 = UntransformByTrx(xin,trx,r)
    % transforms from cropped and rotated heatmap coordinates to global movie coordinates
    function xout1 = UntransformByTrx(xin,trx,r)
      % xin: Typically, [nfrms x ... x 2] with 1st dim representing frames,
      %   last dim representing coordinate
      % trx: fields .y, .x need to be [nfrms] aligned/corresponding to 1st dim of xin
      % r: [2] x/y of origin of xin coord system (that is mapped to trx.x,trx.y)
      
      sz = size(xin);
      assert(sz(end)==2);
      nd = numel(sz);
      if nd >= 3 && sz(1) == numel(trx.theta),
        newsz = [sz(1),prod(sz(2:end-1)),sz(end)];
      else
        newsz = [prod(sz(1:end-1)),1,sz(end)];
        assert(newsz(1)==numel(trx.theta));
      end
      xin = reshape(xin,newsz);
      idxgood = any(any(~isnan(xin),2),3);
      ngood = nnz(idxgood);
      if ngood == 0,
        xout1 = reshape(xin,sz);
        return;
      end
      xin = xin(idxgood,:,:);
      
      ct = cos(pi/2+trx.theta(idxgood));
      st = sin(pi/2+trx.theta(idxgood));
      
      xout = xin - reshape(r,[1,1,2]);
      
      % unrotate
      xout = cat(3,xout(:,:,1).*ct(:) - xout(:,:,2).*st(:),...
                   xout(:,:,1).*st(:) + xout(:,:,2).*ct(:));
      
      % translate
      xout = xout + cat(3,reshape(trx.x(idxgood),[ngood,1]),...
                          reshape(trx.y(idxgood),[ngood,1]));
      
      xout1 = nan(newsz);
      xout1(idxgood,:,:) = xout;
      
      xout1 = reshape(xout1,sz);      
    end
    
    % xout1 = TransformByTrx(xin,trx,r)
    % transforms from global movie coordinates to cropped and rotated heatmap coordinates
    function xout1 = TransformByTrx(xin,trx,r,f)
      % xin: Typically, [nfrms x ... x 2] with 1st dim representing frames,
      %   last dim representing coordinate
      % trx: fields .y, .x need to be [nfrms] aligned/corresponding to 1st dim of xin
      % r: [2] x/y of origin of xout1 coord system (that is mapped to trx.x,trx.y)
      % f: (opt) indices into trx fields corresponding to first dim of xin.

      sz = size(xin);
      assert(sz(end)==2);
      nd = numel(sz);
      if nd >= 3 && sz(1) == numel(trx.theta),
        newsz = [sz(1),prod(sz(2:end-1)),sz(end)];
      else
        newsz = [prod(sz(1:end-1)),1,sz(end)];
      end
      xin = reshape(xin,newsz);
      idxgood = any(any(~isnan(xin),2),3);
      ngood = nnz(idxgood);
      if ngood == 0,
        xout1 = reshape(xin,sz);
        return;
      end
      xin = xin(idxgood,:,:);

      if nargin < 4,
        idxtrx = idxgood;
      else
        assert(all(idxgood)); % f labels first dim of xin; could also set idxtrx to f(idxgood) maybe
        idxtrx = f;
        assert(numel(idxtrx)==ngood);
      end
      ntrx = nnz(idxtrx);
      
      ct = cos(-pi/2-trx.theta(idxtrx));
      st = sin(-pi/2-trx.theta(idxtrx));
      
      % translate
      xout = xin - cat(3,reshape(trx.x(idxtrx),[ntrx,1]),...
                         reshape(trx.y(idxtrx),[ntrx,1]));
      
      % rotate
      xout = cat(3,xout(:,:,1).*ct(:) - xout(:,:,2).*st(:),...
                   xout(:,:,1).*st(:) + xout(:,:,2).*ct(:));

      % translate so that origin is top corner of image
      xout = xout + reshape(r,[1,1,2]);
            
      xout1 = nan(newsz);
      xout1(idxgood,:,:) = xout;
      
      xout1 = reshape(xout1,sz);      
    end
    
    function [trksnew] = triangulate(trks,rois,crig,pp3dtype,varargin)
      %
      % trk1/trk2: "Trkfile" structs, as if generated by MDN. Reqd fields: 
      %   .pTrk
      %   .pTrkFrm
      %   .pTrkiTgt
      %   .pTrklocs_unet
      %   .pTrkconf
      %   .pTrkconf_unet
      % rois: [nview x 4]. Currently only used for
      %   pp3dtype=='experimental'. Can be all nans for other pp3dtype etc.
      % crig: scalar CalRig object
      % pp3dtype: either 'triangulate' or 'experimental'
      %
      % trk1new/trk2new: trkfiles containing reconstructed results with
      % diagnostics
      %
      % This can throw.
      trksnew = {};
      DXYZ = myparse(varargin,...
        'dxyz',0.005...
        );
      
      nvw = length(trks);
      if iscell(crig)
        crig = crig{1};
      end
      
      for i = 2:length(trks)
        if ~isequal(trks{i-1}.pTrkFrm, trks{i}.pTrkFrm) || ~isequal(trks{i-1}.pTrkiTgt,trks{i}.pTrkiTgt)
          error('Cannot perform 3D postprocessing; trkfiles differ in frames/targets tracked.');
        end
      end
      % if ~isequal(trk1.pTrkFrm,trk2.pTrkFrm) || ...
      %    ~isequal(trk1.pTrkiTgt,trk2.pTrkiTgt)
      %   error('Cannot perform 3D postprocessing; trkfiles differ in frames/targets tracked.');
      % end
      
      if isfield(trks{1},'pTrk3D')
        error('Trkfile for view 1 already contains ''pTrk3D'' field. This trkfile has already been 3d-postprocessed.');
      end
      
      ptrks = cell(nvw, 1);
      for i = 1:nvw
        ptrks{i} = trks{i}.pTrk{1};
      end
      % end
      % ptrk1 = trk1.pTrk{1};
      % ptrk2 = trk2.pTrk{1};
        
      [npt,d,nfrm,ntgt] = size(ptrks{1});
      for i = 2:nvw
        assert(isequal(size(ptrks{1}),size(ptrks{i})),...
          'Trkfiles contain position arrays with inconsistent sizes.');
      end

      % if ~isa(crig,'CalRig')
      %   error('Expected ''crig'' to be a CalRig instance.');
      % end
        
      fprintf(1,'Performing 3d reconciliation: %s...\n',pp3dtype);
      wbObj = WaitBarWithCancelCmdline('3d reconciliation');
      oc = onCleanup(@()delete(wbObj));
      switch pp3dtype
        case 'triangulate'
          % See PostProcess.ReconstructSampleMultiView
          
          assert(ntgt==1,'Expected single-target data in trkfiles.');
          ptrk_reshaped = cell(nvw, 1);
          for i = 1:nvw
            ptrk_reshaped{i} = reshape(permute(ptrks{i},[2 3 1]),2,nfrm*npt); % coord, frm*pt
          end
          all_ptrks = cat(3,ptrk_reshaped{:});
          
          X = nan(3,nfrm*npt);
          ptrkrp = nan(size(all_ptrks));
          % wbObj.startPeriod('Triangulation','shownumden',true,...
          %   'denominator',nfrm*npt);
          % wbObjFrmShow = 500;
          for i=1:nfrm*npt
            % if mod(i,wbObjFrmShow)==0
            %   wbObj.updateFracWithNumDen(i);
            % end
            [X(:, i), ptrkrp(:, i, :)] = crig.triangulate(all_ptrks(:, i, :));
          end
          % wbObj.endPeriod();
          X = permute(reshape(X,[3 nfrm npt]),[3 1 2]); % npt x 3 x nfrm
          ptrkrp = reshape(ptrkrp,[2 nfrm npt 1 nvw]);
          ptrkrp = permute(ptrkrp,[3 1 2 4 5]); % npt x 2 x nfrm x 1 x nvw

          trksnew = cell(nvw, 1);
          for i = 1:nvw
            trksnew{i} = struct( ...
              'pTrkSingleView',{trks{i}.pTrk},...
              'pTrk',{{ptrkrp(:,:,:,:,i)}},...
              'pTrk3d',{{X}} ...
            );
          end

        case 'experimental'
          %rois = obj.lObj.getMovieRoiMovIdx(mIdx);
          
          if ~isempty(rois),
            szassert(rois,[nvw 4]);
          end
          
          [X,ptrkrp,tMD,isspecial,prefview] = viewpref3drecon(...
            trk1,trk2,crig,'roisEPline',rois,'dxyz',DXYZ,...
            'wbObj',wbObj);
          X = permute(X,[3 2 1]); % npt x 3 x nfrm
          ptrkrp = permute(ptrkrp,[4 2 1 5 3]); % npt x 2 x nfrm x 1 x nvw
          
          trk1new = struct(...
            'pTrkSingleView',{trk1.pTrk},...
            'pTrk',{{ptrkrp(:,:,:,:,1)}},...
            'pTrk3d',{{X}},...
            'recon3d_prefview',prefview');
          trk2new = struct(...
            'pTrkSingleView',{trk2.pTrk},...
            'pTrk',{{ptrkrp(:,:,:,:,2)}});
          
          %save(trkfiles{1},'-append','-struct','trk1save');
%           fprintf(1,'New variables ''pTrkSingleView'', ''pTrk'', ''pTrk3d''.\n');
          %trkfiles{1});
          %save(trkfiles{2},'-append','-struct','trk2save');
%           fprintf(1,'New variables ''pTrkSingleView'', ''pTrk''.\n');
          %trkfiles{2});
          
        otherwise
          assert(false);
          
      end
      
    end
    
    function [v,msg] = canPostProcess(lObj,tblMFT)
      
      v = true;
      msg = '';
      
      if lObj.nview == 1,
        return;
      end
      
      sPrmAll = lObj.trackGetTrainingParams();
      pptype = APTParameters.getPostProcessReconcile3dType(sPrmAll);
      if strcmpi(pptype,'None'),
        return;
      end
      
      mIdxes = unique(tblMFT.mov);
      hasCal = true(size(mIdxes));
      for i = 1:numel(mIdxes),
        vcd = lObj.getViewCalibrationDataMovIdx(mIdxes(i));
        if isempty(vcd),
          hasCal(i) = false;
        end
      end
      
      if any(~hasCal),
        v = false;
        msg = sprintf('Postprocessing includes triangulation, but movies with the following indices are missing calibration files: %s. Please add calibration files for these movies.',mat2str(mIdxes(~hasCal)));
      end
      
    end
      
  end
  
end