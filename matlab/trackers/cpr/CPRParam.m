classdef CPRParam
  % CPR Parameter notes 20170502
  % 
  % CPRLabelTracker (CPRLT) owns the ultimately-used parameter structure 
  % (.sPrm) used in training/tracking. It is partially replicated in 
  % RegressorCascade but this is an impl detail.
  %
  % CPRLabelTracker uses the older/more-complete form of parameters that
  % has redundancies, (many) currently unused fields, etc. These
  % parameters come from the original CPR code. The defaults for these are
  % given in .../cpr/param.example.yaml.
  %
  % For ease-of-use, APT users are shown a reduced/simplified set of
  % parameters, sometimes called "new" parameters. The defaults for these
  % are given in .../cpr/params_cpr.yaml.
  %
  % This file contains conversion routines (below) for going from new-style
  % parameters to old-style parameters and vice-versa.
  %
  % Forward/maintenance plan.
  % * If user loads an existing project, CPRLT modernizes .sPrm on load. 
  % This is done by reading CPRLabelTracker.readDefaultParams and 
  % overlaying the project contents on top. (This is in "Old-style" 
  % parameter space.). Old-style params are then converted to new-style 
  % params if/when modification in the UI is desired.
  % * Over time, "backend" code (trackers etc) that use old-style
  % parameters should migrade over to using the new-style parameters. In
  % this way the old-style parameters will be gradually phased out.
  
  methods (Static)
    
    function [sPrmCPR,varargout] = all2cpr(sPrmAll,nPhysPoints,nview)
      sPrmPPandCPR = sPrmAll;
      sPrmPPandCPR.ROOT = rmfield(sPrmPPandCPR.ROOT,'DeepTrack');
      if nargout > 1,
        [sPrmPPandCPRold,varargout{:}] = CPRParam.new2old(sPrmPPandCPR,nPhysPoints,nview);
      else
        [sPrmPPandCPRold] = CPRParam.new2old(sPrmPPandCPR,nPhysPoints,nview);
      end
      sPrmCPR = rmfield(sPrmPPandCPRold,'PreProc');
    end
    
    function [sOld,trkNFrmsSm,trkNFrmsLg,trkNFrmsNear] = ...
        new2old(sNew,nphyspts,nviews)
      % Convert new-style parameters to old-style parameters. Defaults are
      % used for old fields when appropriate. Some old-style fields that 
      % are currently unnecessary are omitted.
      %
      % The additional return args trkNFrms* are b/c the new-style
      % parameters now store some general tracking-related parameters that
      % are stored on lObj rather than in the legacy CPR params.
            
      sNew = APTParameters.enforceConsistency(sNew);
      
      sOld = struct();
      sOld.Model.name = '';
      sOld.Model.nfids = nphyspts;
      sOld.Model.d = 2;
      sOld.Model.nviews = nviews;
      sOld.Model.D = sOld.Model.d*sOld.Model.nfids;
      
%       trkType = TrackerType.(sNew.ROOT.Track.Type);
      if nargout >= 2,
        trkNFrmsSm = APTParameters.getTrackNFramesSmall(sNew);
        trkNFrmsLg = APTParameters.getTrackNFramesLarge(sNew);
        trkNFrmsNear = APTParameters.getTrackNFramesNeighborhood(sNew);
      end

      sOld.PreProc.BackSub = APTParameters.getBackSubParams(sNew);
      sOld.PreProc.histeq = APTParameters.getUseHistEq(sNew);
%       sOld.PreProc.histeqH0NumFrames = he.NSampleH0;
      sOld.PreProc.TargetCrop = APTParameters.getMATargetCropParams(sNew);
      %sOld.PreProc.TargetCropMA = sNew.ROOT.ImageProcessing.MultiTarget.TargetCropMA;
%       sOld.PreProc.NeighborMask = sNew.ROOT.ImageProcessing.MultiTarget.NeighborMask;
      sOld.PreProc.channelsFcn = [];
      
      cpr = APTParameters.getCPRParams(sNew);
      sOld.Reg.T = cpr.NumMajorIter;
      sOld.Reg.K = cpr.NumMinorIter;
      sOld.Reg.type = 1; 
      sOld.Reg.M = cpr.Ferns.Depth;
      sOld.Reg.R = 0;
      sOld.Reg.loss = 'L2';
      sOld.Reg.prm.thrr = [cpr.Ferns.Threshold.Lo cpr.Ferns.Threshold.Hi];
      sOld.Reg.prm.reg = cpr.Ferns.RegFactor;
      switch cpr.RotCorrection.OrientationType
        case 'fixed'
          sOld.Reg.rotCorrection.use = false;
%           if sOld.PreProc.TargetCrop.AlignUsingTrxTheta
%             warningNoTrace('.OrientationType incompatible with .AlignUsingTrxTheta.');
%           end
        case 'arbitrary'
          sOld.Reg.rotCorrection.use = true;
        otherwise
          assert(false);
      end
      sOld.Reg.rotCorrection.iPtHead = cpr.RotCorrection.HeadPoint;
      sOld.Reg.rotCorrection.iPtTail = cpr.RotCorrection.TailPoint;
      sOld.Reg.occlPrm.Stot = 1; 
      
      sOld.Ftr.type = cpr.Feature.Type;
      sOld.Ftr.metatype = cpr.Feature.Metatype;
      sOld.Ftr.F = cpr.Feature.NGenerate;
      sOld.Ftr.nChn = 1;
      sOld.Ftr.radius = cpr.Feature.Radius;
      sOld.Ftr.abratio = cpr.Feature.ABRatio;
      sOld.Ftr.nsample_std = cpr.Feature.Nsample_std;
      sOld.Ftr.nsample_cor = cpr.Feature.Nsample_cor;
      sOld.Ftr.neighbors = [];
      
      sOld.TrainInit.Naug = cpr.Replicates.NrepTrain;
      sOld.TrainInit.augrotate = []; % obsolete
      sOld.TrainInit.doptjitter = cpr.Replicates.DoPtJitter;
      sOld.TrainInit.ptjitterfac = cpr.Replicates.PtJitterFac;
      sOld.TrainInit.doboxjitter = cpr.Replicates.DoBBoxJitter;
      sOld.TrainInit.augjitterfac = cpr.Replicates.AugJitterFac;
      sOld.TrainInit.augUseFF = cpr.Replicates.AugUseFF;
      sOld.TrainInit.iPt = [];
      
      sOld.TestInit.Nrep = cpr.Replicates.NrepTrack;
      sOld.TestInit.augrotate = []; % obsolete
      sOld.TestInit.doptjitter = cpr.Replicates.DoPtJitter;
      sOld.TestInit.ptjitterfac = cpr.Replicates.PtJitterFac;
      sOld.TestInit.doboxjitter = cpr.Replicates.DoBBoxJitter;
      sOld.TestInit.augjitterfac = cpr.Replicates.AugJitterFac;
      sOld.TestInit.augUseFF = cpr.Replicates.AugUseFF;
      sOld.TestInit.movChunkSize = APTParameters.getTrackChunkSize(sNew);
      
      sOld.Prune.method = cpr.Prune.Method;
      sOld.Prune.maxdensity_sigma = cpr.Prune.DensitySigma;
      sOld.Prune.poslambdafac = cpr.Prune.PositionLambdaFactor;
    end
    
%     function [sOld,trkNFrmsSm,trkNFrmsLg,trkNFrmsNear] = ...
%         new2oldPre20181108(sNew,nphyspts,nviews)
%       % Convert new-style parameters to old-style parameters. Defaults are
%       % used for old fields when appropriate. Some old-style fields that 
%       % are currently unnecessary are omitted.
%       %
%       % The additional return args trkNFrms* are b/c the new-style
%       % parameters now store some general tracking-related parameters that
%       % are stored on lObj rather than in the legacy CPR params.
%             
%       sOld = struct();
%       sOld.Model.name = '';
%       sOld.Model.nfids = nphyspts;
%       sOld.Model.d = 2;
%       sOld.Model.nviews = nviews;
%       sOld.Model.D = sOld.Model.d*sOld.Model.nfids;
%       
% %       trkType = TrackerType.(sNew.ROOT.Track.Type);
%       trkNFrmsSm = sNew.ROOT.Track.NFramesSmall;
%       trkNFrmsLg = sNew.ROOT.Track.NFramesLarge;
%       trkNFrmsNear = sNew.ROOT.Track.NFramesNeighborhood;
% 
%       he = sNew.ROOT.Track.HistEq;
%       sOld.PreProc.BackSub = sNew.ROOT.Track.BackSub;
%       sOld.PreProc.histeq = he.Use;
%       sOld.PreProc.histeqH0NumFrames = he.NSampleH0;
%       sOld.PreProc.TargetCrop = sNew.ROOT.Track.MultiTarget.TargetCrop;
%       sOld.PreProc.NeighborMask = sNew.ROOT.Track.MultiTarget.NeighborMask;
%       sOld.PreProc.channelsFcn = [];
%       
%       cpr = sNew.ROOT.CPR;
%       sOld.Reg.T = cpr.NumMajorIter;
%       sOld.Reg.K = cpr.NumMinorIter;
%       sOld.Reg.type = 1; 
%       sOld.Reg.M = cpr.Ferns.Depth;
%       sOld.Reg.R = 0;
%       sOld.Reg.loss = 'L2';
%       sOld.Reg.prm.thrr = [cpr.Ferns.Threshold.Lo cpr.Ferns.Threshold.Hi];
%       sOld.Reg.prm.reg = cpr.Ferns.RegFactor;
%       switch cpr.RotCorrection.OrientationType
%         case 'fixed'
%           sOld.Reg.rotCorrection.use = false;
%         case {'arbitrary' 'arbitrary trx-specified'}
%           sOld.Reg.rotCorrection.use = true;
%         otherwise
%           assert(false);
%       end
%       sOld.Reg.rotCorrection.iPtHead = cpr.RotCorrection.HeadPoint;
%       sOld.Reg.rotCorrection.iPtTail = cpr.RotCorrection.TailPoint;
%       sOld.Reg.occlPrm.Stot = 1; 
% %       sOld.Reg.occlPrm.nrows = 3;
% %       sOld.Reg.occlPrm.ncols = 3;
% %       sOld.Reg.occlPrm.nzones = 1;
% %       sOld.Reg.occlPrm.th = 0.5;
%       
%       sOld.Ftr.type = cpr.Feature.Type;
%       sOld.Ftr.metatype = cpr.Feature.Metatype;
%       sOld.Ftr.F = cpr.Feature.NGenerate;
%       sOld.Ftr.nChn = 1;
%       sOld.Ftr.radius = cpr.Feature.Radius;
%       sOld.Ftr.abratio = cpr.Feature.ABRatio;
%       sOld.Ftr.nsample_std = cpr.Feature.Nsample_std;
%       sOld.Ftr.nsample_cor = cpr.Feature.Nsample_cor;
%       sOld.Ftr.neighbors = [];
%       
%       sOld.TrainInit.Naug = cpr.Replicates.NrepTrain;
%       sOld.TrainInit.augrotate = []; % obsolete
%       sOld.TrainInit.usetrxorientation = ...
%         strcmp(cpr.RotCorrection.OrientationType,'arbitrary trx-specified');
%       sOld.TrainInit.doptjitter = cpr.Replicates.DoPtJitter;
%       sOld.TrainInit.ptjitterfac = cpr.Replicates.PtJitterFac;
%       sOld.TrainInit.doboxjitter = cpr.Replicates.DoBBoxJitter;
%       sOld.TrainInit.augjitterfac = cpr.Replicates.AugJitterFac;
%       sOld.TrainInit.augUseFF = cpr.Replicates.AugUseFF;
%       sOld.TrainInit.iPt = [];
%       
%       sOld.TestInit.Nrep = cpr.Replicates.NrepTrack;
%       sOld.TestInit.augrotate = []; % obsolete
%       sOld.TestInit.usetrxorientation = sOld.TrainInit.usetrxorientation;
%       sOld.TestInit.doptjitter = cpr.Replicates.DoPtJitter;
%       sOld.TestInit.ptjitterfac = cpr.Replicates.PtJitterFac;
%       sOld.TestInit.doboxjitter = cpr.Replicates.DoBBoxJitter;
%       sOld.TestInit.augjitterfac = cpr.Replicates.AugJitterFac;
%       sOld.TestInit.augUseFF = cpr.Replicates.AugUseFF;
%       sOld.TestInit.movChunkSize = sNew.ROOT.Track.ChunkSize;
%       
%       sOld.Prune.method = cpr.Prune.Method;
%       sOld.Prune.maxdensity_sigma = cpr.Prune.DensitySigma;
%       sOld.Prune.poslambdafac = cpr.Prune.PositionLambdaFactor;
%     end
    
    
%     function [sNew,npts,nviews] = old2newPre20181108(sOld,lObj)
%       % Convert old-style CPR parameters to APT-style parameters.
%       %
%       % lObj: Labeler instance. Need this b/c the new parameters include
%       % general tracking-related parameters that are set in lObj.
%       
%       npts = sOld.Model.nfids;
%       assert(sOld.Model.d==2);
%       assert(sOld.Model.D==sOld.Model.d*sOld.Model.nfids);
%       nviews = sOld.Model.nviews;      
%       
%       sNew = struct();
% %       sNew.ROOT.Track.Type = char(lObj.trackerType);
%       sNew.ROOT.Track.BackSub = sOld.PreProc.BackSub;
%       sNew.ROOT.Track.HistEq.Use = sOld.PreProc.histeq;
%       sNew.ROOT.Track.HistEq.NSampleH0 = sOld.PreProc.histeqH0NumFrames;
%       sNew.ROOT.Track.MultiTarget.TargetCrop = sOld.PreProc.TargetCrop;
%       sNew.ROOT.Track.MultiTarget.NeighborMask = sOld.PreProc.NeighborMask;
%       sNew.ROOT.Track.ChunkSize = sOld.TestInit.movChunkSize;
%       sNew.ROOT.Track.NFramesSmall = lObj.trackNFramesSmall;
%       sNew.ROOT.Track.NFramesLarge = lObj.trackNFramesLarge;
%       sNew.ROOT.Track.NFramesNeighborhood = lObj.trackNFramesNear;
%       assert(isempty(sOld.PreProc.channelsFcn));
%       
%       sNew.ROOT.CPR.NumMajorIter = sOld.Reg.T;
%       sNew.ROOT.CPR.NumMinorIter = sOld.Reg.K;
%       assert(sOld.Reg.type==1);
%       sNew.ROOT.CPR.Ferns.Depth = sOld.Reg.M;
%       assert(sOld.Reg.R==0);
%       assert(strcmp(sOld.Reg.loss,'L2'));
%       sNew.ROOT.CPR.Ferns.Threshold.Lo = sOld.Reg.prm.thrr(1);
%       sNew.ROOT.CPR.Ferns.Threshold.Hi = sOld.Reg.prm.thrr(2);
%       sNew.ROOT.CPR.Ferns.RegFactor = sOld.Reg.prm.reg;
% 
%       % dups assert below for doc purposes
%       assert(sOld.TrainInit.usetrxorientation==sOld.TestInit.usetrxorientation);
%       if sOld.Reg.rotCorrection.use
%         if sOld.TrainInit.usetrxorientation
%           sNew.ROOT.CPR.RotCorrection.OrientationType = 'arbitrary trx-specified'; 
%         else
%           sNew.ROOT.CPR.RotCorrection.OrientationType = 'arbitrary';        
%         end
%       else
%         sNew.ROOT.CPR.RotCorrection.OrientationType = 'fixed';
%       end
%       sNew.ROOT.CPR.RotCorrection.HeadPoint = sOld.Reg.rotCorrection.iPtHead;
%       sNew.ROOT.CPR.RotCorrection.TailPoint = sOld.Reg.rotCorrection.iPtTail;
%       
%       assert(sOld.Reg.occlPrm.Stot==1); 
% %       sOld.Reg.occlPrm.nrows = 3;
% %       sOld.Reg.occlPrm.ncols = 3;
% %       sOld.Reg.occlPrm.nzones = 1;
% %       sOld.Reg.occlPrm.th = 0.5;
%       
%       sNew.ROOT.CPR.Feature.Type = sOld.Ftr.type;
%       sNew.ROOT.CPR.Feature.Metatype = sOld.Ftr.metatype;
%       sNew.ROOT.CPR.Feature.NGenerate = sOld.Ftr.F;
%       assert(sOld.Ftr.nChn==1);
%       sNew.ROOT.CPR.Feature.Radius = sOld.Ftr.radius;
%       sNew.ROOT.CPR.Feature.ABRatio = sOld.Ftr.abratio;
%       sNew.ROOT.CPR.Feature.Nsample_std = sOld.Ftr.nsample_std;
%       sNew.ROOT.CPR.Feature.Nsample_cor = sOld.Ftr.nsample_cor;
%       assert(isempty(sOld.Ftr.neighbors));
%       
%       sNew.ROOT.CPR.Replicates.NrepTrain = sOld.TrainInit.Naug;
%       sNew.ROOT.CPR.Replicates.NrepTrack = sOld.TestInit.Nrep;
%       if ~isempty(sOld.TrainInit.augrotate)
%         assert(sOld.TrainInit.augrotate==sOld.TestInit.augrotate);
%         if sOld.TrainInit.augrotate~=sOld.Reg.rotCorrection.use
%           warningNoTrace('CPRParam:rot',...
%             'TrainInit.augrotate (%d) differs from Reg.rotCorrection.use (%d). Ignoring value of TrainInit.augrotate.',...
%             sOld.TrainInit.augrotate,sOld.Reg.rotCorrection.use);
%         end
%       end
%       assert(sOld.TrainInit.usetrxorientation==sOld.TestInit.usetrxorientation);
%       assert(sOld.TrainInit.doptjitter==sOld.TestInit.doptjitter);
%       assert(sOld.TrainInit.ptjitterfac==sOld.TestInit.ptjitterfac);
%       assert(sOld.TrainInit.doboxjitter==sOld.TestInit.doboxjitter);
%       assert(sOld.TrainInit.augjitterfac==sOld.TestInit.augjitterfac);
%       assert(sOld.TrainInit.augUseFF==sOld.TestInit.augUseFF);
%       %sNew.ROOT.CPR.Replicates.AugRotate = sOld.TrainInit.augrotate;
%       
%       sNew.ROOT.CPR.Replicates.DoPtJitter = sOld.TrainInit.doptjitter;
%       sNew.ROOT.CPR.Replicates.PtJitterFac = sOld.TrainInit.ptjitterfac;
%       sNew.ROOT.CPR.Replicates.DoBBoxJitter = sOld.TrainInit.doboxjitter;
%       sNew.ROOT.CPR.Replicates.AugJitterFac = sOld.TrainInit.augjitterfac;
%       sNew.ROOT.CPR.Replicates.AugUseFF = sOld.TrainInit.augUseFF;
%       assert(isempty(sOld.TrainInit.iPt));
%       
%       sNew.ROOT.CPR.Prune.Method = sOld.Prune.method;
%       sNew.ROOT.CPR.Prune.DensitySigma = sOld.Prune.maxdensity_sigma;
%       sNew.ROOT.CPR.Prune.PositionLambdaFactor = sOld.Prune.poslambdafac;
%     end
    
    function [sNew,npts,nviews] = old2new(sOld,lObj)
      % Convert old-style CPR parameters to APT-style parameters.
      %
      % lObj: Labeler instance. Need this b/c the new parameters include
      % general tracking-related parameters that are set in lObj.
      
      if nargin < 2,
        lObj = [];
      end
      
      npts = sOld.Model.nfids;
      assert(sOld.Model.d==2);
      assert(sOld.Model.D==sOld.Model.d*sOld.Model.nfids);
      nviews = sOld.Model.nviews;      
      
      sNew = CPRParam.old2newPPOnly(sOld.PreProc);
      sNew = APTParameters.setCPRParams(sNew,sOld);
      if nargin > 1 && ~isempty(lObj),
        sNew = setNFramesTrackParams(sNew,lObj);
      end
      
      sNew = APTParameters.enforceConsistency(sNew);
    end
    
    function [sNew] = old2newPPOnly(sOld)
      
      sNew = struct();
%       sNew.ROOT.Track.Type = char(lObj.trackerType);
      sNew = APTParameters.setBackSubParams(sNew,sOld.BackSub);
      sNew = APTParameters.setUseHistEq(sNew,sOld.histeq);
%       sNew.ROOT.ImageProcessing.HistEq.NSampleH0 = sOld.histeqH0NumFrames;
      sNew = APTParameters.setMATargetCropParams(sNew,sOld.TargetCrop);
      %sNew.ROOT.MultiAnimal.TargetCropMA = sOld.TargetCropMA;
      assert(isfield(sOld.TargetCrop,'AlignUsingTrxTheta'));
      %sNew.ROOT.ImageProcessing.MultiTarget.NeighborMask = sOld.NeighborMask;
      assert(isempty(sOld.channelsFcn));
    end
    
    
    function [sPrmCPR,chunkSize] = old2newCPROnly(sOld)
      % Convert old-style CPR parameters to APT-style parameters.
      %
      % lObj: Labeler instance. Need this b/c the new parameters include
      % general tracking-related parameters that are set in lObj.
      
      sPrmCPR = struct();
      sPrmCPR.NumMajorIter = sOld.Reg.T;
      sPrmCPR.NumMinorIter = sOld.Reg.K;
      assert(sOld.Reg.type==1);
      sPrmCPR.Ferns.Depth = sOld.Reg.M;
      assert(sOld.Reg.R==0);
      assert(strcmp(sOld.Reg.loss,'L2'));
      sPrmCPR.Ferns.Threshold.Lo = sOld.Reg.prm.thrr(1);
      sPrmCPR.Ferns.Threshold.Hi = sOld.Reg.prm.thrr(2);
      sPrmCPR.Ferns.RegFactor = sOld.Reg.prm.reg;

      % dups assert below for doc purposes
      if sOld.Reg.rotCorrection.use
        sPrmCPR.RotCorrection.OrientationType = 'arbitrary';
      else
        sPrmCPR.RotCorrection.OrientationType = 'fixed';
        % enforceConsistency called below
%         if sNew.ROOT.ImageProcessing.MultiTarget.TargetCrop.AlignUsingTrxTheta
%           warningNoTrace('.OrientationType incompatible with .AlignUsingTrxTheta.');
%         end
      end
      sPrmCPR.RotCorrection.HeadPoint = sOld.Reg.rotCorrection.iPtHead;
      sPrmCPR.RotCorrection.TailPoint = sOld.Reg.rotCorrection.iPtTail;
      
      assert(sOld.Reg.occlPrm.Stot==1); 
      
      sPrmCPR.Feature.Type = sOld.Ftr.type;
      sPrmCPR.Feature.Metatype = sOld.Ftr.metatype;
      sPrmCPR.Feature.NGenerate = sOld.Ftr.F;
      assert(sOld.Ftr.nChn==1);
      sPrmCPR.Feature.Radius = sOld.Ftr.radius;
      sPrmCPR.Feature.ABRatio = sOld.Ftr.abratio;
      sPrmCPR.Feature.Nsample_std = sOld.Ftr.nsample_std;
      sPrmCPR.Feature.Nsample_cor = sOld.Ftr.nsample_cor;
      assert(isempty(sOld.Ftr.neighbors));
      
      sPrmCPR.Replicates.NrepTrain = sOld.TrainInit.Naug;
      sPrmCPR.Replicates.NrepTrack = sOld.TestInit.Nrep;
      if ~isempty(sOld.TrainInit.augrotate)
        assert(sOld.TrainInit.augrotate==sOld.TestInit.augrotate);
        if sOld.TrainInit.augrotate~=sOld.Reg.rotCorrection.use
          warningNoTrace('CPRParam:rot',...
            'TrainInit.augrotate (%d) differs from Reg.rotCorrection.use (%d). Ignoring value of TrainInit.augrotate.',...
            sOld.TrainInit.augrotate,sOld.Reg.rotCorrection.use);
        end
      end
      assert(sOld.TrainInit.doptjitter==sOld.TestInit.doptjitter);
      assert(sOld.TrainInit.ptjitterfac==sOld.TestInit.ptjitterfac);
      assert(sOld.TrainInit.doboxjitter==sOld.TestInit.doboxjitter);
      assert(sOld.TrainInit.augjitterfac==sOld.TestInit.augjitterfac);
      assert(sOld.TrainInit.augUseFF==sOld.TestInit.augUseFF);
      %sPrmCPR.Replicates.AugRotate = sOld.TrainInit.augrotate;
      
      sPrmCPR.Replicates.DoPtJitter = sOld.TrainInit.doptjitter;
      sPrmCPR.Replicates.PtJitterFac = sOld.TrainInit.ptjitterfac;
      sPrmCPR.Replicates.DoBBoxJitter = sOld.TrainInit.doboxjitter;
      sPrmCPR.Replicates.AugJitterFac = sOld.TrainInit.augjitterfac;
      sPrmCPR.Replicates.AugUseFF = sOld.TrainInit.augUseFF;
      assert(isempty(sOld.TrainInit.iPt));
      
      sPrmCPR.Prune.Method = sOld.Prune.method;
      sPrmCPR.Prune.DensitySigma = sOld.Prune.maxdensity_sigma;
      sPrmCPR.Prune.PositionLambdaFactor = sOld.Prune.poslambdafac;
      chunkSize = sOld.TestInit.movChunkSize;
      
    end
    
  end
end