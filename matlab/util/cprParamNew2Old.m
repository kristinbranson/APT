function [sOld,trkNFrmsSm,trkNFrmsLg,trkNFrmsNear] = ...
    cprParamNew2Old(sNew,nphyspts,nviews)
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
    trkNFrmsSm = sNew.ROOT.Track.NFramesSmall;
    trkNFrmsLg = sNew.ROOT.Track.NFramesLarge;
    trkNFrmsNear = sNew.ROOT.Track.NFramesNeighborhood;
  end

  he = sNew.ROOT.ImageProcessing.HistEq;
  sOld.PreProc.BackSub = sNew.ROOT.ImageProcessing.BackSub;
  sOld.PreProc.histeq = he.Use;
%       sOld.PreProc.histeqH0NumFrames = he.NSampleH0;
  sOld.PreProc.TargetCrop = sNew.ROOT.MultiAnimal.TargetCrop;
  %sOld.PreProc.TargetCropMA = sNew.ROOT.ImageProcessing.MultiTarget.TargetCropMA;
%       sOld.PreProc.NeighborMask = sNew.ROOT.ImageProcessing.MultiTarget.NeighborMask;
  sOld.PreProc.channelsFcn = [];

  cpr = sNew.ROOT.CPR;
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
  sOld.TestInit.movChunkSize = sNew.ROOT.Track.ChunkSize;

  sOld.Prune.method = cpr.Prune.Method;
  sOld.Prune.maxdensity_sigma = cpr.Prune.DensitySigma;
  sOld.Prune.poslambdafac = cpr.Prune.PositionLambdaFactor;
end % function
