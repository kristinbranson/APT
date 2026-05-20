function [sPrmCPR,chunkSize] = cprParamOld2NewCPROnly(sOld)
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

end % function
