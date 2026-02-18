function [sNew] = cprParamOld2NewPPOnly(sOld)

  sNew = struct();
%       sNew.ROOT.Track.Type = char(lObj.trackerType);
  sNew.ROOT.ImageProcessing.BackSub = sOld.BackSub;
  sNew.ROOT.ImageProcessing.HistEq.Use = sOld.histeq;
%       sNew.ROOT.ImageProcessing.HistEq.NSampleH0 = sOld.histeqH0NumFrames;
  sNew.ROOT.MultiAnimal.TargetCrop = sOld.TargetCrop;
  %sNew.ROOT.MultiAnimal.TargetCropMA = sOld.TargetCropMA;
  assert(isfield(sOld.TargetCrop,'AlignUsingTrxTheta'));
  %sNew.ROOT.ImageProcessing.MultiTarget.NeighborMask = sOld.NeighborMask;
  assert(isempty(sOld.channelsFcn));
end % function
