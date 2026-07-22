function [sOld,trkNFrmsSm,trkNFrmsLg,trkNFrmsNear] = ...
    preProcParamsNew2Old(sNew,nphyspts,nviews)
  % Convert new-style parameters to the old-style preprocessing parameter
  % struct still consumed by the tracking pipeline. Defaults are used for
  % old fields when appropriate.
  %
  % This formerly also produced the old-style CPR algorithm parameters, but
  % CPR is no longer supported and the ROOT.CPR parameter subtree has been
  % removed, so only the Model/PreProc portion remains.
  %
  % The additional return args trkNFrms* are b/c the new-style
  % parameters now store some general tracking-related parameters that
  % are stored on lObj rather than in the params.

  sNew = APTParameters.enforceConsistency(sNew);

  sOld = struct();
  sOld.Model.name = '';
  sOld.Model.nfids = nphyspts;
  sOld.Model.d = 2;
  sOld.Model.nviews = nviews;
  sOld.Model.D = sOld.Model.d*sOld.Model.nfids;

  if nargout >= 2,
    trkNFrmsSm = sNew.ROOT.Track.NFramesSmall;
    trkNFrmsLg = sNew.ROOT.Track.NFramesLarge;
    trkNFrmsNear = sNew.ROOT.Track.NFramesNeighborhood;
  end

  he = sNew.ROOT.ImageProcessing.HistEq;
  sOld.PreProc.BackSub = sNew.ROOT.ImageProcessing.BackSub;
  sOld.PreProc.histeq = he.Use;
  sOld.PreProc.TargetCrop = sNew.ROOT.MultiAnimal.TargetCrop;
  sOld.PreProc.channelsFcn = [];
end % function
