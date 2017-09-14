classdef FrameSetVariable < FrameSet
  properties
    prettyStringHook % fcn with sig str = fcn(labeler)
    
    % fcn with sig frms = fcn(labeler,iMov,nfrm,iTgt). Returns "base"
    % frames. nfrm is number of frames in iMov (purely convenience).
    getFramesBase 
  end
  methods
    function obj = FrameSetVariable(psFcn,frmfcn)
      obj.prettyStringHook = psFcn;
      obj.getFramesBase = frmfcn;
    end
    function str = getPrettyString(obj,labelerObj)
      str = obj.prettyStringHook(labelerObj);
    end
    function frms = getFrames(obj,labelerObj,iMov,iTgt,decFac)
      % Get frames to track for given movie/target/decimation
      %
      % iMov: scalar movie index
      % iTgt: scalar target
      % decFac: positive int, decimation factor
      %
      % frms: vector of frames for given movie. Can be empty for various 
      % reasons, eg:
      %   * iTgt is not present in iMov
      %   * frames where iTgt is live in iMov do not intersect with obj
      
      assert(isscalar(iMov));
      movInfo = labelerObj.movieInfoAll{iMov,1}; % XXX GT MERGE. Multiview, info.nframes should be common minimum
      nfrm = movInfo.nframes;
      
      % Step 1: figure out "base" frms, independent of target/decimation
      frms = obj.getFramesBase(labelerObj,iMov,nfrm,iTgt);
      frms = unique(frms);
      frms = frms(:)';
      
      % Step 2: restrict based on target live-ness
      tfOOB = frms<1 | frms>nfrm;
      if any(tfOOB)
        warningNoTrace('Discarding %d out-of-bounds frames.',nnz(tfOOB));
      end
      frms = frms(~tfOOB);
      
      if labelerObj.hasTrx
        tfaf = labelerObj.trxFilesAllFull(iMov,:);
        [~,frm2trxI] = cellfun(@(x)labelerObj.getTrx(x,nfrm),tfaf,'uni',0);
        frm2trxOverallTgt = frm2trxI{1}(:,iTgt);
        for iView=2:numel(frm2trxI)
          frm2trxOverallTgt = and(frm2trxOverallTgt,frm2trxI{iView}(:,iTgt));
        end
        % frm2trxOverallTgt is [nfrmx1] logical, true at frames where iTgt 
        % is live in all views
        
        tfFrmOK = frm2trxOverallTgt(frms);
        frms(~tfFrmOK) = [];
      else
        % no target-based restriction
        assert(iTgt==1);
      end
      
      % Step 3: decimate
      frms = frms(1:decFac:numel(frms));
    end
  end
  
  properties (Constant) % canned/enumerated vals
    AllFrm = FrameSetVariable(@(lo)'All frame',@lclAllFrmGetFrms);
    SelFrm = FrameSetVariable(@(lo)'Selected frames',@lclSelFrmGetFrms);
    WithinCurrFrm = FrameSetVariable(@lclWithinCurrFrmPrettyStr,@lclWithinCurrFrmGetFrms);
    LabeledFrm = FrameSetVariable(@(lo)'Labeled frames',@lclLabeledFrmGetFrms);
  end  
end

function str = lclWithinCurrFrmPrettyStr(lObj)
str = sprintf('Within %d frames of current frame',lObj.trackNFramesNear);
end
function frms = lclAllFrmGetFrms(lObj,iMov,nfrm,iTgt)
frms = 1:nfrm;
end
function frms = lclSelFrmGetFrms(lObj,iMov,nfrm,iTgt)
% .selectedFrames are conceptually wrt current movie, which in general 
% differs from iMov; action may still make sense however, eg "frames 
% 100-300"
frms = lObj.selectedFrames;
end
function frms = lclWithinCurrFrmGetFrms(lObj,iMov,nfrm,iTgt)
currFrm = lObj.currFrame; % Note currentMovie~=iMov in general
df = lObj.trackNFramesNear;
frm0 = max(currFrm-df,1);
frm1 = min(currFrm+df,nfrm);
frms = frm0:frm1;
end
function frms = lclLabeledFrmGetFrms(lObj,iMov,nfrm,iTgt)
npts = lObj.nLabelPoints;
lpos = lObj.labeledpos{iMov}; % [nptsx2xnfrmxntgt] % XXX GT MERGE
lposTgt = reshape(lpos(:,:,:,iTgt),[2*npts nfrm]);
tfLbledFrm = any(~isnan(lposTgt),1); % considered labeled if any x- or y-coord is non-nan
frms = find(tfLbledFrm);
end
