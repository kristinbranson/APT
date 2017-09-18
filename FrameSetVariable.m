classdef FrameSetVariable < FrameSet
  properties
    prettyStringHook % fcn with sig str = fcn(labeler)
    
    % fcn with sig frms = fcn(labeler,mIdx,nfrm,iTgt). Returns "base"
    % frames. nfrm is number of frames in mIdx (purely convenience).
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
    function frms = getFrames(obj,labelerObj,mIdx,iTgt,decFac)
      % Get frames to track for given movie/target/decimation
      %
      % mIdx: scalar MovieIndex
      % iTgt: scalar target
      % decFac: positive int, decimation factor
      %
      % frms: vector of frames for given movie. Can be empty for various 
      % reasons, eg:
      %   * iTgt is not present in iMov
      %   * frames where iTgt is live in iMov do not intersect with obj
      
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      
      nfrm = labelerObj.getNFramesMovIdx(mIdx);
      
      % Step 1: figure out "base" frms, independent of target/decimation
      frms = obj.getFramesBase(labelerObj,mIdx,nfrm,iTgt);
      frms = unique(frms);
      frms = frms(:)';
      
      % Step 2: restrict based on target live-ness
      tfOOB = frms<1 | frms>nfrm;
      if any(tfOOB)
        warningNoTrace('Discarding %d out-of-bounds frames.',nnz(tfOOB));
      end
      frms = frms(~tfOOB);
      
      if labelerObj.hasTrx
        tfaf = labelerObj.getTrxFilesAllFullMovIdx(mIdx);
        [~,~,frm2trxTotAnd] = Labeler.getTrxCacheAcrossViewsStc(...
          labelerObj.trxCache,tfaf,nfrm);
        frm2trxTotAndTgt = frm2trxTotAnd(:,iTgt);        
        % frm2trxOverallTgt is [nfrmx1] logical, true at frames where iTgt 
        % is live in all views
        
        tfFrmOK = frm2trxTotAndTgt(frms);
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
    AllFrm = FrameSetVariable(@(lo)'All frames',@lclAllFrmGetFrms);
    SelFrm = FrameSetVariable(@(lo)'Selected frames',@lclSelFrmGetFrms);
    WithinCurrFrm = FrameSetVariable(@lclWithinCurrFrmPrettyStr,@lclWithinCurrFrmGetFrms);
    LabeledFrm = FrameSetVariable(@(lo)'Labeled frames',@lclLabeledFrmGetFrms);
  end  
end

function str = lclWithinCurrFrmPrettyStr(lObj)
str = sprintf('Within %d frames of current frame',lObj.trackNFramesNear);
end
function frms = lclAllFrmGetFrms(lObj,mIdx,nfrm,iTgt)
frms = 1:nfrm;
end
function frms = lclSelFrmGetFrms(lObj,mIdx,nfrm,iTgt)
% .selectedFrames are conceptually wrt current movie, which in general 
% differs from iMov; action may still make sense however, eg "frames 
% 100-300"
frms = lObj.selectedFrames;
end
function frms = lclWithinCurrFrmGetFrms(lObj,mIdx,nfrm,iTgt)
currFrm = lObj.currFrame; % Note currentMovie~=iMov in general
df = lObj.trackNFramesNear;
frm0 = max(currFrm-df,1);
frm1 = min(currFrm+df,nfrm);
frms = frm0:frm1;
end
function frms = lclLabeledFrmGetFrms(lObj,mIdx,nfrm,iTgt)
npts = lObj.nLabelPoints;
lpos = lObj.getLabeledPosMovIdx(mIdx); % [nptsx2xnfrmxntgt]
lposTgt = reshape(lpos(:,:,:,iTgt),[2*npts nfrm]);
tfLbledFrm = any(~isnan(lposTgt),1); % considered labeled if any x- or y-coord is non-nan
frms = find(tfLbledFrm);
end
