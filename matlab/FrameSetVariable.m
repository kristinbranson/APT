classdef FrameSetVariable < FrameSet
  properties
    prettyStringHook % fcn with sig str = fcn(labeler)
    prettyCompactStringHook % fcn with sig str = fcn(labeler)
    
    % fcn with sig frms = fcn(labeler,mIdx,nfrm,iTgt). Returns "base"
    % frames. nfrm is number of frames in mIdx (purely convenience).
    getFramesBase 
    
    avoidTbl % MFTtable of frames to avoid
    avoidRadius % avoidance radius-- 1=>avoidrow frame itself is avoided, but adjacent frame is not. 0=>no avoidance
  end
  methods
    function obj = FrameSetVariable(psFcn,pcsFun,frmfcn,varargin)
      [avdTbl,avdRad] = myparse(varargin,...
        'avoidTbl',[],... % specify both avoid* params or none
        'avoidRadius',[]);
      
      obj.prettyStringHook = psFcn;
      obj.prettyCompactStringHook = pcsFun;
      obj.getFramesBase = frmfcn;
      
      if ~isempty(avdTbl)
        assert(isa(avdTbl.mov,'MovieIndex'));
      end
      obj.avoidTbl = avdTbl;
      obj.avoidRadius = avdRad;
    end
    function str = getPrettyString(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;
      str = obj.prettyStringHook(labelerObj);
    end
    function str = getPrettyCompactString(obj,labelerObj)
      assert(isstruct(labelerObj), 'labelerObj, despite the name, must be a struct') ;
      str = obj.prettyCompactStringHook(labelerObj);
    end
    function frms = getFrames(obj,labelerObj,mIdx,iTgt,decFac)
      % Get frames to track for given movie/target/decimation
      %
      % mIdx: scalar MovieIndex
      % iTgt: scalar target. nan <=> "any target MA"
      % decFac: positive int, decimation factor
      %
      % frms: vector of frames for given movie. Can be empty for various 
      % reasons, eg:
      %   * iTgt is not present in iMov
      %   * frames where iTgt is live in iMov do not intersect with obj
      
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      
      nfrm = labelerObj.getNFramesMovIdx(mIdx);
      
      % Step 1: figure out "base" frms, independent of target/decimation
      frms = feval(obj.getFramesBase,labelerObj,mIdx,nfrm,iTgt);
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
        tfafempty = cellfun(@isempty,tfaf);
        assert(all(tfafempty) || all(~tfafempty)); % for given movie, either all views have a trx or all don't
        if all(tfafempty)
          % unusual, given that labeledObj.hasTrx -- but currently allowed
          assert(iTgt==1);
        else
          [~,~,frm2trxTotAnd] = Labeler.getTrxCacheAcrossViewsStc(...
            labelerObj.trxCache,tfaf,nfrm);
          frm2trxTotAndTgt = frm2trxTotAnd(:,iTgt);        
          % frm2trxOverallTgt is [nfrmx1] logical, true at frames where iTgt 
          % is live in all views

          tfFrmOK = frm2trxTotAndTgt(frms);
          frms(~tfFrmOK) = [];
        end
      elseif labelerObj.maIsMA
        % pass
      else
        % no target-based restriction
        assert(iTgt==1);
      end
      
      % Step 2.5: Avoid avoidrows, if applicable
      tblAvoid = obj.avoidTbl;
      tfAvoid = ~isempty(tblAvoid);
      if tfAvoid
        assert(issorted(frms));
        frmlast = frms(end);
        tffrms = false(1,frmlast);
        tffrms(frms) = true;
        
        rad = obj.avoidRadius-1; % radius==1 => only the avoidrow itself is avoided
        tfThisMovTgt = tblAvoid.mov==mIdx & tblAvoid.iTgt==iTgt;
        favoidctrs = tblAvoid.frm(tfThisMovTgt);
        for j=1:numel(favoidctrs)
          favoid = max(1,favoidctrs(j)-rad):min(frmlast,favoidctrs(j)+rad);
          tffrms(favoid) = false;
        end
        
        frms = find(tffrms);
      end
      
      % Step 3: decimate
      frms = frms(1:decFac:numel(frms));
    end
  end
  
  properties (Constant) % canned/enumerated vals
    AllFrm = FrameSetVariable(@(lo)'All frames',@(lo)'All fr',@FrameSetVariable.allFrmGetFrms);
    SelFrm = FrameSetVariable(@(lo)'Selected frames',@(lo)'Sel fr',@lclSelFrmGetFrms);
    WithinCurrFrm = FrameSetVariable(@lclWithinCurrFrmPrettyStr,@lclWithinCurrFrmPrettyCompactStr,@lclWithinCurrFrmGetFrms);
    WithinCurrFrmLarge = FrameSetVariable(@lclWithinCurrFrmPrettyStrLarge,@lclWithinCurrFrmPrettyCompactStrLarge,@lclWithinCurrFrmGetFrmsLarge);
    LabeledFrm = FrameSetVariable(@(lo)'Labeled frames',@(lo)'Lab fr',@FrameSetVariable.labeledFrmGetFrms); % AL 20180125: using parameterized anon fcnhandle that directly calls lclLabeledFrmGetFrmsCore fails in 17a, suspect class init issue
    Labeled2Frm = FrameSetVariable(@(lo)'Labeled frames',@(lo)'Lab fr',@lclLabeledFrmGetFrms2);
  end
  
  methods (Static)
    function frms = allFrmGetFrms(lObj,mIdx,nfrm,iTgt)
      frms = 1:nfrm;
    end
    function frms = labeledFrmGetFrms(lObj,mIdx,nfrm,iTgt)
      frms = lclLabeledFrmGetFrmsCore(lObj,mIdx,nfrm,iTgt,false);
    end
  end
end

function str = lclWithinCurrFrmPrettyStr(lObj)
if isunix && ~ismac
  str = sprintf('Nearest %d frames',2*lObj.trackNFramesNear);
else
  str = sprintf('Within %d frames of current frame',lObj.trackNFramesNear);
end
end
function str = lclWithinCurrFrmPrettyStrLarge(lObj)
if isunix && ~ismac
  str = sprintf('Nearest %d frames',2*lObj.trackNFramesNear*5);
else
  str = sprintf('Within %d frames of current frame',lObj.trackNFramesNear*5);
end
end
function str = lclWithinCurrFrmPrettyCompactStr(lObj)
str = sprintf('+/-%d fr',lObj.trackNFramesNear);
end
function str = lclWithinCurrFrmPrettyCompactStrLarge(lObj)
str = sprintf('+/-%d fr',5*lObj.trackNFramesNear);
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
function frms = lclWithinCurrFrmGetFrmsLarge(lObj,mIdx,nfrm,iTgt)
currFrm = lObj.currFrame; % Note currentMovie~=iMov in general
df = lObj.trackNFramesNear*5;
frm0 = max(currFrm-df,1);
frm1 = min(currFrm+df,nfrm);
frms = frm0:frm1;
end
function frms = lclLabeledFrmGetFrms2(lObj,mIdx,nfrm,iTgt)
frms = lclLabeledFrmGetFrmsCore(lObj,mIdx,nfrm,iTgt,true);
end
function frms = lclLabeledFrmGetFrmsCore(lObj,mIdx,nfrm,iTgt,tfLbls2)
% iTgt=nan <=> "any target MA"
if tfLbls2 & (~lObj.maIsMA)
  s = lObj.getLabels2MovIdx(mIdx);
  frms = s.isLabeledT(iTgt);
else
  s = lObj.getLabelsMovIdx(mIdx);
  frms = Labels.isLabeledT(s,iTgt);
end
end
