classdef FrameSet < handle
  % A FrameSet represents a conceptual set of frames for a movie in an APT 
  % project. An explicit list of frame numbers can be generated given a 
  % labeler instance, movie/target index, and decimation.

  enumeration
    % For projects w/out trx, this is "all frames in movie."
    % For projects with trx, this is "all frames in movie where given
    %   target is live"
    % This treatment pertains to all FrameSet enums
    AllFrm ('All frames',[],[])
    SelFrm ('Selected frames',[],[])
    WithinCurrFrm ('Within %d frames of current frame','trackNFramesNear',[])
    LabeledFrm ('Labeled frames',[],[])
    % Weird, single Custom instance across entire MATLAB session.
    % FrameSet.Custom objects should be very short-lived (only instantiated
    % at cmdline or from APTCluster)
    Custom ('Custom',[],[])
  end

  properties
    prettyStringPat
    labelerProp
    info
  end
  
  methods
    
    function obj = FrameSet(ps,lprop,ifo)
      obj.prettyStringPat = ps;
      obj.labelerProp = lprop;
      obj.info = ifo;
    end
    
    function v = getPrettyString(obj,labelerObj)
      lprop = obj.labelerProp;
      if isempty(lprop)
        v = obj.prettyStringPat;
      else
        val = labelerObj.(lprop);
        v = sprintf(obj.prettyStringPat,val);
      end
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
      
      % Step 1: figure out frms independent of trx status
      switch obj
        case FrameSet.AllFrm
          frms = 1:nfrm;
        case FrameSet.SelFrm
          % .selectedFrames are conceptually wrt current movie, which 
          % in general differs from iMov; action may still make sense
          % however, eg "frames 100-300"
          frms = labelerObj.selectedFrames; 
        case FrameSet.WithinCurrFrm
          currFrm = labelerObj.currFrame; % Again, currentMovie~=iMov in general
          df = labelerObj.(obj.labelerProp);
          frm0 = max(currFrm-df,1);
          frm1 = min(currFrm+df,nfrm);
          frms = frm0:frm1;
        case FrameSet.LabeledFrm
          % XXX GT MERGE
          npts = labelerObj.nLabelPoints;
          lpos = labelerObj.labeledpos{iMov}; % [nptsx2xnfrmxntgt]
          lposTgt = reshape(lpos(:,:,:,iTgt),[2*npts nfrm]);
          tfLbledFrm = any(~isnan(lposTgt),1); % considered labeled if any x- or y-coord is non-nan
          frms = find(tfLbledFrm);
        case FrameSet.Custom
          frms = obj.info;
        otherwise
          assert(false);
      end
      
      frms = unique(frms);
      frms = frms(:)';
      
      % Step 2: restrict based on trx status
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
        % no restriction
        assert(iTgt==1);
      end
      
      % Step 3: decimate
      frms = frms(1:decFac:numel(frms));
    end
    
  end
  
end