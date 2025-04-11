classdef Interpolator < LabelTracker
    
  %% Ctor/Dtor
  methods
    
    function obj = Interpolator(lObj)
      obj@LabelTracker(lObj);
    end
        
  end
    
  
  %%
  methods
    
    function train(obj) %#ok<MANU>
      warningNoTrace('No training required for this tracker.');
    end
    
    function track(obj,iMovs,frms)
      
      prm = obj.readParamFileYaml();
      smoothspan = prm.SmoothSpan;
      
      labeler = obj.lObj;
      assert(labeler.nTargets==1);
      npts = labeler.nLabelPoints;
      for i=1:numel(iMovs)
        iM = iMovs(i);
        fs = frms{i};
        fprintf(1,'Interpolating movie %d\n',iM);
        
        lpos = labeler.labeledposGTaware{iM}; % [npts x 2 x nfrm]
        for iPt = 1:npts
        for j = 1:2
          lpos(iPt,j,:) = smooth(squeeze(lpos(iPt,j,:)),smoothspan);
        end
        end
        
        xyfrms = lpos(:,:,fs);
        labeler.labelPosSetUnmarkedFramesMovieFramesUnmarked(...
          xyfrms,iM,fs);
        labeler.setFrameGUI(labeler.currFrame,'tfforcelabelupdate',true);
      end
    end
    
  end
  
end