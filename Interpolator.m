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
    
    function track(obj,~,~)
      % Currently always tracks all frames.
      
      prm = obj.readParamFileYaml();
      smoothspan = prm.SmoothSpan;
      
      labeler = obj.lObj;
      npts = labeler.nLabelPoints;
      lposnew = cell(labeler.nmovies,1);
      for iMov = 1:labeler.nmovies
        fprintf(1,'Interpolating movie %d\n',iMov);
        
        lpos = labeler.labeledpos{iMov}; % nptsx2xnfrm
        for iPt = 1:npts
        for j = 1:2
          lpos(iPt,j,:) = smooth(squeeze(lpos(iPt,j,:)),smoothspan);
        end
        end
        lposnew{iMov} = lpos;        
      end
      
      % call labeler to set new "real" labels. 
      labeler.labelPosSuperBulkImport(lposnew);
    end
    
  end
  
end