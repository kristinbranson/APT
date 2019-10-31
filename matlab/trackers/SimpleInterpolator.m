classdef SimpleInterpolator < LabelTracker
    
  %% Ctor/Dtor
  methods
    
    function obj = SimpleInterpolator(lObj)
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

      labeler = obj.lObj;
      assert(labeler.nTargets==1);
      npts = labeler.nLabelPoints;
      for i=1:numel(iMovs)
        iM = iMovs(i);
        fs = frms{i};
        if any( (fs(2:end)-fs(1:end-1))>1.5)
          error('Interpolator:FramesIncorrect','frames must be consecutive');
        end
        
        fprintf(1,'Interpolating movie %d\n',iM);
        
        lpos = labeler.labeledposGTaware{iM}; % [npts x 2 x nfrm]
        marked = labeler.labeledposMarked{iM};
        newPts = lpos(:,:,fs);
        for iPt = 1:npts
          
          curPts = lpos(iPt,:,fs);
          fixed = marked(iPt,fs);
          
          fpts = find(fixed);
          if numel(fpts)<2,
            warning('Fewer than 2 labeled points for the point %d, cannot interpolate',iPt);
            continue;
          end
          for ndx = 1:numel(fpts)-1
            iStart = fpts(ndx);
            iEnd = fpts(ndx+1);
            iSz = iEnd-iStart+1;
            if (iSz-1) > prm.maxInterval
              continue;
            end
            for dd = 1:size(lpos,2);
              newPts(iPt,dd,iStart:iEnd) = linspace(...
                curPts(:,dd,iStart),curPts(:,dd,iEnd),iSz);
            end
          end
            
        end
        
        labeler.labelPosSetUnmarkedFramesMovieFramesUnmarked(...
          newPts,iM,fs);
        labeler.setFrame(labeler.currFrame,'tfforcelabelupdate',true);
      end
    end
    
  end
  
end