classdef GMMTracker < LabelTracker
    
  %% Ctor/Dtor
  methods
    
    function obj = GMMTracker(lObj)
      obj@LabelTracker(lObj);
    end
        
  end
    
  
  %%
  methods
    
    function train(obj) %#ok<MANU>
      warningNoTrace('No training required for this tracker.');
    end
    
    function track(obj,iMovs,frms)
      
%       prm = obj.readParamFileYaml();
      
      dampen = 0.5;
      poslambdafixed = 100;
      
      if exist('trkfiles','var')==0
        movfiles = obj.lObj.movieFilesAllFull(iMovs);
        trkfiles = cellfun(@Labeler.defaultTrkFileName,movfiles,'uni',0);
      end

      labeler = obj.lObj;
      assert(labeler.nTargets==1);
      npts = labeler.nLabelPoints;
      for i=1:numel(iMovs)
        iM = iMovs(i);
        fs = frms{i};
        if any( (fs(2:end)-fs(1:end-1))>1.5)
          error('Interpolator:FramesIncorrect','frames must be consecutive');
        end
        
        T = load(trkfiles{i},'-mat');
        
        fprintf(1,'Interpolating movie %d\n',iM);
        
        lpos = labeler.labeledpos{iM}; % [npts x 2 x nfrm]
        marked = labeler.labeledposMarked{iM};
        newPts = nan(npts,size(lpos,2),numel(fs));
        hwait = waitbar(0,sprintf('Interpolating pt %d for movie %d',1,iM));
        for iPt = 1:npts
          
          curPts = squeeze(lpos(iPt,:,fs));
          fixed = marked(iPt,fs);
          curS = squeeze(permute(T.pSample(iPt,:,:,fs),[2,4,3,1]));
          curWt = permute( squeeze(T.weights(:,iPt,fs)),[2,1]);
          
          % Add fixed points to the samples.
          curS(:,:,end+1) = curPts; %#ok<AGROW>
          curWt(fixed,:) = 0;
          curWt(:,end+1) = fixed; %#ok<AGROW>
          
          fixed = double(fixed);
          fixed(~fixed) = nan;
          fixed(~isnan(fixed)) = size(curWt,2);
          
          cost = -log( curWt);
            
          [curX,~,~,~] = ChooseBestTrajectory(curS,cost,'dampen',dampen,...
            'poslambda',poslambdafixed,'fix',fixed);
          newPts(iPt,:,:) = curX;
          waitbar(iPt/npts,hwait,sprintf('Interpolating pt %d for movie %d',iPt+1,iM));
        end
        delete(hwait);
        
        labeler.labelPosSetUnmarkedFramesMovieFramesUnmarked(...
          newPts,iM,fs);
        labeler.setFrame(labeler.currFrame,'tfforcelabelupdate',true);
      end
    end
    
  end
  
end