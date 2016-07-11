classdef GMMTracker < LabelTracker
  
  properties
    menu_gmm
    menu_simple
    curmode  % 0 for gmm, 1 for simple
    showhidesamples %0 for hide, 1 for show.
  end
  
  %% Ctor/Dtor
  methods
    
    function obj = GMMTracker(lObj)
      obj@LabelTracker(lObj);
      tmenu = uimenu(lObj.gdata.menu_track,'Label','Tracking Type');
      obj.menu_gmm = uimenu(tmenu,'Label','GMM','Callback',@obj.setmodeGMM);
      obj.menu_simple = uimenu(tmenu,'Label','Simple','Callback',@obj.setmodeSimple);
      set(obj.menu_gmm,'Checked','on');
      set(obj.menu_simple,'Checked','off');
      obj.curmode = 0;
      obj.menu_track_showsamples = uimenu(tmenu,'Label','Show Samples','Callback',@obj.showhideSamples);
      obj.showhidesamples = 0;
    end
        
  end
    
  
  %%
  methods
    
    function train(obj) %#ok<MANU>
      warningNoTrace('No training required for this tracker.');
    end
    
    function track(obj,iMovs,frms)
      switch obj.curmode,
        case 0
          obj.GMMtrack(iMovs,frms);
        case 1
          obj.SimpleTrack(iMovs,frms);
      end
    end
    
    function GMMtrack(obj,iMovs,frms)
      
%       prm = obj.readParamFileYaml();
      
      dampen = 0.5;
      poslambdafixed = 100;
      
      labeler = obj.lObj;

      if exist('trkfiles','var')==0
        movfiles = obj.lObj.movieFilesAllFull(iMovs);
        trkfiles = cellfun(@labeler.defaultTrkFileName,movfiles,'uni',0);
      end

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
    
    function SimpleTrack(obj,iMovs,frms)

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
        
        lpos = labeler.labeledpos{iM}; % [npts x 2 x nfrm]
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
    
    
    %% Update display
    function newLabelerFrame(obj)
      npts = labeler.nLabelPoints;
      if 
%       assert(numel(obj.sampleScatterH)==npts);
      
    end
    
    
    %% GUI Menu calls.
    function setmodeGMM(obj,src,evt)
      set(obj.menu_gmm,'Checked','on');
      set(obj.menu_simple,'Checked','off');
      obj.curmode = 0;
    end
    
    function setmodeSimple(obj,src,evt)
      set(obj.menu_gmm,'Checked','off');
      set(obj.menu_simple,'Checked','on');
      obj.curmode = 1;
    end
    
    function showhideSamples(obj,src,evt)
      if obj.showhidesamples < 0.5
        obj.showhidesamples = 1;
        set(obj.menu_track_showsamples,'Label','Hide Samples');
      else
        obj.showhidesamples = 0;
        set(obj.menu_track_showsamples,'Label','Hide Samples');
      end
    end
    
  end
  
end