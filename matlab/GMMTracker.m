classdef GMMTracker < LabelTracker
  
  properties
    menu_gmm
    menu_simple
    curmode  % 0 for gmm, 1 for simple
    
    menu_track_showsamples
    showhidesamples %0 for hide, 1 for show.
    sampleScatterH
    colors 
    
    currMovie
    sampleInfo
    
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
      obj.menu_track_showsamples = uimenu(lObj.gdata.menu_track,'Label','Show Samples','Callback',@obj.showhideSamples);
      obj.showhidesamples = 0;
      obj.colors = obj.lObj.labelPointsPlotInfo.Colors;
      
      npts = obj.lObj.nLabelPoints;
      obj.sampleScatterH = gobjects(1,npts);
      hold(obj.ax,'off')
      for ndx = 1:npts
        obj.sampleScatterH(ndx) = scatter(obj.ax,nan,nan,[],...
          obj.colors(ndx,:),'x','Visible','off');
        if ndx ==1, hold(obj.ax, 'on'); end
      end
      hold(obj.ax,'off');
      % scatter() changes many axis props. Revert at least these props;
      % there are others as well that may not be important for now.
      set(obj.ax,'Visible','off','HitTest','off','Color','None'); 
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

      movfiles = obj.lObj.movieFilesAllFull(iMovs);
      trkfiles = cellfun(@labeler.defaultTrkFileName,movfiles,'uni',0);

      assert(labeler.nTargets==1);
      npts = labeler.nLabelPoints;
      for i=1:numel(iMovs)
        iM = iMovs(i);
        fs = frms{i};
        if any( (fs(2:end)-fs(1:end-1))>1.5)
          error('Interpolator:FramesIncorrect','frames must be consecutive');
        end
        
        if i == obj.currMovie
          T = obj.sampleInfo;
        else
          T = load(trkfiles{i},'-mat');
        end
        
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
        labeler.setFrameGUI(labeler.currFrame,'tfforcelabelupdate',true);
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
        labeler.setFrameGUI(labeler.currFrame,'tfforcelabelupdate',true);
      end
    end
    
    
    %% Update display
    function newLabelerFrame(obj)
      npts = obj.lObj.nLabelPoints;
      assert(numel(obj.sampleScatterH)==npts);
      obj.updateScatterH();
      
      if obj.lObj.currMovie ~= obj.currMovie,
        obj.newLabelerMovie();
      end
      curf = obj.lObj.currFrame;
      
      for ndx = 1:npts
        curS = squeeze(permute(obj.sampleInfo.pSample(ndx,:,:,curf),[2,3,4,1]));
        curWt = squeeze(obj.sampleInfo.weights(:,ndx,curf));
        if obj.showhidesamples>0.5
          set(obj.sampleScatterH(ndx),'XData',curS(1,:),'YData',curS(2,:),...
            'SizeData',max(0.0001,curWt)*30,'Visible','on');
        else
          set(obj.sampleScatterH(ndx),'XData',nan,'YData',nan,...
            'SizeData',1,'Visible','off');
        end
      end
    end
    
    function newLabelerMovie(obj)
      curmov = obj.lObj.currMovie;
      movfile = obj.lObj.movieFilesAllFull{curmov};
      trkfile = obj.lObj.defaultTrkFileName(movfile);
      if ~exist(trkfile,'file'),
        fprintf('Could not find trkfile %s. Please locate it.\n',trkfile);
        [p,f] = fileparts(movfile);
        [f,p] = uigetfile('*.trk',sprintf('Locate trkfile %s',f),p);
        if ~ischar(f),
          error('Could not find trkfile %s',trkfile);
        end
        tmptrkfile = fullfile(p,f);
        if ~exist(tmptrkfile,'file'),
          error('Trkfile %s does not exist',tmptrkfile);
        end
        [success,msg] = copyfile(tmptrkfile,trkfile);
        if ~success,
          error('Error copying %s to %s: %s',tmptrkfile,trkfile,msg);
        end
      end
      T = load(trkfile,'-mat');
      obj.sampleInfo = T;
      obj.currMovie = curmov;
    end
    
    function updateScatterH(obj)
      hold(obj.ax,'off');
      for ndx = 1:numel(obj.sampleScatterH),
        if ~ishghandle(obj.sampleScatterH(ndx)),
          obj.sampleScatterH(ndx) = scatter(obj.ax,nan,nan,[],...
            obj.colors(ndx,:),'x','Visible','off');
        end
        if ndx == 1
          hold(obj.ax,'on');
        end

      end
      hold(obj.ax,'off')
      set(obj.ax,'Color','None'); % Adding a scatter adds color to the axis apparently.


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
        set(obj.menu_track_showsamples,'Label','Show Samples');
      end
      obj.newLabelerFrame();
    end
    
  end
  
end