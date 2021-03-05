classdef LabelTracker < handle
% Tracker base class

  % LabelTracker has two responsibilities:
  % 1. Take a bunch of images+labels and learn a classifier to 
  %    predict/track labels on new images.
  % 2. Store the predictions generated in 1, as well as any related 
  %    tracking diagnostics
  %
  % LabelTracker does not implement tracking visualization, but deleowns a 
  % TrackingVisualizer* that 
  %
  % LabelTracker is a base class intended to be concretized with a 
  % particular tracking algo.
  
  % Note re "iMovSgned". Specification of movies in this API is done via
  % variables named "iMovSgned" which represent vectors of indices into
  % .lObj.movieFilesAll (when iMovSgned is positive) and 
  % .lObj.movieFilesAllGT (when iMovSgned is negative).
  %
  % In this way a single scalar (integer) value continues to serve as a
  % unique key/ID for a movie in the project; these values are used in APIs
  % as well as Metadata tables (eg in .mov field). This implicitly assumes
  % that a LabelTracker handles only a single view in a multiview project.
  %
  % Moving forward, if another "list of movie(set)s" is required, or if 
  % LabelTrackers must handle multiple views, then this signed index vector 
  % will need to be replaced with some other ID mechanism (eg a uuid or 
  % 2-ple ID etc.
  
  properties (Abstract)
    algorithmName % char
    trackerInfo; % struct with whatever information we want to save about the current tracker. 
  end  
  
  properties
    lObj % (back)handle to Labeler object
    paramFile; % char, current parameter file
    ax % axis for viewing tracking results
    sPrmAll; % all parameters - KB 20190214: store all parameters with each tracker
    
    trkVizInterpolate % scalar logical. If true, interpolate tracking results when visualizing
    
    lastTrainStats = []; % struct with information about the last training for visualization
    
    hLCurrMovie; % listener to lObj.currMovie
    hLCurrFrame; % listener to lObj.currFrame
    hLCurrTarget; % listener to lObj.currTarget
    hLMovieRemoved % " lObj/movieRemoved
    hLMoviesReordered % "
  end  
  
  properties (SetObservable,SetAccess=protected)
    hideViz = false; % scalar logical. If true, hide visualizations
    showPredsCurrTargetOnly = false;
  end
  
%   properties (Constant)    
%     INFOTIMELINE_PROPS_TRACKER = EmptyLandmarkFeatureArray();
%   end
      
  methods
    
    function obj = LabelTracker(labelerObj)
      obj.lObj = labelerObj;
      
      trkPrefs = labelerObj.projPrefs.Track;
      val = logical(trkPrefs.PredictInterpolate);
      assert(isscalar(val),'Expected scalar value for ''PredictInterpolate''.');
      if val
        warningNoTrace('Turning off tracking interpolation.');
        labelerObj.projPrefs.Track.PredictInterpolate = false;
        val = false;
      end
      obj.trkVizInterpolate = val;
      
      obj.hLCurrMovie = addlistener(labelerObj,'newMovie',@(s,e)obj.newLabelerMovie());
      %obj.hLCurrFrame = addlistener(labelerObj,'currFrame','PostSet',@(s,e)obj.newLabelerFrame());
      obj.hLCurrTarget = addlistener(labelerObj,'currTarget','PostSet',@(s,e)obj.newLabelerTarget());
      obj.hLMovieRemoved = addlistener(labelerObj,'movieRemoved',@(s,e)obj.labelerMovieRemoved(e));
      obj.hLMoviesReordered = addlistener(labelerObj,'moviesReordered',@(s,e)obj.labelerMoviesReordered(e));
    end
    
    function init(obj)
      % Called when a new project is created/loaded, etc
      obj.ax = obj.lObj.gdata.axes_all;
      obj.initHook();
    end
        
    function delete(obj)
      if ~isempty(obj.hLCurrMovie)
        delete(obj.hLCurrMovie);
      end
      if ~isempty(obj.hLCurrFrame)
        delete(obj.hLCurrFrame);
      end
      if ~isempty(obj.hLCurrTarget)
        delete(obj.hLCurrTarget);
      end
      if ~isempty(obj.hLMovieRemoved)
        delete(obj.hLMovieRemoved);
      end
      if ~isempty(obj.hLMoviesReordered)
        delete(obj.hLMoviesReordered);
      end      
    end    
	
  end
  
  methods
    
    function initHook(obj) %#ok<*MANU>
      % Called when a new project is created/loaded, etc
    end
        
    function sPrm = getParams(obj)
      sPrm = struct();
    end
       
    function ppdata = fetchPreProcData(obj,tblP,ppPrms)
      % Fetch preprocessed data per this tracker. Don't update any cache
      % b/c the preproc params supplied may be "trial"/random.
      % 
      % tblP: MFTable
      % ppPrms: scalar struct, preproc params only.
      % 
      % ppdata: CPRData
      
      assert(false,'Overload required.');
    end
    
    function train(obj)
      % (Incremental) Train
      % - If it's the first time, it's a regular/full train
      % - If a tracker is trained, it's an incremental train
    end
    
    function [tfCanTrain,reason] = canTrain(obj)
      
      tfCanTrain = true;
      reason = '';
      
    end

    function [tfCanTrack,reason] = canTrack(obj)
      
      tfCanTrack = true;
      reason = '';
      
    end 
        
    function retrain(obj)
      % Full Train from scratch; existing/previous results cleared 
    end
    
    function tf = getHasTrained(obj)
      %
    end
    
    function track(obj,tblMFT,varargin)
      % Apply trained tracker to the specified frames.
      % 
      % tblMFT: MFTable with cols MFTable.FLDSID
      %
      %
      % DEPRECATED Legacy/Single-target API:
      %   track(obj,iMovs,frms,...)
      %
      % iMovsSgned: [M] indices into .lObj.movieFilesAll to track; negative
      %   indices are into .lObj.movieFilesAllGT.
      % frms: [M] cell array. frms{i} is a vector of frames to track for iMovs(i).
    end
    
    function xy = getPredictionCurrentFrame(obj)
      % Convenience meth
      %
      % xy: [nPtsx2xnTgt] tracked results for current Labeler frame
      
      xy = [];
    end

    function [tpos,taux,tauxlbl] = getTrackingResultsCurrMovieTgt(obj)
      % Get current tracking results for current movie, tgt
      %
      % MA: current tgt is currently-selected tracklet
      % 
      % This is a convenience method as it is a special case of 
      % getTrackingResults. Concrete LabelTrackers will also typically have 
      % the current movie's tracking results cached.
      % 
      % tpos: [npts d nfrm], or empty/[] will be accepted if no
      % results are available. 
      % taux: [npts nfrm naux], or empty/[]
      % tauxlbl: [naux] cellstr 
      tpos = [];
      taux = [];
      tauxlbl = cell(0,1);
    end
      
    function [trkfiles,tfHasRes] = getTrackingResults(obj,iMovsSgned)
      % Get tracking results for movie(set) iMovs.
      % Default implemation returns all NaNs and tfHasRes=false.
      %
      % iMovsSgned: [nMov] vector of movie(set) indices, negative for GT
      %
      % trkfiles: [nMovxnView] cell of TrkFile objects
      % tfHasRes: [nMov] logical. If true, corresponding movie(set) has 
      % tracking nontrivial (nonempty) tracking results
      
      trkfiles = [];
      tfHasRes = [];
    end
    
    function tblTrk = getAllTrackResTable(obj)
      % Get all tracking results known to tracker in a single table.
      %
      % tblTrk: fields .mov, .frm, .iTgt, .pTrk
      
      tblTrk = [];
    end
    
    function s = getTrainedTrackerMetadata(obj)
      % Get standardized form of metadata for this tracker. Metadata should
      % include parameters, uniquely identify a trained model, etc.
      
      s = struct('class',class(obj));
    end
    
    function importTrackingResults(obj,iMovSgned,trkfiles)
      % Import tracking results for movies iMovs.
      % Default implemation ERRORS
      %
      % Currently no set policy on whether to merge or overwrite existing 
      % tracking results.
      %
      % iMovs: vector of movie indices
      % trkfiles: vector of TrkFile objects, same numel as iMovs

      assert(false,'Import tracking results is unsupported for this tracker.');   
    end
    
    function clearTrackingResults(obj)
      % Clear all current/cached tracking results. Trained tracker should
      % remain untouched. Used in two situations:
      %  
      % - It is desired to explicitly clear the current tracking DB b/c eg
      % it will be out of date after a retrain.
      % - For trackers with in-memory tracking DBs, to control memory 
      % usage.
      %
      % Default impl: none
    end    
        
    function newLabelerFrame(obj)
      % Called when Labeler is navigated to a new frame
    end
    
    function newLabelerTarget(obj)
      % Called when Labeler is navigated to a new target
    end
    
    function newLabelerMovie(obj)
      % Called when Labeler is navigated to a new movie
    end
    
    function updateLandmarkColors(obj)
      % Called when colors for landmarks have changed
    end
    
    function labelerMovieRemoved(obj,eventdata)
      % Called on Labeler/movieRemoved
    end

    function labelerMoviesReordered(obj,eventdata)
    end
        
    function s = getSaveToken(obj)
      % Get a struct to serialize
      s = struct();
    end

    function loadSaveToken(obj,s) %#ok<*INUSD>
      
    end
    
    function setHideViz(obj,tf)
      obj.hideViz = tf;
    end
    
    function hideVizToggle(obj)
      obj.setHideViz(~obj.hideViz);
    end

    function setShowPredsCurrTargetOnly(obj,tf)
      obj.showPredsCurrTargetOnly = tf;
    end
    
    function showPredsCurrTargetOnlyToggle(obj)
      obj.setShowPredsCurrTargetOnly(~obj.showPredsCurrTargetOnly);
    end

    % update information about the current tracker
    % placeholder - should be defined by child classes
    function updateTrackerInfo(obj)
      
    end
    
    % return a cell array of strings with information about the current
    % tracker
    % placeholder - should be defined by child classes
    function [infos] = getTrackerInfoString(obj,doupdate)
      infos = {'Not implemented'};
    end
    
  end
  
  methods % For infotimeline display
    
    function props = propList(obj)
      props = EmptyLandmarkFeatureArray();
    end
    
    function data = getPropValues(obj,prop)
      % Return the values of a particular property for
      % infotimeline
      
      labeler = obj.lObj;
      npts = labeler.nLabelPoints;
      nfrms = labeler.nframes;
      ntgts = labeler.nTargets;
      iTgt = labeler.currTarget;
      if iTgt == 0,
        iTgt = 1;
      end
      iMov = labeler.currMovie;
      [tpos,taux,tauxlbl] = obj.getTrackingResultsCurrMovieTgt();
      
      needtrx = obj.lObj.hasTrx && strcmpi(prop.coordsystem,'Body');
      if needtrx,
        trxFile = obj.lObj.trxFilesAllFullGTaware{iMov,1};
        bodytrx = obj.lObj.getTrx(trxFile,obj.lObj.movieInfoAllGTaware{iMov,1}.nframes);
        bodytrx = bodytrx(iTgt);
      else
        bodytrx = [];
      end
      
      if isempty(tpos)
        % edge case uninitted (not sure why)
        tpos = nan(npts,2,nfrms);
      end
      
      plist = obj.propList();
      plistcodes = {plist.code}';
      tfaux = any(strcmp(prop.code,plistcodes));
      if tfaux
        iaux = find(strcmp(tauxlbl,prop.feature));
        assert(isscalar(iaux));
        data = taux(:,:,iaux);
        
        % cf ComputeLandmarkFeatureFromPos
        if strcmpi(prop.transform,'none')
          % none; data unchanged
        else
          fun = sprintf('compute_landmark_transform_%s',prop.transform);
          if ~exist(fun,'file'),
            warningNoTrace('Unknown property transformation ''%s'' for timeline display.',...
              prop.transform);
            % data unchanged
          else
            data = feval(fun,struct('data',data));
            data = data.data;
          end
        end
      else 
        tpostag = false(npts,nfrms,ntgts);
        data = ComputeLandmarkFeatureFromPos(tpos,...
          tpostag(:,:,iTgt),bodytrx,prop);
      end
    end
    
  end
  
  methods % Utilities
    
    function prm = readParamFileYaml(obj)
      prmFile = obj.paramFile;
      if isempty(prmFile)
        error('LabelTracker:noParams',...
          'Tracking parameter file needs to be set.');
      end
      prm = ReadYaml(prmFile);
    end
            
  end
  
  methods (Static)
    
    function tObj = create(lObj,trkClsAug,trkData)
      % Factory meth
      
      tCls = trkClsAug{1};
      tClsArgs = trkClsAug(2:end);
      
      if exist(tCls,'class')==0
        error('Labeler:projLoad',...
          'Project tracker class ''%s'' cannot be found.',tCls);
      end
      tObj = feval(tCls,lObj,tClsArgs{:});
      tObj.init();
      if ~isempty(trkData)
        tObj.loadSaveToken(trkData);
      end
    end
    
    function info = getAllTrackersCreateInfo(isMA)
      dlnets = enumeration('DLNetType');
      dlnets = dlnets([dlnets.doesMA]==isMA);
      info = arrayfun(@(x){'DeepTracker' 'trnNetType' x},dlnets,'uni',0);
      if ~isMA
        info = [info; {{'CPRLabelTracker'}}];
      end
    end
    
    function [tf,loc] = trackersCreateInfoIsMember(infocell1,infocell2)
      keyfcn = @(infocell)cellfun(@(x)sprintf('%s#',x{:}),infocell,'uni',0);
      keys1 = keyfcn(infocell1);
      keys2 = keyfcn(infocell2);
      [tf,loc] = ismember(keys1,keys2);      
    end
    
  end
  
end
