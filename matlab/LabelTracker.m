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
    
    hListeners; % cell vec of Labeler listeners
  end  
  
  properties (SetObservable, SetAccess=protected)
    hideViz = false; % scalar logical. If true, hide visualizations
    showPredsCurrTargetOnly = false;
  end
  
  methods (Abstract)
    % return cellstr, (deep) nets used by this tracker
    v = getNetsUsed(obj)
  end
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
      
      listeners = { ...
        addlistener(labelerObj,'newMovie',@(s,e)obj.newLabelerMovie());
        %addlistener(labelerObj,'currFrame','PostSet',@(s,e)obj.newLabelerFrame());
        addlistener(labelerObj,'didSetCurrTarget',@(s,e)(obj.newLabelerTarget()));
        addlistener(labelerObj,'movieRemoved',@(s,e)obj.labelerMovieRemoved(e));
        addlistener(labelerObj,'moviesReordered',@(s,e)obj.labelerMoviesReordered(e));
        };
      obj.hListeners = listeners;
    end
    
    function init(obj)
      % Called when a new project is created/loaded, etc
      obj.ax = obj.lObj.gdata.axes_all;
      obj.initHook();
    end
        
    function delete(obj)
      obj.deleteListeners();
    end    
    
    function deleteListeners(obj)
      cellfun(@delete,obj.hListeners);
      obj.hListeners = cell(0,1);
    end
    
    function setEnableListeners(obj,val)
      hs = obj.hListeners;
      for i=1:numel(hs)
        hs{i}.Enabled = val;
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
    
    function setParams(obj,sPrm)
      % this should only be done if one knows what one is doing! 
      obj.sPrmAll = sPrm;
    end
    function setTrackParams(obj,sPrmTrack)
      if ~isempty(obj.sPrmAll)
        obj.sPrmAll = APTParameters.setTrackParams(obj.sPrmAll,sPrmTrack);
      end
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
    
    function [tfhaspred,xy,tfocc] = getTrackingResultsCurrFrm(obj)
      % Convenience meth
      %
      % xy: [nPtsx2xnTgt] tracked results for current Labeler frame
      
      tfhaspred = [];
      xy = [];
      tfocc = [];
    end

    function [tfhasdata,xy,occ,sf,ef,aux,auxlbl] = ...
                                    getTrackingResultsCurrMovieTgt(obj)
      % Get current tracking results for current movie, tgt
      %
      % MA: current tgt is currently-selected tracklet
      % 
      % This is a convenience method as it is a special case of 
      % getTrackingResults. Concrete LabelTrackers will also typically have 
      % the current movie's tracking results cached.
      %
      % tfhasdata: true if data is present. if false, remaining outputs
      %   are indeterminate
      % xy: [npt x 2 x numfrm]. numfrm = ef-sf+1
      % occ: [npt x numfrm]
      % sf: start frame, labels xy(:,:,1)
      % ef: end frame, labels xy(:,:,end)
      % aux (opt): [npt x numfrm x numaux] Auxiliary stats for this tracker
      % auxlbl: [numaux] cellstr 
      
      tfhasdata = false;
      xy = [];
      occ = [];
      sf = nan;
      ef = nan;
      aux = [];
      auxlbl = cell(0,1);
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
    
    function tblTrk = getTrackingResultsTable(obj)
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
    
    function updateDLCache(obj,dlcachedir)
      % For DL tracker portability across save/load
      
      % none
    end
    
    function deactivate(obj)
      % called when a tracker is no longer active. for performance      
      obj.setEnableListeners(false);      
    end
    
    function activate(obj)
      obj.setEnableListeners(true);
    end
    
  end
  
  methods % For infotimeline display
    
    function props = propList(obj)
      props = EmptyLandmarkFeatureArray();
    end
    
    function data = getPropValues(obj,prop)
      % Return the values of a particular property for timeline
      %
      % data: [labeler.nframes] timeseries
      
      labeler = obj.lObj;
      npts = labeler.nLabelPoints;
      nfrms = labeler.nframes;
      %ntgts = labeler.nTargets;
      iTgt = labeler.currTarget;
      if iTgt == 0,
        iTgt = 1;
      end
      iMov = labeler.currMovie;
            
      %[tpos,taux,tauxlbl] = obj.getTrackingResultsCurrMovieTgt();      
      [tfhasdata,xy,occ,sf,ef,aux,auxlbl] = obj.getTrackingResultsCurrMovieTgt();
      if ~tfhasdata
        data = nan(npts,nfrms);
        return;
      end
      
      needtrx = obj.lObj.hasTrx && strcmpi(prop.coordsystem,'Body');
      if needtrx,
        trxFile = obj.lObj.trxFilesAllFullGTaware{iMov,1};
        bodytrx = obj.lObj.getTrx(trxFile,obj.lObj.movieInfoAllGTaware{iMov,1}.nframes);
        bodytrx = bodytrx(iTgt);
      else
        bodytrx = [];
      end      
      
      plist = obj.propList();
      plistcodes = {plist.code}';
      tfaux = any(strcmp(prop.code,plistcodes));
      if tfaux
        % 20220919: appears auxiliary props won't ever need bodytrx
        
        iaux = find(strcmp(auxlbl,prop.feature));
        assert(isscalar(iaux));
        data = aux(:,:,iaux); % [npts x (ef-sf+1)]
        
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
            %data = data.data;
          end
        end
        
        data = padData(data,sf,ef,nfrms);
      else
        [data,units] = ComputeLandmarkFeatureFromPos(xy,occ,sf,ef,nfrms,bodytrx,prop);
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
      % This will need updating. DLNetType will include all types of nets
      % such as objdetect which will not qualify as eg regular/SA trackers.
      if isMA
        info = cat(1,DeepTrackerBottomUp.getTrackerInfos,...
          DeepTrackerTopDown.getTrackerInfos,...
          DeepTrackerTopDownCustom.getTrackerInfos...
          );
        % For custom 2stage trackers add the DeepTrackerTownDown again.
      else        
        dlnets = enumeration('DLNetType');
        dlnets = dlnets(~[dlnets.isMultiAnimal]);
        info = arrayfun(@(x){'DeepTracker' 'trnNetType' x},dlnets,'uni',0);        
        %info = [info; {{'CPRLabelTracker'}}];
      end
    end
    
    function [tf,loc] = trackersCreateInfoIsMember(infocell1,infocell2)
      n1 = numel(infocell1);
      n2 = numel(infocell2);
      tf = false(n1,1);
      loc = zeros(n1,1);
      for i=1:n1
        for j=1:n2
          if isequal(infocell1{i},infocell2{j})
            tf(i) = true;
            loc(i) = j;
            break;
          end
        end
      end
%       keyfcn = @(infocell)cellfun(@(x)sprintf('%s#',x{:}),infocell,'uni',0);
%       keys1 = keyfcn(infocell1);
%       keys2 = keyfcn(infocell2);
%       [tf,loc] = ismember(keys1,keys2);      
    end
    
  end
  
end
