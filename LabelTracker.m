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
  
%   properties (Constant)
%     % Known concrete LabelTrackers
%     subclasses = {...
%       'Interpolator'
%       'SimpleInterpolator'
%       'GMMTracker'
%       'CPRLabelTracker'
%       };
%   end
  
  properties (Abstract)
    algorithmName % char
  end  
  
  properties    
    lObj % (back)handle to Labeler object
    paramFile; % char, current parameter file
    ax % axis for viewing tracking results
    
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
  end
  
  properties (Constant)
    APT_DEFAULT_TRACKERS = {
      {'CPRLabelTracker'}
      {'DeepTracker' 'trnNetType' DLNetType.mdn}
      {'DeepTracker' 'trnNetType' DLNetType.deeplabcut}
      {'DeepTracker' 'trnNetType' DLNetType.unet}
      };
    INFOTIMELINE_PROPS_TRACKER = EmptyLandmarkFeatureArray();
  end
    
  methods
    
    function obj = LabelTracker(labelerObj)
      obj.lObj = labelerObj;
      
      trkPrefs = labelerObj.projPrefs.Track;
      if isfield(trkPrefs,'PredictInterpolate')
        val = logical(trkPrefs.PredictInterpolate);
        assert(isscalar(val),'Expected scalar value for ''PredictInterpolate''.');
        if val
          warningNoTrace('Turning off tracking interpolation.');
          labelerObj.projPrefs.Track.PredictInterpolate = false;
          val = false;
        end
      else
        val = false;
      end
%       if obj.lObj.hasTrx && val
%         warningNoTrace('LabelTracker:interp',...
%           'Project has trajectories; turning off tracking interpolation.');
%         val = false;
%       end
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
    
%     function setParamHook(obj)
%       % Called when a new parameter file is specified
%       
%       % See setParams.
%     end
%     
%     function setParams(obj,sPrm)
%       % Directly set params. Note, methods .setParamFile and .setParams
%       % "overlap". Subclasses should do something intelligent.
%     end
    
    function sPrm = getParams(obj)
      sPrm = struct();
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

    
    function [tfsucc,tblPTrn,dataPreProc] = preretrain(obj,tblPTrn,wbObj,prmpp)
      % Right now this figures out which rows comprise the training set.
      %
      % PostConditions (tfsucc==true):
      %   - If initially unknown, training set is determined/returned in
      %   tblPTrn
      %   - lObj.preProcData has been updated to include all rows of
      %   tblPTrn; lObj.preProcData.iTrn has been set to those rows
      %
      % PostConditions (tfsucc=false): other outputs indeterminte
      %
      % tblPTrn (in): Either [], or a MFTable.
      % wbObj: Either [], or a WaitBarWithCancel.
      %
      % tfsucc: see above
      % tblPTrn (out): MFTable
      % dataPreProc: CPRData handle, obj.lObj.preProcData
      
      tfsucc = false;
      dataPreProc = [];
      tfWB = ~isempty(wbObj);
      if ~exist('prmpp','var'),
        prmpp = [];
      end
      
      % Either use supplied tblPTrn, or use all labeled data
      if isempty(tblPTrn)
        % use all labeled data
        tblPTrn = obj.lObj.preProcGetMFTableLbled('wbObj',wbObj);
        if tfWB && wbObj.isCancel
          % Theoretically we are safe to return here as of 201801. We
          % have only called obj.asyncReset() so far.
          % However to be conservative/nonfragile/consistent let's reset
          % as in other cancel/early-exits          
          return;
        end
      end
      if obj.lObj.hasTrx
        tblfldscontainsassert(tblPTrn,[MFTable.FLDSCOREROI {'thetaTrx'}]);
      elseif obj.lObj.cropProjHasCrops
        tblfldscontainsassert(tblPTrn,[MFTable.FLDSCOREROI]);
      else
        tblfldscontainsassert(tblPTrn,MFTable.FLDSCORE);
      end
      
      if isempty(tblPTrn)
        error('CPRLabelTracker:noTrnData','No training data set.');
      end
      
      [dataPreProc,dataPreProcIdx,tblPTrn,tblPTrnReadFail] = ...
        obj.lObj.preProcDataFetch(tblPTrn,'wbObj',wbObj,'preProcParams',prmpp);
      if tfWB && wbObj.isCancel
        % none
        return;
      end
      nMissedReads = height(tblPTrnReadFail);
      if nMissedReads>0
        warningNoTrace('Removing %d training rows, failed to read images.\n',...
          nMissedReads);
      end
      fprintf(1,'Training with %d rows.\n',height(tblPTrn));
      
      dataPreProc.iTrn = dataPreProcIdx;
      fprintf(1,'Training data summary:\n');
      dataPreProc.summarize('mov',dataPreProc.iTrn);
      
      tfsucc = true;      
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
    
    function tpos = getTrackingResultsCurrMovie(obj)
      % This is a convenience method as it is a special case of 
      % getTrackingResults. Concrete LabelTrackers will also typically have 
      % the current movie's tracking results cached.
      % 
      % tpos: [npts d nfrm ntgt], or empty/[] will be accepted if no
      % results are available. 
      tpos = [];
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
      
      s = struct('class',classname(obj));
    end
    
    function xy = getPredictionCurrentFrame(obj)
      % xy: [nPtsx2xnTgt] tracked results for current Labeler frame
      xy = [];
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
        
  end
  
  methods % For infotimeline display
    
    function props = propList(obj)
      %props = {'x' 'y' 'dx' 'dy' '|dx|' '|dy|'}';
      if isempty(obj.INFOTIMELINE_PROPS_TRACKER),
        props = {};
      else
        props = {obj.INFOTIMELINE_PROPS_TRACKER.name};
      end
    end
    
    function data = getPropValues(obj,prop)
      % Return the values of a particular property for
      % infotimeline
      
      labeler = obj.lObj;
      npts = labeler.nLabelPoints;
      nfrms = labeler.nframes;
      ntgts = labeler.nTargets;
      iTgt = labeler.currTarget;
      iMov = labeler.currMovie;
      tpos = obj.getTrackingResultsCurrMovie();
      
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
        tpos = nan(npts,2,nfrms,ntgts);
      end
      
      if ismember(prop.code,{obj.INFOTIMELINE_PROPS_TRACKER.code}),
        error('Not implemented');
      else      
        tpostag = false(npts,nfrms,ntgts);
        data = ComputeLandmarkFeatureFromPos(tpos(:,:,:,iTgt),tpostag(:,:,iTgt),bodytrx,prop);
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
    
  end
%     
%     function sc = findAllSubclasses
%       % sc: cellstr of LabelTracker subclasses in APT.Root
%       
%       scnames = LabelTracker.subclasses; % candidates
%       nSC = numel(scnames);
%       tf = false(nSC,1);
%       for iSC=1:nSC
%         name = scnames{iSC};
%         mc = meta.class.fromName(name);
%         tf(iSC) = ~isempty(mc) && any(strcmp('LabelTracker',{mc.SuperclassList.Name}));
%       end
%       sc = scnames(tf);
%     end
%     
%   end
  
end
