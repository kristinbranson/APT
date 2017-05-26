classdef LabelTracker < handle
  % LabelTracker knows how to take a bunch of images+labels and learn a
  % classifier to predict/track labels on new images.
  %
  % LabelTracker is a base class intended to be concretized with a 
  % particular tracking algo.
  
  properties (Constant)
    % Known concrete LabelTrackers
    subclasses = {...
      'Interpolator'
      'SimpleInterpolator'
      'GMMTracker'
      'CPRLabelTracker'
      };
  end
  
  properties
    lObj % (back)handle to Labeler object
    paramFile; % char, current parameter file
    ax % axis for viewing tracking results
    
    trkVizInterpolate % scalar logical. If true, interpolate tracking results when visualizing
    
    hLCurrMovie; % listener to lObj.currMovie
    hLCurrFrame; % listener to lObj.currFrame
  end  
  
  properties (SetObservable,SetAccess=protected)
    hideViz = false; % scalar logical. If true, hide visualizations
  end
    
  methods
    
    function obj = LabelTracker(labelerObj)
      obj.lObj = labelerObj;      
      
      trkPrefs = labelerObj.projPrefs.Track;
      if isfield(trkPrefs,'PredictInterpolate')
        val = logical(trkPrefs.PredictInterpolate);
        if ~isscalar(val)
          error('LabelTracker:init','Expected scalar value for ''PredictInterpolate'' preference.');
        end
      else
        val = false;
      end
      obj.trkVizInterpolate = val;
      
      obj.hLCurrMovie = addlistener(labelerObj,'currMovie','PostSet',@(s,e)obj.newLabelerMovie());
      obj.hLCurrFrame = addlistener(labelerObj,'currFrame','PostSet',@(s,e)obj.newLabelerFrame());
    end
    
    function init(obj)
      % Called when a new project is created/loaded, etc

      deleteValidHandles(obj.ax);     
      axAll = obj.lObj.gdata.axes_all;
      axOver = gobjects(size(axAll));
      for i=1:numel(axAll)
        axOver(i) = axisOverlay(axAll(i));
        axOver(i).LineWidth = 2;
      end
      obj.ax = axOver;
      
      obj.initHook();
    end
    
    function setParamFile(obj,prmFile)
      % See also setParams.
      
      obj.paramFile = prmFile;
      obj.setParamHook();
    end
    
    function delete(obj)
      deleteValidHandles(obj.ax);
      if ~isempty(obj.hLCurrMovie)
        delete(obj.hLCurrMovie);
      end
      if ~isempty(obj.hLCurrFrame)
        delete(obj.hLCurrFrame);
      end
    end
    
  end
  
  methods
    
    function initHook(obj) %#ok<*MANU>
      % Called when a new project is created/loaded, etc
    end
    
    function setParamHook(obj)
      % Called when a new parameter file is specified
      
      % See setParams.
    end
    
    function setParams(obj,sPrm)
      % Directly set params. Note, methods .setParamFile and .setParams
      % "overlap". Subclasses should do something intelligent.
    end
    
    function sPrm = getParams(obj)
      sPrm = struct();
    end
       
    function train(obj)
      % (Incremental) Train
      % - If it's the first time, it's a regular/full train
      % - If a tracker is trained, it's an incremental train
    end
    
    function retrain(obj)
      % Full Train from scratch; existing/previous results cleared 
    end
    
    function track(obj,iMovs,frms,varargin)
      % Apply trained tracker to the specified frames.
      %
      % Legacy/Single-target API:
      %   track(obj,iMovs,frms,...)
      %
      % iMovs: [M] indices into .lObj.movieFilesAll to track
      % frms: [M] cell array. frms{i} is a vector of frames to track for iMovs(i).
      %
      % Newer/multi-target API:
      %     track(obj,[],[],'tblP',tblMF)
      %
      % tblMF: MFTable with rows specifying movie/frame/target
      %
      % Optional PVs.
    end
    
    function [trkfiles,tfHasRes] = getTrackingResults(obj,iMovs)
      % Get tracking results for movie(set) iMovs.
      % Default implemation returns all NaNs and tfHasRes=false.
      %
      % iMovs: [nMov] vector of movie(set) indices
      %
      % trkfiles: [nMovxnView] vector of TrkFile objects
      % tfHasRes: [nMov] logical. If true, corresponding movie(set) has 
      % tracking nontrivial (nonempty) tracking results
      
      validateattributes(iMovs,{'numeric'},{'vector' 'positive' 'integer'});
      
      assert(~obj.lObj.isMultiView,'Multiview unsupported.');
      
      nMov = numel(iMovs);
      for i = nMov:-1:1
        trkpos = nan(size(obj.lObj.labeledpos{iMovs(i)}));
        trkfiles(i) = TrkFile(trkpos);
        tfHasRes(i) = false;
      end
    end
            
    function importTrackingResults(obj,iMovs,trkfiles)
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
      % remain untouched. Used when tracking many movies to avoid memory
      % overflow.

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
    
    % AL 20160715: Don't use/overload me, still rationalizing save/load
    function s = getSaveToken(obj)
      % Get a struct to serialize
      s = struct();
    end
    % AL 20160715: Don't use/overload me, still rationalizing save/load    
    function loadSaveToken(obj,s) %#ok<*INUSD>
      
    end
    
    function vizHide(obj)
      obj.hideViz = true;
    end
    
    function vizShow(obj)
      obj.hideViz = false;
    end
    
    function hideVizToggle(obj)
      if obj.hideViz
        obj.vizShow();
      else
        obj.vizHide();
      end
    end
    
    function xy = getPredictionCurrentFrame(obj)
      % xy: [nPtsx2xnTgt] tracked results for current Labeler frame
      xy = [];
    end
    
  end
  
  methods % For infotimeline display
    
    function props = propList(obj)
      % Return a list of properties that could be shown in the
      % infotimeline
      props = {};
    end
    
    function data = getPropValues(obj,prop)
      % Return the values of a particular property for
      % infotimeline
      data = [];
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
        
    %#MV
    function tblP = getTblP(obj,iMovs,frms) % obj CONST
      % From .lObj, read tblP for given movies/frames.
            
      labelerObj = obj.lObj;
      assert(~labelerObj.hasTrx,...
        'Legacy codepath not intended for multitarget projects.');
      movID = labelerObj.movieFilesAll;
      movID = FSPath.standardPath(movID);
      [~,tblP] = Labeler.lblCompileContentsRaw(labelerObj.movieFilesAllFull,...
        labelerObj.labeledpos,labelerObj.labeledpostag,iMovs,frms,...
        'noImg',true,'lposTS',labelerObj.labeledposTS,'movieNamesID',movID);
    end
    
  end
  
  methods (Static)
    
    function sc = findAllSubclasses
      % sc: cellstr of LabelTracker subclasses in APT.Root
      
      scnames = LabelTracker.subclasses; % candidates
      nSC = numel(scnames);
      tf = false(nSC,1);
      for iSC=1:nSC
        name = scnames{iSC};
        mc = meta.class.fromName(name);
        tf(iSC) = ~isempty(mc) && any(strcmp('LabelTracker',{mc.SuperclassList.Name}));
      end
      sc = scnames(tf);
    end
    
  end
  
end
