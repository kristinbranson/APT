classdef LabelTracker < handle
  % LabelTracker knows how to take a bunch of images+labels and learn a
  % classifier to predict/track labels on new images.
  %
  % LabelTracker is a base class intended to be concretized with a 
  % particular tracking algo.
  
  properties
    lObj % (back)handle to Labeler object
    paramFile; % char, current parameter file
    ax % axis for viewing tracking results
    
    trkVizInterpolate % scalar logical. If true, interpolate tracking results when visualizing
    
    hLCurrMovie; % listener to lObj.currMovie
    hLCurrFrame; % listener to lObj.currFrame    
  end
  
  methods
    
    function obj = LabelTracker(labelerObj)
      obj.lObj = labelerObj;   
      
      axOver = axisOverlay(obj.lObj.gdata.axes_curr);
      axOver.LineWidth = 2;
      obj.ax = axOver;
      
      trkPrefs = labelerObj.trackPrefs;
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
      axisOverlay(obj.lObj.gdata.axes_curr,obj.ax);
      obj.initHook();
    end
    
    function setParamFile(obj,prmFile)
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
    end
       
    function train(obj)
      % (Incremental) Train
      % - If it's the first time, it's a regular/full train
      % - If a tracker is trained, it's an incremental train
    end
    
    function retrain(obj)
      % Full Train from scratch; existing/previous results cleared 
    end
    
%     function inspectTrainingData(obj)
%     end

    function track(obj,iMovs,frms)
      % Apply trained tracker to the specified frames.
      %
      % iMovs: [M] indices into .lObj.movieFilesAll to track
      % frms: [M] cell array. frms{i} is a vector of frames to track for iMovs(i).
    end
    
    function trkfiles = getTrackingResults(obj,iMovs)
      % Get tracking results for movie iMov.
      % Default implemation here RETURNS ALL NANS
      %
      % iMovs: vector of movie indices
      %
      % trkfiles: vector of TrkFile objects, same numel as iMovs
      
      validateattributes(iMovs,{'numeric'},{'vector' 'positive' 'integer'});
      
      nMov = numel(iMovs);
      for i = nMov:-1:1
        trkpos = nan(size(obj.lObj.labeledpos{iMovs(i)}));
        trkfiles(i) = TrkFile(trkpos);
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
    
    function tblP = getTblPLbled(obj)
      % From .lObj, read tblP for all movies/labeledframes. Currently,
      % exclude partially-labeled frames.
      %
      % tblP: table of labeled frames, one row per frame.       
      
      labelerObj = obj.lObj;
      [~,tblP] = Labeler.lblCompileContents(labelerObj.movieFilesAllFull,labelerObj.labeledpos,...
        labelerObj.labeledpostag,'lbl','noImg',true,'lposTS',labelerObj.labeledposTS);
      
      p = tblP.p;
      tfnan = any(isnan(p),2);
      nnan = nnz(tfnan);
      if nnan>0
        warningNoTrace('CPRLabelTracker:nanData','Not including %d partially-labeled rows.',nnan);
      end
      tblP = tblP(~tfnan,:);
    end
    
  end
  
end