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
    
    hLCurrMovie; % listener to lObj.currMovie
    hLCurrFrame; % listener to lObj.currFrame
  end
  
  methods
    
    function obj = LabelTracker(labelerObj)
      obj.lObj = labelerObj;   
      
      axOver = axisOverlay(obj.lObj.gdata.axes_curr);
      axOver.LineWidth = 2;
      obj.ax = axOver;
      
      obj.hLCurrMovie = addlistener(labelerObj,'currMovie','PostSet',@(s,e)obj.newLabelerMovie());
      obj.hLCurrFrame = addlistener(labelerObj,'currFrame','PostSet',@(s,e)obj.newLabelerFrame());
    end
    
    function init(obj)
      % Called when a new project is created/loaded, etc
      axisOverlay(obj.lObj.gdata.axes_curr,obj.ax);
      obj.initHook();
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
    
    function setParamFile(obj,prmFile)
      obj.paramFile = prmFile;
    end
    
    function initHook(obj) %#ok<*MANU>
      % Called when a new project is created/loaded, etc
    end    
       
    function train(obj)
    end
    
%     function inspectTrainingData(obj)
%     end

    function track(obj,iMovs,frms)
      % Apply trained tracker to the specified frames.
      %
      % iMovs: [M] indices into .lObj.movieFilesAll to track
      % frms: [M] cell array. frms{i} is a vector of frames to track for iMovs(i).
    end    
    
    function newLabelerFrame(obj)
      % Called when Labeler is navigated to a new frame
    end
    
    function newLabelerMovie(obj)
      % Called when Labeler is navigated to a new movie
    end
    
    function s = getSaveToken(obj)
      % Get a struct to serialize
      s = struct();
    end
    
    function loadSaveToken(obj,s) %#ok<*INUSD>
      
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