classdef (Abstract) LabelTracker < handle
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
    algorithmName  % char
    trackerInfo  % struct with whatever information we want to save about the current tracker.     
  end  
  
  properties
    lObj % (back)handle to Labeler object
    paramFile; % char, current parameter file
    %ax % axis for viewing tracking results
    sPrmAll; % all parameters - KB 20190214: store all parameters with each tracker
    
    trkVizInterpolate % scalar logical. If true, interpolate tracking results when visualizing
    
    lastTrainStats = []; % struct with information about the last training for visualization
    
    % hListeners  % cell vec of Labeler listeners
  end  
  
  properties (SetAccess=protected)
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
      
      % listeners = { ...
      %   addlistener(labelerObj,'newMovie',@(s,e)obj.newLabelerMovie());
      %   %addlistener(labelerObj,'currFrame','PostSet',@(s,e)obj.newLabelerFrame());
      %   addlistener(labelerObj,'didSetCurrTarget',@(s,e)(obj.newLabelerTarget()));
      %   addlistener(labelerObj,'movieRemoved',@(s,e)obj.labelerMovieRemoved(e));
      %   addlistener(labelerObj,'moviesReordered',@(s,e)obj.labelerMoviesReordered(e));
      %   };
      % obj.hListeners = listeners;
    end
    
    function init(obj)
      % Called when a new project is created/loaded, etc.
      % Also used to reset a tracker to a state with no trained models, no tracking
      % results, etc.
      obj.initHook();
    end
        
    function delete(obj)
      %obj.deleteListeners();
    end    
    
    % function deleteListeners(obj)
    %   % cellfun(@delete,obj.hListeners);
    %   % obj.hListeners = cell(0,1);
    % end
    
    % function setEnableListeners(obj,val)
    %   % hs = obj.hListeners;
    %   % for i=1:numel(hs)
    %   %   hs{i}.Enabled = val;
    %   % end
    % end
	
  end
  
  methods
    
    function initHook(obj) %#ok<*MANU>
      % Called from init() when a new project is created/loaded, etc
      % Designed to be overloaded by subclasses.
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
       
    function ppdata = fetchPreProcData(obj,tblP,ppPrms)  %#ok<STOUT>
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
    
    function tf = hasBeenTrained(obj)  %#ok<STOUT>
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
      % called when a tracker is no longer active.
      %obj.setEnableListeners(false);      
    end
    
    function activate(obj)
      %obj.setEnableListeners(true);
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
      % tfaux = any(strcmp(prop.code,plistcodes)) ;  
      tfaux = any(strcmp(prop.code,plistcodes)) && ~isempty(auxlbl) ;  
        % Added the ~isempty() check above b/c the lack of it was causing
        % occasional errors.  -- ALT, 2024-11-07
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
        [data,~] = ComputeLandmarkFeatureFromPos(xy,occ,sf,ef,nfrms,bodytrx,prop);
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
      prm = yaml.ReadYaml(prmFile);
    end
            
  end  % methods
  
  methods (Static)
    
    function trackerObj = create(lObj, tci, saveToken)
      % Factory method

      % Handle missing tci arg
      if ~exist('tci', 'var') || isempty(tci)
        % Use default network types
        if lObj.maIsMA
          tci = TrackerCreateInfo('DeepTracker', DLNetType.multi_mdn_joint_torch, DLNetMode.multiAnimalBU);
        else
          tci = TrackerCreateInfo('DeepTracker', DLNetType.mdn_joint_fpn, DLNetMode.singleAnimal);
        end
      end
      % Type-check args
      assert(isa(tci, 'TrackerCreateInfo') && isscalar(tci));

      trackerClassName = tci.className ;
      trackerClassConstructorArgs = tci.constructorArgs() ;
      
      if ~exist(trackerClassName,'class') ,
        error('Labeler:projLoad',...
              'Project tracker class ''%s'' cannot be found.',trackerClassName);
      end
      trackerObj = feval(trackerClassName, lObj, trackerClassConstructorArgs{:});
      trackerObj.init();
      if exist('saveToken', 'var')
        trackerObj.loadSaveToken(saveToken);
      end
    end
    
    % function result = getAllTrackersCreateInfo(isMA)
    %   % Get information about all of the kinds of trackers that the user can choose
    %   % from.
    %   %
    %   % This will need updating. DLNetType will include all types of nets
    %   % such as objdetect which will not qualify as eg regular/SA trackers.
    %   if isMA
    %     info = horzcat(DeepTrackerBottomUp.getTrackerInfos(), ...
    %                    DeepTrackerTopDown.getTrackerInfos(), ...
    %                    DeepTrackerTopDownCustom.getTrackerInfos() ) ;
    %     % For custom 2-stage trackers add the DeepTrackerTopDown again.
    %   else        
    %     netTypes = enumeration('DLNetType');
    %     netTypes = netTypes(~[netTypes.isMultiAnimal]);
    %     % info = arrayfun(@(x){'DeepTracker' 'trnNetType' x}, dlnets, 'UniformOutput', false) ;
    %     info = arrayfun(@(netType)(TrackerCreateInfo('DeepTracker', netType, DLNetMode.singleAnimal)), ...
    %                     netTypes, ...
    %                     'UniformOutput', true) ;
    %   end
    %   result = info(:)' ;  % want row vector
    %   assert(isa(result, 'TrackerCreateInfo') && isrow(result)) ;
    % end
    
    % function [tf,loc] = trackersCreateInfoIsMember(tcis1, tcis2)
    %   % Need to re-do this.  During loading, the first arg won't be a
    %   % TrackerClassInfo, it will be builtin object of some kind.
    % 
    %   % For each element of tcis1, determine whether some element of tcis2
    %   % matches it, and if so, at what index into tcis2 the matching element is
    %   % found at.  If tcis1 has n elements, on return tf is an n x 1 logical
    %   % array, and loc is an n x 1 double array of indices into tcis2.  Used to
    %   % match up trackers in a being-loaded .lbl file to the list of available
    %   % trackers according to LabelTracker.getAllTrackersCreateInfo().  tcis1
    %   % should come from the read-in .lbl file, and tcis2 from a call to
    %   % LablerTracker.getAllTrackersCreateInfo().
    %   assert(isa(tcis1, 'TrackerCreateInfo') && isrow(tcis1)) ;
    %   assert(isa(tcis2, 'TrackerCreateInfo') && isrow(tcis2)) ;
    %   n1 = numel(tcis1);
    %   n2 = numel(tcis2);
    %   tf = false(1,n1);
    %   loc = zeros(1,n1);
    %   for i=1:n1
    %     tci1 = tcis1(i) ;  % "tci" for Tracker Create Info
    %     for j=1:n2
    %       tci2 = tcis2(j) ;
    %       if isequal(tci1,tci2)
    %         tf(i) = true;
    %         loc(i) = j;
    %         break
    %       end
    %       tci1_class_name = tci1.className ;
    %       tci2_class_name = tci2.className ;
    %       if strcmp(tci1_class_name,'DeepTrackerTopDownCustom') && ...
    %          strcmp(tci2_class_name,'DeepTrackerTopDownCustom')
    %         % since custom don't have trnNetType defined. MK 20240228
    %         tci1_stage_1_trnNetMode = tci1.netMode(1) ;
    %         tci1_stage_2_trnNetMode = tci1.netMode(2) ;
    %         tci2_stage_1_trnNetMode = tci2.netMode(1) ;
    %         tci2_stage_2_trnNetMode = tci2.netMode(2) ;           
    %         if isequal(tci1_stage_1_trnNetMode,tci2_stage_1_trnNetMode) && ...
    %            isequal(tci1_stage_2_trnNetMode,tci2_stage_2_trnNetMode)
    %           tf(i) = true;
    %           loc(i) = j;  
    %           break
    %         end
    %       end
    %     end  % for j
    %   end  % for i
    % end  % function
    
  end  % methods (Static)
  
  methods
    function set.hideViz(obj, value)
      obj.hideViz = value ;
      obj.lObj.doNotify('didSetTrackerHideViz') ;
    end    

    function set.showPredsCurrTargetOnly(obj, value)
      obj.showPredsCurrTargetOnly = value ;
      obj.lObj.doNotify('didSetTrackerShowPredsCurrTargetOnly') ;
    end    
    
    % function copyProperties_(obj, other)
    %   % Make the independent properties defined in the the LabelerTracker class like
    %   % those of other.  Both obj and other should be scalar LabelTracker objects.
    %   % The underscore suffix indicates this class is intended only to be used by
    %   % LabelTracker and friends, not by external client code.
    % 
    %   % Added this when added the twin() method to LabelTracker subclasses.  Could
    %   % have rewritten the LabelTracker constructor to take property-value pairs,
    %   % but didn't want to get into that.
    % 
    %   % Type-check the arguments
    %   assert(isscalar(obj)) ;
    %   assert(isscalar(other)) ;
    % 
    %   % Define a helper function
    %   className = 'LabelTracker' ;
    %   should_property_be_copied = @(x)(strcmp(x.DefiningClass.Name, className) && ~x.Dependent && ~x.Constant) ;
    % 
    %   % Actually get the prop names that satisfy the predicate
    %   raw_property_names = property_names_satisfying_predicate(obj, should_property_be_copied) ;
    %   %excluded_property_names = {'lObj', 'bgTrackPoller', 'bgTrainPoller', 'bgTrkMonitor', 'bgTrnMonitor', 'trkVizer'}' ;
    %   excluded_property_names = {'lObj'}' ;
    %   property_names = setdiff(raw_property_names, excluded_property_names) ;
    % 
    %   % Set each property in turn
    %   for i = 1:length(property_names)
    %     property_name=property_names{i} ;
    %     raw_property_value = obj.(property_name) ;
    %     if ~isempty(raw_property_value) && isa(raw_property_value, 'handle')
    %       nop() ;
    %     end
    %     if isa(raw_property_value, 'matlab.mixin.Copyable')
    %       property_value = raw_property_value.copy() ;
    %     else
    %       property_value = raw_property_value ;
    %     end
    %     obj.(property_name) = property_value ;
    %   end
    % end  % function
    
  end  % methods
end  % classdef
