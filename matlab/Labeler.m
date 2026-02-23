classdef Labeler < handle
% Bransonlab Animal Video Labeler/Tracker

  properties (Constant, Hidden)
    VERSION = '3.1'
    DEFAULT_LBLFILENAME = '%s.lbl'
    DEFAULT_CFG_FILENAME = Labeler.defaultCfgFilePath()
    MAX_MOVIENAME_LENGTH = 80
    
    % non-config props
  	% KB 20190214 - replaced trackDLParams, preProcParams with trackParams
    SAVEPROPS = { ...
      'VERSION' 'projname' 'maIsMA' ...
      'movieReadPreLoadMovies' ...
      'movieFilesAll' 'movieInfoAll' 'trxFilesAll' 'trxInfoAll' 'projMacros'...
      'movieFilesAllGT' 'movieInfoAllGT' ...
      'movieFilesAllCropInfo' 'movieFilesAllGTCropInfo' ...
      'movieFilesAllHistEqLUT' 'movieFilesAllGTHistEqLUT' ...
      'trxFilesAllGT' 'trxInfoAllGT' ...
      'cropIsCropMode' ...
      'viewCalibrationData' 'viewCalProjWide' ...
      'viewCalibrationDataGT' ...
      'labels' 'labels2' 'labelsGT' 'labels2GT' ...
      'labelsRoi' 'labelsRoiGT' ...
      'currMovie' 'currFrame' 'currTarget' ...
      'gtIsGTMode' 'gtSuggMFTable' 'gtTblRes' ...
      'labelTemplate' ...
      'trackModeIdx' 'trackDLBackEnd' ...
      'suspScore' 'suspSelectedMFT' 'suspComputeFcn' ...
      'trackParams' ...
      'trackAutoSetParams' ...
      'xvResults' 'xvResultsTS' ...
      'fgEmpiricalPDF'...
      'projectHasTrx'...
      'skeletonEdges' 'showSkeleton' 'showMaRoi' 'showMaRoiAux' 'flipLandmarkMatches' 'skelHead' 'skelTail' 'skelNames' ...
      'isTwoClickAlign'...
      'trkResIDs' 'trkRes' 'trkResGT' 'trkResViz' 'saveVersionInfo' ...
      'nLabelPointsAdd' 'track_id'};
    
     % props to update when replace path is given during initialization
     % MK 20220418
    MOVIEPROPS = { ...
      'movieFilesAll' 'trxFilesAll' 'projMacros' ...
      'movieFilesAllGT' 'trxFilesAllGT' }
    
    SAVEBUTNOTLOADPROPS = { ...
       'VERSION' 'currFrame' 'currMovie' 'currTarget'};     
     
    DLCONFIGINFOURL = 'https://github.com/kristinbranson/APT/wiki/Deep-Neural-Network-Tracking'; 
    
    NEIGHBORING_FRAME_MAXRADIUS = 10;
    DEFAULT_RAW_LABEL_FILENAME = 'label_file.lbl';

    PROPS_GTSHARED = struct('reg',...
      struct('MFA','movieFilesAll',...
             'MFAF','movieFilesAllFull',...
             'MFAHL','movieFilesAllHaveLbls',...
             'MFACI','movieFilesAllCropInfo',...
             'MFALUT','movieFilesAllHistEqLUT',...
             'MIA','movieInfoAll',...
             'TFA','trxFilesAll',...
             'TFAF','trxFilesAllFull',...
             'TIA','trxInfoAll',...
             'LBL','labels',...
             'LBL2','labels2',...
             'LPOS','labeledpos',...
             'LPOSTS','labeledposTS',...
             'LPOSTAG','labeledpostag',...
             'LPOS2','labeledpos2',...
             'VCD','viewCalibrationData',...
             'TRKRES','trkRes'),...
             'gt',...
      struct('MFA','movieFilesAllGT',...
             'MFAF','movieFilesAllGTFull',...
             'MFAHL','movieFilesAllGTHaveLbls',...
             'MFACI','movieFilesAllGTCropInfo',...
             'MFALUT','movieFilesAllGTHistEqLUT',...
             'MIA','movieInfoAllGT',...
             'TFA','trxFilesAllGT',...
             'TFAF','trxFilesAllGTFull',...
             'TIA','trxInfoAllGT',...
             'LBL','labelsGT',...
             'LBL2','labels2GT',...
             'LPOS','labeledposGT',...
             'LPOSTS','labeledposTSGT',...
             'LPOSTAG','labeledpostagGT',...
             'LPOS2','labeledpos2GT',...
             'VCD','viewCalibrationDataGT',...
             'TRKRES','trkResGT'));
  end  % properties (Constant, Hidden)
  
  events
    newProject
    didLoadProject
    newMovie
    %startAddMovie
    %finishAddMovie
    %startSetMovie
      
    % This event is thrown immediately before .currMovie is updated (if 
    % necessary). Listeners should not rely on the value of .currMovie at
    % event time. currMovie will be subsequently updated (in the usual way) 
    % if necessary. 
    % EventData is a MoviesRemappedEventData
    moviesReordered
    
    dataImported
    updateDoesNeedSave
    updateStatusAndPointer
    didSetTrx
    updateTrxSetShowTrue
    updateTrxSetShowFalse
    updateTrxTable
    updateFrameTableIncremental
    updateFrameTableComplete

    didSetProjectName
    didSetProjFSInfo
    didSetMovieFilesAll
    didSetMovieFilesAllGT
    didSetMovieFilesAllHaveLbls
    didSetMovieFilesAllGTHaveLbls    

    didSetMovieCenterOnTarget
    didSetMovieRotateTargetUp
    didSetMovieForceGrayscale
    didSetMovieInvert
    didSetMovieViewBGsubbed
    
    didSetTrxFilesAll
    didSetTrxFilesAllGT

    didSetShowTrx
    didSetShowTrxCurrTargetOnly
    didSetShowOccludedBox
    didSetShowSkeleton
    didSetShowMaRoi 
    didSetShowMaRoiAux

    % didSetLabels
    didSetLabelMode
    didSetLabels2Hide

    didSetLabels2ShowCurrTargetOnly
    didSetLastLabelChangeTS
    didSetLblCore

    gtIsGTModeChanged 
    gtSuggUpdated  % general update occurred of gtSuggMFTable*
    gtSuggMFTableLbledUpdated  % incremental update of gtSuggMFTableLbled occurred
    gtResUpdated  % update of GT performance results occurred
    
    % update_menu_track_tracking_algorithm
    % update_menu_track_tracking_algorithm_quick
    update_menu_track_tracker_history
    didSetCurrTracker
    didSetCurrTarget

    didSetTrackModeIdx
    didSetTrackDLBackEnd
    didSetTrackNFramesSmall
    didSetTrackNFramesLarge
    didSetTrackNFramesNear
    didSetTrackParams
    didSpawnTrackingForGT
    didComputeGTResults
    newProgressMeter

    cropIsCropModeChanged  % cropIsCropMode mutated
    cropCropsChanged  % something in .movieFilesAll*CropInfo mutated
    cropUpdateCropGUITools

    update_text_trackerinfo
    refreshTrackMonitorViz
    updateTrackMonitorViz
    refreshTrainMonitorViz
    updateTrainMonitorViz
    % raiseTrainingStoppedDialog
    updateTargetCentrationAndZoom
    updateMainAxisHighlight
    updateBackendTestText
    updateAfterCurrentFrameSet
    update
    updateTimelineSelection
    updateTimelineProps
    updateTimelineStatThresh
    updateTimelineTraces
    updateTimelineLandmarkColors
    updateCurrImagesAllViews
    updatePrevAxesImage
    updatePrevAxesLabels
    initializePrevAxesTemplate
    updatePrevAxes
    downdateCachedAxesProperties
    updateShortcuts
  end

  events  % used to come from labeler.tracker
    % Thrown when new tracking results are loaded for the current lObj
    % movie
    % newTrackingResults 
    
    updateTrainingMonitor
    trainEnd
    updateTrackingMonitor
    trackEnd    

    didSetTrackerHideViz
    didSetTrackerShowPredsCurrTargetOnly
  end
  
  %% Project
  properties
    projname              % init: PN
    projFSInfo            % filesystem info
    projTempDir           % temp dir name to save the raw label file
    infoTimelineModel_     % InfoTimelineModel object for timeline selection state
  end

  properties (Dependent)
    infoTimelineModel     % InfoTimelineModel object for timeline selection state
  end

  properties
    projTempDirDontClearOnDestructor = false  % transient. set to true for eg CI testing
  end

  properties (SetAccess=private)
    projMacros = struct()  % scalar struct, filesys macros. init: PN
  end

  properties
    % TODO: rename this to "initialConfig" or similar. This is now a
    % "configuration" not "preferences". For the most part this does not 
    % dup Labeler properties, b/c many of the configuration 
    %
    % scalar struct containing noncore (typically cosmetic) prefs
    % TODO: rename to projConfig or similar
    %
    % Notes 20160818: Configuration properties
    % projPrefs captures minor cosmetic/navigation configuration
    % parameters. These are all settable, but currently there is no fixed
    % policy about how the UI updates in response. Depending on the
    % (sub)property, changes may be reflected in the UI "immediately" or on
    % next use; changes may only be reflected after a new movie; or chnages
    % may never be reflected (for now). Changes will always be saved to
    % project files however.
    %
    % A small subset of more important configuration (legacy) params are
    % Dependent and forward (both set and get) to a subprop of projPrefs.
    % This gives the user explicit property access while still storing the
    % property within the cfg structure.
    %
    % Finally, a small subset of the most important configuration params 
    % (eg labelMode, nview) have their own nonDependent properties.
    % When configurations are loaded/saved, these props are explicitly 
    % copyied from/to the configuration struct.
    %
    % Maybe an improved/ultimate implementation:
    % * Make .projPrefs and all subprops handle objects. This improves the
    % set API by i) eliminating mistyped fieldnames, ii) making set-checks
    % convenient, and iii) making set-observation and UI updates
    % convenient.
    % * Eliminate the 2nd set of params above, just use the .projPrefs
    % structure which now has listeners attached.
    % * The most important configuration params can still have their own
    % props.
    projPrefs  % init: C
    
    projVerbose = 0  % transient, unmanaged
    
    isgui = false  % whether there is a GUI
    isInDebugMode = false  % whether the Labeler is in debug mode.  Controls e.g. whether the Debug menu is shown.
    isInAwsDebugMode = false  % whether the Labeler is in AWS debug mode.  Controls e.g. whether AWS is shutdown at exit.
    unTarLoc = ''  % location that project has most recently been untarred to
    
    projRngSeed = 17 
    
    saveVersionInfo  % info about versions of stuff when proj last saved
    currentTrackerIndexInTrackersAll_
  end

  properties (Dependent)
    hasProject             % scalar logical
    projectfile            % Full path to current project 
    projectroot            % Parent dir of projectfile, if it exists
    bgTrnIsRunning         % True iff background training is running
    bgTrkIsRunning         % True iff background tracking is running
    % trackersAll            % All the 'template' trackers
    % trackersAllCreateInfo  % The creation info for each tracker in trackersAll
    trackerHistory        
    lastTrainEndCause 
      % Did the last bout of training complete or error or was it aborted by user.
      % Only meaningful if training has been run at least once in the current session.
      % Defaults to EndCause.undefined if training has not been run in the current session.
      % In other words not persisted to the .lbl file in any way.
    lastTrackEndCause  
      % Did the last bout of tracking complete or error or was it aborted by user.
      % Only meaningful if tracking has been run at least once in the current session.
      % Defaults to EndCause.undefined if tracking has not been run in the current session.
      % In other words not persisted to the .lbl file in any way.
  end

  properties (Dependent, Hidden)
    backend
  end

  %% Movie/Video
  % Originally "Movie" referred to high-level data units/elements, eg
  % things added, removed, managed by the MovieManager etc; while "Video"
  % referred to visual details, display, playback, etc. But I sort of
  % forgot and mixed them up so that Movie sometimes applies to the latter.
  properties (SetAccess=private)
    nview  % number of views. init: C
    viewNames  % [nview] cellstr. init: C
    
    % States of viewCalProjWide/viewCalData:
    % .viewCalProjWide=[], .vCD=any. Here .vcPW is uninitted and .vCD is unset/immaterial.
    % .viewCalProjWide=true, .vCD=<scalar Calrig obj>. Scalar Calrig obj apples to all movies, including GT
    % .viewCalProjWide=false, .vCD=[nMovSet] cell array of calRigs. .vCD
    % applies element-wise to movies. .vCD{i} can be empty indicating unset
    % calibration object for that movie.
    viewCalProjWide  % [], true, or false. init: PN
    viewCalibrationData  % Opaque calibration 'useradata' for multiview. init: PN
    viewCalibrationDataGT  % etc. 
    
    movieReadPreLoadMovies = false  % scalar logical. Set .preload property on any MovieReaders per this prop
    movieReader = []  % [1xnview] MovieReader objects. init: C
    movieInfoAll = {}  % cell-of-structs, same size as movieFilesAll
    movieInfoAllGT = {}  % same as .movieInfoAll but for GT mode
    movieDontAskRmMovieWithLabels = false  % If true, won't warn about removing-movies-with-labels    
    projectHasTrx = false  % whether there are trx files for any movie
  end

  properties (Dependent)
    movieInfoAllGTaware  
    viewCalibrationDataGTaware  % Either viewCalData or viewCalDataGT
    viewCalibrationDataCurrent  % view calibration data applicable to current movie (gt aware)
  end

  properties
    movieFilesAll = {}  % [nmovset x nview] column cellstr, full paths to movies; can include macros 
    movieFilesAllGT = {}  % same as .movieFilesAll but for GT mode
  end

  properties
    % Using cells here so movies do not have to all have the same bitDepth
    % See HistEq.genHistEqLUT for notes on how to apply LUTs
    %
    % These should prob be called "preProcMovieFilesAllHistEqLUT" since
    % they are preproc-parameter dependent etc
    movieFilesAllHistEqLUT  % [nmovset x nview] cell. Each el is a scalar struct containing lut + related info, or [] 
    movieFilesAllGTHistEqLUT  % [nmovsetGT x nview] "
    cmax_auto = nan(0,1) 
    clim_manual = zeros(0,2) 
  end

  properties
    movieFilesAllHaveLbls = zeros(0,1)  % [nmovsetx1] double; actually, "numLbledTgts"
        % How MFAHL is maintained
        % - At project load, it is updated fully.
        % - Trivial update on movieRm/movieAdd.
        % - Otherwise, all labeling operations can only affect the current
        % movie; meanwhile the FrameTable contains all necessary info to
        % update movieFilesAllHaveLbls. So we piggyback onto
        % updateFrameTable*(). 
        %
        % For MultiView, MFAHL is true if any movie in a movieset has
        % labels.
    movieFilesAllGTHaveLbls = false(0,1)  % etc
  end

  properties
    moviename  % short 'pretty' name, cosmetic purposes only. For multiview, primary movie name.
    movieCenterOnTarget = false  % scalar logical.
    movieRotateTargetUp = false
  end

  properties (Transient)
    % I don't see where either of these are ever changed -- ALT, 2023-05-14
    movieCenterOnTargetLandmark = false  % scalar logical. If true, see movieCenterOnTargetIpt. Transient, unmanaged.
    movieCenterOnTargetIpt = []  % scalar point index, used if movieCenterOnTargetLandmark=true. Transient, unmanaged
  end

  properties
    movieForceGrayscale = false  % scalar logical. In future could make [1xnview].
    movieFrameStepBig  % scalar positive int
    movieShiftArrowNavMode  % scalar ShiftArrowMovieNavMode
  end

  properties (SetAccess=private)
    movieShiftArrowNavModeThresh  % scalar double. This is separate prop from the ShiftArrowMode so it persists even if the ShiftArrowMode changes.
  end

  properties
    movieShiftArrowNavModeThreshCmp  % char, eg '<' or '>='
    moviePlaySegRadius  % scalar int
    moviePlayFPS  
    movieInvert  % [1xnview] logical. If true, movie should be inverted when read. This is to compensate for codec issues where movies can be read inverted 
                 % on platform A wrt platform B
      % Not much care is taken wrt interactions with cropInfo. If you 
      % change your .movieInvert, then your crops will likely be wrong.
      % A warning is thrown but nothing else
  end

  properties (Transient, SetAccess = protected)
    movieViewBGsubbed = false
  end

  properties (Dependent)
    isMultiView 
    movieFilesAllGTaware 
    movieFilesAllFull  % like movieFilesAll, but macro-replaced and platformized
    movieFilesAllGTFull  % etc
    movieFilesAllFullGTaware 
    movieFilesAllHaveLblsGTaware 
    hasMovie 
    moviefile 
    nframes 
    movierawnr  % [nview]. numRows in original/raw movies
    movierawnc  % [nview]. numCols in original/raw movies
    movienr  % [nview]. always equal to numRows in .movieroi
    movienc  % [nview]. always equal to numCols in .movieroi
    movieroi  % [nview x 4]. Each row is [xlo xhi ylo yhi]. If no crop present, then this is just [1 nc 1 nr].
    movieroictr  % [nview x 2]. Each row is [xc yc] center of current roi in that view.
    nmovies 
    nmoviesGT 
    nmoviesGTaware 
    moviesSelected  % [nSel] vector of MovieIndices currently selected in MovieManager. GT mode ok.
    doesNeedSave
  end
  
  %% Crop
  properties
    movieFilesAllCropInfo  % [nmovset x 1] cell. Each el is a [nview] array of cropInfos, or [] if no crop info 
    movieFilesAllGTCropInfo  % [nmovsetGT x 1] "
    cropIsCropMode  % scalar logical
  end

  properties (Dependent)
    movieFilesAllCropInfoGTaware
    cropProjHasCrops  % scalar logical. If true, all elements of movieFilesAll*CropInfo are populated. If false, all elements of " are []
  end
  
  %% Trx
  properties
    trxFilesAll = {}  % column cellstr, full paths to trxs. Same size as movieFilesAll.
    trxInfoAll = {}
    trxFilesAllGT = {}  % etc. Same size as movieFilesAllGT.
    trxInfoAllGT = {}
  end

  properties (SetAccess=private)
    trxCache = []             % containers.Map. Keys: fullpath. vals: lazy-loaded structs with fields: .trx and .frm2trx
    trx = []                  % trx object
    frm2trx = []              % nFrm x nTrx logical. frm2trx(iFrm,iTrx) is true if trx iTrx is live on frame iFrm (for current movie)
  end

  properties (Dependent)
    targetZoomRadiusDefault
  end

  properties (Dependent)
    trxFilesAllFull  % like .movieFilesAllFull, but for .trxFilesAll
    trxFilesAllGTFull  % etc
    trxFilesAllFullGTaware
    trxInfoAllGTaware
    hasTrx
    currTrx
    nTrx
    nTargets  % nTrx, or 1 if no Trx
  end
  
  %% ShowTrx
  properties
    showTrx                   % true to show trajectories
    showTrxCurrTargetOnly     % if true, plot only current target
    showTrxIDLbl              % true to show id label. relevant if .hasTrx or .maIsMa
    showOccludedBox           % whether to show the occluded box    
    showSkeleton              % true to plot skeleton 
    showMaRoi 
    showMaRoiAux
  end 
  
  %% Labeling
  properties
    labelMode             % scalar LabelMode. init: C
    % Multiview. Right now all 3d pts must live in all views, eg
    % .nLabelPoints=nView*NumLabelPoints. first dim of labeledpos is
    % ordered as {pt1vw1,pt2vw1,...ptNvw1,pt1vw2,...ptNvwK}
    labels 
    labels2  % [nmov] cell array of TrkFile. See notes in %% Labels2 section
    labelsGT 
    labels2GT 
    labelsRoi 
    labelsRoiGT
    labels2Hide           % scalar logical
    labels2ShowCurrTargetOnly   % scalar logical, transient    
    skeletonEdges = zeros(0,2)  % nEdges x 2 matrix containing indices of vertex landmarks
                                %
                                % Multiview: currently, els of skeletonEdges
                                % are expected to be in (1..nPhysPts), ie 
                                % edges defined wrt 3d/physical pts with 
                                % pts identified across views
    skelHead = []  % [], or scalar pt index for head. 
                   % Multiview: indices currently expected to be in (1..nPhysPts)
    skelTail = []
    skelNames    % [nptsets] cellstr names labeling rows of .labeledposIPtSetMap.
                 % NOTE: arguably the "point names" should be. init: C
                 % used to be labeledposSetNames
    flipLandmarkMatches = zeros(0,2)  % nPairs x 2 matrix containing indices of vertex landmarks    
  end

  properties  
    labelPointsPlotInfo   % struct containing cosmetic info for labelPoints. init: C
    predPointsPlotInfo   % " predicted points. init: C
    impPointsPlotInfo 
    isTwoClickAlign = true  % KB 20220506 store the state of whether two-click alignment is selected
  end

  properties (SetAccess=private)
    nLabelPoints          % scalar integer. This is the total number of 2D labeled points across all views. Contrast with nPhysPoints. init: C
    labelTemplate 
    nLabelPointsAdd = 0   % scalar integer. This is set when LabelController::projAddLandmarks() is called
    
    labeledposIPtSetMap   % [nptsets x nview] 3d 'point set' identifications. labeledposIPtSetMap(iSet,:) gives
                          % point indices for set iSet in various views. init: C
    labeledposIPt2View    % [npts] vector of indices into 1:obj.nview. Convenience prop, derived from .labeledposIPtSetMap. init: C
    labeledposIPt2Set     % [npts] vector of set indices for each point. Convenience prop. init: C
  end

  properties
    labeledposNeedsSave   % scalar logical, .labeledpos has been touched since last save. Currently does NOT account for labeledpostag
    lastLabelChangeTS     % last time training labels were changed
  end

  properties (Dependent)
    hFig  % This is a temporary crutch.  Eventually it will not be needed, and then we eliminate it.
          % It is no longer used internally by the Labeler methods.
  end

  properties (Transient)  % private by convention
    controller_  % This is a temporary crutch.  Eventually it will not be needed, and then we eliminate it.
  end

  properties (Transient)  % private by convention
    doesNeedSave_ = false
  end

  properties (Transient)  % private by convention
    howBusy_ = 0  % increases with calls to pushBusyStatus(), decreases with calls to popBusyStatus()
    %isStatusBusy_ = false
    rawStatusStringStack_  = cell(1,0)
    rawClearStatusString_ = 'Ready.'
    % rawStatusString_  = 'Ready.'
    % rawStatusStringWhenClear_ = 'Ready.'
    progressMeter_
    backgroundProcessingStatusString_ = '' 
  end

  properties (Dependent)
    isStatusBusy
    rawStatusString
    %rawStatusStringWhenClear
    % didSpawnTrackingForGT
    progressMeter
    backgroundProcessingStatusString
  end

  properties (Dependent, Hidden)
    labeledpos            % column cell vec with .nmovies elements. labeledpos{iMov} is npts x 2 x nFrm(iMov) x nTrx(iMov) double array; 
                          % labeledpos{1}(:,1,:,:) is X-coord, labeledpos{1}(:,2,:,:) is Y-coord. init: PN
    labeledposTS          % labeledposTS{iMov} is nptsxnFrm(iMov)xnTrx(iMov). It is the last time .labeledpos or .labeledpostag was touched. init: PN
    labeledpostag         % column cell vec with .nmovies elements. labeledpostag{iMov} is npts x nFrm(iMov) x nTrx(iMov) logical indicating *occludedness*. 
                          % ("tag" for legacy reasons) init: PN
    labeledposGT          % like .labeledpos    
    labeledposTSGT        % like .labeledposTS
    labeledpostagGT       % like .labeledpostag
    labeledpos2GT         % like .labeledpos2
    labeledpos2           % identical size/shape with labeledpos. aux labels (eg predicted, 2nd set, etc). init: PN

    labeledposGTaware 
    labeledposTSGTaware 
    labeledpostagGTaware 
    labelsGTaware 

    labeledpos2GTaware 
    labels2GTaware 
    
    labeledposCurrMovie 
    labeledpos2CurrMovie 
    labeledpostagCurrMovie 
    
    labelsCurrMovie 
    labels2CurrMovie 
    
    nPhysPoints  % number of physical/3D points
  end

  properties
    lblCore  % init: L
  end

  properties
    labeledpos2trkViz  % scalar TrackingVisualizer*, or [] if no imported results for currMovie
  end
  
  properties
    fgEmpiricalPDF  % struct containing empirical FG pdf and metadata
  end
  
  %% MA
  properties
    maIsMA
  end
  
  %% GT mode
  properties (SetAccess=private)
    gtIsGTMode = false; % scalar logical
  end

  properties
    gtSuggMFTable  % [nGTSugg x ncol] MFTable for suggested frames to label. .mov values are MovieIndexes
    gtSuggMFTableLbled  % [nGTSuggx1] logical flags indicating whether rows of .gtSuggMFTable were gt-labeled

    gtTblRes  % [nGTcomp x ncol] table, or []. Most recent GT performance results. 
      % gtTblRes(:,MFTable.FLDSID) need not match
      % gtSuggMFTable(:,MFTable.FLDSID) because eg GT performance can be 
      % computed even if some suggested frames are not be labeled.
    gtPlotParams = struct('prc_vals',[50,75,90,95,98],...
      'nbins',50); % parameters for ShowGTResults
  end

  properties (Dependent)
    gtNumSugg  % height(gtSuggMFTable)
  end
  
  %% Suspiciousness
  properties (SetAccess=private)
    suspScore  % column cell vec same size as labeledpos. suspScore{iMov} is nFrm(iMov) x nTrx(iMov)
    suspSelectedMFT  % MFT table of selected suspicous frames.
    suspComputeFcn  
      % Function with sig [score,tblMFT,diagstr]=fcn(labelerObj) that 
      % computes suspScore, suspSelectedMFT.
      % See .suspScore for required size/dims of suspScore and contents.
      % diagstr is arbitrary diagnostic info (assumed char for now).     
  end

  properties (SetAccess=private, Transient)
    suspDiag  % Transient "userdata", diagnostic output from suspComputeFcn    
  end
  
  %% PreProc
  properties
    ppdb  % PreProcDB for DL
  end

  properties (Dependent)    
    preProcParams  % struct - KB 20190214 -- made this a dependent property, derived from trackParams
  end  

  %% Tracking
  properties
    % trackersAll_ = cell(1,0)
    %   % cell row vector of concrete LabelTracker objects. init: PNPL
    %   % Since the introduction of trackerHistory_, these are used only as templates.
    %   % Calling .hasBeenTrained() on any of these should return false, always.
    % trackersAllCreateInfo_ = cell(1,0)
    %   % cell row vector of "tracker-create-info" structs, with same number of
    %   % elements as trackersAll, and with a one-to-one correspondence between them.
    %   % Exists to ease creation of new working trackers.
    trackerHistory_ = cell(1,0)
      % Cell row vector of concrete LabelTracker objects.  Contains a history of all
      % trained trackers, with age increasing with index.  trackerHistory_{1} is the
      % current tracker.
  end

  properties (Dependent)
    tracker  % The current tracker, or []
    trackerAlgo  % The current tracker algorithm, or ''
    trackerNetsUsed  % cellstr
    trackerIsDL
    trackerIsTwoStage
    trackerIsBotUp
    trackerIsObjDet
    trackDLParams  % scalar struct, common DL params
    DLCacheDir  % string, location of DL cache dir
  end

  properties
    trackModeIdx  % index into MFTSetEnum.TrackingMenu* for current trackmode. 
     %Note MFTSetEnum.TrackingMenuNoTrx==MFTSetEnum.TrackingMenuTrx(1:K).
     %Values of trackModeIdx 1..K apply to either the NoTrx or Trx cases; 
     %larger values apply only the Trx case.
     
    trackDLBackEnd  % scalar DLBackEndClass
    
    trackNFramesSmall  % small/fine frame increment for tracking. init: C
    trackNFramesLarge  % big/coarse ". init: C
    trackNFramesNear  % neighborhood radius. init: C
    trackParams  % all tracking parameters. init: C
    trackAutoSetParams = true
  end

  properties
    trkResIDs  % [nTR x 1] cellstr unique IDs
    trkRes  % [nMov x nview x nTR] cell. cell array of TrkFile objs
    trkResGT  % [nMovGT x nview x nTR] cell. etc
    trkResViz  % [nTR x 1] cell. TrackingVisualizer vector
    track_id = false
  end

  properties (Dependent)
    trkResGTaware
  end
  
  %% CrossValidation
  properties
    xvResults  % table of most recent cross-validation results. This table
      % has a row for every labeled frame present at the time xvalidation 
      % was run. So it should be fairly explicit if/when it is out-of-date 
      % relative to the project (eg when labels are added or removed)
    xvResultsTS  % timestamp for xvResults
  end
  
  %% Prev
  properties
    %prevIm = struct('CData',0,'XData',0,'YData',0)  % struct, like a stripped image handle (.CData, .XData, .YData). 'primary' view only
    prevIm = []
    prevImRoi = [] 
    prevAxesMode_ = PrevAxesMode.LASTSEEN  % scalar PrevAxesMode
    prevAxesModeTargetSpec_ = PrevAxesTargetSpec()  % identity + rendering data for frozen/prev-axes frame
    prevAxesYDir_ = 'reverse'  % cached YDir of prev axes, set by downdateCachedAxesProperties
    currAxesProps_ = struct('XDir', 'normal', 'YDir', 'reverse', 'XLim', [0.5 1024.5], 'YLim', [0.5 1024.5])  % cached props of curr axes
    prevAxesSizeInPixels_ = [256 256]  % cached [w h] of prev axes in pixels
    lblPrev_ptsH  % [npts] VirtualLine. init: L
    lblPrev_ptsTxtH  % [npts] VirtualText. init: L
  end

  properties (Dependent)
    prevAxesMode
    prevAxesModeTargetSpec
  end
  
  %% Misc
  properties
    prevFrame = nan       % last previously VISITED frame
    currTarget = 1      % always 1 if proj doesn't have trx    
    currImHud  % scalar AxisHUD object TODO: move to LabelerGUI. init: C
  end
%   properties
%     keyPressHandlers  % [nhandlerx1] cell array of LabelerKeyEventHandlers.
%   end

  properties
    currMovie  % idx into .movieFilesAll (row index, when obj.multiView is true), or .movieFilesAllGT when .gtIsGTmode is on
  end

  properties (Dependent)
    currMovIdx  % scalar MovieIndex
    selectedFrames  % vector of frames currently selected frames; typically t0:t1
  end

  properties 
    currFrame = 1  % current frame
    currIm = []             % [nview] cell vec of image data. init: C
    currImRoi = []
    % selectedFrames_ = []     % vector of frames currently selected frames; typically t0:t1
    drag = false 
    drag_pt = [] 
    silent_ = false  % Don't open dialogs. Use defaults. For testing and debugging
  end

  properties (SetAccess=private)
    isinit = false          % scalar logical; true during initialization, when some invariants not respected
  end

  properties (Dependent)
    gdata  % handles structure for LabelerGUI.  This is a temporary crutch.  Eventually it will not be needed, and then we eliminate it.
           % It is no longer used by the Labeler internally.
    silent
  end

  
  % Primary lifecycle methods
  methods 
    function obj = Labeler(varargin)
      obj.rc = RC();
      [isgui, isInDebugMode, isInAwsDebugMode] = ...
        myparse_nocheck(varargin, ...
                        'isgui', false, ...
                        'isInDebugMode', false, ...
                        'isInAwsDebugMode', false) ;
      obj.isgui = isgui ;
      obj.isInDebugMode = isInDebugMode ;
      obj.isInAwsDebugMode = isInAwsDebugMode ;
      obj.progressMeter_ = ProgressMeter() ;
      obj.infoTimelineModel_ = InfoTimelineModel(obj.hasTrx);
      if ~isgui ,
        % If a GUI is attached, this is done by the controller, after it has
        % registered itself with the Labeler.
        % If no GUI attached, we do it ourselves.
        obj.handleCreationTimeAdditionalArgumentsGUI_(varargin{:}) ;
      end
    end

    function registerController(obj, controller)
      % This function is a temporary hack.  The long-term goal is to get rid of
      % labeller_controller_ property b/c the model shouldn't need to talk directly
      % to it.  It's here now as a crutch until we eliminate reliance on it, then we
      % get rid of it. directly talk to either of those.
      obj.isgui = true ;
      obj.controller_ = controller ;
    end

    function handleCreationTimeAdditionalArgumentsGUI_(obj, varargin)
      [projfile, replace_path] = ...
        myparse_nocheck(varargin, ...
                        'projfile',[], ...
                        'replace_path',{'',''}) ;
      if projfile ,
        obj.projLoadGUI(projfile, 'replace_path', replace_path) ;
      end      
    end

    function delete(obj)
      obj.controller_ = [] ;  % this is a weak reference (by convention), so don't delete
      % Backend should release resources properly when deleted now
      % be = obj.trackDLBackEnd;
      % if ~isempty(be)
      %   be.shutdown();
      % end
      if ~isempty(obj.projTempDir) 
        if obj.projTempDirDontClearOnDestructor ,
          fprintf('As requested, leaving temp dir %s in place.\n', obj.projTempDir) ;
        else
          obj.projRemoveTempDirAsync();
        end
      end
    end  % function    
  end  % methods
  
          
  %% Prop access
  methods % dependent prop getters
    function v = get.viewCalibrationDataGTaware(obj)
      v = obj.getViewCalibrationDataGTawareArg(obj.gtIsGTMode);
    end

    function v = getViewCalibrationDataGTawareArg(obj,gt)
      if gt
        v = obj.viewCalibrationDataGT;
      else
        v = obj.viewCalibrationData;
      end
    end

    function v = get.viewCalibrationDataCurrent(obj)
      % Nearly a forward to getViewCalibrationDataMovIdx except edge cases
      vcdPW = obj.viewCalProjWide;
      if isempty(vcdPW)
        v = [];
      elseif vcdPW
        vcd = obj.viewCalibrationData; % applies to regular and GT movs
        assert(isequal(vcd,[]) || isscalar(vcd));
        v = vcd;
      else % ~vcdPW
        vcd = obj.viewCalibrationDataGTaware;
        nmov = obj.nmoviesGTaware;
        assert(iscell(vcd) && numel(vcd)==nmov);
        if nmov==0 || obj.currMovie==0
          v = [];
        else
          v = vcd{obj.currMovie};
        end
      end
    end

    function v = getViewCalibrationDataMovIdx(obj,mIdx)
      vcdPW = obj.viewCalProjWide;
      if isempty(vcdPW)
        v = [];
      elseif vcdPW
        vcd = obj.viewCalibrationData; % applies to regular and GT movs
        assert(isequal(vcd,[]) || isscalar(vcd));
        v = vcd;
      else % ~vcdPW
        [iMov,gt] = mIdx.get();
        vcd = obj.getViewCalibrationDataGTawareArg(gt);
        nmov = obj.getnmoviesGTawareArg(gt);
        assert(iscell(vcd) && numel(vcd)==nmov);
        v = vcd{iMov};
      end
    end

    function v = get.isMultiView(obj)
      v = obj.nview>1;
    end

    function v = get.movieFilesAllGTaware(obj)
      if obj.gtIsGTMode
        v = obj.movieFilesAllGT;
      else        
        v = obj.movieFilesAll;
      end
    end

    function v = get.movieFilesAllFull(obj)
      % See also .projLocalizePath()
      sMacro = obj.projMacrosGetWithAuto();
      v = FSPath.fullyLocalizeStandardize(obj.movieFilesAll,sMacro);
      FSPath.warnUnreplacedMacros(v);
    end

    function v = get.movieFilesAllGTFull(obj)
      sMacro = obj.projMacrosGetWithAuto();
      v = FSPath.fullyLocalizeStandardize(obj.movieFilesAllGT,sMacro);
      FSPath.warnUnreplacedMacros(v);
    end

    function v = get.movieFilesAllFullGTaware(obj)
      if obj.gtIsGTMode
        v = obj.movieFilesAllGTFull;
      else        
        v = obj.movieFilesAllFull;
      end
    end

    function [tffound,mIdx] = getMovIdxMovieFilesAllFull(obj,movsets,varargin)
      % Get the MovieIndex vector corresponding to a set of movies by
      % comparing against .movieFilesAll*Full.
      %
      % Defaults to using .movieFilesAll*Full per .gtIsGTMode.
      %
      % movsets: [n x nview] movie fullpaths. Can have repeated
      % rows/moviesets.
      %
      % tffound: [n x 1] logical
      % mIdx: [n x 1] MovieIndex vector. mIdx(i)==0<=>tffound(i)==false.
      
      gt = myparse(varargin,...
        'gt',obj.gtIsGTMode);
      
      if gt
        mfaf = obj.movieFilesAllGTFull;
      else        
        mfaf = obj.movieFilesAllFull;
      end
      
      [nmovset,nvw] = size(movsets);
      assert(nvw==obj.nview);
      
      movsets = FSPath.standardPath(movsets);
      movsets = cellfun(@FSPath.platformizePath,movsets,'uni',0);
      [iMov1,iMov2] = Labeler.identifyCommonMovSets(movsets,mfaf);
      
      tffound = false(nmovset,1);
      tffound(iMov1,:) = true;
      mIdx = zeros(nmovset,1);
      mIdx(iMov1) = iMov2; % mIdx is all 0s, or positive movie indices
      mIdx = MovieIndex(mIdx,gt);
    end

    function v = get.movieFilesAllHaveLblsGTaware(obj)
      v = obj.getMovieFilesAllHaveLblsArg(obj.gtIsGTMode);
    end

    function v = getMovieFilesAllHaveLblsArg(obj,gt)
      if gt
        v = obj.movieFilesAllGTHaveLbls;
      else
        v = obj.movieFilesAllHaveLbls;
      end      
    end

    function v = get.movieFilesAllCropInfoGTaware(obj)
      if obj.gtIsGTMode
        v = obj.movieFilesAllGTCropInfo;
      else
        v = obj.movieFilesAllCropInfo;
      end
    end

    function v = getMovieFilesAllHistEqLUTGTawareStc(obj,gt)
      if gt
        v = obj.movieFilesAllGTHistEqLUT;
      else
        v = obj.movieFilesAllHistEqLUT;
      end
    end

    function v = getMovieFilesAllHistEqLUTMovIdx(obj,mIdx)
      % v: [1xnview] row of .movieFilesAll*HistEqLUT
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [imov,gt] = mIdx.get();
      if gt
        v = obj.movieFilesAllGTHistEqLUT(imov,:);
      else
        v = obj.movieFilesAllHistEqLUT(imov,:);
      end
    end

    function v = get.cropProjHasCrops(obj)
      % Returns true if proj has at least one movie and has crops
      v = obj.nmovies>0 && ~isempty(obj.movieFilesAllCropInfo{1});
      
%       ci = [obj.movieFilesAllCropInfo; obj.movieFilesAllGTCropInfo];
%       tf = ~cellfun(@isempty,ci);
%       v = all(tf);
%       assert(v || ~any(tf));
    end

    function v = getMovieFilesAllCropInfoGTAware(obj)
      if obj.gtIsGTMode,
        v = obj.movieFilesAllGTCropInfo;
      else
        v = obj.movieFilesAllCropInfo;
      end
    end

    function v = getMovieFilesAllCropInfoMovIdx(obj,mIdx)
      % mIdx: scalar MovieIndex 
      % v: empty, or [nview] CropInfo array 
      
      assert(isa(mIdx,'MovieIndex'));
      assert(isscalar(mIdx));
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.movieFilesAllGTCropInfo{iMov};
      else
        v = obj.movieFilesAllCropInfo{iMov};
      end
    end

    function v = get.trxFilesAllFull(obj)
      % Warning: Expensive to call. Call me once and then index rather than
      % using a compound indexing-expr.
      v = Labeler.trxFilesLocalize(obj.trxFilesAll,obj.movieFilesAllFull);
    end    

    function v = get.trxFilesAllGTFull(obj)
      % Warning: Expensive to call. Call me once and then index rather than
      % using a compound indexing-expr.
      v = Labeler.trxFilesLocalize(obj.trxFilesAllGT,obj.movieFilesAllGTFull);
    end

    function v = get.trxFilesAllFullGTaware(obj)
      % Warning: Expensive to call. Call me once and then index rather than
      % using a compound indexing-expr.
      if obj.gtIsGTMode
        v = obj.trxFilesAllGTFull;
      else
        v = obj.trxFilesAllFull;
      end
    end

    function v = get.trxInfoAllGTaware(obj)
      if obj.gtIsGTMode
        v = obj.trxInfoAllGT;
      else
        v = obj.trxInfoAll;
      end
    end

    function v = getTrxFilesAllFullMovIdx(obj,mIdx)
      % Warning: Expensive to call. Call me once and then index rather than
      % using a compound indexing-expr.      
      assert(all(isa(mIdx,'MovieIndex')));
      [iMov,gt] = mIdx.get();
      n = numel(iMov);
      v = cell(n,obj.nview);
      
      sMacro = obj.projMacrosGetWithAuto;
      mfa = FSPath.fullyLocalizeStandardize(obj.movieFilesAll,sMacro);
      mfaGT = FSPath.fullyLocalizeStandardize(obj.movieFilesAllGT,sMacro);
      tfa = obj.trxFilesAll;
      tfaGT = obj.trxFilesAllGT;
      for i=1:n
        j = iMov(i);
        if gt(i)
          v(i,:) = Labeler.trxFilesLocalize(tfaGT(j,:),mfaGT(j,:));
        else
          v(i,:) = Labeler.trxFilesLocalize(tfa(j,:),mfa(j,:));
        end
      end
    end
    
    function v = getTrxFileInfoAllMovIdx(obj,mIdx)
      assert(all(isa(mIdx,'MovieIndex')));
      [iMov,gt] = mIdx.get();
      n = numel(iMov);
      v = cell(n,obj.nview);
      for i=1:n
        if gt
          v(i,:) = obj.trxInfoAllGT(iMov(i),:);
        else
          v(i,:) = obj.trxInfoAll(iMov(i),:);
        end
      end
    end
    
    function v = get.hasMovie(obj)
      v = obj.hasProject && obj.movieReader(1).isOpen;
    end    

    function v = get.moviefile(obj)
      mr = obj.movieReader(1);
      if isempty(mr)
        v = [];
      else
        v = mr.filename;
      end
    end

    function v = get.movierawnr(obj)
      mr = obj.movieReader;
      if mr(1).isOpen
        v = [mr.nr]';
      else
        v = nan(obj.nview,1);
      end
    end

    function v = get.movierawnc(obj)
      mr = obj.movieReader;
      if mr(1).isOpen
        v = [mr.nc]';          
      else
        v = nan(obj.nview,1);
      end
    end    

    function v = get.movienr(obj)
      v = obj.movieroi;
      v = v(:,4)-v(:,3)+1;
    end

    function v = get.movienc(obj)
      v = obj.movieroi;
      v = v(:,2)-v(:,1)+1;
    end    

    function v = get.movieroi(obj)
      mr = obj.movieReader;
      if mr(1).isOpen
        v = cat(1,mr.roiread);
      else
        v = nan(obj.nview,4);
      end
    end

    function v = get.movieroictr(obj)
      rois = obj.movieroi;
      v = [rois(:,1)+rois(:,2),rois(:,3)+rois(:,4)]/2;
    end

    function rois = getMovieRoiMovIdx(obj,mIdx)
      % v: [nview x 4] roi
      
      if obj.cropProjHasCrops
        ci = obj.getMovieFilesAllCropInfoMovIdx(mIdx);
        assert(~isempty(ci));        
        rois = cat(1,ci.roi);
      else
        [iMov,gt] = mIdx.get();
        mia = obj.getMovieInfoAllGTawareArg(gt);
        mia = mia(iMov,:);
        rois = cellfun(@(x)[1 x.info.nc 1 x.info.nr],mia,'uni',0);
        rois = cat(1,rois{:});
      end
      szassert(rois,[obj.nview 4]);
    end

    function v = get.nframes(obj)
      if isempty(obj.currMovie) || obj.currMovie==0
        v = nan;
      else
        % multiview case: ifos have .nframes set identically if movies have
        % different lengths
        ifo = obj.movieInfoAllGTaware{obj.currMovie,1};
        v = ifo.nframes;
      end
    end

    function [ncmin,nrmin] = getMinMovieWidthHeight(obj)
      movInfos = [obj.movieInfoAll; obj.movieInfoAllGT];
      nrall = cellfun(@(x)x.info.nr,movInfos); % [(nmov+nmovGT) x nview]
      ncall = cellfun(@(x)x.info.nc,movInfos); % etc
      nrmin = min(nrall,[],1); % [1xnview]
      ncmin = min(ncall,[],1); % [1xnview]      
    end

    function v = getNFramesMovIdx(obj,mIdx)
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt        
        movInfo = obj.movieInfoAllGT{iMov,1};
      else
        movInfo = obj.movieInfoAll{iMov,1};
      end
      v = movInfo.nframes;
    end

    function v = getNFramesMovFile(obj,movFile)
      
      movfilefull = obj.projLocalizePath(movFile);
      assert(exist(movfilefull,'file')>0,'Cannot find file ''%s''.',movfilefull);
      v = MovieReader.getNFrames(movfilefull);

%       mr = MovieReader;
%       mr.open(movfilefull);
%       v = mr.nframes;
%       mr.close();
    end    
    
    function v = get.moviesSelected(obj) %#%GUIREQ
      % Get the currently selected movies.

      % Need to restructure code s.t. the MovieManagerController sets an instance
      % variable in Labeler when the selected movies changes.  Then this function
      % will return the value of that instance variable.  Having the Labeller touch
      % a controller (other than via notify()) breaks the view-independence of the
      % Labeler.
      
      if ~obj.isinit,
        v = [];
        return;
      end
      
      mmc = obj.controller_.movieManagerController_ ;  % suboptimal to have to touch obj.controller_, which is deprecated
      if ~isempty(mmc) && isvalid(mmc)
        v = mmc.getSelectedMovies();
      else
        v = [] ;
        % error('Labeler:getMoviesSelected',...
        %            'Cannot access Movie Manager. Make sure your desired movies are selected in the Movie Manager.');
      end
    end  % function

    function v = get.hasTrx(obj)
      v = ~isempty(obj.trx);
    end

    function v = get.currTrx(obj)
      if obj.hasTrx
        v = obj.trx(obj.currTarget);
      else
        v = [];
      end
    end

    function v = get.nTrx(obj)
      v = numel(obj.trx);
    end

    function v = getnTrxMovIdx(obj,mIdx)
      % Warning: Slow/expensive, call in bulk and index rather than calling
      % piecemeal 
      %
      % mIdx: [n] MovieIndex vector
      % v: [n] array; v(i) == num targets in ith mov
      
      assert(isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      assert(all(gt) || ~any(gt));
      gt = all(gt);
      
      [~,v] = obj.getNFrmNTrx(gt,iMov);      
    end

    function [nfrms,ntgts] = getNFrmNTrx(obj,gt,iMov)
      % Both gt, iMov optional
      %
      % nfrms, ntgts: [numel(iMov) x 1]
      
      if nargin<2
        gt = false;
      end
      if nargin<3
        nmov = obj.getnmoviesGTawareArg(gt);
        iMov = 1:nmov;
      end
        
      PROPS = obj.gtGetSharedPropsStc(gt);
      %tfaf = obj.(PROPS.TFAF);
      mia = obj.(PROPS.MIA);
      tia = obj.(PROPS.TIA);
      nfrms = zeros(size(iMov));
      ntgts = ones(size(iMov));
      
      for i=1:numel(ntgts)        
        %trxfile = tfaf{iMov(i)};
        nfrms(i) = mia{iMov(i)}.nframes;
        if ~isempty(tia{iMov(i),1}), % KB todo -- hacked so this didn't error
          ntgts(i) = tia{iMov(i),1}.ntgts;
        end
%         if isempty(trxfile)
%           % none; ntgts(i) is 1
%         else
%           trxI = obj.getTrx(trxfile,nfrms(i));
%           ntgts(i) = numel(trxI);
%         end
      end
    end

    function v = get.nTargets(obj)
      if obj.hasTrx
        v = obj.nTrx;
      else
        v = 1;
      end
    end

    function v = getNTargets(obj,gt,imov)
      PROPS = obj.gtGetSharedPropsStc(gt);
      if obj.hasTrx,
        v = obj.(PROPS.TIA){imov,1}.ntgts;
      elseif obj.maIsMA,
        labelscurr = obj.(PROPS.LBL){imov,1};
        if isstruct(labelscurr) && isfield(labelscurr,'tgt') && ~isempty(labelscurr.tgt),
          v = max(labelscurr.tgt);
        else
          v = 1;
        end
      else
        v = 1;
      end
    end

    function v = get.targetZoomRadiusDefault(obj)
      v = obj.projPrefs.Trx.ZoomFactorDefault;
    end

    function v = get.hasProject(obj)
      % AL 20160710: debateable utility/correctness, but if you try to do
      % some things (eg open MovieManager) right on bootup from an empty
      % Labeler you get weird errors.
      v = size(obj.movieFilesAll,2)>0; % AL 201806: want first dim instead?
    end

    function v = get.projectfile(obj)
      info = obj.projFSInfo;
      if ~isempty(info)
        v = info.filename;
      else
        v = [];
      end
    end

    function v = get.projectroot(obj)
      v = obj.projectfile;
      if ~isempty(v)
        v = fileparts(v);
      end
    end

    function v = get.nmovies(obj)
      % for multiview labeling, this is really 'nmoviesets'
      v = size(obj.movieFilesAll,1);
    end

    function v = get.nmoviesGT(obj)
      v = size(obj.movieFilesAllGT,1);
    end

    function v = get.nmoviesGTaware(obj)
      v = obj.getnmoviesGTawareArg(obj.gtIsGTMode);
    end

    function v = getnmoviesGTawareArg(obj,gt)
      if ~gt
        v = size(obj.movieFilesAll,1);
      else
        v = size(obj.movieFilesAllGT,1);
      end
    end

    function v = get.movieInfoAllGTaware(obj)
      if obj.gtIsGTMode
        v = obj.movieInfoAllGT;
      else
        v = obj.movieInfoAll;
      end
    end

    function v = getMovieInfoAllGTawareArg(obj,gt)
      if gt
        v = obj.movieInfoAllGT;
      else
        v = obj.movieInfoAll;
      end
    end

    function v = get.labeledpos(obj)
      v = Labels.lObjGetLabeledPos(obj,'labels',false);
    end

    function v = get.labeledposTS(obj)
      [~,v] = Labels.lObjGetLabeledPos(obj,'labels',false);
    end    

    function v = get.labeledpostag(obj)
      [~,~,v] = Labels.lObjGetLabeledPos(obj,'labels',false);
    end    

    function v = get.labeledposGT(obj)
      v = Labels.lObjGetLabeledPos(obj,'labelsGT',true);
    end

    function v = get.labeledposTSGT(obj)
      [~,v] = Labels.lObjGetLabeledPos(obj,'labelsGT',true);
    end

    function v = get.labeledpostagGT(obj)
      [~,~,v] = Labels.lObjGetLabeledPos(obj,'labelsGT',true);
    end

    function v = get.labeledpos2GT(obj)
      v = Labels.lObjGetLabeledPos(obj,'labels2GT',true);
    end    

    function v = get.labeledpos2(obj)
      v = Labels.lObjGetLabeledPos(obj,'labels2',false);
    end

    function v = get.labeledposGTaware(obj)
      v = obj.getlabeledposGTawareArg(obj.gtIsGTMode);
    end

    function v = getlabeledposGTawareArg(obj,gt)      
      if gt
        v = obj.labeledposGT;
      else
        v = obj.labeledpos;
      end      
    end

    function v = get.labelsGTaware(obj)
      v = obj.getlabelsGTawareArg(obj.gtIsGTMode);
    end

    function v = getlabelsGTawareArg(obj,gt)
      if gt
        v = obj.labelsGT;
      else
        v = obj.labels;
      end
    end    

    function v = getLabeledPosMovIdx(obj,mIdx)
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.labeledposGT{iMov};
      else 
        v = obj.labeledpos{iMov};
      end
    end

    function v = getLabelsMovIdx(obj,mIdx)
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.labelsGT{iMov};
      else 
        v = obj.labels{iMov};
      end
    end

    function v = getLabeledPos2MovIdx(obj,mIdx)
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.labeledpos2GT{iMov};
      else 
        v = obj.labeledpos2{iMov};
      end
    end

    function v = getLabels2MovIdx(obj,mIdx)
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.labels2GT{iMov};
      else 
        v = obj.labels2{iMov};
      end
      v = Labels.fromTrkfile(v);
    end

    function v = get.labeledposTSGTaware(obj)
      if obj.gtIsGTMode
        v = obj.labeledposTSGT;
      else
        v = obj.labeledposTS;
      end
    end

    function v = get.labeledpostagGTaware(obj)
      if obj.gtIsGTMode
        v = obj.labeledpostagGT;
      else
        v = obj.labeledpostag;
      end
    end

    function v = get.labeledpos2GTaware(obj)
      if obj.gtIsGTMode
        v = obj.labeledpos2GT;
      else
        v = obj.labeledpos2;
      end
    end

    function v = get.labels2GTaware(obj)
      if obj.gtIsGTMode
        v = obj.labels2GT;
      else
        v = obj.labels2;
      end
    end    

    function v = get.labeledposCurrMovie(obj)
      if obj.currMovie==0
        v = [];
      elseif obj.gtIsGTMode
        v = obj.labeledposGT{obj.currMovie};
      else
        v = obj.labeledpos{obj.currMovie};
      end
    end

    function v = get.labelsCurrMovie(obj)
      if obj.currMovie==0
        v = [];
      elseif obj.gtIsGTMode
        v = obj.labelsGT{obj.currMovie};
      else
        v = obj.labels{obj.currMovie};
      end
    end

    function v = get.labeledpos2CurrMovie(obj)
      if obj.currMovie==0
        v = [];
      elseif obj.gtIsGTMode
        v = obj.labeledpos2GT{obj.currMovie};
      else
        v = obj.labeledpos2{obj.currMovie};
      end
    end

    function v = get.labels2CurrMovie(obj)
      if obj.currMovie==0
        v = [];
      elseif obj.gtIsGTMode
        v = obj.labels2GT{obj.currMovie};
      else
        v = obj.labels2{obj.currMovie};
      end
    end

    function v = get.labeledpostagCurrMovie(obj)
      if obj.currMovie==0
        v = [];
      elseif obj.gtIsGTMode
        v = obj.labeledpostagGT{obj.currMovie};
      else
        v = obj.labeledpostag{obj.currMovie};
      end
    end

    function v = getIsLabeled(obj,tbl)
      v = Labels.lObjGetIsLabeled(obj,'labels',tbl,false);
    end

    function v = getIsLabeledGT(obj,tbl,varargin)
      v = Labels.lObjGetIsLabeled(obj,'labelsGT',tbl,true,varargin{:});
    end

    
    
    function v = get.nPhysPoints(obj)
      v = size(obj.labeledposIPtSetMap,1);
    end

    function v = get.currMovIdx(obj)
      v = MovieIndex(obj.currMovie,obj.gtIsGTMode);
    end

    function v = get.gdata(obj)
      v = obj.controller_ ;
    end

    function v = get.hFig(obj)
      v = obj.controller_.mainFigure_ ;
    end

    function v = get.gtNumSugg(obj)
      v = height(obj.gtSuggMFTable);
    end

    function v = get.tracker(obj)
      if isempty(obj.trackerHistory_),
        v = [];
      else
        v = obj.trackerHistory_{1};
      end
    end

    function v = get.trackerAlgo(obj)
      v = obj.tracker;
      if isempty(v)
        v = '';
      else
        v = v.algorithmName;
      end
    end

    function v = get.trackerNetsUsed(obj)
      v = obj.tracker;
      if isempty(v)
        v = cell(0,1);
      else
        v = v.getNetsUsed();
      end
    end    

    function v = get.trackerIsDL(obj)
      v = obj.tracker;
      if isempty(v)
        v = [];
      else
        v = isa(v,'DeepTracker');
      end
    end 

    function v = get.trackerIsTwoStage(obj)
      % here we actually mean MA-TD
      v = obj.tracker;
      v = ~isempty(v) && isa(v,'DeepTracker') && v.trnNetMode.isTwoStage;
    end

    function v = get.trackerIsObjDet(obj)
      v = obj.tracker;
      v = ~isempty(v) && isa(v,'DeepTracker') && v.trnNetMode.isObjDet;
    end

    function v = get.trackerIsBotUp(obj)
      v = obj.tracker;
      v = ~isempty(v) && isa(v,'DeepTracker') && ...
        v.trnNetMode.isMA && ~v.trnNetMode.isTopDown;
    end
    % KB 20190214 - store all parameters together in one struct. these dependent functions emulate previous behavior
    function v = get.preProcParams(obj)      
      if isempty(obj.trackParams),
        v = [];
      else
        v = APTParameters.all2PreProcParams(obj.trackParams);
      end
    end
    
    function v = get.trackDLParams(obj)      
      if isempty(obj.trackParams),
        v = [];
      else
        v = APTParameters.all2TrackDLParams(obj.trackParams);
      end
    end
    
    function v = get.DLCacheDir(obj)      
      v = obj.projTempDir;
    end
    
    function v = get.trkResGTaware(obj)
      gt = obj.gtIsGTMode;
      if gt
        v = obj.trkResGT;
      else
        v = obj.trkRes;
      end
    end
  end
  
  methods % prop access
    function set.labels(obj,v)
      obj.labels = v;
      if ~obj.isinit %#ok<MCSUP>
        obj.notify('updateTrxTable');
        obj.syncPropsMfahl_() ;
        obj.notify('updateFrameTableIncremental');
      end
      %obj.notify('didSetLabels') ;
      obj.notify('updateTimelineTraces') ;
    end

    function set.labelsGT(obj,v)
      obj.labelsGT = v;
      if ~obj.isinit %#ok<MCSUP>
        obj.notify('updateTrxTable');
        obj.syncPropsMfahl_() ;
        obj.notify('updateFrameTableIncremental');
        obj.gtUpdateSuggMFTableLbledIncremental();
      end
    end

    function set.labelsRoi(obj,v)
      obj.labelsRoi = v;
      if ~obj.isinit %#ok<MCSUP>
        obj.syncPropsMfahl_() ;
        obj.notify('updateFrameTableIncremental');
      end
    end

    function set.labelsRoiGT(obj,v)
      obj.labelsRoiGT = v;
      if ~obj.isinit %#ok<MCSUP>
        obj.syncPropsMfahl_() ;
        obj.notify('updateFrameTableIncremental');
      end
    end

    function set.movieForceGrayscale(obj,v)
      if isscalar(v) && islogical(v) ,
        [obj.movieReader.forceGrayscale] = deal(v); %#ok<MCSUP>
        obj.movieForceGrayscale = v;
        obj.notify('didSetMovieForceGrayscale') ;
      else
        obj.notify('didSetMovieForceGrayscale') ;
        error('APT:invalidValue', 'Invalid value for movieForceGrayscale') ;
      end
    end

    function set.movieInvert(obj,v)
      if islogical(v) && numel(v)==obj.nview , %#ok<MCSUP>
        movieInvert0 = obj.movieInvert;
        if ~isequal(movieInvert0,v)
          mrs = obj.movieReader; %#ok<MCSUP>
          for i=1:obj.nview %#ok<MCSUP>
            mrs(i).flipVert = v(i);
          end
          if ~obj.isinit %#ok<MCSUP>
            obj.preProcNonstandardParamChanged();
            if obj.cropProjHasCrops %#ok<MCSUP>
              warningNoTrace('Project cropping will NOT be inverted/updated. Please check your crops.')
            end
          end
          obj.movieInvert = v;
        end
        obj.notify('didSetMovieInvert') ;
      else        
        obj.notify('didSetMovieInvert') ;
        error('APT:invalidValue', 'Invalid value for movieInvert') ;
      end
    end

    function set.movieCenterOnTarget(obj,v)
      obj.movieCenterOnTarget = v;
      if ~v && obj.movieRotateTargetUp %#ok<MCSUP>
        obj.movieRotateTargetUp = false; %#ok<MCSUP>
      end
      obj.notify('updateTargetCentrationAndZoom') ;
      obj.notify('didSetMovieCenterOnTarget') ;
    end

    function set.movieRotateTargetUp(obj,v)
      if obj.maIsMA && obj.currTarget==0 ,
        warningNoTrace('Labeler:MA', 'No labeled target selected. Not rotating');
      else
        if v && ~obj.movieCenterOnTarget %#ok<MCSUP>
          %warningNoTrace('Labeler:prop','Setting .movieCenterOnTarget to true.');
          obj.movieCenterOnTarget = true; %#ok<MCSUP>
        end
        obj.movieRotateTargetUp = v;
        obj.notify('updateTargetCentrationAndZoom') ;
      end
      obj.notify('didSetMovieRotateTargetUp') ;
    end

    function set.targetZoomRadiusDefault(obj,v)
      obj.projPrefs.Trx.ZoomFactorDefault = v;
    end
    
    function tfIsReady = isReady(obj)
      tfIsReady = ~obj.isinit && obj.hasMovie && obj.hasProject;
    end
    
    function setMovieShiftArrowNavModeThresh(obj,v)
      assert(isscalar(v) && isnumeric(v));
      obj.movieShiftArrowNavModeThresh = v;
      obj.infoTimelineModel.statThresh = v;
      obj.notify('updateTimelineStatThresh');
    end
    
    function toggleTimelineIsStatThreshVisible(obj)
      obj.infoTimelineModel.toggleIsStatThreshVisible();
      obj.notify('updateTimelineStatThresh');
    end
  end
  
  %% Configurations
  methods (Hidden)

  % Property init legend
  % IFC: property initted during initFromConfig_()  
  %      (but initFromConfig_ is called from both projNew() and projLoadGUI()...
  %       -- ALT, 2025-02-05)
  % PNPL: property initted during projNew() or projLoadGUI()
  % L: property initted during labelingInit()
  % (todo) TI: property initted during trackingInit()
  %
  % There are only two ways to start working on a project.
  % 1. New/blank project: projNew().
  % 2. Existing project: projLoadGUI(), which is (initFromConfig(), then
  %    property-initialization-equivalent-to-projNew().)
  
    function initFromConfig_(obj, cfg)
      % Initialize obj from cfg, a configuration struct.  This is used e.g. when loading
      % a project from a .lbl file, or when creating a new project.
      % Note: Configuration struct must be modernized
    
      isinit0 = obj.isinit ;
      obj.isinit = true ;
            
      % Views
      obj.nview = cfg.NumViews;
      if isempty(cfg.ViewNames)
        obj.viewNames = arrayfun(@(x)sprintf('view%d',x),1:obj.nview,'uni',0);
      else
        if numel(cfg.ViewNames) ~= obj.nview
          error('Labeler:prefs',...
            'ViewNames: must specify %d names (one for each view)',obj.nview);
        end
        obj.viewNames = cfg.ViewNames;
      end
     
      npts3d = cfg.NumLabelPoints;
      npts = obj.nview*npts3d;      
      obj.nLabelPoints = npts;
      if isempty(cfg.LabelPointNames)
        cfg.LabelPointNames = arrayfun(@(x)sprintf('pt%d',x),(1:cfg.NumLabelPoints)','uni',0);
      end
      assert(numel(cfg.LabelPointNames)==cfg.NumLabelPoints);
         
      % pts, sets, views
      setnames = cfg.LabelPointNames;
      nSet = size(setnames,1);
      ipt2view = nan(npts,1);
      ipt2set = nan(npts,1);
      setmap = nan(nSet,obj.nview);
      for iSet = 1:nSet
        set = setnames{iSet};
        iPts = iSet:npts3d:npts;
        if numel(iPts)~=obj.nview
          error('Labeler:prefs',...
            'Number of point indices specified for ''%s'' does not equal number of views (%d).',set,obj.nview);
        end
        setmap(iSet,:) = iPts;
        
        iViewNZ = find(iPts>0);
        ipt2view(iPts(iViewNZ)) = iViewNZ;
        ipt2set(iPts(iViewNZ)) = iSet;
      end
      iptNotInAnyView = find(isnan(ipt2view));
      if ~isempty(iptNotInAnyView)
        iptNotInAnyView = arrayfun(@num2str,iptNotInAnyView,'uni',0);
        error('Labeler:prefs',...
          'The following points are not located in any view or set: %s',...
           String.cellstr2CommaSepList(iptNotInAnyView));
      end
      obj.labeledposIPt2View = ipt2view;
      obj.labeledposIPt2Set = ipt2set;
      obj.labeledposIPtSetMap = setmap;
      obj.skelNames = setnames;
      
      didsetlabelmode = false;
      if isfield(cfg,'LabelMode'),
        iscompatible = (cfg.NumViews==1 && cfg.LabelMode ~= LabelMode.MULTIVIEWCALIBRATED2) || ...
          (cfg.NumViews==2 && cfg.LabelMode == LabelMode.MULTIVIEWCALIBRATED2);
        if iscompatible,
          obj.labelMode = cfg.LabelMode;
          didsetlabelmode = true;
        end
      end
      if ~didsetlabelmode,
        if cfg.NumViews==1
          obj.labelMode = LabelMode.TEMPLATE;
        else
          obj.labelMode = LabelMode.MULTIVIEWCALIBRATED2;
        end
      end
      
      lpp = cfg.LabelPointsPlot;
      % Some convenience mods to .LabelPointsPlot
      % KB 20181022: Colors will now be nSet x 3
      if ~isfield(lpp,'Colors') || size(lpp.Colors,1)~=nSet
        lpp.Colors = feval(lpp.ColorMapName,nSet);
      end
      % .LabelPointsPlot invariants:
      % - lpp.ColorMapName, lpp.Colors both exist
      % - lpp.Colors is [nSet x 3]
      obj.labelPointsPlotInfo = lpp;
      % AL 20190603: updated .PredictPointsPlot reorg
      if ~isfield(cfg.Track.PredictPointsPlot,'Colors') || ...
          size(cfg.Track.PredictPointsPlot.Colors,1)~=nSet
        cfg.Track.PredictPointsPlot.Colors = ...
          feval(cfg.Track.PredictPointsPlot.ColorMapName,nSet);
      end
      if ~isfield(cfg.Track.ImportPointsPlot,'Colors') || ...
          size(cfg.Track.ImportPointsPlot.Colors,1)~=nSet
        cfg.Track.ImportPointsPlot.Colors = ...
          feval(cfg.Track.ImportPointsPlot.ColorMapName,nSet);
      end
      % .PredictPointsPlot color nvariants:
      % - ppp.ColorMapName, ppp.Colors both exist
      % - ppp.Colors is [nSet x 3]
      obj.predPointsPlotInfo = cfg.Track.PredictPointsPlot;
      obj.impPointsPlotInfo = cfg.Track.ImportPointsPlot;
            
      obj.trackNFramesSmall = cfg.Track.PredictFrameStep;
      obj.trackNFramesLarge = cfg.Track.PredictFrameStepBig;
      obj.trackNFramesNear = cfg.Track.PredictNeighborhood;
      obj.trackModeIdx = 1;
      cfg.Track = rmfield(cfg.Track,...
        {'PredictPointsPlot' 'PredictFrameStep' 'PredictFrameStepBig' ...
        'PredictNeighborhood'});
                  
      arrayfun(@delete,obj.movieReader);
      obj.movieReader = [];
      for i=obj.nview:-1:1
        mr(1,i) = MovieReader;
      end
      obj.movieReader = mr;
      obj.currIm = cell(obj.nview,1);
      obj.currImRoi = cell(obj.nview,1);
      delete(obj.currImHud);
      controller = obj.controller_;
      obj.currImHud = AxisHUD(controller.axes_curr.Parent,controller.axes_curr); 
      %obj.movieSetNoMovie();
      
      obj.movieForceGrayscale = logical(cfg.Movie.ForceGrayScale);
      obj.movieFrameStepBig = cfg.Movie.FrameStepBig;
      obj.movieShiftArrowNavMode = ShiftArrowMovieNavMode.(cfg.Movie.ShiftArrowNavMode);
      obj.movieShiftArrowNavModeThresh = cfg.Movie.ShiftArrowNavModeThresh;
      obj.movieShiftArrowNavModeThreshCmp = cfg.Movie.ShiftArrowNavModeThreshCmp;
      obj.moviePlaySegRadius = cfg.Movie.PlaySegmentRadius;
      obj.moviePlayFPS = cfg.Movie.PlayFPS;
           
      fldsRm = intersect(fieldnames(cfg),...
        {'NumViews' 'ViewNames' 'NumLabelPoints' 'LabelPointNames' ...
        'LabelMode' 'LabelPointsPlot' 'ProjectName' 'Movie'});
      obj.projPrefs = rmfield(cfg,fldsRm);
      % A few minor subprops of projPrefs have explicit props

      obj.maIsMA = cfg.MultiAnimal && ~cfg.Trx.HasTrx;
      obj.projectHasTrx = cfg.Trx.HasTrx;

      obj.notify('newProject');

      % order important: this needs to occur after 'newProject' event so
      % that figs are set up. (names get changed)
      movInvert = ViewConfig.getMovieInvert(cfg.View);
      obj.movieInvert = movInvert;
      obj.movieCenterOnTarget = cfg.View(1).CenterOnTarget;
      obj.movieRotateTargetUp = cfg.View(1).RotateTargetUp;
      
      obj.preProcInit();
      
      % % Reset .trackersAll
      % cellfun(@delete, obj.trackersAll_) ;
      % obj.trackersAll_ = cell(1,0);

      % Also clear tracker history
      cellfun(@delete, obj.trackerHistory_) ;
      obj.trackerHistory_ = cell(1,0);

      obj.trackDLBackEnd = DLBackEndClass();
      obj.trackDLBackEnd.isInAwsDebugMode = obj.isInAwsDebugMode ;
      obj.trackParams = [];
      
      obj.showOccludedBox = cfg.View.OccludedBox;
      
      obj.showTrx = cfg.Trx.ShowTrx;
      obj.showTrxCurrTargetOnly = cfg.Trx.ShowTrxCurrentTargetOnly;
      obj.showTrxIDLbl = cfg.Trx.ShowTrxIDLbl;
            
      obj.labels2Hide = false;
      obj.labels2ShowCurrTargetOnly = false;

      obj.skeletonEdges = zeros(0,2);
      obj.skelHead = [];
      obj.skelTail = [];
      obj.showSkeleton = false;
      obj.showMaRoi = obj.labelMode == LabelMode.MULTIANIMAL;
      obj.showMaRoiAux = obj.labelMode == LabelMode.MULTIANIMAL;
      obj.flipLandmarkMatches = zeros(0,2);
      
      % New projs set to LASTSEEN, since in general no reference target can have
      % been set yet.
      obj.setPrevAxesMode(PrevAxesMode.LASTSEEN);
      
      % maybe useful to clear/reinit and shouldn't hurt
      obj.trxCache = containers.Map();
      
      if obj.isgui,
        obj.rcSaveProp('lastProjectConfig',obj.getCurrentConfig());
      end
      
      obj.isinit = isinit0;
    end  % function
    
    function cfg = getCurrentConfig(obj)
      % cfg is modernized

      cfg = obj.projPrefs;
      
      cfg.NumViews = obj.nview;
      cfg.ViewNames = obj.viewNames;
      cfg.NumLabelPoints = obj.nPhysPoints;
      cfg.LabelPointNames = obj.skelNames;
      cfg.LabelMode = char(obj.labelMode);
      % View stuff: read off current state of axes
      controller = obj.controller_;
      viewCfg = ViewConfig.readCfgOffViews(controller.figs_all,controller.axes_all);
      for i=1:obj.nview
        viewCfg(i).InvertMovie = obj.movieInvert(i);
        viewCfg(i).CenterOnTarget = obj.movieCenterOnTarget;
        viewCfg(i).RotateTargetUp = obj.movieRotateTargetUp;
      end
      cfg.View = viewCfg;

      % misc config props that have an explicit labeler prop

      cfg.Movie = struct(...
        'ForceGrayScale',obj.movieForceGrayscale,...
        'FrameStepBig',obj.movieFrameStepBig,...
        'ShiftArrowNavMode',char(obj.movieShiftArrowNavMode),...
        'ShiftArrowNavModeThresh',obj.movieShiftArrowNavModeThresh,...
        'ShiftArrowNavModeThreshCmp',obj.movieShiftArrowNavModeThreshCmp,...
        'PlaySegmentRadius',obj.moviePlaySegRadius,...
        'PlayFPS',obj.moviePlayFPS);

      cfg.LabelPointsPlot = obj.labelPointsPlotInfo;      
      cfg.Trx.ShowTrx = obj.showTrx;
      cfg.Trx.ShowTrxCurrentTargetOnly = obj.showTrxCurrTargetOnly;
      cfg.Trx.ShowTrxIDLbl = obj.showTrxIDLbl;
      cfg.Trx.HasTrx = obj.projectHasTrx;
      
      cfg.Track.PredictFrameStep = obj.trackNFramesSmall;
      cfg.Track.PredictFrameStepBig = obj.trackNFramesLarge;
      cfg.Track.PredictNeighborhood = obj.trackNFramesNear;
      cfg.Track.PredictPointsPlot = obj.predPointsPlotInfo;
      cfg.Track.ImportPointsPlot = obj.impPointsPlotInfo;
      
      cfg.PrevAxes.Mode = char(obj.prevAxesMode);
      cfg.PrevAxes.ModeInfo = obj.prevAxesModeTargetSpec.toStructForPersistence();
    end

    function shortcuts = getShortcuts(obj)
      prefs = obj.projPrefs;
      if isfield(prefs,'Shortcuts'),
        shortcuts = prefs.Shortcuts;
      else
        shortcuts = struct;
      end
    end

    function setShortcuts(obj,scs)
      obj.projPrefs.Shortcuts = scs;
      notify(obj, 'updateShortcuts');
    end

  end
    
  % Consider moving this stuff to Config.m
  methods (Static)
    
    function cfg = cfgGetLastProjectConfigNoView
      cfgBase = yaml.ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      cfg = RC.getprop('lastProjectConfig');
      if isempty(cfg)
        cfg = cfgBase;
      else
        cfg = structoverlay(cfgBase,cfg,'dontWarnUnrecog',true);
        
        % use "fresh"/empty View stuff on the theory that this 
        % often/usually doesn't generalize across projects
        cfg.View = repmat(cfgBase.View,cfg.NumViews,1); 
      end
    end
    
    function cfg = cfgModernizeBase(cfg)
      % 20190602 cfg.LabelPointsPlot, cfg.Track.PredictPointsPlot rearrange
      
      if ~isfield(cfg.LabelPointsPlot,'MarkerProps')
        FLDS = {'Marker' 'MarkerSize' 'LineWidth'};
        cfg.LabelPointsPlot.MarkerProps = structrestrictflds(cfg.LabelPointsPlot,FLDS);
        cfg.LabelPointsPlot = rmfield(cfg.LabelPointsPlot,FLDS);
      end
      if ~isfield(cfg.LabelPointsPlot,'TextProps')
        FLDS = {'FontSize'};
        cfg.LabelPointsPlot.TextProps = structrestrictflds(cfg.LabelPointsPlot,FLDS);
        cfg.LabelPointsPlot = rmfield(cfg.LabelPointsPlot,FLDS);
        
        cfg.LabelPointsPlot.TextOffset = cfg.LabelPointsPlot.LblOffset;
        cfg.LabelPointsPlot = rmfield(cfg.LabelPointsPlot,'LblOffset');
      end
      
      if ~isfield(cfg.Track.PredictPointsPlot,'MarkerProps')
        cfg.Track.PredictPointsPlot = struct('MarkerProps',cfg.Track.PredictPointsPlot);
      end        
      if isfield(cfg.Track,'PredictPointsPlotColorMapName')
        cfg.Track.PredictPointsPlot.ColorMapName = cfg.Track.PredictPointsPlotColorMapName;
        cfg.Track = rmfield(cfg.Track,'PredictPointsPlotColorMapName');
      end
      if isfield(cfg.Track,'PredictPointsPlotColors')
        cfg.Track.PredictPointsPlot.Colors = cfg.Track.PredictPointsPlotColors;
        cfg.Track = rmfield(cfg.Track,'PredictPointsPlotColors');
      end
      if isfield(cfg.Track,'PredictPointsShowTextLbl')
        cfg.Track.PredictPointsPlot.TextProps.Visible = ...
          onIff(cfg.Track.PredictPointsShowTextLbl);
        cfg.Track = rmfield(cfg.Track,'PredictPointsShowTextLbl');
      end
    end    

    function cfg = cfgModernize(cfg)
      % Bring a cfg up-to-date with latest by adding in any new fields from
      % config.default.yaml.

      cfg = Labeler.cfgModernizeBase(cfg);
      
      cfgBase = yaml.ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      
      cfg = structoverlay(cfgBase,cfg,'dontWarnUnrecog',true,...
        'allowedUnrecogFlds',{'Colors'});% 'ColorsSets'});
      view = augmentOrTruncateVector(cfg.View,cfg.NumViews);
      cfg.View = view(:);
    end
    
    function cfg = cfgDefaultOrder(cfg)
      % Reorder fields of cfg struct to default order
      
      cfg0 = yaml.ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      flds0 = fieldnames(cfg0);
      flds = fieldnames(cfg);
      flds0 = flds0(ismember(flds0,flds)); 
      fldsExtra = setdiff(flds,flds0);
      cfg = orderfields(cfg,[flds0(:);fldsExtra(:)]);
    end
    
    function cfg = cfgRmLabelPoints(cfg,iptsrm)
      % Update/Massage config, removing given landmarks/pts 
      %
      % iptsrm: vector of ipt indices

      nptsrm = numel(iptsrm);
      nptsremain = cfg.NumLabelPoints-nptsrm;
      if nptsremain<=0
        error('Cannot remove %d points as config.NumLabelPoints is %d.',...
          nptsrm,cfg.NumLabelPoints);
      end
      
      cfg.LabelPointNames(iptsrm,:) = [];
      cfg.NumLabelPoints = nptsremain;
    end
    
    function cfg = cfgAddLabelPoints(cfg,nptsadd)
      % Update/Massage config, adding nptsadd new landmarks/pts 
      %
      % nptsadd: scalar positive int
      
      npts0 = cfg.NumLabelPoints;
      newptnames = arrayfun(@(x)sprintf('pt%d',x),...
        (npts0+1:npts0+nptsadd)','uni',0);

      cfg.LabelPointNames = cat(1,cfg.LabelPointNames,newptnames);
      cfg.NumLabelPoints = cfg.NumLabelPoints + nptsadd;
    end      
    
    % moved this from ProjectSetup
    function sMirror = cfg2mirror(cfg)
      % Convert true/full data struct to 'mirror' struct for adv table. (The term
      % 'mirror' comes from implementation detail of adv table/propertiesGUI.)
      
      nViews = cfg.NumViews;
      nPoints = cfg.NumLabelPoints;
      
      cfg = Labeler.cfgDefaultOrder(cfg);
      sMirror = rmfield(cfg,{'NumViews' 'NumLabelPoints'}); % 'LabelMode'});
      sMirror.Track = rmfield(sMirror.Track,{'Enable' 'Type'});
      
      assert(isempty(sMirror.ViewNames) || numel(sMirror.ViewNames)==nViews);
      flds = arrayfun(@(i)sprintf('view%d',i),(1:nViews)','uni',0);
      if isempty(sMirror.ViewNames)
        vals = repmat({''},nViews,1);
      else
        vals = sMirror.ViewNames(:);
      end
      sMirror.ViewNames = cell2struct(vals,flds,1);
      
      assert(isempty(sMirror.LabelPointNames) || numel(sMirror.LabelPointNames)==nPoints);
      flds = arrayfun(@(i)sprintf('point%d',i),(1:nPoints)','uni',0);
      if isempty(sMirror.LabelPointNames)
        vals = repmat({''},nPoints,1);
      else
        vals = sMirror.LabelPointNames;
      end
      sMirror.LabelPointNames = cell2struct(vals,flds,1);
    end
    
    % moved this from ProjectSetup
    function sMirror = hlpAugmentOrTruncNameField(sMirror,fld,subfld,n)
      v = sMirror.(fld);
      flds = fieldnames(v);
      nflds = numel(flds);
      if nflds>n
        v = rmfield(v,flds(n+1:end));
      elseif nflds<n
        for i=nflds+1:n
          v.([subfld num2str(i)]) = '';
        end
      end
      sMirror.(fld) = v;
    end
    
    function sMirror = hlpAugmentOrTruncStructField(sMirror,fld,n)

      v = sMirror.(fld);
      v = augmentOrTruncateVector(v,n);
      sMirror.(fld) = v(:);
      
    end
        
  end

  properties
    rc = [];
  end

  methods % interacting with RC file
    
    function rcSaveProp(obj,name,v)
      obj.rc.set(name,v);
    end

    function v = rcGetProp(obj,name)
      v = obj.rc.get(name);
    end

  end
  
  %% Project/Lbl files
  methods
   
    function projNew(obj, cfg)
      % Create new project based on configuration in cfg.
      obj.pushBusyStatus('Configuring new project...') ;
      oc = onCleanup(@()(obj.popBusyStatus())) ;

      obj.initFromConfig_(cfg) ;

      obj.isinit = true;

      obj.projname = cfg.ProjectName ;
      obj.projFSInfo = [];
      obj.projGetEnsureTempDir('cleartmp',true);
      obj.movieFilesAll = cell(0,obj.nview);
      obj.movieFilesAllGT = cell(0,obj.nview);
      obj.movieFilesAllHaveLbls = zeros(0,1);
      obj.movieFilesAllGTHaveLbls = zeros(0,1);
      obj.movieInfoAll = cell(0,obj.nview);
      obj.movieInfoAllGT = cell(0,obj.nview);
      obj.movieFilesAllCropInfo = cell(0,1);
      obj.movieFilesAllGTCropInfo = cell(0,1);
      obj.movieFilesAllHistEqLUT = cell(0,obj.nview);
      obj.movieFilesAllGTHistEqLUT = cell(0,obj.nview);
      obj.cropIsCropMode = false;
      obj.trxFilesAll = cell(0,obj.nview);
      obj.trxFilesAllGT = cell(0,obj.nview);
      obj.trxInfoAll = cell(0,obj.nview);
      obj.trxInfoAllGT = cell(0,obj.nview);
      obj.projMacros = struct();
      obj.viewCalProjWide = [];
      obj.viewCalibrationData = [];
      obj.viewCalibrationDataGT = [];
      obj.labelTemplate = []; % order important here
      obj.gtIsGTMode = false;
      obj.movieSetNoMovie(); % order important here      
      obj.labels = cell(0,1);
      obj.labels2 = cell(0,1);
      obj.labelsGT = cell(0,1);
      obj.labels2GT = cell(0,1);      
      obj.labelsRoi = cell(0,1);
      obj.labelsRoiGT = cell(0,1);
      obj.lastLabelChangeTS = 0;
      obj.gtIsGTMode = false;
      obj.gtSuggMFTable = MFTable.emptyTable(MFTable.FLDSID);
      obj.gtSuggMFTableLbled = false(0,1);
      obj.gtTblRes = [];
      
      obj.trackResInit();
      
      obj.isinit = false;
      
      obj.syncPropsMfahl_() ;
      obj.notify('updateFrameTableComplete');
      obj.labeledposNeedsSave = false;
      obj.doesNeedSave_ = false;

      trkPrefs = obj.projPrefs.Track ;
      if trkPrefs.Enable
        % % Create default trackers (now only used as templates)
        % assert(isempty(obj.trackersAll_));
        % obj.initializeTrackersAllAndFriends_() ;

        % Also create a working tracker
        tracker = LabelTracker.create(obj) ;
        obj.trackerHistory_ = { tracker } ;
        
        tPrm = APTParameters.defaultParamsTree() ;
        sPrm = tPrm.structize();
        obj.trackParams = sPrm;
      end

      % Note that the project now needs saving
      obj.setDoesNeedSave(true, 'New project') ;      
      
      % Do a full update of the GUI
      obj.notify('update') ;
    end
    
    function projSave(obj, fname)      
      % This is the proper model save operation for clients.  (That does not require
      % a GUI.)
      obj.pushBusyStatus('Saving project...');
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      try
        obj.saveVersionInfo = GetGitMatlabStatus(APT.Root);
      catch
        obj.saveVersionInfo = [];
      end      
      s = obj.projGetSaveStruct();      
      try
        rawLblFileName = obj.projGetRawLblFile();
        save(rawLblFileName,'-mat','-struct','s');
        obj.projBundleSave_(fname);
      catch ME
        save(fname,'-mat','-struct','s');
        msg = ME.getReport();
        warningNoTrace('Saved raw project file %s. Error caught during bundled project save:\n%s\n',...
                       fname,msg);
      end
      obj.labeledposNeedsSave = false;
      obj.doesNeedSave_ = false;
      obj.projFSInfo = ProjectFSInfo('saved',fname);
      obj.rcSaveProp('lastLblFile',fname);      
      % Assign the projname from the proj file name if appropriate
      if isempty(obj.projname) && ~isempty(obj.projectfile)
        [~,fname] = fileparts(obj.projectfile);
        obj.projname = fname;
      end
    end  % function
    
%     function projSaveModified(obj,fname,varargin)
%       try
%         [~,obj.saveVersionInfo] = GetGitMatlabStatus(APT.Root);
%       catch
%         obj.saveVersionInfo = [];
%       end
% 
%       s = obj.projGetSaveStructWithMassage(varargin{:});
%       save(fname,'-mat','-struct','s');
%       fprintf('Saved modified project file %s.\n',fname);
%     end
        
    function s = projGetSaveStruct(obj,varargin)
      [~,forceIncDataCache,forceExcDataCache,macroreplace,...
        savepropsonly,massageCropProps] = ...
        myparse(varargin,...
        'sparsify',false,...
        'forceIncDataCache',false,... % include .ppdb even if normally excluded
        'forceExcDataCache',false, ... 
        'macroreplace',false, ... % if true, use mfaFull/tfaFull for mfa/tfa
        'savepropsonly',false, ... % if true, just get the .SAVEPROPS with no further massage
        'massageCropProps',false ... % if true, structize crop props
        );
      assert(~(forceExcDataCache&&forceIncDataCache));      

      s = struct();
      s.cfg = obj.getCurrentConfig();
      
      for f_wrapped = obj.SAVEPROPS , 
        f=f_wrapped{1};
        if strcmp(f, 'trackDLBackEnd') ,
          % Special handling for this field
          % We want to use an "encoding container" for it, b/c saving custom objects is
          % fraught.
          backend = obj.(f) ;  % reference copy only
          if isempty(backend) ,
            s.(f) = backend ;
          else
            container = encode_for_persistence(backend) ;
            s.(f) = container ;
          end
        elseif ismember(f,{'viewCalibrationData','viewCalProjWide','viewClaibrationDataGT'}),
          if iscell(obj.(f)),
            s.(f) = cell(size(obj.(f)));
            for i = 1:numel(obj.(f)),
              if isa(obj.(f){i},'CalRig'),
                s.(f){i} = obj.(f){i}.getSaveStruct();
              else
                s.(f){i} = obj.(f){i};
              end
            end
          else
            if isa(obj.(f),'CalRig'),
              s.(f) = obj.(f).getSaveStruct();
            else
              s.(f) = obj.(f);
            end
          end
        else
          % Used for most fields
          s.(f) = obj.(f);
        end
      end
      
      if macroreplace
        s.movieFilesAll = obj.movieFilesAllFull;
        s.movieFilesAllGT = obj.movieFilesAllGTFull;
        s.trxFilesAll = obj.trxFilesAllFull;
        s.trxFilesAllGT = obj.trxFilesAllGTFull;
        s.trxInfoAll = obj.trxInfoAll;
        s.trxInfoAllGT = obj.trxInfoAllGT;
      end
      if massageCropProps
        cellOfObjArrs2CellOfStructArrs = ...
          @(x)cellfun(@(y)arrayfun(@struct,y),x,'uni',0); % note, y can be []
        warnst = warning('off','MATLAB:structOnObject');
        s.movieFilesAllCropInfo = cellOfObjArrs2CellOfStructArrs(obj.movieFilesAllCropInfo);
        s.movieFilesAllGTCropInfo = cellOfObjArrs2CellOfStructArrs(obj.movieFilesAllGTCropInfo);
        warning(warnst);
        s.cropProjHasCrops = obj.cropProjHasCrops;
      end
      if savepropsonly
        return
      end
      
      switch obj.labelMode
        case LabelMode.TEMPLATE
          %s.labelTemplate = obj.lblCore.getTemplate();
      end

      trackerHistory = obj.trackerHistory_ ;
      s.trackerClass = cellfun(@getTCICellArray,trackerHistory,'uni',0);
      s.trackerData = cellfun(@getSaveToken,trackerHistory,'uni',0);
      
      if ~forceExcDataCache && forceIncDataCache
        s.ppdb = obj.ppdb;
      end
    end  % function
    
    function currMovInfo = projLoadGUI(obj,fname,varargin)
      % Load a lbl file
      %
      % currProjInfo: scalar struct containing diagnostic info. When the
      % project is loaded, APT attempts to set the movie to the last-known
      % (saved) movie. If this movie is not found in the filesys, the
      % project will be set to "nomovie" and currProjInfo will contain:
      %   .iMov: movie(set) index for desired-but-not-found movie
      %   .badfile: path of file-not-found
      %
      % If the movie is able to set the project correctly, currProjInfo
      % will be [].
            
      obj.pushBusyStatus('Loading project...') ;
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      
      starttime = tic();
      
      [nomovie, replace_path] = myparse(varargin,...
        'nomovie',false, ... % If true, call movieSetNoMovie() instead of movieSetGUI(currMovie)
        'replace_path',{'',''} ...
        );
      if isempty(replace_path) ,
        replace_path = {'',''} ;
      end

      currMovInfo = [];
      
      if exist('fname','var')==0
        lastLblFile = obj.rcGetProp('lastLblFile');
        if isempty(lastLblFile)
          lastLblFile = pwd;
        end
        filterspec = sprintf(obj.DEFAULT_LBLFILENAME,'*');
        [fname,pth] = uigetfile(filterspec,'Load label file',lastLblFile);
        if isequal(fname,0)
          return;
        end
        fname = fullfile(pth,fname);
      end
      
      if exist(fname,'file')==0
        error('Labeler:file','File ''%s'' not found.',fname);
      else
        tmp = which(fname);
        if ~isempty(tmp)
          fname = tmp; % use fullname
        else
          % fname exists, but cannot be expanded into a fullpath; ideally 
          % the only possibility is that it is already a fullpath
        end
      end
      
      % MK 20190204. Use Unbundling instead of loading.
      [success,tlbl,wasbundled] = obj.projUnbundleLoad(fname);
      if ~success, error('Could not unbundle the label file %s',fname); end
      
      % AL 20191002 occlusion-prediction viz, DLNetType enum changed
      % should-be-harmlessly
      % AL 20210923 net removal (see below)
      warnst0 = warning('off','MATLAB:class:EnumerationValueChanged');
      warnst1 = warning('off','MATLAB:class:EnumerationNameMissing'); 
      s = load(tlbl,'-mat');
      fprintf('Loaded project data from file.\n');
      warning([warnst0 warnst1]);

      % ALT 2023-05-02      
      % Sometimes trackDLBackEnd is a DLBackEndClass object, because history.  
      % If that's the case, we just pass it through.  It will get rectified when the
      % project is next saved.
      if isfield(s, 'trackDLBackEnd') ,
        backend_thang = s.trackDLBackEnd ;
        if is_an_encoding_container(backend_thang) ,
          backend = decode_encoding_container(backend_thang) ;
        elseif isa(backend_thang, 'DLBackEndClass') ,
          backend = backend_thang ;
        else
          error('Don''t know how to decode something of class %s', class(baackend_thang)) ;
        end
        s.trackDLBackEnd = backend ;
      end

      if ~all(isfield(s,{'VERSION' 'movieFilesAll'}))
        error('Labeler:load','Unexpected contents in Label file.');
      end
      obj.rcSaveProp('lastLblFile',fname);

      t0 = tic;
      s = Labeler.lblModernize(s);
      fprintf('Modernized project (%f s).\n',toc(t0));

      % convert CalRig structs to CalRig objects
      for fn1 = {'viewCalibrationData','viewCalibrationDataGT','viewValProjWide'},
        fn = fn1{1};
        if ~isfield(s,fn),
          continue;
        end
        if iscell(s.(fn)),
          for i = 1:numel(s.(fn)),
            if isstruct(s.(fn){i}),
              try
                ss = CalRig.createCalRigObjFromStruct(s.(fn){i});
              catch ME,
                warningNoTrace('Load error creating CalRig object from %s{%d} struct:\n',fn,i,getReport(ME));
                ss = s.(fn){i};
              end
              s.(fn){i} = ss;
            end
          end
        elseif isstruct(s.(fn)),
          try
            ss = CalRig.createCalRigObjFromStruct(s.(fn));
          catch ME
            warningNoTrace('Load error creating CalRig object from %s struct:\n%s',fn,getReport(ME));
            ss = s.(fn);
          end
          s.(fn) = ss;
        end
      end

      % Set this so all the prop-setting below doesn't create issues 
      % when the associated events fire.
      obj.isinit = true;
      
      t0 = tic;
      obj.initFromConfig_(s.cfg);
      fprintf('Initialized configuration (%f s).\n',toc(t0));
      
      % From here to the end of this method is a parallel initialization to
      % projNew()
      
      % For all the loadable properties in s, load them into the obj, doing path
      % replacement along the way if called for.
      t0 = tic;
      LOADPROPS = Labeler.SAVEPROPS(~ismember(Labeler.SAVEPROPS,...
                                              Labeler.SAVEBUTNOTLOADPROPS));
      path_to_replace = replace_path{1} ;
      target_path = replace_path{2} ;
      for i = 1 : numel(LOADPROPS) 
        prop_name = LOADPROPS{i} ;
        if ~isfield(s, prop_name)          
          warningNoTrace('Labeler:load','Missing load field ''%s''.',prop_name);
          continue
        end
        saved_value = s.(prop_name) ;
        if isempty(path_to_replace) ,
          % If there is no path replacement to be done, just assign the saved value to the object property.
          obj.(prop_name) = saved_value ;
        else          
          if any(strcmp(prop_name,Labeler.MOVIEPROPS))
            % If prop_name is a movie property, then we want to do path replacement
            if isstruct(saved_value)
              value = structfun(@(path)(strrep(path, path_to_replace, target_path)), ...
                               saved_value, ...
                               'UniformOutput', false) ;              
            else              
              value = strrep(saved_value, path_to_replace, target_path) ;
            end
            obj.(prop_name) = value ;
          else
            % If prop_name is not a movie property, then just assign the saved value to
            % the object property.
            obj.(prop_name) = saved_value ;
          end
        end
      end  % for

      % need this before setting movie so that .projectroot exists
      obj.projFSInfo = ProjectFSInfo('loaded',fname);
      
      % check that all movie files exist, allow macro fixes
      doesFileExistFromRegularMovieIndex = false(1, obj.nmovies) ;
      for i = 1:obj.nmovies ,
        doesThisRegularMovieExist = obj.movieCheckFilesExist(MovieIndex(i,false)) ;
        doesFileExistFromRegularMovieIndex(i) = doesThisRegularMovieExist ;
      end
      doesFileExistFromGTMovieIndex = false(1, obj.nmoviesGT) ;
      for i = 1:obj.nmoviesGT,
        doesThisGTMovieExist = obj.movieCheckFilesExist(MovieIndex(i,true)) ;
        doesFileExistFromGTMovieIndex(i) = doesThisGTMovieExist ;
      end
      doAllRegularMoviesExist = all(doesFileExistFromRegularMovieIndex) ;
      doAllGTMoviesExist = all(doesFileExistFromGTMovieIndex) ;

      obj.initTrxInfo();      

      obj.computeLastLabelChangeTS_Old();
      fcnNumLbledRows = @Labels.numLbls;
      obj.movieFilesAllHaveLbls = cellfun(fcnNumLbledRows,obj.labels);
      obj.movieFilesAllGTHaveLbls = cellfun(fcnNumLbledRows,obj.labelsGT);      
%       obj.movieFilesAllHaveLbls = cellfun(@Labels.hasLbls,obj.labels);
%       obj.movieFilesAllGTHaveLbls = cellfun(@Labels.hasLbls,obj.labelsGT);      
      obj.gtUpdateSuggMFTableLbledComplete();      

      % % Populate obj.trackersAll_
      % obj.initializeTrackersAllAndFriends_() ;  % do I need this here?

      % Populate obj.trackerHistory_
      nTracker = numel(s.trackerData);
      assert(nTracker==numel(s.trackerClass));
      assert(isempty(obj.trackerHistory_));
      trackerCreateInfos = cellfun(@(tciAsCellArray)(TrackerCreateInfo.fromTCICellArray(tciAsCellArray, s.maIsMA)), ...
                                   s.trackerClass(:)') ;
      rawTrackerHistory = ...
        cellfun(@(tc,td)(LabelTracker.create(obj, tc, td)), ...
                num2cell(trackerCreateInfos), s.trackerData(:)', ...
                'UniformOutput', false) ;
      isFilePreTrackerHistory = isfield(s, 'currTracker') ;
        % indicates whether the file predates the introduction of tracker history
      if isFilePreTrackerHistory
        currTracker = s.currTracker ;
      else
        currTracker = [] ;
      end
      trackerHistory = apt.trimTrackersAfterLoad(rawTrackerHistory, isFilePreTrackerHistory, currTracker) ;
      obj.trackerHistory_ = trackerHistory;
      
      obj.isinit = false;

      % Initialize preprocessed data cache
      if isfield(s, 'ppdb') && ~isempty(s.ppdb)
        fprintf('Loading DL data cache: %d rows.\n',s.ppdb.dat.N);
        obj.ppdb = s.ppdb;
      end
      fprintf('Initialized properties from loaded data (%f s).\n',toc(t0));

      t0 = tic;
      if obj.nmoviesGTaware==0 || s.currMovie==0 || nomovie
        obj.movieSetNoMovie();
      else
        [tfok,badfile] = obj.movieCheckFilesExist(s.currMovie,s.gtIsGTMode);
        if ~tfok
          currMovInfo.iMov = s.currMovie;
          currMovInfo.badfile = badfile;
          obj.movieSetNoMovie();
        else
          obj.movieSetGUI(s.currMovie);
          [tfok] = obj.checkFrameAndTargetInBounds(s.currFrame,s.currTarget);
          if ~tfok,
            warning('Cached frame number %d and target number %d are out of bounds for movie %d, reverting to using first frame of first target.',...
                    s.currFrame,s.currTarget,s.currMovie);
            s.currFrame = 1;
            s.currTarget = 1;
          end
          obj.setFrameAndTargetGUI(s.currFrame,s.currTarget,true); % force updates of everything
        end
      end
      fprintf('Opened current movie (%f s).\n',toc(t0));
      
%       % Needs to occur after tracker has been set up so that labelCore can
%       % communicate with tracker if necessary (in particular, Template Mode 
%       % <-> Hide Predictions)
%       obj.labelingInit();

      t0 = tic;

      obj.labeledposNeedsSave = false;
      obj.doesNeedSave_ = false;
%       obj.suspScore = obj.suspScore;

      obj.syncPropsMfahl_() ;
      obj.notify('updateFrameTableComplete');
      
      if obj.currMovie>0
        obj.labelsUpdateNewFrame(true);
      end
      
      % Set up the prev_axes 
      % This needs to occur after .labeledpos etc has been set
      obj.prevAxesModeTargetSpec_ = PrevAxesTargetSpec.fromPersistedStruct(s.cfg.PrevAxes.ModeInfo) ;
      pamode = PrevAxesMode.(s.cfg.PrevAxes.Mode) ;
      obj.setPrevAxesMode(pamode) ;

      % Make sure the AWS debug mode of the backend is consistent with the Labeler AWS debug
      % mode
      obj.trackDLBackEnd.isInAwsDebugMode = obj.isInAwsDebugMode ;
 
      % obj.setPropertiesToFireCallbacksToInitializeUI_() ;
      %obj.notify('update') ;

      % The fact that we (presumably) have to update before doing these next few
      % things suggests to me (along with their visual-centric names) that maybe
      % these methods belong in the LabelerController.  -- ALT, 2025-02-11
      obj.setSkeletonEdges(obj.skeletonEdges);
      obj.setShowSkeleton(obj.showSkeleton);
      obj.setShowMaRoi(obj.showMaRoi);
      obj.setShowMaRoiAux(obj.showMaRoiAux);
      obj.setFlipLandmarkMatches(obj.flipLandmarkMatches);
      % AL20220113 otherwise labels2 cosmetics (eg text show/hide) not applied
      obj.labels2VizShowHideUpdate();
      
      if ~wasbundled
        % DMC.rootDir point to original model locs
        % Immediately modernize model wrt dl-cache-related invariants
        % After this branch the model is as good as a bundled load
        
        fprintf('\n\n### Raw/unbundled project migration.\n');
        fprintf('Copying Deep Models into %s.\n',obj.projTempDir);
        backend = obj.trackDLBackEnd ;
        for iTrker = 1:numel(obj.trackerHistory_)
          tracker = obj.trackerHistory_{iTrker} ;
          if isprop(tracker,'trnLastDMC') && ~isempty(tracker.trnLastDMC)            
            try
              if backend.isProjectCacheRemote
                warningNoTrace('Remote model detected for net type %s. This will not migrated/preserved.',tracker.trnNetType);
              else
                tracker.copyModelFiles(obj.projTempDir,true);
              end
            catch ME
              warningNoTrace('Nettype ''%s'': error caught trying to save model. Trained model will not be migrated for this net type:\n%s',...
                tracker.trnNetType,ME.getReport());
            end
          end
        end
      end
      obj.projUpdateDLCache_();  % this can fail (output arg not checked)

      % % Surely this should have happened when the current tracker was set up...
      % % Update the tracker info (are we sure this didn't happen already?)
      % if ~isempty(obj.tracker)
      %   obj.tracker.updateTrackerInfo();
      % end

      % Send a bunch of notifications to update the UI, if present
      obj.notify('update') ;
      obj.notify('didLoadProject');  % should phase this out eventually
      % obj.notify('cropUpdateCropGUITools');
      % obj.notify('gtIsGTModeChanged');
      % obj.notify('gtSuggUpdated');
      % obj.notify('gtResUpdated');
      % obj.notify('update_menu_track_tracking_algorithm') ;
      % obj.notify('update_menu_track_tracker_history') ;
      % obj.notify('didSetCurrTracker') ;
      % obj.notify('update_text_trackerinfo') ;
      fprintf('Updated GUI (%f s).\n',toc(t0));

      % If any movies were missing, error now.  We do this late so that the
      % project still gets loaded, and the user can fix any missing movies in the
      % movie manager.
      if doAllRegularMoviesExist ,
        % All is well, do nothing
      else
        % Not all regular movies exist
        missingRegularMovieFilePaths = obj.movieFilesAll(~doesFileExistFromRegularMovieIndex) ;
        fprintf('During loading, there were missing (non-GT) movies:\n') ;
        cellfun(@(path)(fprintf('%s\n', path)), ...
                missingRegularMovieFilePaths) ;
      end
      if doAllGTMoviesExist ,
        % All is well, do nothing
      else
        % All regular movies exist, but not all GT movies exist
        missingGTMovieFilePaths = obj.movieFilesAllGT(~doesFileExistFromGTMovieIndex) ;
        fprintf('During loading, there were missing GT movies:\n') ;
        cellfun(@(path)(fprintf('%s\n', path)), ...
                missingGTMovieFilePaths) ;
      end
      if nomovie || (doAllRegularMoviesExist && doAllGTMoviesExist)
        % All is well, do nothing.
      else
        error('Labeler:movie_missing', 'At least one movie is missing (see console for list).  Use Movie Manager to fix.') ;
      end
      % Final sign-off
      fprintf('\nFinished loading project, elapsed time %f s.\n',toc(starttime)); 


    end  % function projLoadGUI
    
    function [movs,tgts,frms] = findPartiallyLabeledFrames(obj)
      
      labels = obj.labelsGTaware();
      frms = zeros(0,1);
      tgts = zeros(0,1);
      movs = zeros(0,1);
      nold = obj.nLabelPoints-obj.nLabelPointsAdd;
      for mov = 1:size(labels,1),
        [frmscurr,tgtscurr] = Labels.isPartiallyLabeledT(labels{mov},nan,nold);
        ncurr = numel(frmscurr);
        if ~isempty(frmscurr),
          frms(end+1:end+ncurr,1) = frmscurr;
          tgts(end+1:end+ncurr,1) = tgtscurr;
          movs(end+1:end+ncurr,1) = mov;
        end
      end
      
    end
    
    function printPartiallyLabeledFrameInfo(obj)
      % Get a list of what movies, targets, frames are still only partially labeled.
      % Not exposed in GUI, but used in important user workflows.
      [movs,tgts,frms] = obj.findPartiallyLabeledFrames();
      if isempty(frms),
        fprintf('No partially labeled frames left.\n');
        return;
      end
      [~,order] = sortrows([movs,tgts,frms]);
      fprintf('Mov\tTgt\tFrm\n');
      for i = order(:)',
        fprintf('%d\t%d\t%d\n',movs(i),tgts(i),frms(i));
      end
    end
        
%     function projImport(obj,fname)
%       % 'Import' the project fname, MERGING movies/labels into the current project.
%           
%       assert(false,'Unsupported');
%       
% %       if exist(fname,'file')==0
% %         error('Labeler:file','File ''%s'' not found.',fname);
% %       else
% %         tmp = which(fname);
% %         if ~isempty(tmp)
% %           fname = tmp; % use fullname
% %         else
% %           % fname exists, but cannot be expanded into a fullpath; ideally 
% %           % the only possibility is that it is already a fullpath
% %         end
% %       end
% %        
% %       [success, tlbl] = obj.projUnbundleLoad(fname);
% %       if ~success, error('Could not unbundle the label file %s',fname); end
% %       s = load(tlbl,'-mat');
% %       obj.projClearTempDir();
% % %       s = load(fname,'-mat');
% %       if s.nLabelPoints~=obj.nLabelPoints
% %         error('Labeler:projImport','Project %s uses nLabelPoints=%d instead of %d for the current project.',...
% %           fname,s.nLabelPoints,obj.nLabelPoints);
% %       end
% %       
% %       assert(~obj.isMultiView && iscolumn(s.movieFilesAll));
% %       
% %       if isfield(s,'projMacros') && ~isfield(s.projMacros,'projdir')
% %         s.projMacros.projdir = fileparts(fname);
% %       else
% %         s.projMacros = struct();
% %       end
% %       
% %       nMov = size(s.movieFilesAll,1);
% %       for iMov = 1:nMov
% %         movfile = s.movieFilesAll{iMov,1};
% %         movfileFull = Labeler.platformize(FSPath.macroReplace(movfile,s.projMacros));
% %         movifo = s.movieInfoAll{iMov,1};
% %         trxfl = s.trxFilesAll{iMov,1};
% %         lpos = s.labeledpos{iMov};
% %         lposTS = s.labeledposTS{iMov};
% %         lpostag = s.labeledpostag{iMov};
% %         if isempty(s.suspScore)
% %           suspscr = [];
% %         else
% %           suspscr = s.suspScore{iMov};
% %         end
% %         
% %         if exist(movfileFull,'file')==0 || ~isempty(trxfl)&&exist(trxfl,'file')==0
% %           warning('Labeler:projImport',...
% %             'Missing movie/trxfile for movie ''%s''. Not importing this movie.',...
% %             movfileFull);
% %           continue;
% %         end
% %            
% %         obj.movieFilesAll{end+1,1} = movfileFull;
% %         obj.movieFilesAllHaveLbls(end+1,1) = any(~isnan(lpos(:)));
% %         obj.movieInfoAll{end+1,1} = movifo;
% %         obj.trxFilesAll{end+1,1} = trxfl;
% %         obj.labeledpos{end+1,1} = lpos;
% %         obj.labeledposTS{end+1,1} = lposTS;
% %         obj.lastLabelChangeTS = max(obj.lastLabelChangeTS,max(lposTS(:)));
% %         obj.labeledposMarked{end+1,1} = false(size(lposTS));
% %         obj.labeledpostag{end+1,1} = lpostag;
% %         obj.labeledpos2{end+1,1} = s.labeledpos2{iMov};
% %         if ~isempty(obj.suspScore)
% %           obj.suspScore{end+1,1} = suspscr;
% %         end
% % %         if ~isempty(obj.suspNotes)
% % %           obj.suspNotes{end+1,1} = [];
% % %         end
% %       end
% % 
% %       obj.labeledposNeedsSave = true;
% %       obj.projFSInfo = ProjectFSInfo('imported',fname);
% %       
% %       % TODO prob would need .preProcInit() here
% %       
% %       if ~isempty(obj.tracker)
% %         warning('Labeler:projImport','Re-initting tracker.');
% %         obj.tracker.init();
% %       end
% %       % TODO .trackerDeep
%     end
    
    % function projAssignProjNameFromProjFileIfAppropriate_(obj)
    %   if isempty(obj.projname) && ~isempty(obj.projectfile)
    %     [~,fname] = fileparts(obj.projectfile);
    %     obj.projname = fname;
    %   end
    % end
    
    % Functions to handle bundled label files
    % MK 20190201
    function tname = projGetEnsureTempDir(obj,varargin) % throws
      % tname: project tempdir, assigned to .projTempDir. Guaranteed to
      % exist, contents not guaranteed
      
      cleartmp = myparse(varargin,...
        'cleartmp',false...
        );
      
      if isempty(obj.projTempDir)
        obj.projTempDir = tempname(APT.getdotaptdirpath());
      end
      tname = obj.projTempDir;
      
      if exist(tname,'dir')==0
        [success,message,~] = mkdir(tname);
        if ~success
          error('Could not create temp directory %s: %s',tname,message);
        end        
      elseif cleartmp
        obj.projClearTempDir();
      else
        % .projTempDir exists and may have stuff in it
      end
      
    end
    
    function [success,rawLblFile,isbundled] = projUnbundleLoad(obj,fname) % throws
      % Unbundles the lbl file if it is a tar bundle.
      %
      % This can throw. If it throws you know nothing.
      %
      % If it doesn't throw:
      %  If success and isbundled, then .projTempDir/<rawLblFile> is the
      %   raw label file and .projTempDir/ contains the exploded DL model
      %   cache tree.
      %  If success and ~isbundled, then .projTempDir/<rawLblFile> is the
      %   raw label file, but there are no DL models under .projTempDir.
      %  If ~success, then .projTempDir is set and exists on the filesystem
      %   but that's it. .projTempDir/<rawLblFile> probably doesn't exist.
      %   
      % fname: fullpath to projectfile, either tarred/bundled or untarred
      %
      % success, isbundled: as described above
      % rawLblFile: full path to untarred raw label file in .projTempDir
      % MK 20190201
      
      [rawLblFile,tname] = obj.projGetRawLblFile('cleartmp',true); % throws; sets .projTempDir
      
      try
        starttime = tic;
        fprintf('Untarring project %s into %s\n',fname,tname);
        untar(fname,tname);
        obj.unTarLoc = tname;
        fprintf('... done with untar, elapsed time = %fs.\n',toc(starttime));
      catch ME
        if endsWith(ME.identifier,'invalidTarFile')
          warningNoTrace('Label file %s is not bundled. Using it in raw (mat) format.',fname);
          [success,message,~] = copyfile(fname,rawLblFile);
          if ~success
            warningNoTrace('Could not copy raw label file: %s',message);
            isbundled = [];
          else
            isbundled = false;  
            obj.unTarLoc = tname;
          end          
          return;
        else
          ME.rethrow(); % most unfortunate
        end
      end
      
      if ~exist(rawLblFile,'file')
        warning('Could not find raw label file in the bundled label file %s',fname);
        success = false;
        isbundled = [];
        return;
      end
      
      success = true;
      isbundled = true;
    end
    
    function success = cleanUpProjTempDir(obj,verbose)
      
      success = false;
      if nargin < 2,
        verbose = true;
      end
      if ~ischar(obj.unTarLoc) || isempty(obj.unTarLoc),
        success = true;
        return;
      end
      if ~exist(obj.unTarLoc,'dir'),
        if verbose,
          fprintf('Temporary tar directory %s does not exist. Not cleaning.\n',obj.unTarLoc);
        end
        return;
      end
      [success,msg] = rmdir(obj.unTarLoc,'s');
      if ~success && verbose,
        fprintf('Error deleting temporary tar directory %s:\n%s\n',obj.unTarLoc,msg);
      end
      if success,
        if verbose,
          fprintf('Removed temporary tar directory %s.\n',obj.unTarLoc);
        end
        obj.unTarLoc = '';
      end
      
    end
    
    function projUpdateDLCache_(obj)
      % Updates project DL state to point to new cache in .projTempDir      

      % Get the project cache dir path (e.g. '/home/janeuser/.apt/tpkjasdfkuhawe') ;
      projectCacheDirPathAsChar = obj.projTempDir;  % native path, as char
   
      % It seems like this warning is thrown often even when nothing is wrong.
      % Disabling.  -- ALT, 2024-10-10
      % Check for exploded cache in tempdir      
      % tCacheDir = fullfile(projectCacheDirPathAsChar,obj.projname);
      % if ~exist(tCacheDir,'dir')
      %   % warningNoTrace('Could not find model data for %s in temp directory %s. Deep Learning trackers not restored.',...
      %   %                obj.projname,cacheDir);
      %   return
      % end
      
      % Update the project cache path in the backend and trackers
      if obj.backend.isProjectCacheRemote ,
        warningNoTrace('Unexpected remote project cache detected');
      else
        obj.backend.nativeProjectCachePath = projectCacheDirPathAsChar ;
        % Update/set all DMC.rootDirs to cacheDir
        trackers = obj.trackerHistory_ ;
        cellfun(@(t)(t.updateDLCache(projectCacheDirPathAsChar)), trackers) ;
      end
    end  % function
    
    function [rawLblFile,projtempdir] = projGetRawLblFile(obj,varargin) % throws
      projtempdir = obj.projGetEnsureTempDir(varargin{:});
      rawLblFile = fullfile(projtempdir,obj.DEFAULT_RAW_LABEL_FILENAME);
    end
    
    function projBundleSave_(obj,outFile,varargin) % throws 
      % bundle contents of projTempDir into outFile
      %
      % throws on err, hopefully cleans up after itself (projtempdir) 
      % regardless. 
      
      verbose = myparse(varargin,...
        'verbose',obj.projVerbose ...
      );
      
      [rawLblFile,projtempdir] = obj.projGetRawLblFile();
      if ~exist(rawLblFile,'file')
        error('Raw label file %s does not exist. Could not create bundled label file.',...
          rawLblFile);
      end
      
      % allModelFiles will contain all projtempdir artifacts to be tarred
      allModelFiles = {rawLblFile};
      
      % If the project cache is remote, make it local
      obj.downloadProjectCacheIfNeeded() ;

      % find the model files and then bundle them into the tar directory.
      % but since there isn't much in way of relative path support in
      % matlabs tar/zip functions, we will also have to copy them first the
      % temp directory. sigh.

      for iTrker = 1:numel(obj.trackerHistory_)
        tracker = obj.trackerHistory_{iTrker};
        if isa(tracker,'DeepTracker')
          % a lot of unnecessary moving around is to maintain the directory
          % structure - MK 20190204
          dmc = tracker.trnLastDMC ;
          if isempty(dmc),
            continue;
          end
          try
            if verbose,
              fprintf('Saving model for nettype ''%s'' from %s.\n',...
                      tracker.trnNetType,dmc.getRootDir);
            end
            modelFilesDst = tracker.copyModelFiles(projtempdir,verbose);
            allModelFiles = [allModelFiles; modelFilesDst(:)]; %#ok<AGROW>
          catch ME
            warningNoTrace('Nettype ''%s'': obj.lerror caught trying to save model. Trained model will not be saved for this net type:\n%s',...
                           tracker.trnNetType,ndx,ME.getReport());
          end
        else
          error('Not implemented') ;
        end
      end
      
      % - all DL models exist under projtempdir
      % - obj...Saving.CacheDir is unchanged
      % - all DMCs need not have .rootDirs that point to projtempdir
                  
      pat = [regexprep(projtempdir,'\\','\\\\') '[/\\]'];
      allModelFiles = cellfun(@(x) regexprep(x,pat,''),...
        allModelFiles,'UniformOutput',false);
      fprintf('Tarring %d model files into %s\n',numel(allModelFiles),projtempdir);
      tar([outFile '.tar'],allModelFiles,projtempdir);
      movefile([outFile '.tar'],outFile); 
      fprintf('Project saved to %s\n',outFile);

      % matlab by default adds the .tar. So save it to tar
      % and then move it.
      
      % Don't clear the tempdir here, user may still be using project.
      %obj.clearTempDir();
    end
    
    function downloadProjectCacheIfNeeded(obj)  % throws on error
      % Copy any training/tracking artifacts on the backend back to the frontend.
      % Throws on err.            
      backend = obj.trackDLBackEnd ;
      backend.downloadProjectCacheIfNeeded(obj.DLCacheDir) ;
    end  % function

    function projExportTrainData(obj,outfile)
      obj.pushBusyStatus(sprintf('Exporting training data to %s',outfile));
      oc = onCleanup(@()(obj.popBusyStatus())) ;            
      [tfsucc,~,s] = ...
        obj.trackCreateDeepTrackerStrippedLbl();
      if ~tfsucc,
        error('Could not collect data for exporting.');
      end
      % preProcData_P is [nLabels,nViews,nParts,2]
      save(outfile,'-mat','-v7.3','-struct','s');
      fprintf('Saved training data to file ''%s''.\n',outfile);
    end
        
    function success = projRemoveTempDir(obj) % throws
      success = true;
      if isempty(obj.projTempDir)
        return
      end
      [success, message, ~] = rmdir(obj.projTempDir,'s');
      if success
        fprintf('Cleared temp directory: %s\n',obj.projTempDir);
      else
        warning('Could not clear the temp directory: %s',message);
      end
    end

    function projRemoveTempDirAsync(obj) % throws
      nativeProjTempDir = obj.projTempDir ;
      if isempty(nativeProjTempDir)
        return
      end
      wslProjTempDir = wsl_path_from_native(nativeProjTempDir) ;
      escapedWslProjTempDir = escape_string_for_bash(wslProjTempDir) ;
      command = sprintf('nohup rm -rf %s &> /dev/null &', escapedWslProjTempDir) ;
      apt.syscmd(command, 'failbehavior', 'err') ;
      fprintf('Clearing temp directory %s in a background process...\n',obj.projTempDir);
    end
    
    function projBundleTempDir(obj, tfile)
      obj.pushBusyStatus('Bundling the temp directory...') ;
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      tar(tfile,obj.projTempDir);
    end
    
    function projClearTempDir(obj) % throws
      if isempty(obj.projTempDir)
        return
      end
      obj.projRemoveTempDir();
      [success, message, ~] = mkdir(obj.projTempDir);
      if ~success
        error('Could not clear the temp directory %s',message);
      end
    end
    
    function v = projDeterministicRandFcn(obj,randfcn)
      % Wrapper around rand() that sets/resets RNG seed
      s = rng(obj.projRngSeed);
      v = randfcn();
      rng(s);
    end  % function    
  end  % methods
  
  methods % projMacros
    
    % Macros defined in .projMacros are conceptually for movie files (or
    % "experiments", which are typically located at arbitrary locations in
    % the filesystem (which bear no relation to eg the location of the
    % project file).
    %
    % Other macros can be defined depending on the context. For instance,
    % trxFiles most typically are located in relation to their
    % corresponding movie and currently only a limited number of 
    % movie-related macros are supported. TrkFiles (tracking outputs) might 
    % be located in relation to their movies or perhaps their projects. etc
    
    function projMacroAdd(obj,macro,val)
      if isfield(obj.projMacros,macro)
        error('Labeler:macro','Macro ''%s'' already defined.',macro);
      end
      obj.projMacros.(macro) = val;
    end
    
    function projMacroSet(obj,macro,val)
      if ~ischar(val)
        error('Labeler:macro','Macro value must be a string.');
      end
      
      s = obj.projMacros;
      if ~isfield(s,macro)
        error('Labeler:macro','''%s'' is not a macro in this project.',macro);
      end
      s.(macro) = val;
      obj.projMacros = s;
    end

    function s = projMacrosGetWithAuto(obj)
      % append auto-generated macros to .projMacros
      %
      % over time, many/most clients of .projMacros should probably call
      % this
      
      s = obj.projMacros;
      if ~isfield(s,'projdir') && ~isempty(obj.projectroot)
        % This conditional allows user to explictly specify project root
        % Useful use case here: testproject 'modules' (lbl + data in portable folder)
        s.projdir = obj.projectroot;
      end
    end
    
    function projMacroRm(obj,macro)
      if ~isfield(obj.projMacros,macro)
        error('Labeler:macro','Macro ''%s'' is not defined.',macro);
      end
      obj.projMacros = rmfield(obj.projMacros,macro);
    end
    
    function projMacroClear(obj)
      obj.projMacros = struct();
    end
    
    function projMacroClearUnused(obj)
      mAll = fieldnames(obj.projMacros);
      mfa = [obj.movieFilesAll(:); obj.movieFilesAllGT(:)];
      for macro=mAll(:)',macro=macro{1}; %#ok<FXSET>
        tfContainsMacro = cellfun(@(x)FSPath.hasMacro(x,macro),mfa);
        if ~any(tfContainsMacro)
          obj.projMacroRm(macro);
        end
      end
    end
    
    function tf = projMacroIsMacro(obj,macro)
      tf = isfield(obj.projMacros,macro);
    end
   
    function s = projMacroStrs(obj)
      m = obj.projMacrosGetWithAuto();
      flds = fieldnames(m);
      vals = struct2cell(m);
      s = cellfun(@(x,y)sprintf('%s -> %s',x,y),flds,vals,'uni',0);
    end    
    
    function p = projLocalizePath(obj,p)
      p = FSPath.fullyLocalizeStandardizeChar(p,obj.projMacros);
    end
    
    function [nlabels,nlabelspermovie,nlabelspertarget] = getNLabels(obj,gt)
      
      if nargin < 2,
        gt = false;
      end
      
      if gt,
        nmovies = obj.nmoviesGT;
        labelsfld = 'labelsGT';
      else
        nmovies = obj.nmovies;
        labelsfld = 'labels';
      end
      imovs = (1:nmovies)';
      itgts = cell(size(imovs));
      for i = 1:nmovies,
        ntgts = obj.getNTargets(gt,i);
        itgts{i} = 1:ntgts;
      end
      nlabelspertarget = Labels.lObjNLabeled(obj,labelsfld,'movi',imovs,'itgt',itgts);
      nlabelspermovie = cellfun(@sum,nlabelspertarget);
      nlabels = sum(nlabelspermovie);
      
    end
    
    function printAllTrackerInfo_(obj, do_include_fileinfo)
      % Print a bunch of info about the trained trackers to stdout.

      if nargin < 2,
        do_include_fileinfo = false;
      end
      
      for i = 1:numel(obj.trackerHistory_),
        tObj = obj.trackerHistory_{i};
        if ~isprop(tObj,'trnLastDMC') || isempty(tObj.trnLastDMC),
          continue
        end
        for j = 1:numel(tObj.trnLastDMC.n),
          nettype = tObj.trnLastDMC.getNetType(j);
          nettype = char(nettype{1});
          netmode = tObj.trnLastDMC.getNetMode(j);
          netmode = char(netmode{1});
          trainid = tObj.trnLastDMC.getTrainID(j);
          trainid = trainid{1};
          fprintf('Tracker %d: %s, view %d, stage %d, mode %s\n',i,nettype,tObj.trnLastDMC.getView(j),tObj.trnLastDMC.getStages(j),netmode);
          fprintf('  Trained %s for %d iterations on %d labels\n',trainid,tObj.trnLastDMC.getIterCurr(j),tObj.trnLastDMC.getNLabels(j));
          if do_include_fileinfo,
            fprintf('  Train config file: %s\n',tObj.trnLastDMC.trainConfigLnx(j));
            fprintf('  Current trained model: %s\n',tObj.trnLastDMC.trainCurrModelLnx(j));
          end
        end
      end
      
    end  % function
    
    function printInfo(obj)
      % This method is refered to in user-facing docs, so need to keep it.

      fprintf('Lbl file: %s\n',obj.projectfile);
      fprintf('Info printed: %s\n',datestr(now,'yyyymmddTHHMMSS'));

      fprintf('Project type: ');
      if obj.labelMode == LabelMode.MULTIANIMAL,
        fprintf('Multi-animal\n');
      elseif obj.hasTrx,
        fprintf('Trx\n');
      else
        fprintf('Single-animal\n');
      end

      fprintf('Number of views: %d\n',obj.nview);
      
      fprintf('Number of landmarks: %d\n',obj.nPhysPoints);
      
      obj.printAllTrackerInfo_();
      
      if isempty(obj.trackDLBackEnd),
        fprintf('Back-end: None\n');
      else
        fprintf('Back-end: %s\n',char(obj.trackDLBackEnd.type));
      end
      
      fprintf('N. train movies: %d\n',obj.nmovies);
      [nlabels,nlabelspermovie,nlabelspertarget] = obj.getNLabels();
      fprintf('N. train labels: %d\n',nlabels);
      fprintf('N. labeled train movies: %d\n',nnz(nlabelspermovie));
      if obj.hasTrx,
        fprintf('N. labeled train trajectories: %d\n',sum(cellfun(@nnz,nlabelspertarget)));
      end
      
      fprintf('N. GT movies: %d\n',obj.nmoviesGT);
      [nlabelsGT,nlabelspermovieGT,nlabelspertargetGT] = obj.getNLabels(true);
      fprintf('N. GT labels: %d\n',nlabelsGT);
      fprintf('N. labeled GT movies: %d\n',nnz(nlabelspermovieGT));
      if obj.hasTrx,
        fprintf('N. labeled GT trajectories: %d\n',sum(cellfun(@nnz,nlabelspertargetGT)));
      end
      
      fprintf('Save code info:\n');
      if isempty(obj.saveVersionInfo),
        fprintf('No saved version info available.\n');
      else
        fprintf(GitMatlabBreadCrumbString(obj.saveVersionInfo));
      end
      fprintf('Load code info:\n');
      info = GetGitMatlabStatus(fileparts(mfilename('fullpath'))) ;
      fprintf(GitMatlabBreadCrumbString(info));
    end  % function
            
  end  % methods
  
  methods (Static)
    
    function s = lblModernize(s)
      % s: struct, .lbl contents
      
	    % whether trackParams is stored -- update from 20190214
      isTrackParams = isfield(s,'trackParams');
      assert(isTrackParams);
            
      s.cfg = Labeler.cfgModernize(s.cfg);
      
      if ~isfield(s,'maIsMA')
        s.maIsMA = false;
      end
      
      % defaultTrackersInfo = LabelTracker.getAllTrackersCreateInfo(s.maIsMA);
      assert(iscell(s.trackerClass));

      % s.trackerClass and s.trackerData are eventually used to restore
      % labeler.trackersHistory

      % update interim/dev MA-BU projs
      for i=1:numel(s.trackerClass)
        if numel(s.trackerClass{i})==3 && ...
            strcmp(s.trackerClass{i}{1},'DeepTracker') && ...
            (~isempty(s.trackerClass{i}{3}) && s.trackerClass{i}{3}==DLNetType.multi_mdn_joint_torch) ,
          s.trackerClass{i}([1 4 5]) = ...
            {'DeepTrackerBottomUp' 'trnNetMode' DLNetMode.multiAnimalBU};
        end
      end

      % Determine which elements of s.trackerClass match some default tracker kind
      % tf = LabelTracker.trackersCreateInfoIsMember(s.trackerClass,...
      %                                              defaultTrackersInfo);
      fakeTrackPrefs = struct('PredictInterpolate', {false}) ;
      fakeProjPrefs = struct('Track', fakeTrackPrefs) ;
      fakeLabeler = struct('maIsMA', {s.maIsMA}, 'projectHasTrx', {s.projectHasTrx}, 'projPrefs', {fakeProjPrefs}) ;
      tfDoKeep = true(size(s.trackerClass)) ;
      for i = 1 : numel(s.trackerClass)
        trackerCreateInfoAsCellArray = s.trackerClass{i} ;
        tci = TrackerCreateInfo.fromTCICellArray(trackerCreateInfoAsCellArray, s.maIsMA) ;
        try
          LabelTracker.create(fakeLabeler, tci) ;  % Just throw away the actual tracker
        catch me
          % If fails, we don't keep this one
          warningNoTrace('Removing obsolete tracker: %d', i);
          tfDoKeep(i) = false ;
        end
      end

      % %assert(all(tf));
      % % AL: removing CPR for now until if/when updated 
      % % AL 20210923: Net removal
      % % When an entry is removed from DLNetType, affected trackerDatas will
      % % have their .trnNetTypes loaded as structs. Eliminate these
      % % trackers.
      % for iTrker=1:numel(s.trackerData)
      %   if ~isempty(s.trackerData{iTrker}) && isfield(s.trackerData{iTrker},'trnNetType')
      %     nt = s.trackerData{iTrker}.trnNetType;
      %     if isstruct(nt)
      %       try
      %         warningNoTrace('Removing obsolete tracker: %s',nt.ValueNames{1});
      %       catch
      %         warningNoTrace('Removing obsolete tracker: %d',iTrker);
      %       end
      %       tf(iTrker) = false;
      %     end
      %   else
      %     % TODO: two-stage trackers
      %   end
      % end
      
      % Delete elements of s.trackerClass, s.trackerData that do not match a default
      % kind of tracker.
      s.trackerClass(~tfDoKeep) = [];
      s.trackerData(~tfDoKeep) = [];

      % % Bring loc into register with s.trackerClass, s.trackerData
      % loc(~tf) = [];      
      % 
      % tclass = defaultTrackersInfo;
      % tclass(loc) = s.trackerClass(:);  % If a default tracker kind matches one in s.trackerClass, replace the default tracker with the one in s.trackerClass
      % tdata = repmat({[]},1,nDfltTrkers);
      % tdata(loc) = s.trackerData(:);  % If a default tracker kind matches one in s.trackerClass, replace the default tracker with the one in s.trackerClass
      % s.trackerClass = tclass;
      % s.trackerData = tdata;      

      % % KB 20201216 update currTracker as well
      % oldCurrTracker = s.currTracker;
      % if oldCurrTracker>0 && ~isempty(loc) && oldCurrTracker <= numel(loc),
      %   s.currTracker = loc(oldCurrTracker);
      % end
      
      % 2019ed0207: added nLabels to dmc
      % 20190404: remove .trnName, .trnNameLbl as these dup DMC
      for i = 1:numel(s.trackerData),
        
        % KB 20220804 refactor DMC
        if isfield(s.trackerData{i},'trnLastDMC') && ~isempty(s.trackerData{i}.trnLastDMC)
          try
            if isfield(s.trackerData{1},'stg1')
              netmode=s.trackerData{1}.stg1.trnNetMode;
            else
              netmode = s.trackerData{1}.trnNetMode;
            end
            s.trackerData{i}.trnLastDMC = ...
              DeepModelChainOnDisk.modernize(s.trackerData{i}.trnLastDMC,...
                                             'netmode',netmode);
          catch ME
            warning('Could not modernize DMC for tracker %d, setting to empty:\n%s',i,getReport(ME));
            s.trackerData{i}.trnLastDMC = [];
          end
        end

        if isfield(s.trackerData{i},'trnName') && ~isempty(s.trackerData{i}.trnName)
          if isfield(s.trackerData{i},'trnLastDMC') && ~isempty(s.trackerData{i}.trnLastDMC)
            assert(all(strcmp(s.trackerData{i}.trnName,...
                              s.trackerData{i}.trnLastDMC.getModelChainID())));
          end
          s.trackerData{i} = rmfield(s.trackerData{i},'trnName');
        end
        if isfield(s.trackerData{i},'trnNameLbl') && ~isempty(s.trackerData{i}.trnNameLbl)
          if isfield(s.trackerData{i},'trnLastDMC') && ~isempty(s.trackerData{i}.trnLastDMC)
            assert(all(strcmp(s.trackerData{i}.trnNameLbl,...
                              s.trackerData{i}.trnLastDMC.getTrainID())));
          end
          s.trackerData{i} = rmfield(s.trackerData{i},'trnNameLbl');
        end
      end  % for
      
      % 20181215 factor dlbackend out of DeepTrackers into single/common
      % prop on Labeler
      if ~isfield(s,'trackDLBackEnd') || ~isa(s.trackDLBackEnd, 'DLBackEndClass') ,
        % maybe change this by looking thru existing trackerDatas
        s.trackDLBackEnd = DLBackEndClass(DLBackEnd.Bsub);
      end
      % 20201028 docker/sing backend img/tag update
      s.trackDLBackEnd.modernize();
      
      % 20181220 DL common parameters
      assert(isTrackParams);      
      
      % KB 20190212: reorganized DL parameters -- many specific parameters
      % were moved to common, and organized common parameters. leaf names
      % should all be the same, and unique, so just match leaves
      s = reorganizeDLParams(s); 
            
      % KB 20191218: replaced scale_range with scale_factor_range
      if isstruct(s.trackParams) && isfield(s.trackParams,'ROOT') && ...
          isstruct(s.trackParams.ROOT) && isfield(s.trackParams.ROOT,'DeepTrack') && ...
          isstruct(s.trackParams.ROOT.DeepTrack) && isfield(s.trackParams.ROOT.DeepTrack,'DataAugmentation') && ...
          isstruct(s.trackParams.ROOT.DeepTrack.DataAugmentation) && ...
          ~isfield(s.trackParams.ROOT.DeepTrack.DataAugmentation,'scale_factor_range') && ...
          isfield(s.trackParams.ROOT.DeepTrack.DataAugmentation,'scale_range'),
        if s.trackParams.ROOT.DeepTrack.DataAugmentation.scale_range ~= 0,
          warningNoTrace(['"Scale range" data augmentation parameter has been replaced by "Scale factor range". ' ...
            'These are very similar, so we have auto-populated "Scale factor range" based on your '...
            '"Scale range" parameter. However, these are not the same. Please examine "Scale factor range" '...
            'in the Tracking parameters GUI.']);
        end
        s.trackParams.ROOT.DeepTrack.DataAugmentation.scale_factor_range = 1+s.trackParams.ROOT.DeepTrack.DataAugmentation.scale_range;
      end
      
      % KB 20190331: adding in post-processing parameters if missing
      % AL 20190507: ... [a subset of] ... .trackParams modernization
      % AL 20190712: (subsumes above) modernizing entire .trackParams
      s.trackParams = APTParameters.modernize(s.trackParams);

      % KB 20190214: store all parameters in each tracker so that we don't
      % have to delete trackers when tracking parameters change
      % AL 20190712: further clarification 
      for i = 1:numel(s.trackerData),
        if isempty(s.trackerData{i}),
          continue;
        end
        
        if ~isfield(s.trackerData{i},'sPrmAll') || isempty(s.trackerData{i}.sPrmAll),
          if isfield(s.trackerData{i},'sPrm') && ~isempty(s.trackerData{i}.sPrm)
            % legacy proj: .sPrm present
            
            tfCPR = strcmp(s.trackerClass{i}{1},'CPRLabelTracker');
            tfDT = strcmp(s.trackerClass{i}{1},'DeepTracker');
            s.trackerData{i}.sPrmAll = s.trackParams;
            
            if tfCPR
              CPRParams1 = APTParameters.all2CPRParams(s.trackerData{i}.sPrmAll,...
                numel(s.cfg.LabelPointNames),s.cfg.NumViews);
              assert(isequaln(CPRParams1,s.trackerData{i}.sPrm));
            elseif tfDT
              DLSpecificParams1 = APTParameters.all2DLSpecificParams(...
                s.trackerData{i}.sPrmAll,s.trackerClass{i}{3});
              assert(isequaln(DLSpecificParams1,s.trackerData{i}.sPrm));
            else
              assert(false);
            end
          else
            if ~isfield(s.trackerData{i},'sPrmAll')
              s.trackerData{i}.sPrmAll = [];
            end
          end
          
          if isfield(s.trackerData{i},'sPrm'),
            s.trackerData{i} = rmfield(s.trackerData{i},'sPrm');
          end          
        else
          % s.trackerData{i}.sPrmAll is present
          assert(~isfield(s.trackerData{i},'sPrm'),'Unexpected legacy parameters.');
        end
        
        % At this point for s.trackerData{i}:
        % 1. For legacy projs that had .sPrm but no .sPrmAll, we created 
        %    .sPrmAll and asserted that those params effectively match the 
        %    old .sPrm. Then we removed the .sPrm. In this case the 
        %   .sPrmAll happens to be modernized already.
        % 2. .sPrm has been removed in all cases.
        % 3. For modern projs that have .sPrmAll, this may not yet be
        % modernized. Responsibility for modernization will now be in the
        % LabelTrackers/loadSaveToken. Hmm not sure this is best.         
      end
      
      if isfield(s,'preProcParams'),
        s = rmfield(s,'preProcParams');
      end
      if isfield(s,'trackDLParams'),
        s = rmfield(s,'trackDLParams');
      end
            
      % KB 20190314: added skeleton
      if ~isfield(s,'skeletonEdges'),
        s.skeletonEdges = zeros(0,2);
      end
      if ~isfield(s,'showSkeleton'),
        s.showSkeleton = false;
      end
      % AL 20201004, 20210324
      if ~isfield(s,'showMaRoi')
        s.showMaRoi = s.cfg.LabelMode == LabelMode.MULTIANIMAL;
      end
      if ~isfield(s,'showMaRoiAux')
        s.showMaRoiAux = s.cfg.LabelMode == LabelMode.MULTIANIMAL;
      end
      if ~isfield(s,'skelHead'),
        s.skelHead = [];
      end      
      if ~isfield(s,'skelTail'),
        s.skelTail = [];
      end      

      % KB 20191203: added landmark matches
      if ~isfield(s,'flipLandmarkMatches'),
        s.flipLandmarkMatches = zeros(0,2);
      end

      
      % 20190429 TrkRes
      if ~isfield(s,'trkRes')
        s = Labeler.resetTrkResFieldsStruct(s);
      end
      if size(s.trkRes,1)~=size(s.movieFilesAll,1) || ...
         size(s.trkResGT,1)~=size(s.movieFilesAllGT,1) 
        % AL20200702: we appear to have had a bug in movieSetAdd which 
        % didn't maintain/update .trkRes/.trkResGT size appropriately. Fix
        % here at load-time. The .trkRes* infrastructure is power-users only
        % so probably very few regular users use it.
        warningNoTrace('Unexpected .trkRes* size. Resetting .trkRes* state. (This is not a commonly-used feature.)');
        s = Labeler.resetTrkResFieldsStruct(s);   
      end
      
      % 202008 lbl compact format
      if ~isfield(s,'labels')
        nmov = numel(s.labeledpos);
        s.labels = cell(nmov,1);
        s.labels2 = cell(nmov,1);
        fullfcn = @SparseLabelArray.full;
        for imov=1:nmov
          s.labels{imov} = Labels.fromarray(fullfcn(s.labeledpos{imov}),...
             'lposTS',fullfcn(s.labeledposTS{imov}),...
             'lpostag',fullfcn(s.labeledpostag{imov}));

          tfo = TrkFile();
          tfo.initFromArraysFull(fullfcn(s.labeledpos2{imov}));
          tfo = tfo.toTrackletFull();
          tfo.initFrm2Tlt(s.movieInfoAll{imov}.nframes);
          s.labels2{imov} = tfo;
        end
        
        nmov = numel(s.labeledposGT);
        s.labelsGT = cell(nmov,1);
        s.labels2GT = cell(nmov,1);
        for imov=1:nmov
          s.labelsGT{imov} = Labels.fromarray(fullfcn(s.labeledposGT{imov}),...
             'lposTS',fullfcn(s.labeledposTSGT{imov}),...
             'lpostag',fullfcn(s.labeledpostagGT{imov}));
           
          tfo = TrkFile();
          tfo.initFromArraysFull(fullfcn(s.labeledpos2GT{imov}));
          tfo = tfo.toTrackletFull();
          tfo.initFrm2Tlt(s.movieInfoAllGT{imov}.nframes);
          s.labels2GT{imov} = tfo;
        end
      end
      % 20210322 labelsRoi
      if ~isfield(s,'labelsRoi')
        nmov = numel(s.labels);
        s.labelsRoi = repmat({LabelROI.new()},nmov,1);
      end
      if ~isfield(s,'labelsRoiGT')
        nmov = numel(s.labelsGT);
        s.labelsRoiGT = repmat({LabelROI.new()},nmov,1);
      end
      
      % 20210317 MA use tracklets in labels2
      % Used Labels earlier in dev
      for i=1:numel(s.labels2)
        stmp = s.labels2{i};
        if ~isa(stmp,'TrkFile')
          if ~isempty(stmp)
            stmp = Labels.toPTrx(stmp);
            tlt = save_tracklet(stmp,[]);
            tfo = TrkFile();
            tfo.initFromTracklet(tlt);
          else                      
            tfo = TrkFile(s.cfg.NumLabelPoints,zeros(0,1));
          end
%           tfo.initFrm2Tlt(s.movieInfoAll{i}.nframes);
          s.labels2{i} = tfo;
        end
        % 20210618: update frm2tltnnz even for existing TrkFiles
        s.labels2{i}.initFrm2Tlt(s.movieInfoAll{i}.nframes);
      end
      for i=1:numel(s.labels2GT)
        stmp = s.labels2GT{i};
        if ~isa(stmp,'TrkFile')
          if ~isempty(stmp)
            stmp = Labels.toPTrx(stmp);
            tlt = save_tracklet(stmp,[]);
            tfo = TrkFile();
            tfo.initFromTracklet(tlt);            
          else
            tfo = TrkFile(s.cfg.NumLabelPoints,zeros(0,1));
          end
          s.labels2GT{i} = tfo;
        end
        s.labels2GT{i}.initFrm2Tlt(s.movieInfoAllGT{i}.nframes);
      end
      
      % KB 20210626 - added info about state of code to saved lbl file
      if ~isfield(s,'saveVersionInfo'),
        s.saveVersionInfo = [];
      end
      
      if ~isfield(s,'trxInfoAll'),
        s.trxInfoAll = {};
      end
      
      % 20210706 Some projs managed to get a non-float value here which
      % causes trouble
      if ~isfloat(s.currFrame)        
        s.currFrame = double(s.currFrame);
      end

      % If the GT suggestions are old-school style for an MA project, fix it.
      if s.maIsMA
        mfTable = s.gtSuggMFTable ;
        if ismember('iTgt', mfTable.Properties.VariableNames)
          % Need to set all the iTgt's to nan, but avoid repeats
          mfTable.iTgt = zeros(size(mfTable.iTgt)) ;  % first set all iTgts to zero
          mfTable = unique(mfTable) ;  % eliminate now-redudant rows
          mfTable.iTgt = nan(size(mfTable.iTgt)) ;  % replace iTgt's with nans (couldn't do before b/c nan~=nan)
          s.gtSuggMFTable = mfTable ;
        end
      end

      % 20250603 - calrig information should be saved as a struct so that
      % we create new objects
      for fn1 = {'viewCalibrationData','viewCalibrationDataGT','viewValProjWide'},
        fn = fn1{1};
        if ~isfield(s,fn),
          continue;
        end
        if iscell(s.(fn)),
          for i = 1:numel(s.(fn)),
            if isa(s.(fn){i},'CalRig'),
              try
                ss = s.(fn){i}.getSaveStruct();
              catch ME,
                warningNoTrace('Modernize error converting %s{%d} CalRig object to struct:\n%s',fn,i,getReport(ME));
                ss = s.(fn){i};
              end
              s.(fn){i} = ss;
            end
          end
        elseif isa(s.(fn),'CalRig'),
          try
            ss = s.(fn).getSaveStruct();
          catch ME,
            warningNoTrace('Modernize error converting %s CalRig object to struct:\n%s',fn,getReport(ME));
            ss = s.(fn);
          end
          s.(fn) = ss;
        end
      end
    end  % function lblModernize()
    
    function s = resetTrkResFieldsStruct(s)
      % see .trackResInit, maybe can combine
      nmov = size(s.movieFilesAll,1);
      nmovGT = size(s.movieFilesAllGT,1);
      nvw = size(s.movieFilesAll,2);
      s.trkResIDs = cell(0,1);
      s.trkRes = cell(nmov,nvw,0);
      s.trkResGT = cell(nmovGT,nvw,0);
      s.trkResViz = cell(0,1);
    end
    
    function [data,tname] = stcLoadLblFile(fname,dodelete)
      if nargin < 2,
        dodelete = true;
      end
      tname = tempname;
      try
        untar(fname,tname);
        data = load(fullfile(tname,'label_file.lbl'),'-mat');
        if dodelete,
          rmdir(tname,'s');
        end
      catch ME,
        if strcmp(ME.identifier,'MATLAB:untar:invalidTarFile'),
          data = load(fname,'-mat');
        else
          throw(ME);
        end
      end
    end

    function stcSaveLblFile(data,tardir,outname)

      save(fullfile(tardir,'label_file.lbl'),'-struct','data','-mat');
      tar([outname,'.tar'],'*',tardir);
      movefile([outname,'.tar'],outname);

    end

  end  % methods (Static) 
  
  %% Movie
  methods
    
    function movieAddAllModes(obj,moviefile,varargin)
      if obj.isMultiView,
        obj.movieSetAdd(moviefile,varargin{:});
      else
        obj.movieAdd(moviefile,varargin{:});
      end
    end

    function updateMovieInfo_(obj, iMov, iView)
      % Update the movie info for movieset iMov, view iView by reading the movie
      % metadata from disk.  Is savvy about normal-vs-GT mode.
      if ~exist('iView', 'var') || isempty(iView)
        iView = 1 ;
      end
      isgt = obj.gtIsGTMode ;
      if isgt,
        movfiles = obj.movieFilesAllGTFull;
      else
        movfiles = obj.movieFilesAllFull;
      end
      movfile = movfiles{iMov,iView};
      mr = MovieReader();
      cleaner = onCleanup(@()(mr.close())) ;
      mr.open(movfile); 
      movieInfo = struct();
      movieInfo.nframes = mr.nframes;
      movieInfo.info = mr.info;
      if isgt,
        obj.movieInfoAllGT{iMov, iView} = movieInfo;
      else
        obj.movieInfoAll{iMov, iView} = movieInfo;
      end        
    end  % function

    % function rereadMovieInfo(obj, updatetrackers)
    %   % there was a bug in ufmf movie info, reread all movie info
    %   % also updates the json file for each trained tracker
    %   mr = MovieReader();
    %   for isgt = [false,true],
    %     if isgt,
    %       movfiles = obj.movieFilesAllGTFull;
    %     else
    %       movfiles = obj.movieFilesAllFull;
    %     end
    %     nMov = numel(movfiles);
    %     for iMov = 1:nMov
    %       movfile = movfiles{iMov};
    %       mr.open(movfile);
    %       ifo = struct();
    %       ifo.nframes = mr.nframes;
    %       ifo.info = mr.info;
    %       mr.close();
    %       fprintf('Before update, movie %d info: \n',iMov);
    %       if isgt,
    %         disp(obj.movieInfoAllGT{iMov}.info);
    %         obj.movieInfoAllGT{iMov} = ifo;
    %       else
    %         disp(obj.movieInfoAll{iMov}.info);
    %         obj.movieInfoAll{iMov} = ifo;
    %       end
    %       fprintf('updated to: \n');
    %       disp(ifo.info);
    %     end
    %   end
    % 
    %   istracker = cellfun(@(x) ~isempty(x) && x.canTrack, obj.trackerHistory);
    %   if any(istracker),
    %     if nargin < 2,
    %       res = questdlg('Update trackers too?','Update trackers?','Yes','No','Yes');
    %       updatetrackers = strcmpi(res,'Yes');
    %     end
    %     if updatetrackers,
    % 
    %       mia = cellfun(@(x)struct('NumRows',x.info.nr,...
    %         'NumCols',x.info.nc),obj.movieInfoAll);
    %       for ivw=1:size(mia,2)
    %         nr = [mia(:,ivw).NumRows];
    %         nc = [mia(:,ivw).NumCols];
    %         assert(all(nr==nr(1) & nc==nc(1)),'Inconsistent movie dimensions for view %d',ivw);
    %       end
    % 
    %       for i = find(istracker(:)'),
    %         dmc = obj.trackerHistory{i}.trnLastDMC;
    %         nativeTrainConfig = DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx());  % native path
    %         trainConfig = wsl_path_from_native(nativeTrainConfig) ;
    %         if exist(trainConfig,'file')
    %           js = TrnPack.hlpLoadJson(trainConfig);
    %           js.MovieInfo = mia(1,:);
    %           TrnPack.hlpSaveJson(js,trainConfig);
    %         end
    %       end
    %     end
    %   end
    % end  % function

    function movieAdd(obj,moviefile,trxfile,varargin)
      % Add movie/trx to end of movie/trx list.
      %
      % moviefile: string or cellstr (can have macros)
      % trxfile: (optional) string or cellstr 
      
      %notify(obj,'startAddMovie');      
      
      assert(~obj.isMultiView,'Unsupported for multiview labeling.');
      
      obj.pushBusyStatus('Adding new movie...');
      oc = onCleanup(@()(obj.popBusyStatus()));
      
      [offerMacroization,gt] = myparse(varargin,...
        'offerMacroization',~isdeployed&&obj.isgui, ... % If true, look for matches with existing macros
        'gt',obj.gtIsGTMode ... % If true, add moviefile/trxfile to GT lists. Could be a separate method, but there is a lot of shared code/logic.
        );
      istrxfile = exist('trxfile','var');
      if istrxfile,
        if iscell(trxfile),
          istrxfile = ~all(cellfun(@isempty,trxfile));
        else
          istrxfile = ~isempty(trxfile);
        end
      end

      PROPS = Labeler.gtGetSharedPropsStc(gt);
      
      moviefile = cellstr(moviefile);
      if istrxfile,
        trxfile = cellstr(trxfile);
        szassert(moviefile,size(trxfile));
      end
      nMov = numel(moviefile);
        
      mr = MovieReader();
      mr.preload = obj.movieReadPreLoadMovies;
      for iMov = 1:nMov
        movFile = moviefile{iMov};
        if istrxfile,
          tFile = trxfile{iMov};
        end
        
        if offerMacroization 
          % Optionally replace movFile, tFile with macroized versions
          [tfCancel,macro,movFileMacroized] = ...
            FSPath.offerMacroization(obj.projMacros,{movFile});
          if tfCancel
            continue;
          end
          tfMacroize = ~isempty(macro);
          if tfMacroize
            assert(isscalar(movFileMacroized));
            movFile = movFileMacroized{1};
          end
          
          % trx
          % Note, tFile could already look like $movdir\trx.mat which would
          % be fine.
          movFileFull = obj.projLocalizePath(movFile);
          if istrxfile,
            [tfMatch,tFileMacroized] = FSPath.tryTrxfileMacroization(...
              tFile,fileparts(movFileFull));
            if tfMatch
              tFile = tFileMacroized;
            end
          end
        end
      
        movfilefull = obj.projLocalizePath(movFile);
        assert(exist(movfilefull,'file')>0,'Cannot find file ''%s''.',movfilefull);
        
        % See movieSetInProj()
        if any(strcmp(movFile,obj.(PROPS.MFA)))
          if nMov==1
            error('Labeler:dupmov',...
              'Movie ''%s'' is already in project.',movFile);
          else
            warningNoTrace('Labeler:dupmov',...
              'Movie ''%s'' is already in project and will not be added to project.',movFile);
            continue;
          end
        end
        if any(strcmp(movfilefull,obj.(PROPS.MFAF)))
          warningNoTrace('Labeler:dupmov',...
            'Movie ''%s'', macro-expanded to ''%s'', is already in project.',...
            movFile,movfilefull);
        end
        
        if istrxfile,
          tFileFull = Labeler.trxFilesLocalize(tFile,movfilefull);
          if ~(isempty(tFileFull) || exist(tFileFull,'file')>0)
            FSPath.throwErrFileNotFoundMacroAware(tFile,tFileFull,'trxfile');
          end
        end
        % Could use movieMovieReaderOpen but we are just using MovieReader 
        % to get/save the movieinfo.
      
        mr.open(movfilefull); 
        ifo = struct();
        ifo.nframes = mr.nframes;
        ifo.info = mr.info;
        mr.close();
        
        if istrxfile,
          [trxinfo] = obj.GetTrxInfo(tFileFull,ifo.nframes);
        else
          trxinfo = [];
        end
                
        nlblpts = obj.nLabelPoints;
        nfrms = ifo.nframes;
        obj.(PROPS.MFA){end+1,1} = movFile;
        obj.(PROPS.MFAHL)(end+1,1) = 0;
        obj.(PROPS.MIA){end+1,1} = ifo;
        obj.(PROPS.TIA){end+1,1} = trxinfo;
        obj.(PROPS.MFACI){end+1,1} = CropInfo.empty(0,0);
        if obj.cropProjHasCrops
          wh = obj.cropGetCurrentCropWidthHeightOrDefault();
          obj.cropInitCropsGen(wh,PROPS.MIA,PROPS.MFACI,...
            'iMov',numel(obj.(PROPS.MFACI)));
        end
        obj.(PROPS.MFALUT){end+1,1} = [];
        if istrxfile,
          obj.(PROPS.TFA){end+1,1} = tFile;
        else
          obj.(PROPS.TFA){end+1,1} = '';
        end
        %obj.(PROPS.LPOS){end+1,1} = nan(nlblpts,2,nfrms,nTgt);
        
        obj.(PROPS.LBL){end+1,1} = Labels.new(nlblpts);
        if obj.maIsMA
          tfo = TrkFile(nlblpts,zeros(0,1));
          tfo.initFrm2Tlt(nfrms);
          obj.(PROPS.LBL2){end+1,1} = tfo;
        else
          if istrxfile,
            ntgts = trxinfo.ntgts;
          else
            ntgts = 1;
          end
          tfo = TrkFile(nlblpts,1:ntgts);
          tfo.initFrm2Tlt(nfrms);          
          obj.(PROPS.LBL2){end+1,1} = tfo;
        end
        if gt
          obj.labelsRoiGT{end+1,1} = LabelROI.new();
        else
          obj.labelsRoi{end+1,1} = LabelROI.new();
        end
%        obj.labeledposY{end+1,1} = nan(4,0);
        
%         obj.(PROPS.LPOSTS){end+1,1} = -inf(nlblpts,nfrms,nTgt);
%         obj.(PROPS.LPOSTAG){end+1,1} = false(nlblpts,nfrms,nTgt);
%         obj.(PROPS.LPOS2){end+1,1} = nan(nlblpts,2,nfrms,nTgt);
        if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
          obj.(PROPS.VCD){end+1,1} = [];
        end
        obj.(PROPS.TRKRES)(end+1,:,:) = {[]};
%         if ~gt
%           obj.labeledposMarked{end+1,1} = false(nlblpts,nfrms,nTgt);
%         end
      end  % for iMov = ...
      
      %notify(obj,'finishAddMovie');            
      
    end  % function
    
    function movieAddBatchFile(obj,bfile)
      % Read movies from batch file
      
      if exist(bfile,'file')==0
        error('Labeler:movieAddBatchFile','Cannot find file ''%s''.',bfile);
      end

      obj.pushBusyStatus(sprintf('Adding movies from file %s...',bfile));
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      
      movs = importdata(bfile);
      try
        movs = regexp(movs,',','split');
        movs = cat(1,movs{:});
      catch ME
        error('Labeler:batchfile',...
          'Error reading file %s: %s',bfile,ME.message);
      end
      trxfiles = cell(size(movs,1),1);
      if obj.hasTrx,
        if size(movs,2) == obj.nview*2,
          trxfiles = movs(:,2:2:end);
          movs = movs(:,1:2:end);
        end
      end
      if size(movs,2)~=obj.nview
        error('Labeler:batchfile',...
          'Expected file %s to have %d column(s), one for each view.',...
          bfile,obj.nview);
      end
      if ~iscellstr(movs)  %#ok<ISCLSTR> 
        error('Labeler:movieAddBatchFile',...
          'Could not parse file ''%s'' for filenames.',bfile);
      end
      nMovSetImport = size(movs,1);
      if obj.nview==1
        fprintf('Importing %d movies from file ''%s''.\n',nMovSetImport,bfile);
        obj.movieAdd(movs,trxfiles,'offerMacroization',false);
      else
        fprintf('Importing %d movie sets from file ''%s''.\n',nMovSetImport,bfile);
        for i=1:nMovSetImport
          try
            obj.movieSetAdd(movs(i,:),'offerMacroization',false);
          catch ME
            warningNoTrace('Labeler:mov',...
              'Error trying to add movieset %d: %s Movieset not added to project.',...
              i,ME.message);
          end
        end
      end
    end

    function movieSetAdd(obj,moviefiles,varargin)
      % Add a set of movies (Multiview mode) to end of movie list.
      %
      % moviefiles: cellstr (can have macros)

      [offerMacroization,gt] = myparse(varargin,...
        'offerMacroization',true, ... % If true, look for matches with existing macros
        'gt',obj.gtIsGTMode ... % If true, add moviefiles to GT lists. 
        );

      if obj.nTargets~=1
        error('Labeler:movieSetAdd','Unsupported for nTargets>1.');
      end
      
      PROPS = Labeler.gtGetSharedPropsStc(gt);
      
      moviefiles = cellstr(moviefiles);
      if numel(moviefiles)~=obj.nview
        error('Labeler:movieAdd',...
          'Number of moviefiles supplied (%d) must match number of views (%d).',...
          numel(moviefiles),obj.nview);
      end
      
      if offerMacroization
        [tfCancel,macro,moviefilesMacroized] = ...
          FSPath.offerMacroization(obj.projMacros,moviefiles);
        if tfCancel
          return;
        end
        tfMacroize = ~isempty(macro);
        if tfMacroize
          moviefiles = moviefilesMacroized;
        end
      end
      
      [tfmatch,imovmatch,tfMovsEq,moviefilesfull] = obj.movieSetInProj(moviefiles);
      if tfmatch
        error('Labeler:dupmov',...
          'Movieset matches current movieset %d in project.',imovmatch);
      end
      
      cellfun(@(x)assert(exist(x,'file')>0,'Cannot find file ''%s''.',x),moviefilesfull);      
      
      for iView=1:obj.nview
        iMFmatches = find(tfMovsEq(:,iView,1));
        iMFFmatches = find(tfMovsEq(:,iView,2));
        iMFFmatches = setdiff(iMFFmatches,iMFmatches);
        if ~isempty(iMFmatches)
          warningNoTrace('Labeler:dupmov',...
            'Movie ''%s'' (view %d) already exists in project.',...
            moviefiles{iView},iView);
        end          
        if ~isempty(iMFFmatches)
          warningNoTrace('Labeler:dupmov',...
            'Movie ''%s'' (view %d), macro-expanded to ''%s'', is already in project.',...
            moviefiles{iView},iView,moviefilesfull{iView});
        end
      end
            
      ifos = cell(1,obj.nview);
      mr = MovieReader();
      mr.preload = obj.movieReadPreLoadMovies;
      for iView = 1:obj.nview
        % Could use movieMovieReaderOpen but we are just using MovieReader 
        % to get/save the movieinfo.
        mr.open(moviefilesfull{iView});
        ifo = struct();
        ifo.nframes = mr.nframes;
        ifo.info = mr.info;
        mr.close();
        ifos{iView} = ifo;
      end
      
      % number of frames must be the same in all movies
      nFrms = cellfun(@(x)x.nframes,ifos);
      if ~all(nFrms==nFrms(1))
        nframesstr = arrayfun(@num2str,nFrms,'uni',0);
        nframesstr = String.cellstr2CommaSepList(nframesstr);
        nFrms = min(nFrms);
        warningNoTrace('Labeler:movieSetAdd',...
          'Movies do not have the same number of frames: %s. The number of frames will be set to %d for this movieset.',...
          nframesstr,nFrms);
        for iView=1:obj.nview
          ifos{iView}.nframes = nFrms;
        end
      else
        nFrms = nFrms(1);
      end
      nTgt = 1;
      
      nLblPts = obj.nLabelPoints;
      obj.(PROPS.MFA)(end+1,:) = moviefiles(:)';
      obj.(PROPS.MFAHL)(end+1,1) = 0;
      obj.(PROPS.MIA)(end+1,:) = ifos;
      obj.(PROPS.MFACI){end+1,1} = CropInfo.empty(0,0);
      if obj.cropProjHasCrops
        wh = obj.cropGetCurrentCropWidthHeightOrDefault();
        obj.cropInitCropsGen(wh,PROPS.MIA,PROPS.MFACI,...
          'iMov',numel(obj.(PROPS.MFACI)));
      end
      obj.(PROPS.MFALUT)(end+1,:) = {[]};
      obj.(PROPS.TFA)(end+1,:) = repmat({''},1,obj.nview);
      trxinfo = cellfun(@(x)obj.GetTrxInfo('',x.nframes),ifos,'uni',0);
      obj.(PROPS.TIA)(end+1,:) = trxinfo;
%       obj.(PROPS.LPOS){end+1,1} = nan(nLblPts,2,nFrms,nTgt);
%       obj.(PROPS.LPOSTS){end+1,1} = -inf(nLblPts,nFrms,nTgt);
%       obj.(PROPS.LPOSTAG){end+1,1} = false(nLblPts,nFrms,nTgt);
%       obj.(PROPS.LPOS2){end+1,1} = nan(nLblPts,2,nFrms,nTgt);
      obj.(PROPS.LBL){end+1,1} = Labels.new(nLblPts);
      %obj.(PROPS.LBL2){end+1,1} = Labels.new(nLblPts);
      assert(~obj.maIsMA);
      tfo = TrkFile(nLblPts,1:nTgt); % one target
      tfo.initFrm2Tlt(nFrms);
      obj.(PROPS.LBL2){end+1,1} = tfo;
      if gt
        obj.labelsRoiGT{end+1,1} = LabelROI.new();
      else
        obj.labelsRoi{end+1,1} = LabelROI.new();
      end
      if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
        obj.(PROPS.VCD){end+1,1} = [];
      end
      obj.(PROPS.TRKRES)(end+1,:,:) = {[]};
%       if ~gt
%         obj.labeledposMarked{end+1,1} = false(nLblPts,nFrms,nTgt);
%       end
      
      % This clause does not occur in movieAdd(), b/c movieAdd is called
      % from UI functions which do this for the user. Currently movieSetAdd
      % does not have any UI so do it here.
      if ~obj.hasMovie && obj.nmoviesGTaware>0
        obj.movieSetGUI(1,'isFirstMovie',true);
      end
    end

    function [tf,imovmatch,tfMovsEq,moviefilesfull] = movieSetInProj(obj,moviefiles)
      % Return true if a movie(set) already exists in the project
      %
      % moviefiles: either char if nview==1, or [nview] cellstr. Can
      %   contain macros.
      % 
      % tf: true if moviefiles exists in a row of .movieFilesAll or
      %   .movieFilesAllFull
      % imovsmatch: if tf==true, the matching movie index; otherwise
      %   indeterminate
      % tfMovsEq: [nmovset x nview x 2] logical. tfMovsEq{imov,ivw,j} is
      %   true if moviefiles{ivw} matches .movieFilesAll{imov,ivw} if j==1
      %   or .movieFilesAllFull{imov,ivw} if j==2. This output contains
      %   "partial match" info for multiview projs.
      %
      % This is GT aware and matches are searched for the current GT state.
      
      if ischar(moviefiles)
        moviefiles = {moviefiles};
      end
      
      nvw = obj.nview;
      assert(iscellstr(moviefiles) && numel(moviefiles)==nvw);
      
      moviefilesfull = cellfun(@obj.projLocalizePath,moviefiles,'uni',0);
      
      PROPS = obj.gtGetSharedProps();

      tfMFeq = arrayfun(@(x)strcmp(moviefiles{x},obj.(PROPS.MFA)(:,x)),1:nvw,'uni',0);
      tfMFFeq = arrayfun(@(x)strcmp(moviefilesfull{x},obj.(PROPS.MFAF)(:,x)),1:nvw,'uni',0);
      tfMFeq = cat(2,tfMFeq{:}); % [nmoviesetxnview], true when moviefiles matches movieFilesAll
      tfMFFeq = cat(2,tfMFFeq{:}); % [nmoviesetxnview], true when movfilefull matches movieFilesAllFull
      tfMovsEq = cat(3,tfMFeq,tfMFFeq); % [nmovset x nvw x 2]. 3rd dim is {mfa,mfaf}
      
      for j=1:2
        iAllViewsMatch = find(all(tfMovsEq(:,:,j),2));
        if ~isempty(iAllViewsMatch)
          assert(isscalar(iAllViewsMatch));
          tf = true;
          imovmatch = iAllViewsMatch;
          return;
        end
      end
      
      tf = false;
      imovmatch = [];
    end
    
%     function tfSucc = movieRmName(obj,movName)
%       % movName: compared to .movieFilesAll (macros UNreplaced)
%       assert(~obj.isMultiView,'Unsupported for multiview projects.');
%       iMov = find(strcmp(movName,obj.movieFilesAll));
%       if isscalar(iMov)
%         tfSucc = obj.movieRm(iMov);
%       end
%     end

    function tfSucc = movieRmGUI(obj,iMov,varargin)
      % tfSucc: true if movie removed, false otherwise
      
      [force,gt] = myparse(varargin,...
        'force',false,... % if true, don't prompt even if mov has labels
        'gt',obj.gtIsGTMode ...
        );
      
      assert(isscalar(iMov));
      
      nMovOrig = obj.getnmoviesGTawareArg(gt);
      assert(any(iMov==1:nMovOrig),'Invalid movie index ''%d''.',iMov);
      if iMov==obj.currMovie
        error('Labeler:movieRm','Cannot remove current movie.');
      end
      
      tfProceedRm = true;
      haslbls1 = obj.labelPosMovieHasLabels(iMov,'gt',gt); % TODO: method should be unnec
      haslbls2 = obj.getMovieFilesAllHaveLblsArg(gt);
      haslbls2 = haslbls2(iMov)>0;
      assert(haslbls1==haslbls2);
      if haslbls1 && ~obj.movieDontAskRmMovieWithLabels && ~force
        str = sprintf('Movie index %d has labels. Are you sure you want to remove?',iMov);
        BTN_NO = 'No, cancel';
        BTN_YES = 'Yes';
        BTN_YES_DAA = 'Yes, don''t ask again';
        btn = questdlg(str,'Movie has labels',BTN_NO,BTN_YES,BTN_YES_DAA,BTN_NO);
        if isempty(btn)
          btn = BTN_NO;
        end
        switch btn
          case BTN_NO
            tfProceedRm = false;
          case BTN_YES
            % none; proceed
          case BTN_YES_DAA
            obj.movieDontAskRmMovieWithLabels = true;
        end
      end
      
      obj.pushBusyStatus('Removing movie...') ;
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      
      if tfProceedRm
        PROPS = Labeler.gtGetSharedPropsStc(gt);
        nMovOrigReg = obj.nmovies;
        nMovOrigGT = obj.nmoviesGT;

        if gt
          movIdx = MovieIndex(-iMov);
          movIdxHasLbls = obj.movieFilesAllGTHaveLbls(iMov)>0;
        else
          movIdx = MovieIndex(iMov);
          movIdxHasLbls = obj.movieFilesAllHaveLbls(iMov)>0;
        end

        obj.(PROPS.MFA)(iMov,:) = [];
        obj.(PROPS.MFAHL)(iMov,:) = [];
        obj.(PROPS.MIA)(iMov,:) = [];
        obj.(PROPS.MFACI)(iMov,:) = [];
        obj.(PROPS.MFALUT)(iMov,:) = [];        
        obj.(PROPS.TFA)(iMov,:) = [];
        obj.(PROPS.TIA)(iMov,:) = [];
        
        tfOrig = obj.isinit;
        obj.isinit = true; % AL20160808. we do not want set.labeledpos side effects, listeners etc.
%         obj.(PROPS.LPOS)(iMov,:) = []; % should never throw with .isinit==true
%         obj.(PROPS.LPOSTS)(iMov,:) = [];
%         obj.(PROPS.LPOSTAG)(iMov,:) = [];
%         obj.(PROPS.LPOS2)(iMov,:) = [];
        obj.(PROPS.LBL)(iMov,:) = []; % should never throw with .isinit==true
        obj.(PROPS.LBL2)(iMov,:) = [];
        if gt
          obj.labelsRoiGT(iMov,:) = [];
        else
          obj.labelsRoi(iMov,:) = [];
        end
        if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
          szassert(obj.(PROPS.VCD),[nMovOrig 1]);
          obj.(PROPS.VCD)(iMov,:) = [];
        end
        obj.(PROPS.TRKRES)(iMov,:,:) = [];

        obj.isinit = tfOrig;
        
        edata = MoviesRemappedEventData.movieRemovedEventData(...
          movIdx,nMovOrigReg,nMovOrigGT,movIdxHasLbls);
        obj.ppdb.dat.movieRemap(edata.mIdxOrig2New);
        if gt
          [obj.gtSuggMFTable,tfRm] = MFTable.remapIntegerKey(...
            obj.gtSuggMFTable,'mov',edata.mIdxOrig2New);
          obj.gtSuggMFTableLbled(tfRm,:) = [];
          if ~isempty(obj.gtTblRes)
            obj.gtTblRes = MFTable.remapIntegerKey(obj.gtTblRes,'mov',...
                                                   edata.mIdxOrig2New);
          end
          obj.notify('gtSuggUpdated');
          obj.notify('gtResUpdated');
        end
        
        obj.prevAxesMovieRemap_(edata.mIdxOrig2New);

        sendMaybe(obj.tracker, 'labelerMovieRemoved', edata) ;
        
        if obj.currMovie>iMov && gt==obj.gtIsGTMode
          % AL 20200511. this may be overkill, maybe can just set 
          % .currMovie directly as the current movie itself cannot be 
          % rm-ed. A lot (if not all) state update here prob unnec
          obj.movieSetGUI(obj.currMovie-1);
        end
      end
      
      tfSucc = tfProceedRm;
    end

    function movieRmAll(obj)
      nmov = obj.nmoviesGTaware;
      obj.movieSetNoMovie();
      for imov=1:nmov
        obj.movieRmGUI(1,'force',true);
      end
    end
    
    function movieReorder(obj,p)
      % Reorder (regular/nonGT) movies 
      %
      % p: permutation of 1:obj.nmovies. Must contain all indices.
      
      nmov = obj.nmovies;
      p = p(:);
      if ~isequal(sort(p),(1:nmov)')
        error('Input argument ''p'' must be a permutation of 1..%d.',nmov);
      end
      
      tfSuspStuffEmpty = (isempty(obj.suspScore) || all(cellfun(@isempty,obj.suspScore))) ...
        && isempty(obj.suspSelectedMFT);
      if ~tfSuspStuffEmpty
        error('Reordering is currently unsupported for projects with suspiciousness.');
      end
      
      iMov0 = obj.currMovie;

      % Stage 1, outside obj.isinit block so listeners can update.
      % Future: clean up .isinit, listener policy etc it is getting too 
      % complex
      FLDS1 = {'movieInfoAll' 'movieFilesAll' 'movieFilesAllHaveLbls'...
        'movieFilesAllCropInfo' 'movieFileAllHistEqLUT' 'trxFilesAll' 'trxInfoAll'};
      for f=FLDS1,f=f{1}; %#ok<FXSET>
        obj.(f) = obj.(f)(p,:);
      end
      
      tfOrig = obj.isinit;
      obj.isinit = true;

      vcpw = obj.viewCalProjWide;
      if isempty(vcpw) || vcpw
        % none
      else
        obj.viewCalibrationData = obj.viewCalibrationData(p);
      end      
      FLDS2 = {'labels' 'labelsRoi' 'labels2'};
%         'labeledpos' 'labeledposTS' 'labeledpostag' ... % 'labeledposMarked' 
%         'labeledpos2'};
      for f=FLDS2,f=f{1}; %#ok<FXSET>
        obj.(f) = obj.(f)(p,:);
      end
      obj.trkRes = obj.trkRes(p,:,:);
      
      obj.isinit = tfOrig;
      
      edata = MoviesRemappedEventData.moviesReorderedEventData(...
        p,nmov,obj.nmoviesGT);
      obj.ppdb.dat.movieRemap(edata.mIdxOrig2New);
      sendMaybe(obj.tracker, 'labelerMoviesReordered', edata) ;
      notify(obj,'moviesReordered',edata);

      if ~obj.gtIsGTMode
        iMovNew = find(p==iMov0);
        obj.movieSetGUI(iMovNew); 
      end
    end
    
    % function movieFilesMacroizeGUI(obj,str,macro)
    %   % Replace a string with a macro throughout .movieFilesAll and *gt. A 
    %   % project macro is also added for macro->string.
    %   %
    %   % str: a string 
    %   % macro: macro which will replace all matches of string (macro should
    %   % NOT include leading $)
    % 
    %   if isfield(obj.projMacros,macro) 
    %     currVal = obj.projMacros.(macro);
    %     if ~strcmp(currVal,str)
    %       qstr = sprintf('Project macro ''%s'' is currently defined as ''%s''. This value can be redefined later if desired.',...
    %         macro,currVal);
    %       btn = questiondlg(qstr,'Existing Macro definition','OK, Proceed','Cancel','Cancel');
    %       if isempty(btn)
    %         btn = 'Cancel';
    %       end
    %       switch btn
    %         case 'OK, Proceed'
    %           % none
    %         otherwise
    %           return;
    %       end           
    %     end
    %   end
    % 
    %   strpat = regexprep(str,'\\','\\\\');
    %   mfa0 = obj.movieFilesAll;
    %   mfagt0 = obj.movieFilesAllGT;
    %   if ispc
    %     mfa1 = regexprep(mfa0,strpat,['$' macro],'ignorecase');
    %     mfagt1 = regexprep(mfagt0,strpat,['$' macro],'ignorecase');
    %   else
    %     mfa1 = regexprep(mfa0,strpat,['$' macro]);
    %     mfagt1 = regexprep(mfagt0,strpat,['$' macro]);
    %   end
    %   obj.movieFilesAll = mfa1;
    %   obj.movieFilesAllGT = mfagt1;
    % 
    %   if ~isfield(obj.projMacros,macro) 
    %     obj.projMacroAdd(macro,str);
    %   end
    % end
    
    function movieFilesUnMacroize(obj,macro)
      % "Undo" macro by replacing $<macro> with its value throughout
      % .movieFilesAll and *gt.
      %
      % macro: must be a currently defined macro
      
      if ~obj.projMacroIsMacro(macro)
        error('Labeler:macro','''%s'' is not a currently defined macro.',...
          macro);
      end
      
      sMacro = struct(macro,obj.projMacros.(macro));
      obj.movieFilesAll = FSPath.macroReplace(obj.movieFilesAll,sMacro);
      obj.movieFilesAllGT = FSPath.macroReplace(obj.movieFilesAllGT,sMacro);
    end
    
    function movieFilesUnMacroizeAll(obj)
      % Replace .movieFilesAll with .movieFilesAllFull and *gt; warn if a 
      % movie cannot be found
      
      mfaf = obj.movieFilesAllFull;
      mfagtf = obj.movieFilesAllGTFull;
      nmov = size(mfaf,1);
      nmovgt = size(mfagtf,1);
      nvw = obj.nview;
      for iView=1:nvw
        for iMov=1:nmov
          mov = mfaf{iMov,iView};
          if exist(mov,'file')==0
            warningNoTrace('Labeler:mov','Movie ''%s'' cannot be found.',mov);
          end
          obj.movieFilesAll{iMov,iView} = mov;
        end
        for iMov=1:nmovgt
          mov = mfagtf{iMov,iView};
          if exist(mov,'file')==0
            warningNoTrace('Labeler:mov','Movie ''%s'' cannot be found.',mov);
          end
          obj.movieFilesAllGT{iMov,iView} = mov;
        end
      end
      
      obj.projMacroClear;
    end
    
    function tfok = checkFrameAndTargetInBounds(obj,frm,tgt)
      tfok = false;
      if obj.nframes < frm,
        return;
      end
      if obj.hasTrx,
        if numel(obj.trx) < tgt,
          return;
        end
        trxtgt = obj.trx(tgt);
        if frm<trxtgt.firstframe || frm>trxtgt.endframe,
          return;
        end
      end
      
      tfok = true;
    end
    
    function [tfok,badfile] = movieCheckFilesExist(obj, varargin)  % obj const
      % Check if the movie files and trx files exist.  This version does not present
      % any UI to help the user correct missing files, if just does what it says on
      % the tin.
      % tfok: true if movie/trxfiles for iMov all exist, false otherwise.
      % badfile: if ~tfok, badfile contains a file that could not be found.

      % Process args, deal with possible MovieIndex input
      whatAmI = varargin{1} ;
      if isa(whatAmI, 'MovieIndex') ,
        [iMov,gt] = whatAmI.get();
      else
        iMov = whatAmI ;
        gt = varargin{2} ;
      end

      PROPS = Labeler.gtGetSharedPropsStc(gt);
      
      if ~all(cellfun(@isempty,obj.(PROPS.TFA)(iMov,:)))
        assert(~obj.isMultiView,...
               'Multiview labeling with targets unsupported.');
      end
      
      for iView = 1:obj.nview
        movfileFull = obj.(PROPS.MFAF){iMov,iView};
        trxFileFull = obj.(PROPS.TFAF){iMov,iView};
        if ~exist(movfileFull,'file')
          tfok = false;
          badfile = movfileFull;
          return
        elseif ~isempty(trxFileFull) && ~exist(trxFileFull,'file')
          tfok = false;
          badfile = trxFileFull;
          return
        end
      end
      
      tfok = true;
      badfile = [];
    end
    
    function tfsuccess = movieSetGUI(obj, iMov, varargin)
      % Set the current movie to the one indicated by iMov.
      % iMov: If multiview, movieSet index (row index into .movieFilesAll)
            
      assert(~isa(iMov,'MovieIndex')); % movieIndices, use movieSetMIdx
      assert(any(iMov==1:obj.nmoviesGTaware),...
                    'Invalid movie index ''%d''.',iMov);

      obj.pushBusyStatus(sprintf('Switching to movie %d...',iMov));
      oc = onCleanup(@()(obj.popBusyStatus())) ;

      [isFirstMovie] = myparse(varargin,...
        'isFirstMovie',~obj.hasMovie... % passing true for the first time a movie is added to a proj helps the UI
        ); 
      
      mIdx = MovieIndex(iMov,obj.gtIsGTMode);
      tfsuccess = obj.controller_.movieCheckFilesExistGUI(mIdx); % throws
      if ~tfsuccess
        return
      end
      
      movsAllFull = obj.movieFilesAllFullGTaware;
      cInfo = obj.movieFilesAllCropInfoGTaware{iMov};
      tfHasCrop = ~isempty(cInfo);

      ppPrms = obj.preProcParams;
      if ~isempty(ppPrms)
        bgsubPrms = ppPrms.BackSub;
        bgArgs = {'bgType',bgsubPrms.BGType,'bgReadFcn',bgsubPrms.BGReadFcn};
      else
        bgArgs = {};
      end        
      for iView=1:obj.nview
        mov = movsAllFull{iMov,iView};
        mr = obj.movieReader(iView);
        mr.preload = obj.movieReadPreLoadMovies;
        mr.open(mov,bgArgs{:}); % should already be faithful to .forceGrayscale, .movieInvert
        if tfHasCrop
          mr.setCropInfo(cInfo(iView)); % cInfo(iView) is a handle/pointer!
        else
          mr.setCropInfo([]);
        end
        obj.rcSaveProp('lbl_lastmovie',mov);
        if iView==1
          if numel(obj.moviefile) > obj.MAX_MOVIENAME_LENGTH,
            obj.moviename = ['..',obj.moviefile(end-obj.MAX_MOVIENAME_LENGTH+3:end)];
          else
            obj.moviename = obj.moviefile;
          end
          %obj.moviename = FSPath.twoLevelFilename(obj.moviefile);
        end
      end
      
      % fix the clim so it doesn't keep flashing
      cmax_auto = nan(1,obj.nview); %#ok<*PROPLC>
      for iView = 1:obj.nview,
        im = obj.movieReader(iView).readframe(1,...
          'doBGsub',obj.movieViewBGsubbed,'docrop',false);
        cmax_auto(iView) = GuessImageMaxValue(im);
        if numel(obj.cmax_auto) >= iView && cmax_auto(iView) == obj.cmax_auto(iView) && ...
            size(obj.clim_manual,1) >= iView && all(~isnan(obj.clim_manual(iView,:))),
          set(obj.controller_.axes_all(iView),'CLim',obj.clim_manual(iView,:));
        else
          obj.clim_manual(iView,:) = nan;
          obj.cmax_auto(iView) = cmax_auto(iView);
          set(obj.controller_.axes_all(iView),'CLim',[0,cmax_auto(iView)]);
        end
      end
      
      isInitOrig = obj.isinit;
      obj.isinit = true; % Initialization hell, invariants momentarily broken
      obj.currMovie = iMov;
      
      obj.labelingInit('dosettemplate',false);
      if isFirstMovie,
        % KB 20161213: moved this up here so that we could redo in initHook
        obj.trkResVizInit();
        % we set template below as it requires .trx to be set correctly. 
        % see below
        % obj.labelingInit('dosettemplate',false); 
      end
      
      trxFile = obj.trxFilesAllFullGTaware{iMov,1};
      tfTrx = ~isempty(trxFile);
      if tfTrx
        assert(~obj.isMultiView,...
          'Multiview labeling with targets is currently unsupported.');
        nfrm = obj.movieInfoAllGTaware{iMov,1}.nframes;
        trxvar = obj.getTrx(trxFile,nfrm);
      else
        trxvar = [];
      end
      obj.trxSet(trxvar);
      %obj.trxfile = trxFile; % this must come after .trxSet() call
        
      obj.setFrameAndTargetGUI(1,1,true);
      
      obj.isinit = isInitOrig; % end Initialization hell      

      % needs to be done after trx are set as labels2trkviz handles
      % multiple targets.
      % 20191017 done for every movie because different movies may have
      % diff numbers of targets.
      obj.labels2TrkVizInit();
      
      if isFirstMovie
        if obj.labelMode==LabelMode.TEMPLATE
          % Setting the template requires the .trx to be appropriately set,
          % so for template mode we redo this (it is part of labelingInit()
          % here.
          obj.labelingInitTemplate();
        end

        for i = 1:obj.nview,
          ud = obj.gdata.axes_all(i).UserData;
          if isstruct(ud) && isfield(ud,'gamma') && ~isempty(ud.gamma),
            gam = ud.gamma;
            ViewConfig.applyGammaCorrection(obj.gdata.images_all,obj.gdata.axes_all,obj.gdata.axes_prev,i,gam);
          end
        end

      end

      % obj.selectedFrames_ = [] ;
      obj.infoTimelineModel_.initNewMovie(obj.isinit, obj.hasMovie, obj.nframes, obj.hasTrx) ;
      obj.notify('updateTimelineTraces');
      obj.notify('updateTimelineLandmarkColors');
      obj.notify('updateTimelineProps');
      obj.notify('updateTimelineSelection');

      % AL20160615: omg this is the plague.
      % AL20160605: These three calls semi-obsolete. new projects will not
      % have empty .labeledpos, .labeledpostag, or .labeledpos2 elements;
      % these are set at movieAdd() time.
      %
      % However, some older projects will have these empty els; and
      % maybe it's worth keeping the ability to have empty els for space
      % reasons (as opposed to eg filling in all els in lblModernize()).
      % Wait and see.
      % AL20170828 convert to asserts 
%       assert(~isempty(obj.labeledpos{iMov}));
%       assert(~isempty(obj.labeledposTS{iMov}));
%       assert(~isempty(obj.labeledposMarked{iMov}));
%       assert(~isempty(obj.labeledpostag{iMov}));
%       assert(~isempty(obj.labeledpos2{iMov}));
           
      edata = NewMovieEventData(isFirstMovie);
      sendMaybe(obj.tracker, 'newLabelerMovie') ;
      notify(obj,'newMovie',edata);
      
      obj.syncPropsMfahl_() ;
      obj.notify('updateFrameTableComplete');
      
      % Proj/Movie/LblCore initialization can maybe be improved
      % Call setFrame again now that lblCore is set up
      if obj.hasTrx
        obj.setFrameAndTargetGUI(obj.currTrx.firstframe,obj.currTarget,true);
      else
        obj.setFrameAndTargetGUI(1,1,true);
      end
            
    end  % function
    
    function tfsuccess = movieSetMIdx(obj,mIdx,varargin)
      assert(isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt~=obj.gtIsGTMode
        obj.gtSetGTMode(gt,'warnChange',true);
      end
      tfsuccess = obj.movieSetGUI(iMov,varargin{:});
    end
    
    function movieSetNoMovie(obj,varargin)
      % Set .currMov to 0
                 
          % Stripped cut+paste form movieSetGUI() for reference 20170714
          %       obj.movieReader(iView).open(movfileFull);
          %       obj.moviename = fullfile(parent,movname);
%       obj.isinit = true; % Initialization hell, invariants momentarily broken
          %       obj.currMovie = iMov;
          %       obj.setFrameAndTarget(1,1);
          %       obj.trxSet(trxvar);
          %       obj.trxfile = trxFile; % this must come after .trxSet() call
%       obj.isinit = false; % end Initialization hell
          %       obj.labelsMiscInit();
          %       obj.labelingInit();
          %       edata = NewMovieEventData(isFirstMovie);
          %       notify(obj,'newMovie',edata);
          %       obj.updateFrameTableComplete();
          %       if obj.hasTrx
          %         obj.setFrameAndTarget(obj.currTrx.firstframe,obj.currTarget);
          %       else
          %         obj.setFrameAndTarget(1,1);
          %       end
              
      for i=1:obj.nview
        obj.movieReader(i).close();
      end
      obj.moviename = '';
      %obj.trxfile = '';
      isInitOrig = obj.isinit;
      obj.isinit = true;
      obj.currMovie = 0;
      obj.trxSet([]);
      obj.currFrame = 1;
      obj.currTarget = 0;
      obj.isinit = isInitOrig;
      
      obj.labels2TrkVizInit();
      obj.trkResVizInit();
      obj.labelingInit('dosettemplate',false);
      % obj.selectedFrames_ = [] ;
      obj.infoTimelineModel_.initNewMovie(obj.isinit, obj.hasMovie, obj.nframes, obj.hasTrx) ;
      obj.notify('updateTimelineTraces');
      obj.notify('updateTimelineLandmarkColors');
      obj.notify('updateTimelineProps');
      obj.notify('updateTimelineSelection');

      edata = NewMovieEventData(false);
      sendMaybe(obj.tracker, 'newLabelerMovie') ;
      notify(obj,'newMovie',edata);
      obj.syncPropsMfahl_() ;
      obj.notify('updateFrameTableComplete');

      % Set state equivalent to obj.setFrameAndTarget();
      controller = obj.controller_;
      imsall = controller.images_all;
      for iView=1:obj.nview
        obj.currIm{iView} = 0;
        obj.currImRoi{iView} = [ 1 1 1 1 ] ;
        set(imsall(iView),'CData',0);
      end
      obj.prevIm = 0 ;
      obj.prevImRoi = [ 1 1 1 1 ] ;
      imprev = controller.image_prev;
      set(imprev,'CData',0);     
      if ~obj.gtIsGTMode
        obj.clearPrevAxesModeTarget();
      end
      obj.currTarget = 1;
      obj.currFrame = 1;
      obj.prevFrame = 1;
      
%       obj.currSusp = [];
    end
    
    function s = moviePrettyStr(obj,mIdx)
      assert(isscalar(mIdx));
      [iMov,gt] = mIdx.get();
      if gt
        pfix = 'GT ';
      else
        pfix = '';
      end
      if obj.isMultiView
        mov = 'movieset';
      else
        mov = 'movie';
      end
      s = sprintf('%s%s %d',pfix,mov,iMov);
    end
    
    % Hist Eq
    
%     function [hgram,hgraminfo] = movieEstimateImHist(obj,varargin) % obj CONST
%       % Estimate the typical image histogram H0 of movies in the project.
%       %
%       % Operates on regular (non-GT) movies. Movies are sampled with all 
%       % movies getting equal weight.
%       %
%       % If movies have crops, cropping occurs before histogram
%       % counting/selection. Otoh Trx have no bearing on this method.
%       %
%       % hgram: [nbin] histogram count vector. See HistEq.selMovCentralImHist
%       % hgraminfo: See HistEq.selMovCentralImHist
%       
%       [nFrmPerMov,nBinHist,iMovsSamp,debugViz] = myparse(varargin,...
%         'nFrmPerMov',20,... % num frames to sample per mov
%         'nBinHist',256, ... % num bins for imhist()
%         'iMovsSamp',[],... % indices into .movieFilesAll to sample. defaults to 1:nmovies
%         'debugViz',false ...
%       );
% 
%       ppPrms = obj.preProcParams;
%       if ~isempty(ppPrms) && ppPrms.BackSub.Use
%         error('Unsupported when background subtraction is enabled.');
%       end
%     
%       wbObj = WaitBarWithCancel('Histogram Equalization','cancelDisabled',true);
%       oc = onCleanup(@()delete(wbObj));
% 
%       if isempty(iMovsSamp)
%         iMovsSamp = 1:size(obj.movieFilesAll,1);
%       end      
%       nmovsetsSamp = numel(iMovsSamp);
%       
%       nvw = obj.nview;
%       fread = nan(nFrmPerMov,nmovsetsSamp,nvw);
%       cntmat = zeros(nBinHist,nmovsetsSamp,nvw);
%       bins0 = [];
%       for ivw=1:nvw
%         wbstr = sprintf('Sampling movies, view %d',ivw);
%         wbObj.startPeriod(wbstr,'shownumden',true,'denominator',nmovsetsSamp);
%         mr = MovieReader;
%         for i=1:nmovsetsSamp
%           tic;
%           wbObj.updateFracWithNumDen(i);
%           
%           imov = iMovsSamp(i);
%           mIdx = MovieIndex(imov);
%           obj.movieMovieReaderOpen(mr,mIdx,ivw);
%           nfrmMov = mr.nframes;
%           if nfrmMov<nFrmPerMov
%             warningNoTrace('View %d, movie %d: sampling %d frames from a total of %d frames in movie.',...
%               ivw,imov,nFrmPerMov,nfrmMov);
%           end
%           fsamp = linspace(1,nfrmMov,nFrmPerMov);
%           fsamp = round(fsamp);
%           fsamp = max(fsamp,1);
%           fsamp = min(fsamp,nfrmMov);
%           
%           for iF=1:nFrmPerMov
%             f = fsamp(iF);
%             im = mr.readframe(f,'docrop',true);
%             nchan = size(im,3);
%             if nchan>1
%               error('Images must be grayscale.');
%             end
% 
%             [cnt,bins] = imhist(im,nBinHist);
%             cntmat(:,i,ivw) = cntmat(:,i,ivw)+cnt;
%             if isempty(bins0)
%               bins0 = bins;
%             elseif ~isequal(bins,bins0)
%               dbins = unique(diff(bins));
%               warningNoTrace('View %d, movie %d, frame %d: unexpected imhist bin vector. bin delta: %d',...
%                 ivw,imov,f,dbins);
%             end
%             fread(iF,i,ivw) = f;
%           end
%           
%           t = toc;
%           fprintf(1,'Elapsed time: %d sec\n',round(t));
%         end
%         wbObj.endPeriod();
%       end
% 
%       [hgram,hgraminfo] = HistEq.selMovCentralImHist(cntmat,...
%         'debugviz',debugViz);      
%     end
    
%     function movieEstimateHistEqLUTs(obj,varargin)
%       % Update .movieFilesAllHistEqLUT, .movieFilesAllGTHistEqLUT based on
%       % .preProcH0. Applying .movieFilesAllHistEqLUT{iMov} to frames of
%       % iMov should have a hgram approximating .preProcH0
%       
%       [nFrmPerMov,wbObj,docheck] = myparse(varargin,...
%         'nFrmPerMov',20, ... % num frames to sample per mov
%         'wbObj',[],...
%         'docheck',false...
%         );
%       
%       if isempty(obj.preProcH0)
%         error('No target image histogram set in property ''%s''.');
%       end
%             
%       obj.movieEstimateHistEqLUTsHlp(false,nFrmPerMov,'docheck',docheck,'wbObj',wbObj);
%       obj.movieEstimateHistEqLUTsHlp(true,nFrmPerMov,'docheck',docheck,'wbObj',wbObj);
%     end
    
%     function movieEstimateHistEqLUTsHlp(obj,isGT,nFrmPerMov,varargin)
%       
%       [wbObj,docheck] = myparse(varargin,...
%         'wbObj',[],...
%         'docheck',false);
%       
%       tfWB = ~isempty(wbObj);
%       
%       PROPS = obj.gtGetSharedPropsStc(isGT);
%       nmovsets = obj.getnmoviesGTawareArg(isGT);
%       nvw = obj.nview;
% 
%       obj.(PROPS.MFALUT) = cell(nmovsets,nvw);
%       
%       for ivw=1:nvw
%         if tfWB
%           wbstr = sprintf('Sampling movies, view %d',ivw);
%           wbObj.startPeriod(wbstr,'shownumden',true,'denominator',nmovsets);
%         end
%         
%         mr = MovieReader;
% %         Isampcat = []; % used if debugViz
% %         Jsampcat = []; % etc
% %         Isampcatyoffs = 0;
%         for imov=1:nmovsets
% %           tic;
%           if tfWB
%             wbObj.updateFracWithNumDen(imov);
%           end
%           
%           mIdx = MovieIndex(imov,isGT);
%           obj.movieMovieReaderOpen(mr,mIdx,ivw);
%           nfrmMov = mr.nframes;
%           if nfrmMov<nFrmPerMov
%             warningNoTrace('View %d, movie %d: sampling %d frames from a total of %d frames in movie.',...
%               ivw,imov,nFrmPerMov,nfrmMov);
%           end
%           fsamp = linspace(1,nfrmMov,nFrmPerMov);
%           fsamp = round(fsamp);
%           fsamp = max(fsamp,1);
%           fsamp = min(fsamp,nfrmMov);
% 
%           Isamp = cell(nFrmPerMov,1);
%           for iF=1:nFrmPerMov
%             f = fsamp(iF);
%             im = mr.readframe(f,'docrop',true);
%             nchan = size(im,3);
%             if nchan>1
%               error('Images must be grayscale.');
%             end
%             Isamp{iF} = im;
%           end
%           
%           try
%             Isamp = cat(2,Isamp{:});
%           catch ME
%             error('Cannot concatenate sampled movie frames: %s',ME.message);
%           end
%         
%           hgram = obj.preProcH0.hgram(:,ivw);
%           s = struct();
%           s.fsamp = fsamp;
%           s.hgram = hgram;
%           [...
%             s.lut,s.lutAL,...
%             Ibin,s.binC,s.binE,s.intens2bin,...
%             Jsamp,JsampAL,...
%             Jbin,JbinAL,...
%             s.hI,s.hJ,s.hJal,cI,cJ,cJal,...
%             s.Tbin,s.TbinAL,Tbininv,TbininvAL] = ...
%             HistEq.histMatch(Isamp,hgram,'docheck',docheck); %#ok<ASGLU>
%           obj.(PROPS.MFALUT){imov,ivw} = s;
%         
% %           t = toc;
% %           fprintf(1,'Elapsed time: %d sec\n',round(t));
%         end
%         
%         if tfWB
%           wbObj.endPeriod();
%         end
%       end
%     end
    
%     function movieHistEqLUTViz(obj)
%       % Viz: 
%       % - hgrams, cgrams, for each movie
%       %   - hilite central hgram/cgram
%       % - LUTs for all movs
%       % - raw image montage Isamp
%       % - sampled image montage
%       %   - Jsamp
%       %   - Jsamp2
%       
%       GT = false;
%       %[iMovs,gt] = mIdx.get();
%       mfaHEifos = obj.getMovieFilesAllHistEqLUTGTawareStc(GT);
%       nmovs = obj.nmovies;
%       nvw = obj.nview;
%       
%       if nmovs==0
%         warningNoTrace('No movies specified.');
%         return;
%       end
% 
%       for ivw=1:nvw
%         nbin = numel(mfaHEifos{1,ivw}.hgram);
%         hI = nan(nbin,nmovs);
%         hJ = nan(nbin,nmovs);
%         hJal = nan(nbin,nmovs);
%         Tbins = nan(nbin,nmovs);
%         TbinALs = nan(nbin,nmovs);
%         Isamp = [];
%         Jsamp = [];
%         JsampAL = [];
%         Isampyoffs = 0;
%         for imov=1:nmovs
%           ifo = mfaHEifos{imov,ivw};
%           hI(:,imov) = ifo.hI;
%           hJ(:,imov) = ifo.hJ;
%           hJal(:,imov) = ifo.hJal;
%           Tbins(:,imov) = ifo.Tbin;
%           TbinALs(:,imov) = ifo.TbinAL;
%           
%           mr = MovieReader;
%           mIdx = MovieIndex(imov);
%           obj.movieMovieReaderOpen(mr,mIdx,ivw);
%           nfrms = numel(ifo.fsamp);
%           Isampmov = cell(nfrms,1);
%           Jsampmov = cell(nfrms,1);
%           JsampALmov = cell(nfrms,1);
%           for iF=1:nfrms
%             f = ifo.fsamp(iF);
%             im = mr.readframe(f,'docrop',true);
%             nchan = size(im,3);
%             if nchan>1
%               error('Images must be grayscale.');
%             end
%             Isampmov{iF} = im;
%             Jsampmov{iF} = ifo.lut(uint32(im)+1);
%             JsampALmov{iF} = ifo.lutAL(uint32(im)+1);
%           end
%           Isampmov = cat(2,Isampmov{:});
%           Jsampmov = cat(2,Jsampmov{:});
%           JsampALmov = cat(2,JsampALmov{:});
%           % normalize ims here before concating to account for possible
%           % different classes
%           Isampmov = HistEq.normalizeGrayscaleIm(Isampmov);
%           Jsampmov = HistEq.normalizeGrayscaleIm(Jsampmov);
%           JsampALmov = HistEq.normalizeGrayscaleIm(JsampALmov);
%           
%           Isamp = [Isamp; Isampmov]; %#ok<AGROW>
%           Jsamp = [Jsamp; Jsampmov]; %#ok<AGROW>
%           JsampAL = [JsampAL; JsampALmov]; %#ok<AGROW>
%           Isampyoffs(end+1,1) = size(Isamp,1); %#ok<AGROW>
%         end
%         
%         hgram = ifo.hgram;
%         
%         cgram = cumsum(hgram);
%         cI = cumsum(hI);
%         cJ = cumsum(hJ);
%         cJal = cumsum(hJal);        
%         
%         x = 1:nbin;
%         figure('Name','imhists and cdfs');
% 
%         axs = mycreatesubplots(2,3,.1);
%         axes(axs(1,1));  %#ok<LAXES> 
%         plot(x,hI);
%         hold on;
%         grid on;
%         hLines = plot(x,hgram,'linewidth',2);
% %         legstr = sprintf('hI (%d movs)',nmovs);
%         legend(hLines,{'hgram'});
%         tstr = sprintf('Raw imhists (%d frms samp)',nfrms);
%         title(tstr,'fontweight','bold');
%         
%         axes(axs(1,2));  %#ok<LAXES> 
%         plot(x,hJ);
%         hold on;
%         grid on;
%         hLines = plot(x,hgram,'linewidth',2);  %#ok<NASGU> 
% %         legend(hLines,{'hgram'});
%         tstr = sprintf('Xformed imhists');
%         title(tstr,'fontweight','bold');
%         
%         axes(axs(1,3));  %#ok<LAXES> 
%         plot(x,hJal);
%         hold on;
%         grid on;
%         hLines = plot(x,hgram,'linewidth',2);  %#ok<NASGU> 
% %         legend(hLines,{'hgram'});
%         tstr = sprintf('Xformed (al) imhists');
%         title(tstr,'fontweight','bold');
% 
%         axes(axs(2,1));  %#ok<LAXES> 
%         plot(x,cI);
%         hold on;
%         grid on;
%         hLines = plot(x,cgram,'linewidth',2);
% %         legstr = sprintf('cI (%d movs)',nmovs);
%         legend(hLines,{'cgram'});
%         tstr = sprintf('cdfs');
%         title(tstr,'fontweight','bold');
%         
%         axes(axs(2,2));  %#ok<LAXES> 
%         hLines = plot(x,cJ);
%         hold on;
%         grid on;
%         hLines(end+1,1) = plot(x,cgram,'linewidth',2); %#ok<AGROW,NASGU> 
% %         legend(hLines,{'cJ','hgram'});
%         
%         axes(axs(2,3));  %#ok<LAXES> 
%         hLines = plot(x,cJal);
%         hold on;
%         grid on;
%         hLines(end+1,1) = plot(x,cgram,'linewidth',2);  %#ok<AGROW,NASGU> 
% %         legend(hLines,{'cJal','cgram'});
%         
%         linkaxes(axs(1,:));
%         linkaxes(axs(2,:));
% 
%         figure('Name','LUTs');
%         x = (1:size(Tbins,1))';
%         axs = mycreatesubplots(1,2,.1);
%         axes(axs(1)); %#ok<LAXES> 
%         plot(x,Tbins,'linewidth',2);
%         grid on;
%         title('Tbins','fontweight','bold');
%         
%         axes(axs(2)); %#ok<LAXES> 
%         plot(x,TbinALs,'linewidth',2);
%         grid on;
%         title('TbinALs','fontweight','bold');
%         
%         linkaxes(axs);
% 
%         figure('Name','Sample Image Montage');
%         axs = mycreatesubplots(3,1);
%         axes(axs(1)); %#ok<LAXES> 
%         imagesc(Isamp);
%         colormap gray
%         yticklocs = (Isampyoffs(1:end-1)+Isampyoffs(2:end))/2;
%         yticklbls = arrayfun(@(x)sprintf('mov%d',x),1:nmovs,'uni',0);
%         set(axs(1),'YTick',yticklocs,'YTickLabels',yticklbls);
%         set(axs(1),'XTick',[]);
%         tstr = sprintf('Raw images, view %d',ivw);
%         if GT
%           tstr = [tstr ' (gt)'];
%         end
%         title(tstr,'fontweight','bold');
%         clim0 = axs(1).CLim;
% 
%         axes(axs(2)); %#ok<LAXES> 
%         imagesc(Jsamp);
%         colormap gray
% %         colorbar
%         axs(2).CLim = clim0;
%         set(axs(2),'XTick',[],'YTick',[]);
%         tstr = sprintf('Converted images, view %d',ivw);
%         if GT
%           tstr = [tstr ' (gt)'];
%         end
%         title(tstr,'fontweight','bold');
%         
%         axes(axs(3)); %#ok<LAXES> 
%         imagesc(JsampAL);
%         colormap gray
% %         colorbar
%         axs(3).CLim = clim0;
%         set(axs(3),'XTick',[],'YTick',[]);
%         tstr = sprintf('Converted images (AL), view %d',ivw);
%         if GT
%           tstr = [tstr ' (gt)'];
%         end
%         title(tstr,'fontweight','bold');
%         
%         linkaxes(axs);
%       end
%     end  % function
    
%     function J = movieHistEqApplyLUTs(obj,I,mIdxs)
%       % Apply LUTs from .movieFilesAll*HistEqLUT to images
%       %
%       % I: [nmov x nview] cell array of raw grayscale images 
%       % mIdxs: [nmov] MovieIndex vector labeling rows of I
%       %
%       % J: [nmov x nview] cell array of transformed/LUT-ed images
%       
%       [nmov,nvw] = size(I);
%       assert(nvw==obj.nview);
%       assert(isa(mIdxs,'MovieIndex'));
%       assert(isvector(mIdxs) && numel(mIdxs)==nmov);
%       
%       J = cell(size(I));      
%       mIdxsUn = unique(mIdxs);
%       for mi = mIdxsUn(:)'
%         mfaHEifo = obj.getMovieFilesAllHistEqLUTMovIdx(mi);
%         assert(isrow(mfaHEifo));
%         rowsThisMov = find(mIdxs==mi);
%         for ivw=1:nvw
%           lut = mfaHEifo{ivw}.lutAL;
%           lutcls = class(lut);
%           for row=rowsThisMov(:)'
%             im = I{row,ivw};
%             assert(isa(im,lutcls));
%             J{row,ivw} = lut(uint32(im)+1);
%           end
%         end
%       end
%     end
      
%     function movieHistEqLUTEffectMontageHlp(obj,isGT,frm,wbObj)
%       % OBSOLETE
%       PROPS = obj.gtGetSharedPropsStc(isGT);
%       Tbins = obj.(PROPS.MFALUT);
%       assert(~isempty(Tbins));
%       nmovsets = obj.getnmoviesGTawareArg(isGT);
%       nvw = obj.nview;
%       for ivw=1:nvw
%         wbstr = sprintf('Sampling movies, view %d',ivw);
%         wbObj.startPeriod(wbstr,'shownumden',true,'denominator',nmovsets);
%         mr = MovieReader;
%         Isamp = cell(nmovsets,1);
%         Jsamp = cell(nmovsets,1);
%         for imov=1:nmovsets
%           wbObj.updateFracWithNumDen(imov);          
%           mIdx = MovieIndex(imov,isGT);
%           obj.movieMovieReaderOpen(mr,mIdx,ivw);
%           Isamp{imov} = mr.readframe(frm,'docrop',true);
%           if size(Isamp{imov},3)>1
%             error('Image must be grayscale.');
%           end
%           lut = Tbins{imov,ivw};
%           Jsamp{imov} = lut(uint32(Isamp{imov})+1);
%         end
%         wbObj.endPeriod();
%         
%         figure;
%         ax = axes;
%         montage(cat(4,Isamp{:}),'Parent',ax);
% %         colormap gray
% %         colorbar
% %         yticklocs = (Isampcatyoffs(1:end-1)+Isampcatyoffs(2:end))/2;
% %         yticklbls = arrayfun(@(x)sprintf('mov%d',x),1:nmovsets,'uni',0);
% %         set(ax,'YTick',yticklocs,'YTickLabels',yticklbls);
%         tstr = sprintf('Raw images, view %d',ivw);
%         if isGT
%           tstr = [tstr ' (gt)']; %#ok<AGROW>
%         end
%         title(tstr,'fontweight','bold');
%         clim0 = ax.CLim;
%         
%         figure;
%         ax = axes;
%         montage(cat(4,Jsamp{:}),'Parent',ax);
% %         colormap gray
% %         colorbar
% %         set(ax,'YTick',yticklocs,'YTickLabels',yticklbls);
%         tstr = sprintf('Corrected images, view %d',ivw);
%         if isGT
%           tstr = [tstr ' (gt)']; %#ok<AGROW>
%         end
%         title(tstr,'fontweight','bold');
%         ax.CLim = clim0;
%       end
%     end    
    
    
    % function movieMovieReaderOpen(obj,movRdr,mIdx,iView) % obj CONST
    %   % Take a movieReader object and open the movie (mIdx,iView), being 
    %   % faithful to obj as per:
    %   %   - .movieForceGrayScale 
    %   %   - .movieInvert(iView)
    %   %   - .preProcParams.BackSub
    %   %   - .cropInfo for (mIdx,iView) as appropriate
    %   %
    %   % movRdr: scalar MovieReader object
    %   % mIdx: scalar MovieIndex
    %   % iView: view index; used for .movieInvert
    % 
    %   ppPrms = obj.preProcParams;
    %   if ~isempty(ppPrms)
    %     bgsubPrms = ppPrms.BackSub;
    %     bgArgs = {'bgType',bgsubPrms.BGType,'bgReadFcn',bgsubPrms.BGReadFcn};
    %   else
    %     bgArgs = {};
    %   end
    % 
    %   movfname = getMovieFilesAllFullMovIdx(obj, mIdx);
    %   movRdr.preload = obj.movieReadPreLoadMovies; % must occur before .open()
    %   movRdr.open(movfname{iView},bgArgs{:});
    %   movRdr.forceGrayscale = obj.movieForceGrayscale;
    %   movRdr.flipVert = obj.movieInvert(iView);      
    %   cInfo = obj.getMovieFilesAllCropInfoMovIdx(mIdx);
    %   if ~isempty(cInfo)
    %     movRdr.setCropInfo(cInfo(iView));
    %   else
    %     movRdr.setCropInfo([]);
    %   end      
    % end
    
    function v = getMovieFilesAllFullMovIdx(obj, mIdx)
      % mIdx: MovieIndex vector
      % v: [numel(mIdx)xnview] movieFilesAllFull/GT 
      % Does not mutate obj.
      
      assert(isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      n = numel(iMov);
      v = cell(n,obj.nview);
      mfaf = obj.movieFilesAllFull;
      mfafGT = obj.movieFilesAllGTFull;
      for i=1:n
        if gt(i)
          v(i,:) = mfafGT(iMov(i),:);
        else
          v(i,:) = mfaf(iMov(i),:);
        end
      end
    end
    
    function movieSetMovieReadPreLoadMovies(obj,tf)
      tf0 = obj.movieReadPreLoadMovies;
      if tf0~=tf && (obj.nmovies>0 || obj.nmoviesGT>0)        
        warningNoTrace('Project already has movies. Checking movie lengths under preloading.');
        obj.hlpCheckWarnMovieNFrames('movieFilesAll','movieInfoAll',true,'');
        obj.hlpCheckWarnMovieNFrames('movieFilesAllGT','movieInfoAllGT',true,' (gt)');
      end
      obj.movieReadPreLoadMovies = tf;
    end

    function hlpCheckWarnMovieNFrames(obj,movFileFld,movIfoFld,preload,gtstr)
      mr = MovieReader;
      mr.preload = preload;
      movfiles = obj.(movFileFld);
      movifo = obj.(movIfoFld);
      [nmov,nvw] = size(movfiles);
      for imov=1:nmov
        fprintf('Movie %d%s\n',imov,gtstr);
        for ivw=1:nvw
          mr.open(movfiles{imov,ivw});
          nf = mr.nframes;
          nf0 = movifo{imov,ivw}.nframes;
          if nf~=nf0
            warningNoTrace('Movie %d%s view %d, old nframes=%d, new nframes=%d.',...
              imov,gtstr,ivw,nf0,nf);
          end
        end
      end
    end  % function
    
  end  % methods
  
  %% Trx
  methods

    % TrxCache notes
    % To avoid repeated loading of trx data from filesystem, we cache
    % previously seen/loaded trx data in .trxCache. The cache is never
    % updated as it is assumed that trxfiles on disk do not mutate over the
    % course of a single APT session.
    
    function [trx,frm2trx] = getTrx(obj,filename,varargin)
      % [trx,frm2trx] = getTrx(obj,filename,[nfrm])
      % Get trx data for iMov/iView from .trxCache; load from filesys if
      % necessary      
      [trx,frm2trx] = Labeler.getTrxCacheStc(obj.trxCache,filename,varargin{:});
    end
    
    function clearTrxCache(obj)
      % Forcibly clear .trxCache
      obj.trxCache = containers.Map();
    end

    function tia = getTrxInfoMovIdx(obj,iMov)
      [movi,gt] = iMov.get;
      if gt,
        tia = obj.trxInfoAll(:,movi);
      else
        tia = obj.trxInfoAllGT(:,movi);
      end
    end

    function frm2trx = getFrm2Trx(obj,movi,varargin)
      [tgts,gt] = myparse(varargin,'tgts','unset','gt',[]);
      if isa(movi,'MovieIndex')
        [movi,gt] = movi.get;
      elseif isempty(gt)
        gt = false;
      end
      PROPS = obj.gtGetSharedPropsStc(gt);
      mia = obj.(PROPS.MIA)(movi,:);
      tia = obj.(PROPS.TIA)(movi,:);
      nframes = max(cellfun(@(x) x.nframes,mia));
      ntgts = max(cellfun(@(x) x.ntgts,tia));
      if ischar(tgts),
        tgts = 1:ntgts;
      end
      frm2trx = false(nframes,numel(tgts),obj.nview);
      if obj.hasTrx,
        for viewi = 1:obj.nview,
          for i = 1:numel(tgts),
            frm2trx(tia{viewi}.firstframes(tgts(i)):tia{viewi}.endframes(tgts(i)),i,viewi) = true;
          end
        end
      else
        frm2trx(:) = true;
      end
      frm2trx = all(frm2trx,3);
    end


  end
  methods (Static)
    
    function [trx,frm2trx] = getTrxCacheStc(trxCache,filename,nfrm)
      % Get trx data for iMov/iView from .trxCache; load from filesys if
      % necessary
      %
      % trxCache: containers.Map
      % filename: fullpath to trxfile
      % nfrm: total number of frames in associated movie
      %
      % trx: struct array
      % frm2trx: [nfrm x ntrx] logical. frm2trx(f,i) is true if trx(i) is
      %  live @ frame f
      
      if nargin < 3,
        nfrm = [];
      end
      
      if trxCache.isKey(filename)
        s = trxCache(filename);
        trx = s.trx;
        frm2trx = s.frm2trx;
        if isempty(nfrm),
          nfrm = size(frm2trx,1);
        end
        szassert(frm2trx,[nfrm numel(trx)]);
          
      else
        if exist(filename,'file')==0
          % Currently user will have to navigate to iMov to fix
          error('Labeler:file','Cannot find trxfile ''%s''.',filename);
        end
        tmp = load(filename,'-mat','trx');
        if isfield(tmp,'trx')
          trx = tmp.trx;
          frm2trx = Labeler.trxHlpComputeF2t(nfrm,trx);          
          trxCache(filename) = struct('trx',trx,'frm2trx',frm2trx); %#ok<NASGU>
          RC.saveprop('lbl_lasttrxfile',filename);
        else
          warningNoTrace('Labeler:trx',...
            'No ''trx'' variable found in trxfile %s.',filename);
          trx = [];
          frm2trx = [];
        end
      end
    end
    
    function [trxCell,frm2trxCell,frm2trxTotAnd] = ...
                          getTrxCacheAcrossViewsStc(trxCache,filenames,nfrm)
      % Similar to getTrxCacheStc, but for an array of filenames and with
      % some checks
      %
      % filenames: cellstr of trxfiles containing trx with the same total 
      %   frame number, eg trx across all views in a single movieset
      % nfrm: common number of frames
      % 
      % trxCell: cell array, same size as filenames, containing trx
      %   structarrays
      % frm2trxCell: cell array, same size as filenames, containing
      %   frm2trx arrays for each trx
      % frm2trxTotAnd: AND(frm2trxCell{:})
      
      assert(~isempty(filenames));
      
      [trxCell,frm2trxCell] = cellfun(@(x)Labeler.getTrxCacheStc(trxCache,x,nfrm),...
          filenames,'uni',0);
      cellfun(@(x)assert(numel(x)==numel(trxCell{1})),trxCell);

      % In multiview multitarget projs, the each view's trx must contain
      % the same number of els and these elements must correspond across
      % views.
      nTrx = numel(filenames);
      if nTrx>1 && isfield(trxCell{1},'id')
        trxids = cellfun(@(x)[x.id],trxCell,'uni',0);
        assert(isequal(trxids{:}),'Trx ids differ.');
      end
      
      frm2trxTotAnd = cellaccumulate(frm2trxCell,@and);
    end
  end
  methods
        
    function trxSet(obj,trx)
      % Set the trajectories for the current movie (.trx).
      %
      % - Call trxSave if you want the new trx to persist somewhere. 
      % Otherwise it will be gone once you switch movies.
      % - This leaves .trxfile and .trxFilesAll unchanged. 
      % - You probably want to call setFrameAndTarget() after this.
      % - This method CANNOT CHANGE the number of trx-- that would require
      % updating .labeledpos and .suspScore as well.
      % - (TODO) Warnings will be thrown if the various .firstframe/.endframes
      % change in such a way that existing labels now fall outside their
      % trx's existence.
      
      if ~obj.isinit
        % AL: seems bizzare, would you ever swap out the trx for the
        % current movie?
        assert(isequal(numel(trx),obj.nTargets)); 

        % TODO: check for labels that are "out of bounds" for new trx
      end
      
      obj.trx = trx;

      if obj.hasTrx
        if ~isfield(obj.trx,'id')
          for i = 1:numel(obj.trx)
            obj.trx(i).id = i;
          end
        end
        %maxID = max([obj.trx.id]);
      else
        %maxID = -1;
      end
%       id2t = nan(maxID+1,1);
%       for i = 1:obj.nTrx
%         id2t(obj.trx(i).id+1) = i;
%       end
%       obj.trxIdPlusPlus2Idx = id2t;
      if isnan(obj.nframes)
        obj.frm2trx = [];
      else
        % Can get this from trxCache
        obj.frm2trx = Labeler.trxHlpComputeF2t(obj.nframes,trx);
      end
      
      obj.currImHud.updateReadoutFields('hasTgt',obj.hasTrx || obj.maIsMA);
      
      obj.notify('didSetTrx') ;
    end
       
    function [sf,ef] = trxGetFrameLimits(obj)
      % frm2trx(iFrm,iTgt) binary, whether target iTgt is alive at frame
      % iFrm
      iTgt = obj.currTarget;
      sf = find(obj.frm2trx(:,iTgt),1);
      ef = find(obj.frm2trx(:,iTgt),1,'last');
    end
    
    function tf = trxCheckFramesLive(obj,frms)
      % Check that current target is live for given frames
      %
      % frms: [n] vector of frame indices
      %
      % tf: [n] logical vector
      
      iTgt = obj.currTarget;
      if obj.hasTrx
        tf = obj.frm2trx(frms,iTgt);
      else
        tf = true(numel(frms),1);
      end
    end
    
    function trxCheckFramesLiveErr(obj,frms)
      % Check that current target is live for given frames; err if not
      
      tf = obj.trxCheckFramesLive(frms);
      if ~all(tf)
        error('Labeler:target',...
          'Target %d is not live during specified frames.',obj.currTarget);
      end
    end   
    
    function trxFilesUnMacroize(obj)
      obj.trxFilesAll = obj.trxFilesAllFull;
      obj.trxFilesAllGT = obj.trxFilesAllGTFull;
    end

    function disarmProgressMeter(obj) 
      obj.progressMeter_.disarm() ;
    end

    function [tfok,tblBig] = hlpTargetsTableUIgetBigTable(obj)
      % Generate the big target summary table.
      obj.progressMeter_.arm('title', 'Target Summary Table') ;
      cleaner = onCleanup(@()(obj.disarmProgressMeter())) ;
      tblBig = obj.trackGetBigLabeledTrackedTable_() ;
      tfok = ~(obj.progressMeter_.wasCanceled) ;
      % if ~tfok, tblBig indeterminate
    end

    % function hlpTargetsTableUIupdate(obj,navTbl)
    %   [tfok,tblBig] = obj.hlpTargetsTableUIgetBigTable();
    %   if tfok
    %     navTbl.setData(obj.trackGetSummaryTable(tblBig));
    %   end
    % end

%     function targetsTableUI(obj)
%       [tfok,tblBig] = obj.hlpTargetsTableUIgetBigTable();
%       if ~tfok
%         return;
%       end
% 
%       tblSumm = obj.trackGetSummaryTable(tblBig);
%       hF = figure('Name','Target Summary (click row to navigate)',...
%         'MenuBar','none','Visible','off');
%       hF.Position(3:4) = [1280 500];
%       centerfig(hF,obj.controller_.mainFigure_);
%       hPnl = uipanel('Parent',hF,'Position',[0 .08 1 .92],'Tag','uipanel_TargetsTable');
%       BTNWIDTH = 100;
%       DXY = 4;
%       btnHeight = hPnl.Position(2)*hF.Position(4)-2*DXY;
%       btnPos = [hF.Position(3)-BTNWIDTH-DXY DXY BTNWIDTH btnHeight];      
%       hBtn = uicontrol('Style','pushbutton','Parent',hF,...
%         'Position',btnPos,'String','Update',...
%         'fontsize',12);
%       FLDINFO = {
%         'mov' 'Movie' 'integer' 30
%         'iTgt' 'Target' 'integer' 30
%         'trajlen' 'Traj. Length' 'integer' 45
%         'frm1' 'Start Frm' 'integer' 30
%         'nFrmLbl' '# Frms Lbled' 'integer' 60
%         'nFrmTrk' '# Frms Trked' 'integer' 60
%         'nFrmImported' '# Frms Imported' 'integer' 90
%         'nFrmLblTrk' '# Frms Lbled&Trked' 'integer' 120
%         'lblTrkMeanErr' 'Track Err' 'float' 60
%         'nFrmLblImported' '# Frms Lbled&Imported' 'integer'  120
%         'lblImportedMeanErr' 'Imported Err' 'float' 60
%         'nFrmXV' '# Frms XV' 'integer' 40
%         'xvMeanErr' 'XV Err' 'float' 40};
%       tblfldsassert(tblSumm,FLDINFO(:,1));
%       nt = NavigationTable(hPnl,[0 0 1 1],...
%         @(row,rowdata)obj.setMFT(rowdata.mov,rowdata.frm1,rowdata.iTgt),...
%         'ColumnName',FLDINFO(:,2)',...
%         'ColumnFormat',FLDINFO(:,3)',...
%         'ColumnPreferredWidth',cell2mat(FLDINFO(:,4)'));
% %      jt = nt.jtable;
%       nt.setData(tblSumm);
% %      cr.setHorizontalAlignment(javax.swing.JLabel.CENTER);
% %      h = jt.JTable.getTableHeader;
% %      h.setPreferredSize(java.awt.Dimension(225,22));
% %      jt.JTable.repaint;
% 
%       hF.UserData = nt;
%       hBtn.Callback = @(s,e)obj.hlpTargetsTableUIupdate(nt);
%       hF.Units = 'normalized';
%       hBtn.Units = 'normalized';
%       hF.Visible = 'on';
% 
%       obj.addDepHandle(hF);
%     end
    
    % initTrxInfo(obj)
    % read in trx files and store number of targets and start and end
    % frames. This data is now kept in the lbl file so that we don't have
    % to keep reading in trx files to count number of targets.
    % added by KB 20210626
    function initTrxInfo(obj)
      if numel(obj.trxInfoAll) == numel(obj.trxFilesAll) && ...
          numel(obj.trxInfoAllGT) == numel(obj.trxFilesAllGT),
        return;
      end
      fprintf('Moderning lbl file to store info about trx files, this may take a minute...\n');
      fprintf('After this is done, please save your new lbl file. This will not need to be run again.\n');

      obj.initTrxInfoHelper(Labeler.gtGetSharedPropsStc(false));
      obj.initTrxInfoHelper(Labeler.gtGetSharedPropsStc(true));
    end
    
    function initTrxInfoHelper(obj,PROPS)
      
      if numel(obj.(PROPS.TIA)) == numel(obj.(PROPS.TFA)),
        return;
      end

      obj.(PROPS.TIA) = cell(size(obj.(PROPS.TFA)));
      tFilesFull = obj.(PROPS.TFAF);
      tFiles = obj.(PROPS.TFA);
      for i = 1:numel(obj.(PROPS.TFA)),
        trxinfo = struct;
        nframes = obj.(PROPS.MIA){i}.nframes;
        if isempty(obj.(PROPS.TFA){i}),
          nTgt = 1;
          trxinfo.ntgts = nTgt;
          trxinfo.firstframes = 1;
          trxinfo.endframes = nframes;
        else
          tFileFull = tFilesFull{i};
          if ~(isempty(tFileFull) || exist(tFileFull,'file')>0)
            FSPath.throwErrFileNotFoundMacroAware(tFiles{i},tFileFull,'trxfile');
          end
          tmptrx = obj.getTrx(tFileFull,nframes);
          nTgt = numel(tmptrx);
          trxinfo.ntgts = nTgt;
          trxinfo.firstframes = [tmptrx.firstframe];
          trxinfo.endframes = [tmptrx.endframe];
        end
        obj.(PROPS.TIA){i} = trxinfo;
      end
    end

    function [trxinfo,tmptrx] = GetTrxInfo(obj,tFileFull,nframes)
      
      trxinfo = struct;
      if ~isempty(tFileFull)
        if nargin < 3,
          nframes = [];
        end
        tmptrx = obj.getTrx(tFileFull,nframes);
        nTgt = numel(tmptrx);
        trxinfo.ntgts = nTgt;
        trxinfo.firstframes = [tmptrx.firstframe];
        trxinfo.endframes = [tmptrx.endframe];
      else
        nTgt = 1;
        trxinfo.ntgts = nTgt;
        trxinfo.firstframes = 1;
        trxinfo.endframes = nframes;
      end
    end

    function sanityCheckTrxInfo(obj)
      assert(size(obj.trxInfoAll,1)==obj.nmovies);
      assert(size(obj.trxInfoAllGT,1)==obj.nmoviesGT);
      for i = 1:numel(obj.movieInfoAll),
        s1 = mat2str(ind2subv(size(obj.movieInfoAll),i));
        if obj.movieInfoAll{i}.nframes==max(obj.trxInfoAll{i}.endframes),
          s2 = 'EQUAL';
        elseif obj.movieInfoAll{i}.nframes<max(obj.trxInfoAll{i}.endframes),
          s2 = 'LESS THAN';
        else
          s2 = 'GREATER THAN';
        end
        fprintf('Movie %s: nframes = %d %s max(endframes) = %d\n',s1,obj.movieInfoAll{i}.nframes,s2,max(obj.trxInfoAll{i}.endframes));
      end
      for i = 1:numel(obj.movieInfoAllGT),
        s1 = mat2str(ind2subv(size(obj.movieInfoAllGT),i));
        fprintf('GT Movie %s: nframes = %d %s max(endframes) = %d\n',s1,obj.movieInfoAllGT{i}.nframes,s2,max(obj.trxInfoAllGT{i}.endframes));
      end
    end
    
  end
  
  methods (Static)
    function f2t = trxHlpComputeF2t(nfrm,trx)
      % Compute f2t array for trx structure
      %
      % nfrm: number of frames in movie corresponding to trx
      % trx: trx struct array
      %
      % f2t: [nfrm x nTrx] logical. f2t(f,iTgt) is true iff trx(iTgt) is
      % live at frame f.
      
      if isempty(nfrm),
        nfrm = max([trx.endframe]);
      end
      nTrx = numel(trx);
      f2t = false(nfrm,nTrx);
      for iTgt=1:nTrx
        frm0 = trx(iTgt).firstframe;
        frm1 = trx(iTgt).endframe;
        f2t(frm0:frm1,iTgt) = true;
      end
    end

    function sMacro = trxFilesMacros(movFileFull)
      sMacro = struct();
      sMacro.movdir = fileparts(movFileFull);
    end

    function trxFilesFull = trxFilesLocalize(trxFiles,movFilesFull)
      % Localize trxFiles based on macros+movFiles
      %
      % trxFiles: can be char or cellstr
      % movFilesFull: like trxFiles, if cellstr same size as trxFiles.
      %   Should NOT CONTAIN macros
      %
      % trxFilesFull: char or cellstr, if cellstr same size as trxFiles
      
      tfChar = ischar(trxFiles);
      trxFiles = cellstr(trxFiles);
      movFilesFull = cellstr(movFilesFull);
      szassert(trxFiles,size(movFilesFull));
      trxFilesFull = cell(size(trxFiles));
      
      for i=1:numel(trxFiles)
        sMacro = Labeler.trxFilesMacros(movFilesFull{i});
        trxFilesFull{i} = FSPath.fullyLocalizeStandardizeChar(trxFiles{i},sMacro);
      end
      FSPath.warnUnreplacedMacros(trxFilesFull);
      
      if tfChar
        trxFilesFull = trxFilesFull{1};
      end
    end
  end  
  methods % show*
        
    function setShowTrx(obj,tf)
      assert(isscalar(tf) && islogical(tf));
      obj.showTrx = tf;
      obj.updateShowTrx();
    end

    function setShowOccludedBox(obj,tf)
      assert(isscalar(tf) && islogical(tf));
      obj.showOccludedBox = tf;
    end
    
    function setShowTrxCurrTargetOnly(obj,tf)
      assert(isscalar(tf) && islogical(tf));
      obj.showTrxCurrTargetOnly = tf;
      obj.updateShowTrx();
    end
    
    function setShowTrxIDLbl(obj,tf)
      assert(isscalar(tf) && islogical(tf));
      obj.showTrxIDLbl = tf;
      obj.updateShowTrx();
    end
    
    function updateShowTrx(obj)
      if obj.maIsMA
        % Consdider/todo: update showtrx/traj for MA        
        % tv = obj.labeledpos2trkViz.      
        %
        % need to decide whether trx/traj should be separately toggleable.
        % currently there is an overall tfHideViz which apples to both
        % tv.tvmt and tv.tvtrx. if we enabled this, there would be another
        % flag controlling display of trx/traj; tv.tvtrx would only show
        % viz when both tfHideViz==false and tfShowTraj==true.
      elseif obj.hasTrx
        obj.notify('updateTrxSetShowTrue') ;
      end
    end
            
    function tfShow = which_trx_are_showing(obj)
      ntgts = obj.nTrx;
      if obj.showTrx
        if obj.showTrxCurrTargetOnly
          tfShow = false(ntgts,1);
          iTgtCurr = obj.currTarget;
          tfShow(iTgtCurr) = true;
        else
          tfShow = true(ntgts,1);
        end
      else
        tfShow = false(ntgts,1);
      end      
    end

    function setSkeletonEdges(obj,se)
      old_se = obj.skeletonEdges ;
      obj.skeletonEdges = se;    
      % If the skeleton is going from nonempty to empty, un-show the skeleton
      % If the skeleton is going from empty to nonempty, show the skeleton
      if isempty(se) && ~isempty(old_se) && obj.showSkeleton ,
        obj.showSkeleton = false ;
      elseif ~isempty(se) && isempty(old_se) && ~obj.showSkeleton ,
        obj.showSkeleton = true ;
      end
      obj.lblCore.updateSkeletonEdges();
      tv = obj.labeledpos2trkViz ;
      if ~isempty(tv)
        tv.initAndUpdateSkeletonEdges(se);
      end
    end

    function setShowSkeleton(obj,tf)
      tf = logical(tf);
      obj.showSkeleton = tf;
      obj.lblCore.updateShowSkeleton();
      tv = obj.labeledpos2trkViz;
      if ~isempty(tv)
        tv.setShowSkeleton(tf);
      end
      dt = obj.tracker;
      if isempty(dt)
        return
      end
      tv = dt.trkVizer;
      if ~isempty(tv)
        tv.setShowSkeleton(tf);
      end
    end

    function setShowMaRoi(obj,tf)
      obj.showMaRoi = logical(tf);
      if obj.labelMode==LabelMode.MULTIANIMAL
        lc = obj.lblCore;
        lc.tv.setShowPches(tf); % lc should be a lblCoreSeqMA      
      end
    end

    function setShowMaRoiAux(obj,tf)
      obj.showMaRoiAux = logical(tf);
      if obj.labelMode==LabelMode.MULTIANIMAL
        lc = obj.lblCore;
        lc.roiSetShow(tf); % lc should be a lblCoreSeqMA      
      end
    end

    function setFlipLandmarkMatches(obj,matches)
      obj.flipLandmarkMatches = matches;
    end

    function setSkelHead(obj,head)
      obj.skelHead = head;
    end

    function setSkelTail(obj,tail)
      obj.skelTail = tail;
    end    

    function setSkelNames(obj,names)
      obj.skelNames = names;
    end
        
  end
  
  %% Labeling
  methods
    
    function labelingInit(obj,varargin)
      % Create LabelCore and call labelCore.init() based on current 
      % .labelMode, .nLabelPoints, .labelPointsPlotInfo, .labelTemplate  
      % For calibrated labelCores, can also require .currMovie, 
      % .viewCalibrationData, .vewCalibrationDataProjWide to be properly 
      % initted
            
      [lblmode,dosettemplate] = myparse(varargin,...
        'labelMode',[],...
        'dosettemplate',true...
        );

      tfLblModeChange = ~isempty(lblmode);
      if tfLblModeChange
        if ischar(lblmode)
          lblmode = LabelMode.(lblmode);
        else
          assert(isa(lblmode,'LabelMode'));
        end
      else
        lblmode = obj.labelMode;
      end
     
      nPts = obj.nLabelPoints;
      lblPtsPlotInfo = obj.labelPointsPlotInfo;
      lblPtsPlotInfo.Colors = obj.LabelPointColors;
      %template = obj.labelTemplate;
      
      lc = obj.lblCore;
      if ~isempty(lc)
        % AL 20150731. Possible/probable MATLAB bug. Existing LabelCore 
        % object should delete and clean itself up when obj.lblCore is 
        % reassigned below. The precise sequence of events is complex 
        % because various callbacks in uicontrols hold refs to the 
        % LabelCore (the callbacks get reassigned in lblCore.init though).
        %
        % However, without the explicit deletion in this branch, the 
        % existing lblCore did not get cleaned up, leading to two sets of 
        % label pts on axes_curr; moreover, when this occurred and the 
        % Labeler GUI was killed (with 'X' button in upper-right), I 
        % consistently got a segv on Win7/64, R2014b.
        %
        % With this explicit cleanup the segv goes away and of course the
        % extra labelpts are fixed as well.
        
        % AL: Need strategy for persisting/carrying lblCore state between
        % movies. newMovie prob doesn't need to call labelingInit.
        hideLabelsPrev = lc.hideLabels;
        if isprop(lc,'streamlined')
          streamlinedPrev = lc.streamlined;
        end
        delete(lc);
        obj.lblCore = [];
      else
        hideLabelsPrev = false;
      end
      obj.lblCore = LabelCore.createSafe(obj.controller_, lblmode) ;  % hack
      obj.lblCore.init(nPts,lblPtsPlotInfo);
      if hideLabelsPrev
        obj.lblCore.labelsHide();
      end
      if isprop(obj.lblCore,'streamlined') && exist('streamlinedPrev','var')>0
        obj.lblCore.streamlined = streamlinedPrev;
      end
      
      % labelmode-specific inits
      if dosettemplate && lblmode==LabelMode.TEMPLATE
        obj.labelingInitTemplate();
      end
      if obj.lblCore.supportsCalibration
        vcd = obj.viewCalibrationDataCurrent;
        if isempty(vcd)
%           warningNoTrace('Labeler:labelingInit',...
%             'No calibration data loaded for calibrated labeling.');
        else
          obj.lblCore.projectionSetCalRig(vcd);
        end
      end
      obj.labelMode = lblmode;
      
      obj.setShowMaRoi(obj.showMaRoi);
      obj.setShowMaRoiAux(obj.showMaRoiAux);
      
      obj.initVirtualPrevAxesLabelPointViz_(lblPtsPlotInfo);
      obj.syncPrevAxesVirtualLabels_();
      notify(obj, 'updatePrevAxesLabels');
      
      if tfLblModeChange
        % sometimes labelcore need this kick to get properly set up
        obj.labelsUpdateNewFrame(true);
      end
    end  % function
    
    function labelingInitTemplate(obj)
      % Call .lblCore.setTemplate based on a labeled frame in the proj
      % Uses .trx as appropriate
      [tffound,iMov,frm,iTgt,xyLbl] = obj.labelFindOneLabeledFrameEarliest();
      if tffound
        if obj.hasTrx
          trxfname = obj.trxFilesAllFullGTaware{iMov};
          movIfo = obj.movieInfoAllGTaware{iMov};
          trximov = obj.getTrx(trxfname,movIfo.nframes);
          trxI = trximov(iTgt);
          idx = trxI.off + frm;
          tt = struct('loc',[trxI.x(idx) trxI.y(idx)],...
            'theta',trxI.theta(idx),'pts',xyLbl);
        else
          tt = struct('loc',[nan nan],'theta',nan,'pts',xyLbl);
        end
        obj.lblCore.setTemplate(tt);
      else
        obj.lblCore.setRandomTemplate();
      end
    end
    
    function [tffound,iMov,frm,iTgt,xyLbl,mints] = labelFindOneLabeledFrameEarliest(obj)
      % Look only in labeledposGTaware, and look for the earliest labeled 
      % frame.  Does not mutate obj.      
      if obj.gtIsGTMode
        lpos = obj.labelsGT;
      else
        lpos = obj.labels;
      end     
      [tffound,iMov,frm,iTgt,xyLbl,mints] = findEarliestLabeledFrameFromLabels(lpos) ;
    end  % function
    
    %%% labelpos
        
%     function labelPosClear(obj)
%       x = rand;
%       if x > 0.5
%         obj.labelPosClear_Old();
%         obj.labelPosClear_New();
%       else
%         obj.labelPosClear_New();
%         obj.labelPosClear_Old();        
%       end
%     end
%     function labelPosClear_Old(obj)
%       % Clear all labels AND TAGS for current movie/frame/target
%       
%       iMov = obj.currMovie;
%       iFrm = obj.currFrame;
%       iTgt = obj.currTarget;
%       
%       PROPS = obj.gtGetSharedProps();
%       x = obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTgt);
%       if all(isnan(x(:)))
%         % none; short-circuit set to avoid triggering .labeledposNeedsSave
%       else        
%         obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTgt) = nan;
%         obj.labeledposNeedsSave = true;
%       end
%       
%       ts = now;
%       obj.(PROPS.LPOSTS){iMov}(:,iFrm,iTgt) = ts;
%       obj.(PROPS.LPOSTAG){iMov}(:,iFrm,iTgt) = false;
%       if ~obj.gtIsGTMode
% %         obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
%         obj.lastLabelChangeTS = ts;
%       end
%     end

    function labelPosClear(obj)
      % Clear all labels AND TAGS for current movie/frame/target
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      [s,tfchanged] = Labels.rmFT(s,iFrm,iTgt);
      if tfchanged
        % avoid triggering .labeledposNeedsSave if poss
        obj.(PROPS.LBL){iMov} = s;
        obj.labeledposNeedsSave = true;
      end
      
      if ~obj.gtIsGTMode
        %obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
        ts = now;
        obj.lastLabelChangeTS = ts;
      end
    end
    
    function labelPosClearPoints(obj,pts)
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      [s,tfchanged] = Labels.rmFTP(s,iFrm,iTgt,pts);
      if tfchanged,
        obj.(PROPS.LBL){iMov} = s;
        obj.labeledposNeedsSave = true;
      end
    end
    
    function labelPosAddLandmarks(obj,new2oldpt)
      % for all movies, for both training labels and gt labels, add new
      % landmarks
      
      for i = 1:numel(obj.labels),
        obj.labels{i} = Labels.remapLandmarks(obj.labels{i},new2oldpt);
      end
      for i = 1:numel(obj.labelsGT),
        obj.labelsGT{i} = Labels.remapLandmarks(obj.labelsGT{i},new2oldpt);
      end
      
    end
    
    function ntgts = labelPosClearWithCompact_New(obj)
      % Clear all labels AND TAGS for current movie/frame/target;
      % AND compactify labels. for MA only

      assert(obj.maIsMA);
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;      
      
      if iTgt==0
        warning('No targets.');
      end
      
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      [s,tfchanged] = Labels.rmFT(s,iFrm,iTgt);
      [s,tfchanged2,ntgts] = Labels.compact(s,iFrm);
      if tfchanged || tfchanged2
        % avoid triggering .labeledposNeedsSave if poss
        obj.(PROPS.LBL){iMov} = s;
        obj.labeledposNeedsSave = true;
      end
      
      if ~obj.gtIsGTMode
        %obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
        ts = now;
        obj.lastLabelChangeTS = ts;
      end      
    end
    
%     function labelPosClearI(obj,iPt)
%       x = rand;
%       if x > 0.5
%         obj.labelPosClearI_Old(iPt);
%         obj.labelPosClearI_New(iPt);
%       else
%         obj.labelPosClearI_New(iPt);
%         obj.labelPosClearI_Old(iPt);
%       end
%     end
%     function labelPosClearI_Old(obj,iPt)
%       % Clear labels and tags for current movie/frame/target, point iPt
%       
%       iMov = obj.currMovie;
%       iFrm = obj.currFrame;
%       iTgt = obj.currTarget;
%       
%       PROPS = obj.gtGetSharedProps();
%       xy = obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt);
%       if all(isnan(xy))
%         % none; short-circuit set to avoid triggering .labeledposNeedsSave
%       else
%         obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt) = nan;
%         obj.labeledposNeedsSave = true;
%       end
%       
%       ts = now;
%       obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = ts;
%       obj.(PROPS.LPOSTAG){iMov}(iPt,iFrm,iTgt) = false;
%       if ~obj.gtIsGTMode
% %         obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
%         obj.lastLabelChangeTS = ts;
%       end
%     end

    function labelPosClearI(obj,iPt)
      % Clear labels and tags for current movie/frame/target, point iPt
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov}(iPt,:,iFrm,iTgt);
      [s,tfchanged] = Labels.clearFTI(s,iFrm,iTgt,iPt);
      if tfchanged
        % avoid triggering .labeledposNeedsSave if poss
        obj.(PROPS.LBL){iMov} = s;
        obj.labeledposNeedsSave = true;
      end
      
      %obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = ts;
      %obj.(PROPS.LPOSTAG){iMov}(iPt,iFrm,iTgt) = false;
      if ~obj.gtIsGTMode
        ts = now;
        %obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end
    end
    
    function [tf,lpos,lpostag] = labelPosIsLabeled(obj,iFrm,iTrx,varargin)
      % For current movie. Labeled includes fullyOccluded
      %
      % tf: scalar logical
      % lpos: [nptsx2] xy coords for iFrm/iTrx
      % lpostag: [npts] logical array 
      % This method does not mutate obj.
      [iMov,gtmode] = myparse(varargin,'iMov',obj.currMovie,'gtmode',obj.gtIsGTMode);
      PROPS = obj.gtGetSharedPropsStc(gtmode);
      s = obj.(PROPS.LBL){iMov};
      [tf,p,occ] = Labels.isLabeledFT(s,iFrm,iTrx);
      lpos = reshape(p,[numel(p)/2 2]);
      lpostag = occ;
    end 

    function [tfperpt,lpos,lpostag] = labelPosIsPtLabeled(obj,iFrm,iTrx)
      % For current movie. Labeled includes fullyOccluded
      %
      % tf: scalar logical
      % lpos: [nptsx2] xy coords for iFrm/iTrx
      % lpostag: [npts] logical array 
      
      iMov = obj.currMovie;
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      [tfperpt,p,occ] = Labels.isLabeledPerPtFT(s,iFrm,iTrx);
      lpos = reshape(p,[numel(p)/2 2]);
      lpostag = occ;
    end 
    
    function [iTgts] = labelPosIsLabeledFrm(obj,iFrm)
      % For current movie, find labeled targets in iFrm (if any)
      %
      % tf: scalar logical
      % iTgts:
      
      iMov = obj.currMovie;
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      iTgts = Labels.isLabeledF(s,iFrm);
    end 
        
%     function [tf0] = labelPosIsLabeledMov(obj,iMov)
%       x = rand;
%       if x > 0.5
%         [tf1] = obj.labelPosIsLabeledMov_Old(iMov);
%         [tf0] = obj.labelPosIsLabeledMov_New(iMov);
%       else
%         [tf0] = obj.labelPosIsLabeledMov_New(iMov);
%         [tf1] = obj.labelPosIsLabeledMov_Old(iMov);
%       end
%       assert(isequal(tf0,tf1));
%     end
%     function tf = labelPosIsLabeledMov_Old(obj,iMov)
%       % iMov: movie index (row index into .movieFilesAll)
%       %
%       % tf: [nframes-for-iMov], true if any point labeled in that mov/frame
% 
%       %#%MVOK
%       
%       ifo = obj.movieInfoAll{iMov,1};
%       nf = ifo.nframes;
%       lpos = obj.labeledpos{iMov};
%       lposnnan = ~isnan(lpos);
%       
%       tf = arrayfun(@(x)nnz(lposnnan(:,:,x,:))>0,(1:nf)');
%     end

    function tf = labelPosIsLabeledMov(obj,iMov)
      ifo = obj.movieInfoAll{iMov,1};
      nf = ifo.nframes;      
      s = obj.labels{iMov};
      tf = Labels.labeledFrames(s,nf);
    end

    function tflbled = labelPosLabeledTgts(obj,iMov)
      zz = obj.getMovieInfoAllGTawareArg(obj.gtIsGTMode);
      ifo = zz{iMov,1};
      nf = ifo.nframes;      
      s = obj.labelsGTaware;
      tflbled = Labels.labeledTgts(s{iMov},nf);
    end
    
    
%     function [tf0] = labelPosIsOccluded(obj,iFrm,iTrx)
%       x = rand;
%       if x > 0.5
%         [tf1] = obj.labelPosIsOccluded_Old(iFrm,iTrx);
%         [tf0] = obj.labelPosIsOccluded_New(iFrm,iTrx);
%       else
%         [tf0] = obj.labelPosIsOccluded_New(iFrm,iTrx);
%         [tf1] = obj.labelPosIsOccluded_Old(iFrm,iTrx);
%       end
%       assert(isequal(tf0,tf1));
%     end
%     function tf = labelPosIsOccluded_Old(obj,iFrm,iTrx)
%       % Here Occluded refers to "pure occluded"
%       % For current movie.
%       % iFrm, iTrx: optional, defaults to current
%       % Note: it is permitted to call eg LabelPosSet with inf coords
%       % indicating occluded
%       
%       iMov = obj.currMovie;
%       if exist('iFrm','var')==0
%         iFrm = obj.currFrame;
%       end
%       if exist('iTrx','var')==0
%         iTrx = obj.currTarget;
%       end
%       PROPS = obj.gtGetSharedProps();
%       lpos = obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTrx);
%       tf = isinf(lpos(:,1));
%     end

    function tf = labelPosIsOccluded(obj,iFrm,iTrx)
      iMov = obj.currMovie;
      if exist('iFrm','var')==0
        iFrm = obj.currFrame;
      end
      if exist('iTrx','var')==0
        iTrx = obj.currTarget;
      end
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      [tflbl,p,~] = Labels.isLabeledFT(s,iFrm,iTrx);
      if tflbl
        tf = isinf(p(1:s.npts));
      else
        tf = false(s.npts,1);
      end
    end
    
%     function labelPosSet(obj,xy)
%       % Set labelpos for current movie/frame/target
%             
%       x = rand;
%       if x > 0.5
%         obj.labelPosprofoi_Old(xy);
%         obj.labelPosSet_New(xy);
%       else
%         obj.labelPosSet_New(xy);
%         obj.labelPosSet_Old(xy);        
%       end
%     end
%     function labelPosSet_Old(obj,xy)
%       iMov = obj.currMovie;
%       iFrm = obj.currFrame;
%       iTgt = obj.currTarget;
%       PROPS = obj.gtGetSharedProps();
%       obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTgt) = xy;
%       ts = now;
%       obj.(PROPS.LPOSTS){iMov}(:,iFrm,iTgt) = ts;
%       if ~obj.gtIsGTMode
%         %obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
%         obj.lastLabelChangeTS = ts;
%       end
%       obj.labeledposNeedsSave = true;
%     end

    function labelPosSet(obj,xy,tfeo)
      if nargin<3
        tfeo = [];
      end
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      ts = now;
      s = obj.(PROPS.LBL){iMov};
      s = Labels.setpFT(s,iFrm,iTgt,xy);
      if ~isempty(tfeo)
        % tfeo will be converted from logical to appropriate cls
        s = Labels.setoccvalFTI(s,iFrm,iTgt,1:s.npts,tfeo);
      end
      obj.(PROPS.LBL){iMov} = s;
      %obj.(PROPS.LPOSTS){iMov}(:,iFrm,iTgt) = ts;

      if ~obj.gtIsGTMode
        %obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end
      obj.labeledposNeedsSave = true;
    end
      
%     function labelPosSetI(obj,xy,iPt)
%       % Set labelpos for current movie/frame/target
%             
%       x = rand;
%       if x > 0.5
%         obj.labelPosSetI_Old(xy,iPt);
%         obj.labelPosSetI_New(xy,iPt);
%       else
%         obj.labelPosSetI_New(xy,iPt);
%         obj.labelPosSetI_Old(xy,iPt);        
%       end
%     end
%     function labelPosSetI_Old(obj,xy,iPt)
%       % Set labelpos for current movie/frame/target, point iPt
%       
%       assert(~any(isnan(xy(:))));
%       
%       iMov = obj.currMovie;
%       iFrm = obj.currFrame;
%       iTgt = obj.currTarget;
%       PROPS = obj.gtGetSharedProps();
%       obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt) = xy;
%       ts = now;
%       obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = ts;
%       if ~obj.gtIsGTMode
% %         obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
%         obj.lastLabelChangeTS = ts;
%       end
%       obj.labeledposNeedsSave = true;
%     end

    function labelPosSetI(obj,xy,iPt)
      % Set labelpos for current movie/frame/target, point iPt
      
      assert(~any(isnan(xy(:))));
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      s = Labels.setpFTI(s, iFrm,iTgt,iPt,xy);
      obj.(PROPS.LBL){iMov} = s;
      if ~obj.gtIsGTMode
        %obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
        obj.lastLabelChangeTS = now;
      end
      obj.labeledposNeedsSave = true;
    end 

    function labelPosSetIFullyOcc(obj,iPt)
      xy = repmat(Labels.getFullyOccValue,[1,2]); % KB is this the right dimensionality??
      obj.labelPosSetI(xy,iPt);
    end
    
%     function labelPosClearFramesI_Old(obj,frms,iPt)
%       xy = nan(2,1);
%       obj.labelPosSetFramesI_Old(frms,xy,iPt);      
%     end
%     function labelPosClearFramesI_New(obj,frms,iPt)
%       xy = nan(2,1);
%       obj.labelPosSetFramesI_New(frms,xy,iPt);      
%     end
    
    % XXX TODO (LabelCoreHT Client)
    function labelPosSetFramesI(obj,frms,xy,iPt)
      % Set labelpos for current movie/target to a single (constant) point
      % across multiple frames
      %
      % frms: vector of frames
      
      assert(false, 'maTODO');
      
      assert(isvector(frms));
      assert(numel(xy)==2);
      assert(isscalar(iPt));
      
      obj.trxCheckFramesLiveErr(frms);

      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(iPt,1,frms,iTgt) = xy(1);
      obj.(PROPS.LPOS){iMov}(iPt,2,frms,iTgt) = xy(2);
      obj.syncPropsMfahl_() ;
      obj.notify('updateFrameTableComplete'); % above sets mutate .labeledpos{obj.currMovie} in more than just .currFrame
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      end
      
      ts = now;
      obj.(PROPS.LPOSTS){iMov}(iPt,frms,iTgt) = ts;
      if ~obj.gtIsGTMode
%         obj.labeledposMarked{iMov}(iPt,frms,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end

      obj.labeledposNeedsSave = true;
    end
   
    
%     function labelPosSetFromLabeledPos2(obj)
%       % copy .labeledpos2 to .labeledpos for current movie/frame/target
%       
%       assert(~obj.gtIsGTMode);
%       
%       iMov = obj.currMovie;
%       if iMov>0
%         frm = obj.currFrame;
%         iTgt = obj.currTarget;
%         lpos = obj.labeledpos2{iMov}(:,:,frm,iTgt);
%         obj.labelPosSet(lpos);
%       else
%         warning('labeler:noMovie','No movie.');
%       end
%     end    
    
%     function labelPosBulkClear(obj,varargin)
%       % Clears ALL labels for current GT state -- ALL MOVIES/TARGETS
%       
%       [cleartype,gt] = myparse(varargin,...
%         'cleartype','all',... % 'all'=>all movs all tgts
%         ...                   % 'tgt'=>all movs current tgt
%         ...                   % WARNING: 'tgt' assumes target i corresponds across all movies!!
%         'gt',obj.gtIsGTMode);
%       
%       PROPS = Labeler.gtGetSharedPropsStc(gt);
%       nMov = obj.getnmoviesGTawareArg(gt);
%       
%       ts = now;
%       iTgt = obj.currTarget;
%       for iMov=1:nMov
%         switch cleartype
%           case 'all'
%             obj.(PROPS.LPOS){iMov}(:) = nan;        
%             obj.(PROPS.LPOSTS){iMov}(:) = ts;
%             obj.(PROPS.LPOSTAG){iMov}(:) = false;
%             if ~gt
%               % unclear what this should be; marked-ness currently unused
%               obj.labeledposMarked{iMov}(:) = false; 
%             end
%           case 'tgt'
%             obj.(PROPS.LPOS){iMov}(:,:,:,iTgt) = nan;
%             obj.(PROPS.LPOSTS){iMov}(:,:,iTgt) = ts;
%             obj.(PROPS.LPOSTAG){iMov}(:,:,iTgt) = false;
%             if ~gt
%               % unclear what this should be; marked-ness currently unused
%               obj.labeledposMarked{iMov}(:,:,iTgt) = false; 
%             end            
%           otherwise
%             assert(false);
%         end
%       end
%       if ~gt,
%         obj.lastLabelChangeTS = ts;
%       end
%       
%       obj.updateFrameTableComplete();
%       if obj.gtIsGTMode
%         obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
%       end
%       obj.labeledposNeedsSave = true;
%     end
    
%     function labelPosBulkImport(obj,xy,ts)
%       % Set ALL labels for current movie/target
%       %
%       % xy: [nptx2xnfrm]
%       
%       assert(~obj.gtIsGTMode);
%       if ~exist('ts','var'),
%         ts = now;
%       end
%       
%       iMov = obj.currMovie;
%       lposOld = obj.labeledpos{iMov};
%       szassert(xy,size(lposOld));
%       obj.labeledpos{iMov} = xy;
%       obj.labeledposTS{iMov}(:) = ts;
%       obj.lastLabelChangeTS = max(obj.labeledposTS{iMov}(:));
%       obj.labeledposMarked{iMov}(:) = true; % not sure of right treatment
% 
%       obj.updateFrameTableComplete();
%       if obj.gtIsGTMode
%         obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
%       else
%         obj.lastLabelChangeTS = ts;
%       end
%       obj.labeledposNeedsSave = true;
%     end
    
    function labelPosBulkImportTblMov(obj,tblFT,iMov,varargin)
      % Set labels for current movie/target from a table. GTmode supported.
      % Existing labels *are cleared*!
      %
      % tblFT: table with fields .frm, .iTgt, .p, .tfocc. CANNOT have field
      % .mov, to avoid possible misunderstandings/bugs. This meth sets
      % labels on the *current movie only*, or the specified movie.
      %   * tblFT.p should have size [n x nLabelPoints*2].
      %       The raster order is (fastest first): 
      %          {physical pt,view,coordinate (x vs y)}
      %   * tblFT.tfocc should be logical of size [n x nLabelPoints]
      %
      % Alternatively, tblFT can be a Labels struct.
      %
      % iMov: optional scalar movie index into which labels are imported.
      %   Defaults to .currMovie.
      % 
      % No checking is done against image or crop size.
      
      docompact = myparse(varargin,...
        'docompact',false ...
        );

      if exist('iMov','var')==0
        iMov = obj.currMovie;
      end
      assert(iMov>0);

      tsnow = now;
      if istable(tblFT)
        % atm pTS are overwritten/set as "now"
        tblfldscontainsassert(tblFT,{'frm' 'iTgt' 'p' 'tfocc'}); 
        tblfldsdonotcontainassert(tblFT,{'mov'});

        n = height(tblFT);
        npts = obj.nLabelPoints;
        if obj.maIsMA
          assert((size(tblFT.p,1)==n)&&(size(tblFT.p,3)==2*npts));
          assert((size(tblFT.tfocc,1)==n)&&(size(tblFT.tfocc,3)==npts));
        else
          szassert(tblFT.p,[n 2*npts]);
          szassert(tblFT.tfocc,[n npts]);
        end
        %assert(islogical(tblFT.tfocc));

        warningNoTrace('Existing labels cleared!');
        tblFT.pTS = tsnow*ones(size(tblFT.tfocc));
        s = Labels.fromtable(tblFT);
      else
        s = tblFT;
      end
      if docompact
        [s,nfrmslbl,nfrmscompact] = Labels.compactall(s);
        fprintf(1,'Movie %d: %d labeled frms, %d frms compactified.\n',...
          iMov,nfrmslbl,nfrmscompact);
      end
      
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LBL){iMov} = s;

      obj.syncPropsMfahl_() ;
      obj.notify('updateFrameTableComplete');
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      else
        obj.lastLabelChangeTS = tsnow;
      end
      obj.labeledposNeedsSave = true;
    end

    function labelPosBulkImportCOCOJson(obj,cocos,varargin)
      % labelPosBulkImportCOCOJson(obj,cocos,...)
      % Import labels and movies from the COCO-structured data cocos.
      % If information is given about the movies that were used to
      % crop out the images that are labeled, then these movies will be
      % added. Otherwise, a new movie will be created which is actually a
      % directory containing consecutively named frames that APT will read
      % as if it is a single movie, with one frame per labeled image. 
      % Labels for each referenced movie will be REPLACED by labels in
      % cocos. 
      %
      % Input:
      % cocos: struct containing data read from COCO json file
      % Fields:
      % .images: struct array with an entry for each labeled image, with
      % the following fields:
      %   .id: Unique id for this labeled image, from 0 to
      %   numel(locs.locdata)-1
      %   .file_name: Relative path to file containing this image
      %   .movid: Id of the movie this image come from, 0-indexed. This is
      %   used iff movie information is available
      %   .frmid: Index of frame this image comes from, 0-indexed. This is
      %   used iff movie information is available
      % .annotations: struct array with an entry for each annotation, with
      % the following fields:
      %   .iscrowd: Whether this is a labeled target (0) or mask (1). If
      %   not available, it is assumed that this is a labeled target (0).
      %   .segmentation: cell of length 1 containing array of length 8 x 1,
      %   containing the x (segmentation{1}(1:2:end)) and y
      %   (segmentation{1}(2:2:end)) coordinates of the mask considered
      %   labeled, 0-indexed. This is only used for label ROIs.
      %   .image_id: Index (0-indexed) of corresponding image
      %   .num_keypoints: Number of keypoints in this target (0 if mask)
      %   .keypoints: array of size nkeypoints*3 containing the x
      %   (keypoints(1:3:end)), y (keypoints(2:3:end)), and occlusion
      %   status (keypoints(3:3:end)). (x,y) are 0-indexed. for occlusion
      %   status, 2 means not occluded, 1 means occluded but labeled, 0
      %   means not labeled. 
      % .info:
      %   .movies: Cell containing paths to movies. If this is available,
      %   then these movies are added to the project. 
      % Optional arguments:
      % outimdir: If no movie info is available, then this is where the
      % dummy movie will be created. Must be provided if movie info is not
      % in the coco structure. 
      % overwrite: Whether to overwrite files that exist. Default: true
      % imname: Base name of created images. Default: 'frame'.
      % cocojsonfile: Path to coco json file. Must be provided if movie
      % info is not in the coco structure. 

      [outimdir,overwrite,imname,cocojsonfile,copyims] = myparse(varargin,...
        'outimdir','','overwrite',true,'imname','frame',...
        'cocojsonfile','','copyims',true);

      % import labels from COCO json file
      hasmovies = isfield(cocos,'info') && isfield(cocos.info,'movies');

      PROPS = obj.gtGetSharedProps();
      tsnow = now;

      if hasmovies,
        moviefilepaths = cocos.info.movies;
        for imov = 1:numel(moviefilepaths),
          moviecurr = moviefilepaths(imov,:); 
          % todo: check multiview, hastrx, gtmode, macros
          % projects
          % is this movie already added to the project?
          [didfind,imovmatch] = obj.movieSetInProj(moviecurr);
          if ~didfind,
            % add movie to project
            fprintf('Adding movie %d:',imov);
            fprintf('  %s',moviecurr{:});
            fprintf('\n');
            obj.movieAddAllModes(moviecurr);
            [didfind,imovmatch] = obj.movieSetInProj(moviecurr);
            assert(didfind,'Failed to add movie');
          else
            fprintf('Movie %d already in project:',imov);
            fprintf('  %s',moviecurr{:});
            fprintf('\n');
          end
          fprintf('Project movie index = %d\n',imovmatch);
          % convert from coco format to label format for this movie
          label_s = Labels.fromcoco(cocos,'imov',imov,'tsnow',tsnow);
          if ~isempty(label_s),
            fprintf('Imported %d labels\n',size(label_s.p,2));
            % store in obj.labels
            obj.(PROPS.LBL){imovmatch} = label_s;
          end
          % add label rois
          % label boxes are stored in labelsRoi as corners (xl,yt);(xl,yb);(xr,yb);(xr,yb)
          labelroi_s = LabelROI.fromcoco(cocos,'imov',imov);
          if ~isempty(labelroi_s),
            fprintf('Imported %d labeled ROIs\n',size(labelroi_s.verts,3));
            % store in obj.labels
            obj.labelsRoi{imovmatch} = labelroi_s;
          end
          if isempty(label_s) && isempty(labelroi_s),
            fprintf('No labels found for this movie\n');
          end
        end
      else

        % create a directory with frames in order
        assert(~isempty(outimdir));
        assert(~isempty(cocojsonfile));
        inrootdir = fileparts(cocojsonfile);
        if copyims,
          if ~exist(outimdir,'dir'),
            mkdir(outimdir);
          end
          nim = numel(cocos.images);
          nz = max(5,ceil(log10(nim)));
          [~,~,imext] = fileparts(cocos.images(1).file_name);
          namepat = sprintf('%s%%0%dd%s',imname,nz,imext);
          for i = 1:nim,
            imcurr = cocos.images(i);
            inp = fullfile(inrootdir,imcurr.file_name);
            outp = fullfile(outimdir,sprintf(namepat,i));
            if i == 1,
              moviepath = outp;
            end
            assert(exist(inp,'file'));
            if overwrite || ~exist(outp,'file'),
              [success,msg] = copyfile(inp,outp);
              assert(success,msg);
            end
          end
        else
          sortedimfiles = sort({cocos.images.file_name});
          moviepath = fullfile(inrootdir,sortedimfiles{1});
        end
        [didfind,~] = obj.movieSetInProj(moviepath);
        if ~didfind,
          obj.movieAddAllModes(moviepath);
        end
        [didfind,imovmatch] = obj.movieSetInProj(moviepath);
        assert(didfind,'Failed to add stitched movie');

        fprintf('Project movie index = %d\n',imovmatch);
        % convert from coco format to label format for this movie
        label_s = Labels.fromcoco(cocos,'tsnow',tsnow);
        if ~isempty(label_s),
          fprintf('Imported %d labels\n',size(label_s.p,2));
          % store in obj.labels
          obj.(PROPS.LBL){imovmatch} = label_s;
        end
        % add label rois
        % label boxes are stored in labelsRoi as corners (xl,yt);(xl,yb);(xr,yb);(xr,yb)
        labelroi_s = LabelROI.fromcoco(cocos);
        if ~isempty(labelroi_s),
          fprintf('Imported %d labeled ROIs\n',size(labelroi_s.verts,3));
          % store in obj.labels
          obj.labelsRoi{imovmatch} = labelroi_s;
        end
        if isempty(label_s) && isempty(labelroi_s),
          fprintf('No labels found');
        end
      end
      obj.updateMovieFilesAllHaveLbls();
      obj.syncPropsMfahl_() ;
      obj.notify('updateFrameTableComplete');
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      else
        obj.lastLabelChangeTS = tsnow;
      end
      obj.labeledposNeedsSave = true;

    end
    
    function labelPosBulkImportTbl(obj,tblMFT)
      % Like labelPosBulkImportTblMov, but table may include movie 
      % fullpaths. 
      %
      % tblMFT: table with fields .mov, .frm, .iTgt, .p, .tfocc. tblFT.mov are 
      % movie full-paths and they must match entries in 
      % obj.movieFilesAllFullGTaware *exactly*. 
      % For multiview projects, tblFT.mov must match 
      % obj.movieFilesAllFullGTaware(:,1).
      
      movs = tblMFT.mov;
      mfaf1 = obj.movieFilesAllFullGTaware(:,1);
      [tf,iMov] = ismember(movs(:,1),mfaf1); % iMov are movie indices
      if ~all(tf)
        movsbad = unique(movs(~tf));
        error('Movies not found in project: %s',...
          String.cellstr2CommaSepList(movsbad));
      end
      
      iMovUn = unique(iMov);
      nMovUn = numel(iMovUn);
      for iiMovUn=1:nMovUn
        imovcurr = iMovUn(iiMovUn);
        tfmov = iMov==imovcurr;
        nrows = nnz(tfmov);
        fprintf('Importing %d rows for movie %d.\n',nrows,imovcurr);
        obj.labelPosBulkImportTblMov(...
          tblMFT(tfmov,{'frm' 'iTgt' 'p' 'tfocc'}),imovcurr);
      end
    end

    function labelImportTrk(obj,iMovs,trkfiles)
      mIdx = MovieIndex(iMovs,obj.gtIsGTMode);
      obj.labelImportTrkGeneric(mIdx,trkfiles,'LBL');
      obj.labelsUpdateNewFrame(true);
      obj.syncPropsMfahl_() ;
      obj.notify('updateFrameTableComplete');
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      else
        obj.lastLabelChangeTS = now;
      end
      obj.labeledposNeedsSave = true;
      obj.notify('dataImported');
      obj.rcSaveProp('lastTrkFileImported',trkfiles{end});
    end
%     
%     function labelPosSetUnmarkedFramesMovieFramesUnmarked(obj,xy,iMov,frms)
%       % Set all unmarked labels for given movie, frames. Newly-labeled 
%       % points are NOT marked in .labeledposmark
%       %
%       % xy: [nptsx2xnumel(frms)xntgts]
%       % iMov: scalar movie index
%       % frms: frames for iMov; labels 3rd dim of xy
%       
%       assert(~obj.gtIsGTMode);
%       
%       npts = obj.nLabelPoints;
%       ntgts = obj.nTargets;
%       nfrmsSpec = numel(frms);
%       assert(size(xy,1)==npts);
%       assert(size(xy,2)==2);
%       assert(size(xy,3)==nfrmsSpec);
%       assert(size(xy,4)==ntgts);
%       validateattributes(iMov,{'numeric'},{'scalar' 'positive' 'integer' '<=' obj.nmovies});
%       nfrmsMov = obj.movieInfoAll{iMov,1}.nframes;
%       validateattributes(frms,{'numeric'},{'vector' 'positive' 'integer' '<=' nfrmsMov});    
%       
%       lposmarked = obj.labeledposMarked{iMov};      
%       tfFrmSpec = false(npts,nfrmsMov,ntgts);
%       tfFrmSpec(:,frms,:) = true;
%       tfSet = tfFrmSpec & ~lposmarked;
%       tfSet = reshape(tfSet,[npts 1 nfrmsMov ntgts]);
%       tfLPosSet = repmat(tfSet,[1 2]); % [npts x 2 x nfrmsMov x ntgts]
%       tfXYSet = ~lposmarked(:,frms,:); % [npts x nfrmsSpec x ntgts]
%       tfXYSet = reshape(tfXYSet,[npts 1 nfrmsSpec ntgts]);
%       tfXYSet = repmat(tfXYSet,[1 2]); % [npts x 2 x nfrmsSpec x ntgts]
%       obj.labeledpos{iMov}(tfLPosSet) = xy(tfXYSet);
%       
%       obj.updateFrameTableComplete();
%       if obj.gtIsGTMode
%         obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
%       else
%         obj.lastLabelChangeTS = now;
%       end
%       obj.labeledposNeedsSave = true;  
%       
% %       for iTgt = 1:ntgts
% %       for iFrm = 1:nfrmsSpec
% %         f = frms(iFrm);
% %         tfset = ~lposmarked(:,f,iTgt); % [npts x 1]
% %         tfset = repmat(tfset,[1 2]); % [npts x 2]
% %         lposFrmTgt = lpos(:,:,f,iTgt);
% %         lposFrmTgt(tfset) = xy(:,:,iFrm,iTgt);
% %         lpos(:,:,f,iTgt) = lposFrmTgt;
% %       end
% %       end
% %       obj.labeledpos{iMov} = lpos;
%     end
    
%     function labelPosSetUnmarked(obj)
%       % Clear .labeledposMarked for current movie/frame/target
%       
%       assert(~obj.gtIsGTMode);
%       iMov = obj.currMovie;
%       iFrm = obj.currFrame;
%       iTgt = obj.currTarget;
%       obj.labeledposMarked{iMov}(:,iFrm,iTgt) = false;
%     end
    
%     function labelPosSetAllMarked(obj,val)
%       % Clear .labeledposMarked for current movie, all frames/targets
% 
%       assert(~obj.gtIsGTMode);
%       obj.labeledposMarked{iMov}(:) = val;
%     end
        
    % XXX TODO (LabelCoreHT client)
    function labelPosSetOccludedI(obj,iPt)
      % Occluded is "pure occluded" here
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt) = inf;
      ts = now;
      obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = ts;
      if ~obj.gtIsGTMode
%         obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end
      obj.labeledposNeedsSave = true;
    end
        
%     function labelPosTagSetI(obj,iPt)
%       x = rand;
%       if x > 0.5
%         obj.labelPosTagSetI_Old(iPt);
%         obj.labelPosTagSetI_New(iPt);
%       else
%         obj.labelPosTagSetI_New(iPt);
%         obj.labelPosTagSetI_Old(iPt);        
%       end
%     end
%     function labelPosTagSetI_Old(obj,iPt)
%       % Set a single tag onto points
%       %
%       % iPt: can be vector
%       %
%       % The same tag value will be set to all elements of iPt.      
%       
%       iMov = obj.currMovie;
%       iFrm = obj.currFrame;
%       iTgt = obj.currTarget;
%       PROPS = obj.gtGetSharedProps();
%       obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = now();
%       obj.(PROPS.LPOSTAG){iMov}(iPt,iFrm,iTgt) = true;
%     end

    function labelPosTagSetI(obj,iPt)
      % Set a single tag onto points
      %
      % iPt: can be vector
      %
      % The same tag value will be set to all elements of iPt.      
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      s = Labels.setoccFTI(s,iFrm,iTgt,iPt);
      obj.(PROPS.LBL){iMov} = s;
    end
    
%     function labelPosTagClearI(obj,iPt)
%       x = rand;
%       if x > 0.5
%         obj.labelPosTagClearI_Old(iPt);
%         obj.labelPosTagClearI_New(iPt);
%       else
%         obj.labelPosTagClearI_New(iPt);
%         obj.labelPosTagClearI_Old(iPt);        
%       end
%     end
%     function labelPosTagClearI_Old(obj,iPt)
%       % iPt: can be vector
%       
%       iMov = obj.currMovie;
%       iFrm = obj.currFrame;
%       iTgt = obj.currTarget;
%       if obj.gtIsGTMode
%         obj.labeledposTSGT{iMov}(iPt,iFrm,iTgt) = now();
%         obj.labeledpostagGT{iMov}(iPt,iFrm,iTgt) = false;
%       else
%         obj.labeledposTS{iMov}(iPt,iFrm,iTgt) = now();
%         obj.labeledpostag{iMov}(iPt,iFrm,iTgt) = false;
%       end
%     end

    function labelPosTagClearI(obj,iPt)
      % iPt: can be vector
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      s = Labels.clroccFTI(s,iFrm,iTgt,iPt);
      obj.(PROPS.LBL){iMov} = s;      
    end
    
    % XXX TODO (LabelCoreHT client)
    function labelPosTagSetFramesI(obj,iPt,frms)
      % Set tag for current movie/target, given pt/frames

      obj.trxCheckFramesLiveErr(frms);
      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOSTAG){iMov}(iPt,frms,iTgt) = true;
    end
    
    % XXX TODO (LabelCoreHT client)
    function labelPosTagClearFramesI(obj,iPt,frms)
      % Clear tags for current movie/target, given pt/frames
      
      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOSTAG){iMov}(iPt,frms,iTgt) = false;
    end
    
%     function [tfneighbor0,iFrm00,lpos00] = ...
%                           labelPosLabeledNeighbor(obj,iFrm,iTrx)
%       x = rand;
%       if x > 0.5
%         [tfneighbor1,iFrm01,lpos01] = obj.labelPosLabeledNeighbor_Old(iFrm,iTrx);
%         [tfneighbor0,iFrm00,lpos00] = obj.labelPosLabeledNeighbor_New(iFrm,iTrx);
%       else
%         [tfneighbor0,iFrm00,lpos00] = obj.labelPosLabeledNeighbor_New(iFrm,iTrx);
%         [tfneighbor1,iFrm01,lpos01] = obj.labelPosLabeledNeighbor_Old(iFrm,iTrx);
%       end
%       assert(isequaln(tfneighbor0,tfneighbor1));
%       if tfneighbor0
%         assert(isequaln(iFrm00,iFrm01));
%         assert(isequaln(lpos00,lpos01));
%       end
%     end
%     function [tfneighbor,iFrm0,lpos0] = ...
%                           labelPosLabeledNeighbor_Old(obj,iFrm,iTrx) % obj const
%       % tfneighbor: if true, a labeled neighboring frame was found
%       % iFrm0: index of labeled neighboring frame, relevant only if
%       %   tfneighbor is true
%       % lpos0: [nptsx2] labels at iFrm0, relevant only if tfneighbor is true
%       %
%       % This method looks for a frame "near" iFrm for target iTrx that is
%       % labeled. This could be iFrm itself if it is labeled. If a
%       % neighboring frame is found, iFrm0 is not guaranteed to be the
%       % closest or any particular neighboring frame although this will tend
%       % to be true.      
%       
%       iMov = obj.currMovie;
%       PROPS = obj.gtGetSharedProps();
%       lposTrx = obj.(PROPS.LPOS){iMov}(:,:,:,iTrx);
%       assert(isrow(obj.NEIGHBORING_FRAME_OFFSETS));
%       for dFrm = obj.NEIGHBORING_FRAME_OFFSETS
%         iFrm0 = iFrm + dFrm;
%         iFrm0 = max(iFrm0,1);
%         iFrm0 = min(iFrm0,obj.nframes);
%         lpos0 = lposTrx(:,:,iFrm0);
%         if ~isnan(lpos0(1))
%           tfneighbor = true;
%           return;
%         end
%       end
%       
%       tfneighbor = false;
%       iFrm0 = nan;
%       lpos0 = [];      
%     end

    function [tfneighbor,iFrm0,lpos0] = ...
                          labelPosLabeledNeighbor(obj,iFrm,iTrx) % obj const
      % tfneighbor: if true, a labeled neighboring frame was found
      % iFrm0: index of labeled neighboring frame, relevant only if
      %   tfneighbor is true
      % lpos0: [nptsx2] labels at iFrm0, relevant only if tfneighbor is true
      %
      % This method looks for a frame "near" iFrm for target iTrx that is
      % labeled. This could be iFrm itself if it is labeled. If a
      % neighboring frame is found, iFrm0 is not guaranteed to be the
      % closest or any particular neighboring frame although this will tend
      % to be true.      
      
      iMov = obj.currMovie;
      PROPS = obj.gtGetSharedProps();
      s = obj.(PROPS.LBL){iMov};
      [tfneighbor,iFrm0,lpos0] = Labels.findLabelNear(s,iFrm,iTrx);
    end
    
%     function [tffound0,mIdx0,frm0,iTgt0,xyLbl0] = labelFindOneLabeledFrame(obj)
%       x = rand;
%       if x > 0.5
%         [tffound1,mIdx1,frm1,iTgt1,xyLbl1] = obj.labelFindOneLabeledFrame_Old();
%         [tffound0,mIdx0,frm0,iTgt0,xyLbl0] = obj.labelFindOneLabeledFrame_New();
%       else
%         [tffound0,mIdx0,frm0,iTgt0,xyLbl0] = obj.labelFindOneLabeledFrame_New();
%         [tffound1,mIdx1,frm1,iTgt1,xyLbl1] = obj.labelFindOneLabeledFrame_Old();
%       end
%       assert(isequaln(tffound0,tffound1));
%       assert(isequaln(mIdx0,mIdx1));
%       assert(isequaln(frm0,frm1));
%       assert(isequaln(iTgt0,iTgt1));
%       assert(isequaln(xyLbl0,xyLbl1));
%     end
%     function [tffound,mIdx,frm,iTgt,xyLbl] = labelFindOneLabeledFrame_Old(obj)
%       % Find one labeled frame, any labeled frame.
%       %
%       % tffound: true if one was found.
%       % mIdx: scalar MovieIndex
%       % frm: scalar frame number
%       % iTgt: etc
%       % xyLbl: [npts x 2] labeled positions
%       
%       iMov = find(obj.movieFilesAllHaveLbls,1);
%       if ~isempty(iMov)
%         mIdx = MovieIndex(iMov,false);        
%       else
%         iMov = find(obj.movieFilesAllGTHaveLbls,1);
%         if isempty(iMov)
%           tffound = false;
%           mIdx = [];
%           frm = [];
%           iTgt = [];
%           xyLbl = [];
%           return;
%         end
%         mIdx = MovieIndex(iMov,true);
%       end
%       
%       lpos = obj.getLabeledPosMovIdx(mIdx);      
%       nFrm = size(lpos,3);
%       nTgt = size(lpos,4);
%       for frm = 1:nFrm
%         for iTgt=1:nTgt
%           xyLbl = lpos(:,:,frm,iTgt);
%           if any(~isnan(xyLbl(:)))
%             tffound = true;
%             return;
%           end
%         end
%       end
%       
%       % Should never reach here
%       tffound = false;
%       mIdx = [];
%       frm = [];
%       iTgt = [];
%       xyLbl = [];
%       return;
%     end

    function [tffound,mIdx,frm,iTgt,xyLbl] = labelFindOneLabeledFrame(obj)
      % Find one labeled frame, any labeled frame.
      %
      % tffound: true if one was found.
      % mIdx: scalar MovieIndex
      % frm: scalar frame number
      % iTgt: etc
      % xyLbl: [npts x 2] labeled positions
      
      iMov = find(obj.movieFilesAllHaveLbls,1);
      if ~isempty(iMov)
        mIdx = MovieIndex(iMov,false);        
      else
        iMov = find(obj.movieFilesAllGTHaveLbls,1);
        if isempty(iMov)
          tffound = false;
          mIdx = [];
          frm = [];
          iTgt = [];
          xyLbl = [];
          return;
        end
        mIdx = MovieIndex(iMov,true);
      end
      
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt
        s = obj.labelsGT{iMov};
      else 
        s = obj.labels{iMov};
      end
      
      % s should be nonempty      
      frm = s.frm(1);
      iTgt = s.tgt(1);
      xyLbl = reshape(s.p(:,1),s.npts,2);
      tffound = true;
    end
    
    function [nTgts,nPts,nRois] = labelPosLabeledFramesStats(obj,frms) % obj const
      % Get stats re labeled frames in the current movie.
      % 
      % frms: SCALAR frame index to consider. If not provided, defaults to
      %   1:obj.nframes.
      %
      % nTgts: numel(frms)-by-1 vector indicating number of targets labeled
      %   for each frame in consideration
      % nPts: numel(frms)-by-1 vector indicating number of points labeled 
      %   for each frame in consideration, across all targets
      % nRois: numel(frms)-by-1 vector indicating number of extra/ma rois
      %   for each frame. (all elements will be 0 for nonMA projs)
      
      tfScalarFrm = exist('frms','var')>0;
      if tfScalarFrm
        assert(isscalar(frms));
      else
        if isnan(obj.nframes)
          frms = [];
        else
          frms = 1:obj.nframes;
        end
      end
      
      if ~obj.hasMovie || obj.currMovie==0 || isempty(frms) % invariants temporarily broken
        nTgts = nan(numel(frms),1);
        nPts = nan(numel(frms),1);
        nRois = nan(numel(frms),1);
        return;
      end
      
      nf = numel(frms);
      %ntgts = obj.nTargets;
      if obj.gtIsGTMode
        lpos = obj.labelsGT;
      else
        lpos = obj.labels;
      end
      s = lpos{obj.currMovie};

      isMA = obj.maIsMA;
      if isMA
        if obj.gtIsGTMode
          sroi = obj.labelsRoiGT{obj.currMovie};
        else
          sroi = obj.labelsRoi{obj.currMovie};
        end
      end
      
      if tfScalarFrm
        is = find(s.frm==frms);
        nTgts = numel(is);
        xs = s.p(1:s.npts,is);
        nPts = nnz(~isnan(xs));
        
        if isMA
          nRois = nnz(sroi.f==frms);
        else
          nRois = 0;
        end
      else
        %frms is guaranteed to be 1:frms(end)
        nTgts = zeros(nf,1);
        nPts = zeros(nf,1);
        nRois = zeros(nf,1);
        for i=1:size(s.p,2)
          f = s.frm(i);
          nTgts(f) = nTgts(f)+1;
          pf = s.p(1:s.npts,i);
          nPts(f) = nPts(f)+nnz(~isnan(pf));
        end
        
        if isMA
          for i=1:numel(sroi.f)
            f = sroi.f(i);
            nRois(f) = nRois(f) + 1;
          end
        end
      end
    end
    
    function ntgts = labelNumLabeledTgts(obj)
      % subset of labelPosLabeledFramesStats
      if obj.gtIsGTMode
        lpos = obj.labelsGT;
      else
        lpos = obj.labels;
      end
      s = lpos{obj.currMovie};
      ntgts = Labels.getNumTgts(s,obj.currFrame);
    end
    
    function [xy,occ] = labelMAGetLabelsFrm(obj,frm)
      if obj.gtIsGTMode
        lpos = obj.labelsGT;
      else
        lpos = obj.labels;
      end
      s = lpos{obj.currMovie};
      [p,occ] = Labels.getLabelsF(s,frm);
      xy = reshape(p,size(p,1)/2,2,[]);
      occ = logical(occ);
    end
    
%     function [tf0] = labelPosMovieHasLabels(obj,iMov,varargin)
%       x = rand;
%       if x > 0.5
%         [tf1] = obj.labelPosMovieHasLabels_Old(iMov,varargin);
%         [tf0] = obj.labelPosMovieHasLabels_New(iMov,varargin);
%       else
%         [tf0] = obj.labelPosMovieHasLabels_New(iMov,varargin);
%         [tf1] = obj.labelPosMovieHasLabels_Old(iMov,varargin);
%       end
%       assert(tf0==tf1);
%     end
%     function tf = labelPosMovieHasLabels_Old(obj,iMov,varargin)
%       gt = myparse(varargin,'gt',obj.gtIsGTMode);
%       if ~gt
%         lpos = obj.labeledpos{iMov};
%       else
%         lpos = obj.labeledposGT{iMov};
%       end
%       tf = any(~isnan(lpos(:)));
%     end

    function tf = labelPosMovieHasLabels(obj,iMov,varargin)
      gt = myparse(varargin,'gt',obj.gtIsGTMode);
      if ~gt
        s = obj.labels{iMov};
      else
        s = obj.labelsGT{iMov};
      end
      tf = ~isempty(s.p);
    end

%     function islabeled0 = currFrameIsLabeled(obj)
%       x = rand;
%       if x > 0.5
%         islabeled0 = currFrameIsLabeled_New(obj);
%         islabeled1 = currFrameIsLabeled_Old(obj);
%       else
%         islabeled0 = currFrameIsLabeled_Old(obj);
%         islabeled1 = currFrameIsLabeled_New(obj);
%       end
%       assert(islabeled0==islabeled1);
%     end

    function islabeled = currFrameIsLabeled(obj)
      % "is fully labeled"
      lpos = obj.labelsGTaware;
      s = lpos{obj.currMovie};
      [tf,p,~] = Labels.isLabeledFT(s,obj.currFrame,obj.currTarget);
      islabeled = tf && ~all(isnan(p));
    end
%     function islabeled = currFrameIsLabeled_Old(obj)
%       % "is fully labeled"
%       lpos = obj.labeledposGTaware;
%       lpos = lpos{obj.currMovie}(:,:,obj.currFrame,obj.currTarget);
%       islabeled = all(~isnan(lpos(:)));
%     end

    function labelroiSet(obj,v)
      % Set/replace all rois for current mov/frm
      %assert(~obj.gtIsGTMode);
      iMov = obj.currMovie;
      frm = obj.currFrame;
      if obj.gtIsGTMode
        s = obj.labelsRoiGT{iMov};
        obj.labelsRoiGT{iMov} = LabelROI.setF(s,v,frm);
      else
        s = obj.labelsRoi{iMov};
        obj.labelsRoi{iMov} = LabelROI.setF(s,v,frm);
      end

      obj.lastLabelChangeTS = now;
      obj.labeledposNeedsSave = true;
    end
    
    function v = labelroiGet(obj,frm)
      % Get rois for current frm
%       assert(~obj.gtIsGTMode);
      iMov = obj.currMovie;
      %frm = obj.currFrame;
      if ~obj.gtIsGTMode
        s = obj.labelsRoi{iMov};
        v = LabelROI.getF(s,frm);
      else
        v = [];
      end
    end

   

    % Label Cosmetics notes 20190601
    %
    % - Cosmetics settings live in PV pairs on .labelPointsPlotInfo
    % - lblCore owns a subset of these PVs in lblCore.ptsPlotInfo. This is
    % a copy (the copyness is an impl detail). lblCore could prob just look 
    % at lObj.labelPointsPlotInfo but having a copy isn't all bad and can 
    % be advantageous at times. A copy of the cosmetics state needs to 
    % exist outside of HG properties in the various handle objects b/c 
    % lblCore subclasses can mutate handle cosmetics in various ways (eg 
    % due to selection, adjustment, occludedness, etc) and these mutations 
    % need to be revertable to the baseline setpoint.
    % - During project save, getCurrentConfig includes .labelPointsPlotInfo
    % to be saved. 
    % - During project load, .labelPointsPlotInfo can be modernized to
    % include/remove fields per the default config (.DEFAULT_CFG_FILENAME)
    %
    % - View> Hide/Show labels applies only to the mainUI (lblCore), and
    % applies to both markers and text. Note that text visibility is also
    % independently toggleable.
    
    % Prediction Cosmetics notes
    % 
    % - Cosmetics live in PV pairs on .predPointsPlotInfo
    % - These props include: Color, Marker-related, Text-related
    % - The .Color prop currently is *per-set*, ie corresponding points in
    % different views currently must have the same color;
    % size(pppi.Color,1) will equal .nPhysPoints. See PredictPointColors().
    % - .pppi applies to:
    %   i) *All* trackers. Currently LabelTrackers all contain a
    %   TrackingVisualizer for visualization. All trackers have their TVs
    %   initted from pppi and thus all trackers' tracking results currently
    %   will look the same. Nothing stops the user from individually
    %   massaging a trackers' TV, but for now there is no facility to 
    %   display preds from multiple tracker objs at the same time, and
    %   any such changes are not currently serialized.
    %   iii) PPPI serves as an initialization point for aux tracking 
    %   results in .trkResViz, but it is expected that the user will mutate
    %   the cosmetics for .trkResViz to facilitate comparison of multiple
    %   tracking results.
    %      a) User-mutations of .trkResViz cosmetic state IS SERIALIZED
    %      with the project. Note in contrast, TrackingVisualizers for
    %      LabelTrackers (in i) currently are NOT serialized at all.
    %
    % Use cases.
    %  1. Comparison across multiple "live" tracking results. 
    %     - Currently this can't really be done, since results from
    %     multiple LabelTrackers cannot be displayed simultaneously, not to
    %     mention that all LabelTrackers share the same cosmetics.
    %  2. Comparison between one "live" tracking result and one imported.
    %     - This is at least possible, but imported results share the same 
    %     cosmetics as "live" results so it's not great.
    %  3. Comparison between multiple imported tracking results.
    %     - This works well as the .trkRes* infrastruture is explicitly 
    %     designed for this (currently cmdline only).
    %
    % Future 20190603.
    %  A. The state of 1. above is unclear as we currently do not even save
    %  tracking results with the project at all. In the meantime, use case
    %  3. does meet the basic need provided tracking results are first
    %  exported. This also serves more general purposes eg when a single
    %  tracker is run across a parameter sweep, or with differing training
    %  data etc.
    %  B. Re 2. above, imported results quite likely should just be a 
    %  special case of the aux (.trkRes*) tracking results. This would
    %  simplify the code a bit, cosmetics would be mutable, and cosmetics
    %  settings would be saved with the project.
    
    function updateLandmarkColors(obj,colorSpecs)
      for i=1:numel(colorSpecs)
        cs = colorSpecs(i);
        lsetType = cs.landmarkSetType;
        lObjUpdateMeth = lsetType.updateColorLabelerMethod();
        obj.(lObjUpdateMeth)(cs.colors,cs.colormapname);
      end
    end
    
    function updateLandmarkCosmetics(obj,mrkrSpecs)
      for i=1:numel(mrkrSpecs)
        ms = mrkrSpecs(i);
        lsetType = ms.landmarkSetType;
        lObjUpdateMeth = lsetType.updateCosmeticsLabelerMethod();
        obj.(lObjUpdateMeth)(ms.MarkerProps,ms.TextProps,ms.TextOffset);
      end
    end
    
    function updateSkeletonCosmetics(obj,skelSpecs)
      for i=1:numel(skelSpecs)
        ss = skelSpecs(i);
        lsetType = ss.landmarkSetType;
      
        ptsPlotInfoFld = lsetType.labelerPropPlotInfo;
        s0 = obj.(ptsPlotInfoFld).SkeletonProps;      
        obj.(ptsPlotInfoFld).SkeletonProps = structoverlay(s0,ss.SkeletonProps);
      
        switch lsetType
          case LandmarkSetType.Label
            lc = obj.lblCore;
            lc.skeletonCosmeticsUpdated();
          case LandmarkSetType.Prediction
            dt = obj.tracker;
            tv = dt.trkVizer;
            if ~isempty(tv)              
              tv.skeletonCosmeticsUpdated();
            end
          case LandmarkSetType.Imported
            lpos2tv = obj.labeledpos2trkViz;
            if ~isempty(lpos2tv)
              lpos2tv.skeletonCosmeticsUpdated();
            end
        end
      end          
    end
    
    % function updateLandmarkLabelColors(obj,colors,colormapname)
    %   % Probably used in conjunction with projAddLandmarks().  -- ALT, 2025-01-28
    %   % colors: "setwise" colors
    % 
    %   szassert(colors,[obj.nPhysPoints 3]);
    %   lc = obj.lblCore;
    %   % Colors apply to lblCore, lblPrev_*, timeline
    % 
    %   obj.labelPointsPlotInfo.ColorMapName = colormapname;
    %   obj.labelPointsPlotInfo.Colors = colors;
    %   ptcolors = obj.Set2PointColors(colors);
    %   lc.updateColors(ptcolors);
    %   LabelCore.setPtsColor(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH,ptcolors);
    %   obj.controller_.labelTLInfo.updateLandmarkColors();
    % end
    
    function updateLandmarkPredictionColors(obj,colors,colormapname)
      % Probably used in conjunction with projAddLandmarks().  -- ALT, 2025-01-28
      
      % colors: "setwise" colors
      szassert(colors,[obj.nPhysPoints 3]);
      
      obj.predPointsPlotInfo.Colors = colors;
      obj.predPointsPlotInfo.ColorMapName = colormapname;
      cellfun(@(t)(t.updateLandmarkColors()), obj.trackerHistory_ ) ;
    end
    
    function updateLandmarkImportedColors(obj,colors,colormapname)
      % Probably used in conjunction with projAddLandmarks().  -- ALT, 2025-01-28
      % colors: "setwise" colors
      szassert(colors,[obj.nPhysPoints 3]);
      
      obj.impPointsPlotInfo.Colors = colors;
      obj.impPointsPlotInfo.ColorMapName = colormapname;
      ptcolors = obj.Set2PointColors(colors);
      
      lpos2tv = obj.labeledpos2trkViz;
      if ~isempty(lpos2tv)        
        lpos2tv.updateLandmarkColors(ptcolors);
      end
      for i=1:numel(obj.trkResViz)
        obj.trkResViz{i}.updateLandmarkColors(ptcolors);
      end
    end

    function updateLandmarkLabelCosmetics(obj,pvMarker,pvText,textOffset)
      % Probably used in conjunction with projAddLandmarks().  -- ALT, 2025-01-28

      lc = obj.lblCore;
      
      % Marker cosmetics apply to lblCore, lblPrev_*
      flds = fieldnames(pvMarker);
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        obj.labelPointsPlotInfo.MarkerProps.(f) = pvMarker.(f);
      end
      lc.updateMarkerCosmetics(pvMarker);      
      set(obj.lblPrev_ptsH,pvMarker);
      
      % Text cosmetics apply to lblCore, lblPrev_*
      flds = fieldnames(pvText);
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        obj.labelPointsPlotInfo.TextProps.(f) = pvText.(f);
      end
      obj.labelPointsPlotInfo.TextOffset = textOffset;
      set(obj.lblPrev_ptsTxtH,pvText);
      obj.syncPrevAxesVirtualLabels_();
      notify(obj, 'updatePrevAxesLabels');
      lc.updateTextLabelCosmetics(pvText,textOffset);
      %obj.labelsUpdateNewFrame(true); % should redraw prevaxes too
    end

    function [tfHideTxt,pvText] = hlpUpdateLandmarkCosmetics(obj,...
        pvMarker,pvText,textOffset,ptsPlotInfoFld)
      % set PVs on .ptsPlotInfo field; mild massage
      
      fns = fieldnames(pvMarker);
      for f=fns(:)',f=f{1}; %#ok<FXSET> 
        % this allows pvMarker to be 'incomplete'; could just set entire
        % struct
        obj.(ptsPlotInfoFld).MarkerProps.(f) = pvMarker.(f);
      end
      fns = fieldnames(pvText);
      for f=fns(:)',f=f{1}; %#ok<FXSET>
        obj.(ptsPlotInfoFld).TextProps.(f) = pvText.(f);
      end
      obj.(ptsPlotInfoFld).TextOffset = textOffset;
      % TrackingVisualizer wants this prop broken out
      tfHideTxt = strcmp(pvText.Visible,'off'); % could make .Visible field optional 
      pvText = rmfield(pvText,'Visible');
    end 

    function updateLandmarkPredictionCosmetics(obj,pvMarker,pvText,textOffset)
      % Probably used in conjunction with projAddLandmarks().  -- ALT, 2025-01-28
      
      [tfHideTxt,pvText] = obj.hlpUpdateLandmarkCosmetics(...
        pvMarker,pvText,textOffset,'predPointsPlotInfo');
      trackers = obj.trackerHistory_ ;
      for i=1:numel(trackers)
        if ~isempty(trackers{i})
          tracker = trackers{i} ;
          tv = tracker.trkVizer;
          if ~isempty(tv)
            tv.setMarkerCosmetics(pvMarker);
            tv.setTextCosmetics(pvText);
            tv.setTextOffset(textOffset);
            tv.setHideTextLbls(tfHideTxt);
          end
        end
      end      
    end  % function
    
    function updateLandmarkImportedCosmetics(obj,pvMarker,pvText,textOffset)
      % Probably used in conjunction with projAddLandmarks().  -- ALT, 2025-01-28
      
       [tfHideTxt,pvText] = obj.hlpUpdateLandmarkCosmetics(...
        pvMarker,pvText,textOffset,'impPointsPlotInfo');      
      
      lpos2tv = obj.labeledpos2trkViz;
      if ~isempty(lpos2tv)
        lpos2tv.setMarkerCosmetics(pvMarker);      
        lpos2tv.setTextCosmetics(pvText);
        lpos2tv.setTextOffset(textOffset);
        lpos2tv.setHideTextLbls(tfHideTxt);
      end      
      % Todo, set on .trkRes*
    end

    function updateTrajImportedColors(obj,colors,colormapname)
      obj.projPrefs.Trx.TrajColor = colors;
      obj.projPrefs.Trx.TrajColorMapName = colormapname;
      lpos2tv = obj.labeledpos2trkViz;
      if ~isempty(lpos2tv) && isa(lpos2tv,'TrackingVisualizerTracklets')
        lpos2tv.updateTrajColors();
      end      
    end
    
  end
  
  methods (Static)
    function sMacro = movTrkFileMacroDescs()
      sMacro = struct;
      sMacro.movdir = '<full path to movie>';
      sMacro.movfile = '<base name of movie>';
      sMacro.expname = '<name of movie parent directory>';
    end

    function trkfile = genTrkFileName(rawname,sMacro,movfile,varargin)      
      % Generate a trkfilename from rawname by macro-replacing.      
      
      enforceExt = myparse(varargin,...
        'enforceExt',true ...
        );
      
      [sMacro.movdir,sMacro.movfile] = fileparts(movfile);
      [~,sMacro.expname] = fileparts(sMacro.movdir);
      trkfile = FSPath.macroReplace(rawname,sMacro);
      if enforceExt
        if ~(numel(rawname)>=4 && strcmp(rawname(end-3:end),'.trk'))
          trkfile = [trkfile '.trk'];
        end
      end
    end

    function LabelPointNames = defaultLandmarkNames(ptidx)
      
      LabelPointNames = arrayfun(@(x)sprintf('pt%d',x),ptidx','uni',0);
      
    end
  end

  methods
    
    function sMacro = baseTrkFileMacros(obj)
      % Set of built-in macros appropriate to exporting files/data
      sMacro = struct();
      sMacro.projname = obj.projname;
      if ~isempty(obj.projectfile)
        [sMacro.projdir,sMacro.projfile] = fileparts(obj.projectfile);
      else
        sMacro.projdir = '';
        sMacro.projfile = '';
      end
      tObj = obj.tracker;
      if ~isempty(tObj)
        algoName = tObj.algorithmName;
        sMacro.trackertype = regexprep(algoName,'\s','_');
      else
        sMacro.trackertype = 'undefinedtracker';
      end
    end
    
    function trkfile = defaultTrkFileName(obj,movfile)
      % Only client is GMMTracker
      trkfile = Labeler.genTrkFileName(obj.defaultExportTrkRawname(),...
                                       obj.baseTrkFileMacros(),movfile);
    end
    
    function rawname = defaultExportTrkRawname(obj,varargin)
      % Default raw/base trkfilename for export (WITH macros, NO extension).
      
      labels = myparse(varargin,...
        'labels',false... % true for eg manual labels (as opposed to automated tracking)
        );
      
      if ~isempty(obj.projectfile)
        basename = '$movfile_$projfile';
      elseif ~isempty(obj.projname)
        basename = '$movfile_$projname';
      else
        basename = '$movfile';
      end
      
      if labels
        gt = obj.gtIsGTMode;
        if gt
          basename = [basename '_gtlabels'];
        else
          basename = [basename '_labels'];
        end
      else
        basename = [basename '_$trackertype'];
      end
      
      rawname = fullfile('$movdir',basename);
    end
    
    function fname = getDefaultFilenameExport(obj,lblstr,ext,varargin)
      includedate = myparse(varargin,'includedate',false);
      rawdir = '$projdir';
      if ~isempty(obj.projectfile)
        rawname = ['$projfile_' lblstr ext];
      elseif ~isempty(obj.projname)
        rawname = ['$projname_' lblstr ext];
      else
        if includedate,
          rawname = [lblstr datestr(now,'yyyymmddTHHMMSS') ext];
        else
          rawname = [lblstr ext];
        end
      end
      sMacro = obj.baseTrkFileMacros();
      fdir = FSPath.macroReplace(rawdir,sMacro);
      fname = FSPath.macroReplace(rawname,sMacro);
      fname = linux_fullfile(fdir,fname);
    end

    function fname = getDefaultFilenameExportStrippedLbl(obj)
      fname = getDefaultFilenameExport(obj,'TrainData','.mat','includedate',true);
    end
    
    function fname = getDefaultFilenameExportLabelTable(obj)
      if obj.gtIsGTMode
        lblstr = 'gtlabels';
      else
        lblstr = 'labels';
      end
      fname = getDefaultFilenameExport(obj,lblstr,'.mat');
    end

    function fname = getDefaultFilenameExportCOCOJson(obj)
      if obj.gtIsGTMode
        lblstr = 'gtcoco';
      else
        lblstr = 'coco';
      end
      fname = getDefaultFilenameExport(obj,lblstr,'.zip');
    end

    function fname = getDefaultFilenameImportCOCOJson(obj)

      rawdir = '$projdir';
      sMacro = obj.baseTrkFileMacros();
      fdir = FSPath.macroReplace(rawdir,sMacro);
      fname = linux_fullfile(fdir,'*.json');

    end        

  end
  
  methods (Static)
	
    function [trkfilesCommon,kwCommon,trkfilesAll] = ...
                      getTrkFileNamesForImport(movfiles)
      % Find available trkfiles for import
      %
      % movfiles: cellstr of movieFilesAllFull
      %
      % trkfilesCommon: [size(movfiles)] cell-of-cellstrs.
      % trkfilesCommon{i} is a cellstr with numel(kwCommon) elements.
      % trkfilesCommon{i}{j} contains the jth common trkfile found for
      % movfiles{i}, where j indexes kwCommon.
      % kwCommon: [nCommonKeywords] cellstr of common keywords found.
      % trkfilesAll: [size(movfiles)] cell-of-cellstrs. trkfilesAll{i}
      % contains all trkfiles found for movfiles{i}, a superset of
      % trkfilesCommon{i}.
      %
      % "Common" trkfiles are those that share the same naming pattern
      % <moviename>_<keyword>.trk and are present for all movfiles.
      
      trkfilesAll = cell(size(movfiles));
      keywords = cell(size(movfiles));
      for i=1:numel(movfiles)
        mov = movfiles{i};
        if exist(mov,'file')==0
          error('Labeler:noMovie','Cannot find movie: %s.',mov);
        end
        
        [movdir,movname] = fileparts(mov);
        trkpat = [movname '*.trk'];
        dd = dir(fullfile(movdir,trkpat));
        trkfilesAllShort = {dd.name}';
        trkfilesAll{i} = cellfun(@(x)fullfile(movdir,x),trkfilesAllShort,'uni',0);
        
        trkpatRE = sprintf('%s(?<kw>.*).trk',movname);
        re = regexp(trkfilesAllShort,trkpatRE,'names');
        re = cellfun(@(x)x.kw,re,'uni',0);
        keywords{i} = re;
        
        assert(numel(trkfilesAll{i})==numel(keywords{i}));
      end
      
      % Find common keywords
      kwUn = unique(cat(1,keywords{:}));
      tfKwUnCommon = cellfun(@(zKW) all(cellfun(@(x)any(strcmp(x,zKW)),keywords(:))),kwUn);
      kwCommon = kwUn(tfKwUnCommon);
      trkfilesCommon = cellfun(@(zTFs,zKWs) cellfun(@(x)zTFs(strcmp(zKWs,x)),kwCommon), ...
        trkfilesAll,keywords,'uni',0);
    end  % function
  end

  methods
    function labelExportTrkGeneric(obj,iMovs,outfiles,lblsFld)
      % Export given labels field for iMovs into trkfiles. 
      %
      % The GT-status of obj is irrelevant, iMovs just indexes lblsFld.
            
      nMov = numel(iMovs);
      nView = obj.nview;
      nPhysPts = obj.nPhysPoints;
      for i=1:nMov
        iMvSet = iMovs(i);
        s = obj.(lblsFld){iMvSet};
        for iView=1:nView
          iPt = (1:nPhysPts) + (iView-1)*nPhysPts;
          sview = Labels.indexPts(s,iPt);
          outf = outfiles{i,iView};
          save(outf,'-mat','-struct','sview');
          fprintf('Saved labels: %s\n',outf);
        end
      end
      msgbox(sprintf('Results for %d moviesets exported.',nMov),'Export complete');      
    end
    
    function labelImportTrkGeneric(obj,mIdx,trkfiles,propsFld,varargin)
      % Import (iMovSets,trkfiles) into the specified labels* fields
      %
      % mIdx: [N] vector of MovieIndex'es
      % trkfiles: [Nxnview] cellstr of trk filenames
      % propsFld: 'LBL' or 'LBL2'

      switch propsFld
        case 'LBL'
          tfLbl2 = false;
        case 'LBL2'
          tfLbl2 = true;
        otherwise
          assert(false,'Unrecognized propsFld.');
      end

      assert(isa(mIdx,'MovieIndex'));
      
      nMovSets = numel(mIdx);
      nView = obj.nview;
      szassert(trkfiles,[nMovSets nView]);
      tfMV = obj.isMultiView;
      
      for i=1:nMovSets
        if tfMV
          fprintf('MovieSet %d...\n',mIdx(i));
        end
        
        mIdxI = mIdx(i);
        movnframes = obj.getNFramesMovIdx(mIdxI);
        
        scell = cell(1,nView);
        for iVw = 1:nView
          tfile = trkfiles{i,iVw};
          scell{iVw} = TrkFile.load(tfile,'movnframes',movnframes);
      	  if iVw == 1
            nLabelPointsInFile = scell{iVw}.npts;
            if nLabelPointsInFile ~= obj.nPhysPoints
              warning('Number of landmarks in the trk file does not match with the project')
            end
          end
          % Display when .trk file was last updated
          tfileDir = dir(tfile);
          disp(['  trk file last modified: ',tfileDir.date]);

          if ~tfLbl2
            scell{iVw} = Labels.fromTrkfile(scell{iVw});
          end
        end
        
        if tfMV
          if tfLbl2
            % assuming tracklet-style TrkFile for now (LBL2)
            s = scell{1};
            s.mergeMultiView(scell{2:end});
          else
            sarr = cell2mat(scell);
            s = Labels.mergeviews(sarr);
          end
        else
          s = scell{1};
        end
                
        % AL20201223 matlab indexing/language bug 2020b
        %[iMov,isGT] = mIdx(i).get();
        [iMov,isGT] = mIdxI.get();
        PROPS = obj.gtGetSharedPropsStc(isGT);
        lblFld = PROPS.(propsFld);        
        obj.(lblFld){iMov} = s; 
      end
    end
    
%     function labelImportTrk_Old(obj,iMovs,trkfiles)
%       % Import label data from trk files.
%       %
%       % iMovs: [nMovie]. Movie(set) indices for which to import.
%       % trkfiles: [nMoviexnview] cellstr. full filenames to trk files
%       %   corresponding to iMov.
%       
%       assert(~obj.gtIsGTMode);
%       
%       obj.labelImportTrkGeneric(iMovs,trkfiles,'labeledpos',...
%         'labeledposTS','labeledpostag');
%       % need to compute lastLabelChangeTS from scratch
%       obj.computeLastLabelChangeTS_Old();
%       
%       obj.movieFilesAllHaveLbls(iMovs) = ...
%         cellfun(@Labeler.computeNumLbledRows,obj.labeledpos(iMovs));
%       
%       obj.updateFrameTableComplete();
%       if obj.gtIsGTMode
%         obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
%       end
%       
%       %obj.labeledposNeedsSave = true; AL 20160609: don't touch this for
%       %now, since what we are importing is already in the .trk file.
%       obj.labelsUpdateNewFrame(true);
%       
%       obj.rcSaveProp('lastTrkFileImported',trkfiles{end});
%     end
    
    % compute lastLabelChangeTS from scratch
    function computeLastLabelChangeTS_Old(obj)      
      % do this because some movies might not be labeled
      maxperlabel = cellfun(@(x) max(x.ts(:)),obj.labels,'Uni',0);
      obj.lastLabelChangeTS = max(cat(1,maxperlabel{:}));

      % this actually takes a few seconds since it reallocates everything
      % as full arrays
      %obj.lastLabelChangeTS = max(cellfun(@(x) max(x(:)),obj.labeledposTS));
    end
    
    % 20180628 iss 202.
    % The following chain of "smart" methods
    % labelImportTrkPromptGenericAuto
    %  labelImportTrkPromptAuto
    %  labels2ImprotTrkPromptAuto
    %
    % were trying too hard for more vanilla use cases, in particular
    % single-view single-movie import situs.
    %
    % However, they may still still be useful for bulk use cases:
    % multiview, or large-numbers-of-movies. So leave them around for now.
    % 
    % Currently these this chain has only a SINGLE CALLER: importTrkResave,
    % which operates in a bulk fashion.
    %
    % The new simplified call is just 
    % labelImportTrkPromptGenericSimple
    % which is called by LabelerGUI for single-movieset situs (including
    % multiview)
    
    function labelImportTrkPromptGenericSimple(obj,iMov,importFcn,varargin)
      % Prompt user for trkfiles to import and import them with given 
      % importFcn. User can cancel to abort
      %
      % iMov: scalar positive index into .movieFilesAll. GT mode not
      %   allowed.
      
      if ~obj.hasMovie
        error('Labeler:noMovie','No movie is loaded.');
      end
      
      obj.pushBusyStatus('Importing tracking results...');
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      
      gtok = myparse(varargin,...
        'gtok',false ... % if true, obj.gtIsGTMode can be true, and iMov 
                  ...% refers per GT state. importFcn needs to support GT
                  ...% state
                  );
      
      assert(isscalar(iMov));      
      if ~gtok
        assert(~obj.gtIsGTMode);
      end
      
      movs = obj.movieFilesAllFullGTaware(iMov,:);
      movdirs = cellfun(@fileparts,movs,'uni',0);
      nvw = obj.nview;
      trkfiles = cell(1,nvw);
      for ivw=1:nvw
        if nvw>1
          promptstr = sprintf('Import trkfile for view %d',ivw);
        else
          promptstr = 'Import trkfile';
        end
        [fname,pth] = uigetfile('*.trk',promptstr,movdirs{ivw});
        if isequal(fname,0)
          return;
        end
        trkfiles{ivw} = fullfile(pth,fname);
      end
      
      feval(importFcn,obj,iMov,trkfiles);
    end
    
    % function labelImportTrkPromptAuto(obj,iMovs)
    %   % Import label data from trk files, prompting if necessary to specify
    %   % which trk files to import.
    %   %
    %   % iMovs: [nMovie]. Optional, movie(set) indices to import.
    %   %
    %   % labelImportTrkPrompt will look for trk files with common keywords
    %   % (consistent naming) in .movieFilesAllFull(iMovs). If there is
    %   % precisely one consistent trkfile pattern, it will import those
    %   % trkfiles. Otherwise it will ask the user which trk files to import.
    % 
    %   assert(~obj.gtIsGTMode);
    % 
    %   if exist('iMovs','var')==0
    %     iMovs = 1:obj.nmovies;
    %   end
    %   obj.labelImportTrkPromptGenericAuto(iMovs,'labelImportTrk');
    % end
    
    function tblMF = labelGetMFTableLabeled(obj,varargin)
      % Compile mov/frm/tgt MFTable; include all labeled frames/tgts. 
      %
      % Includes nonGT/GT rows per current GT state.
      %
      % Can return [] indicating "no labels of requested/specified type"
      %
      % tblMF: See MFTable.FLDSFULLTRX.
      
      [wbObj,useLabels2,useMovNames,tblMFTrestrict,useTrain,tfMFTOnly] = myparse(varargin,...
        'wbObj',[], ... % optional ProgressMeter. If canceled:
                   ... % 1. obj logically const (eg except for obj.trxCache)
                   ... % 2. tblMF (output) indeterminate
        'useLabels2',false,... % if true, use labels2 instead of labels
        'useMovNames',false,... % if true, use movieNames instead of movieIndices
        'tblMFTrestrict',[],... % if supplied, tblMF is the labeled subset 
                           ... % of tblMFTrestrict (within fields .mov, 
                           ... % .frm, .tgt). .mov must be a MovieIndex.
                           ... % tblMF ordering should be as in tblMFTrestrict
        'useTrain',[],... % whether to use training labels (1) gt labels (0), or whatever current mode is ([])
        'MFTOnly',false... % if true, only return mov, frm, target
        ); 
      tfWB = ~isempty(wbObj);
      tfRestrict = ~isempty(tblMFTrestrict);
      
      if useLabels2 
        if isempty(useTrain)
          mfts = MFTSetEnum.AllMovAllLabeled2;
        elseif useTrain
          mfts = MFTSet(MovieIndexSetVariable.AllTrnMov,FrameSetVariable.Labeled2Frm,...
                        FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
        else % useGT
          mfts = MFTSet(MovieIndexSetVariable.AllGTMov,FrameSetVariable.Labeled2Frm,...
                        FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
        end
      else
        if isempty(useTrain)
          mfts = MFTSetEnum.AllMovAllLabeled;
        elseif useTrain
          mfts = MFTSet(MovieIndexSetVariable.AllTrnMov,FrameSetVariable.LabeledFrm,...
                        FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);          
        else % useGT
          mfts = MFTSet(MovieIndexSetVariable.AllGTMov,FrameSetVariable.LabeledFrm,...
                        FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
        end
      end
      tblMF = mfts.getMFTable(obj);
      
      if tfRestrict
        tblMF = MFTable.intersectID(tblMF,tblMFTrestrict);
      end
      
      if tfMFTOnly,
        return;
      end
      
      if isequal(tblMF,[]) % this would have errored below in call to labelAddLabelsMFTableStc
        return;
      end
      
      if obj.hasTrx,
        if isempty(useTrain),
          trxFiles = obj.trxFilesAllFullGTaware;
        elseif useTrain == 0,
          trxFiles = obj.trxFilesAllGTFull;
        else
          trxFiles = obj.trxFilesAllFull;
        end
          
        argsTrx = {'trxFilesAllFull',trxFiles,'trxCache',obj.trxCache};
      else
        argsTrx = {};
      end
      if useLabels2        
        if isempty(useTrain),
          lpos = obj.labels2GTaware;
        elseif useTrain == 0,
          lpos = obj.labels2GT;
        else
          lpos = obj.labels2;
        end
 
        for i=1:numel(lpos)
          if isa(lpos{i},'TrkFile')
            lpos{i} = Labels.fromtable(lpos{i}.tableform('labelsColNames',true));
          end
        end
        % could use tracklet->tableform and merge
      else
        if isempty(useTrain),
          lpos = obj.labelsGTaware;
        elseif useTrain == 0,
          lpos = obj.labelsGT;
        else
          lpos = obj.labels;
        end
      end
      
      if obj.maIsMA
        maxn = obj.trackParams.ROOT.MultiAnimal.Track.max_n_animals;
      else
        maxn = 1;
      end

      tblMF = Labels.labelAddLabelsMFTableStc(tblMF,lpos,argsTrx{:},...
         'wbObj',wbObj,'isma',obj.maIsMA,'maxanimals',maxn);
      if tfWB ,
        if wbObj.wasCanceled ,
          return
        end
      end
      
      if useMovNames
        assert(isa(tblMF.mov,'MovieIndex'));
        tblMF.mov = obj.getMovieFilesAllFullMovIdx(tblMF.mov);
      end
    end
    
    function tblMF = labelMFTableAddROITrx(obj,tblMF,roiRadius,varargin)
      % Add .pRoi and .roi to tblMF using trx info
      %
      % tblMF.pRoi: Just like tblMF.p, but relative to tblMF.roi (p==1 => 
      %   first row/col of ROI)
      % tblMF.roi: [nrow x (2*2*nview)]. Raster order {lo,hi},{x,y},view      
      
      [rmOOB,pfld,proifld] = myparse(varargin,...
        'rmOOB',true,... % if true, remove rows where shape is outside roi
        'pfld','p',...  % optional "source" lbl field
        'proifld','pRoi'... % optional "dest" roi-lbl field
        );
      
      %tblfldscontainsassert(tblMF,MFTable.FLDSFULLTRX);
      % no requirements beyond as accessed in code
      
      tblfldsdonotcontainassert(tblMF,{proifld 'roi'});
      
      nphyspts = obj.nPhysPoints;
      nrow = height(tblMF);
      p = tblMF.(pfld);
      pTrx = tblMF.pTrx;
      xy = Shape.vecs2xys(p.'); % [npts x 2 x nrow]
      xyTrx = Shape.vecs2xys(pTrx.'); % [ntrxpts x 2 x nrow]
      
      [roi,tfOOBview,xyRoi] = Shape.xyAndTrx2ROI(xy,xyTrx,nphyspts,roiRadius);
      % roi: [nrow x 4*nview]
      % tfOOBview: [nrow x nview]
      % xyRoi: [npts x 2 x nrow]
      npts = obj.nLabelPoints;
      szassert(xyRoi,[npts 2 nrow]);
      pRoi = reshape(xyRoi,npts*2,nrow).'; % [nrow x D]
      
      tblMF = [tblMF table(pRoi,roi,'VariableNames',{proifld 'roi'})];

      if rmOOB
        tfRmrow = any(tfOOBview,2);
        nrm = nnz(tfRmrow);
        if nrm>0
          warningNoTrace('Labeler:oob',...
            '%d rows with shape out of bounds of target ROI. These rows will be discarded.',nrm);
          tblMF(tfRmrow,:) = [];
        end
      end                    
    end  % function
    
    function tblMF = labelMFTableAddROICrop(obj,tblMF,varargin)
      % Add .pRoi and .roi to tblMF using crop info
      %
      % tblMF.pRoi: Just like tblMF.p, but relative to tblMF.roi (p==1 => 
      %   first row/col of ROI)
      % tblMF.roi: [nrow x (2*2*nview)]. Raster order {lo,hi},{x,y},view
      %
      % tblMF(out): rows removed if xy are OOB of roi.
      
      [rmOOB,pfld,proifld] = myparse(varargin,...
        'rmOOB',true,... % if true, remove rows where shape is outside roi
        'pfld','p',...  % optional "source" lbl field
        'proifld','pRoi'... % optional "dest" roi-lbl field
        );
      
      %tblfldscontainsassert(tblMF,MFTable.FLDSFULL);
      % no requirements beyond as accessed in code
      tblfldsdonotcontainassert(tblMF,{proifld 'roi'});
      assert(isa(tblMF.mov,'MovieIndex'));
     
      if ~obj.cropProjHasCrops
        error('Project does not contain cropping information.');
      end
      
      obj.cropCheckCropSizeConsistency();
      
      nphyspts = obj.nPhysPoints;
      nvw = obj.nview;
      n = height(tblMF);
      p = tblMF.(pfld);      
      tfRmRow = false(n,1); % true to rm row due to OOB
      pRoi = nan(size(p));
      roi = nan(n,4*obj.nview);
      for i=1:n
        mIdx = tblMF.mov(i);
        cInfo = obj.getMovieFilesAllCropInfoMovIdx(mIdx);
        roiCurr = cat(2,cInfo.roi);
        szassert(roiCurr,[1 4*nvw]);
        
        % See Shape.p2pROI etc
        xy = Shape.vec2xy(p(i,:));
        [xyROI,tfOOBview] = Shape.xy2xyROI(xy,roiCurr,nphyspts);
        if ~rmOOB,
          tfOOBview(:) = false;
        end
        if any(tfOOBview)
          warningNoTrace('Labeler:oob',...
            'Movie(set) %d, frame %d, target %d: shape out of bounds of target ROI. Not including row.',...
            mIdx,tblMF.frm(i),tblMF.iTgt(i));
          tfRmRow(i) = true;
        else
          pRoi(i,:) = Shape.xy2vec(xyROI);
          roi(i,:) = roiCurr;
        end
      end
      
      tblMF = [tblMF table(pRoi,roi,'VariableNames',{proifld 'roi'})];
      tblMF(tfRmRow,:) = [];
    end

    function tblMF = labelAddLabelsMFTable(obj,tblMF,varargin)  % const % todo this is slow
      mIdx = tblMF.mov;
      assert(isa(mIdx,'MovieIndex'));
      [~,gt] = mIdx.get();
      assert(all(gt) || all(~gt),...
        'Currently only all-GT or all-nonGT supported.');
      gt = gt(1);
      PROPS = Labeler.gtGetSharedPropsStc(gt);
      if obj.hasTrx
        tfaf = obj.(PROPS.TFAF);
      else
        tfaf = [];
      end
      if obj.maIsMA
        max_animals = obj.tracker.sPrmAll.ROOT.MultiAnimal.Track.max_n_animals;
      else
        max_animals = 1;
      end

      if obj.gtIsGTMode
        roi = obj.labelsRoiGT;
      else
        roi = obj.labelsRoi;
      end

      tblMF = Labels.labelAddLabelsMFTableStc(tblMF,obj.(PROPS.LBL),...
        'trxFilesAllFull',tfaf,'trxCache',obj.trxCache,varargin{:},...
        'isma',obj.maIsMA,'maxanimals',max_animals,'roi',roi);
    end

%     function tblMF = labelAddLabelsMFTable_Old(obj,tblMF,varargin)
%       mIdx = tblMF.mov;
%       assert(isa(mIdx,'MovieIndex'));
%       [~,gt] = mIdx.get();
%       assert(all(gt) || all(~gt),...
%         'Currently only all-GT or all-nonGT supported.');
%       gt = gt(1);
%       PROPS = Labeler.gtGetSharedPropsStc(gt);
%       if obj.hasTrx
%         tfaf = obj.(PROPS.TFAF);
%       else
%         tfaf = [];
%       end
%       tblMF = Labels.labelAddLabelsMFTableStc_Old(tblMF,...
%                                                   obj.(PROPS.LPOS),...
%                                                   obj.(PROPS.LPOSTAG),...
%                                                   obj.(PROPS.LPOSTS),...
%                                                   'trxFilesAllFull',tfaf,...
%                                                   'trxCache',obj.trxCache,...
%                                                   varargin{:});
%     end  % function
    
  end

  %% MA
  methods (Static)
    function roi = maRoiXY2RoiFixed(xy,rad)
      % fixed-radius roi, centered on kp centroid
      % xy: [npt x 2]
      % roi: [4x2]
      xymu = mean(xy,1,'omitnan');
      xlo = xymu(1)-rad;
      xhi = xymu(1)+rad;
      ylo = xymu(2)-rad;
      yhi = xymu(2)+rad;
      roi = [xlo ylo; xlo yhi; xhi yhi; xhi ylo];      
    end

    function roi = maRoiXY2RoiScaled(xy,scalefac,fixedmargin)
      % scaled roi, centered on center of bbox; equivalent to 'expanding'
      %  bbox by scalefac*bbox
      % xy: [npt x 2]
      % roi: [4x2]
      xymin = min(xy,[],1,'omitnan');
      xymax = max(xy,[],1,'omitnan');
      xyrad = scalefac * (xymax-xymin)/2 + fixedmargin;
      xymid = (xymax+xymin)/2;
      xylo = xymid-xyrad;
      xyhi = xymid+xyrad;
      roi = [xylo; xylo(1) xyhi(2); xyhi; xyhi(1) xylo(2)];
    end

    function roi = maComputeBboxGeneral(kps,minaa,dopad,padfac,padfloor)
      % Axis-aligned keypoint-derived bounding box 
      %
      % kps: npts x 2
      % minaa: scalar in (0,1). minimum aspect ratio 
      % dopad: scalar logical. if true, do padding
      % padfac: padding factor
      % padfloor: minimum pad in px
      % 
      % roi: 4x2. four (x,y) corners of rectangular roi.
      
      xymin = min(kps,[],1,'omitnan');
      xymax = max(kps,[],1,'omitnan');
      
      if dopad
        rads = (xymax-xymin)/2;
        radmax = max(rads);
        pad = radmax*(padfac-1.0);
        pad = max(pad,padfloor); % pad, padfloor both in raw/input px
        xymin = xymin - pad;
        xymax = xymax + pad;
      end
      
      diams = xymax-xymin;
      [~,ilg] = max(diams);
      if ilg==1
        ism = 2;
      else
        ism = 1;
      end
      aa = diams(ism)/diams(ilg);
      if aa < minaa
        % mindiam = maxdiam * minaa
        rsmall = (diams(ilg)*minaa)/2;
        xyc = (xymin+xymax)/2;
        xymin(ism) = xyc(ism) - rsmall;
        xymax(ism) = xyc(ism) + rsmall;
      end
      
      roi = [xymin xymax]; % [xlo ylo xhi yhi]
      %roi = [xlo ylo; xlo yhi; xhi yhi; xhi ylo];
      idxs = [1 2;1 4;3 4;3 2];
      roi = roi(idxs);
      %bb = [xymin (xymax-xymin)];
    end

    function roi = bbox2roi(bb)
      xylohi = bb;
      xylohi(3:4) = xylohi(1:2)+xylohi(3:4); % now xlo ylo xhi yhi
      %roi = [xlo ylo; xlo yhi; xhi yhi; xhi ylo];
      idxs = [1 2;1 4;3 4;3 2];
      roi = xylohi(idxs);
    end
  end
  methods
%     function maSetPtInfo(obj,ptNames)
%       % ht assumed to be correct even if htEnabled==false
%       obj.maPtNames = ptNames;
%       %obj.maPtHeadTail = ht;
%     end
% 
    function crop_sz = get_ma_crop_sz(obj)
      % Get crop sz for MA
      sagg = TrnPack.aggregateLabelsAddRoi(obj,false,...
        obj.trackParams.ROOT.MultiAnimal.Detect.BBox,...
        obj.trackParams.ROOT.MultiAnimal.LossMask);
      min_crop = obj.trackParams.ROOT.MultiAnimal.LossMask.PadFloor;
      maxx = min_crop; maxy = min_crop;
      for ndx = 1:numel(sagg)
        frs = unique(sagg(ndx).frm);
        for fndx = 1:numel(frs)
          fr = frs(fndx);
          idx = (sagg(ndx).frm == fr);
          rois = sagg(ndx).roi(:,idx);
          rect = [rois(1,:); rois(5,:);...
            rois(3,:) - rois(1,:);...
            rois(6,:) - rois(5,:)]';
          
          conn = rectint(rect,rect)>0;
          gr = graph(conn);
          components = conncomp(gr);
          ncomp = max(components);
          for cndx = 1:ncomp
            cur_roi = rois(:,components==cndx);
            xmin = min(cur_roi(1,:));
            xmax = max(cur_roi(3,:));
            ymin = min(cur_roi(5,:));
            ymax = max(cur_roi(6,:));
            maxx = max(maxx,xmax-xmin);
            maxy = max(maxy,ymax-ymin);
          end
        end
      end
      crop_sz = max(maxx,maxy);
      crop_sz = ceil(crop_sz/min_crop)*min_crop;
    end

    function r = maEstimateTgtCropRad(obj,cropszfac)
      % Don't call directly, doesn't apply mod32 constraint
      spanptl = 95;
      npts = obj.nLabelPoints;
    
      s = cat(1,obj.labels{:});
      p = cat(2,s.p); % 2*npts x n
      p = p.';

      n = size(p,1);
      xy = reshape(p,n,npts,2);
      
      xymin = squeeze(min(xy,[],2)); % n x 2
      xymax = squeeze(max(xy,[],2)); % n x 2
      xyspan = xymax-xymin;
      xyspan = prctile(xyspan(:),spanptl);
      r = xyspan/2*cropszfac;
    end
  
    function roi = maGetLossMask(obj,xy,sPrmLoss)
      % Compute mask roi for keypoints xy 
      %
      % xy: [npts x 2]
      % sPrmLoss: .LossMask params
      %
      % roi: [4x2] [x(:) y(:)] corners of rectangular roi

      if nargin<3
        sPrmLoss = obj.trackParams.ROOT.MultiAnimal.LossMask;
      end      

      tfHT = ~isempty(obj.skelHead) && ~isempty(obj.skelTail);
      tfalignHT = sPrmLoss.AlignHeadTail && tfHT;
      minaa = sPrmLoss.MinAspectRatio;
      padfac = sPrmLoss.PadFactor;
      padflr = sPrmLoss.PadFloor;
      dopad = true;
      
      if tfalignHT
        xyH = xy(obj.skelHead,:);
        xyCent = mean(xy,1,'omitnan');
        
        if ~isempty(obj.skelTail)
          xyT = xy(obj.skelTail,:);
        else
          xyT = xyCent;
          warningNoTrace('No tail point defined; using centroid');
        end

        if any(isnan(xyH)) && any(isnan(xyT))
          phi = 0;
        elseif any(isnan(xyH))
          v = xyCent - xyT;
          phi = atan2(v(2),v(1));
        elseif any(isnan(xyT))
          v = xyH - xyCent;
          phi = atan2(v(2),v(1));
        else
          v = xyH-xyT; % vec from tail->head
          phi = atan2(v(2),v(1)); % azimuth of vec from t->h
        end

        R = rotationMatrix(-phi);

        xyc = xy-xyCent; % kps centered about centroid
        Rxyc = R*xyc.'; % [2xnpts] centered, rotated kps
                        % vec from t->h should point to positive x
        Rroi = Labeler.maComputeBboxGeneral(Rxyc.',minaa,dopad,padfac,padflr);
        % Rroi is [4x2]
        roi = R.'*Rroi.'; 
        roi = roi.'+xyCent;
      else
        roi = Labeler.maComputeBboxGeneral(xy,minaa,dopad,padfac,padflr);
      end
    end
      
  end
  
  %% ViewCal
  methods (Access=private)
    function viewCalCheckCalRigObj(obj,crObj)
      % Basic checks on calrig obj
      
      if ~(isequal(crObj,[]) || isa(crObj,'CalRig')&&isscalar(crObj))
        error('Labeler:viewCal','Invalid calibration object.');
      end
      nView = obj.nview;
      if nView~=crObj.nviews
        error('Labeler:viewCal',...
          'Number of views in project inconsistent with calibration object.');
      end
%     AL 20181108 very strict check, don't worry about this for now. Maybe 
%     later
%       if ~all(strcmpi(obj.viewNames(:),crObj.viewNames(:)))
%         warningNoTrace('Labeler:viewCal',...
%           'Project viewnames do not match viewnames in calibration object.');
%       end
    end
    
  end  % methods (Access=private

  methods   
    function viewCalClear(obj)
      obj.viewCalProjWide = [];
      obj.viewCalibrationData = [];
      obj.viewCalibrationDataGT = [];
      % Currently lblCore is not cleared, change will be reflected in
      % labelCore at next movie change etc
      
      % lc = obj.lblCore;      
      % if lc.supportsCalibration
      %   warning('Labeler:viewCal','');
      % end
    end
    
    function viewCalSetCurrMovie(obj,crObj,varargin)
      % Set calibration object for current movie

%       tfSetViewSizes = myparse(varargin,...
%         'tfSetViewSizes',false); % If true, set viewSizes on crObj per current movieInfo
      
      nMov = obj.nmoviesGTaware;
      if nMov==0 || obj.currMovie==0
        error('Labeler:calib',...
          'Add/select a movie first before setting the calibration object.');
      end

      obj.viewCalCheckCalRigObj(crObj);

      vcdPW = obj.viewCalProjWide;
      if isempty(vcdPW)
        obj.viewCalProjWide = false;
        obj.viewCalibrationData = cell(obj.nmovies,1);
        obj.viewCalibrationDataGT = cell(obj.nmoviesGT,1);
      elseif vcdPW
        warningNoTrace('Labeler:viewCal',...
          'Discarding project-wide calibration data. Calibration data will need to be set on other movies.');
        obj.viewCalProjWide = false;
        obj.viewCalibrationData = cell(obj.nmovies,1);
        obj.viewCalibrationDataGT = cell(obj.nmoviesGT,1);
      else
        assert(iscell(obj.viewCalibrationData));
        assert(iscell(obj.viewCalibrationDataGT));
        szassert(obj.viewCalibrationData,[obj.nmovies 1]);
        szassert(obj.viewCalibrationDataGT,[obj.nmoviesGT 1]);
      end
      
%       obj.viewCalSetCheckViewSizes(obj.currMovie,crObj,tfSetViewSizes);
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.VCD){obj.currMovie} = crObj;
      
      lc = obj.lblCore;
      if lc.supportsCalibration
        lc.projectionSetCalRig(crObj);
      else
        warning('Labeler:viewCal',...
          'Current labeling mode does not utilize view calibration.');
      end
    end
    
  end
  
%   methods (Static)
%     function nptsLbled = labelPosNPtsLbled(lpos)
%       % poor man's export of LabelPosLabeledFramesStats
%       %
%       % lpos: [nPt x d x nFrm x nTgt]
%       % 
%       % nptsLbled: [nFrm]. Number of labeled (non-nan) points for each frame
%       
%       [~,d,nfrm,ntgt] = size(lpos);
%       assert(d==2);
%       assert(ntgt==1,'One target only.');
%       lposnnan = ~isnan(lpos);
%       
%       nptsLbled = nan(nfrm,1);
%       for f = 1:nfrm
%         tmp = all(lposnnan(:,:,f),2); % both x/y must be labeled for pt to be labeled
%         nptsLbled(f) = sum(tmp);
%       end
%     end
%   end
  
  methods (Access=private)
    
    function labelsUpdateNewFrame(obj,force)
      %ticinfo = tic;
      if obj.isinit
        return;
      end
      if exist('force','var')==0
        force = false;
      end
      %fprintf('labelsUpdateNewFrame 1: %f\n',toc(ticinfo)); ticinfo = tic;
      if ~isempty(obj.lblCore) && (obj.prevFrame~=obj.currFrame || force)
        obj.lblCore.newFrame(obj.prevFrame,obj.currFrame,obj.currTarget);
      end
      %fprintf('labelsUpdateNewFrame 2: %f\n',toc(ticinfo)); ticinfo = tic;
      obj.syncPrevAxesVirtualLabels_();
      notify(obj, 'updatePrevAxesLabels');
      %fprintf('labelsUpdateNewFrame 3: %f\n',toc(ticinfo)); ticinfo = tic;
      obj.labels2VizUpdate('dotrkres',true);
      %fprintf('labelsUpdateNewFrame 4: %f\n',toc(ticinfo));
    end

    function labelsUpdateNewTarget(obj,prevTarget)
      if ~isempty(obj.lblCore)
        obj.lblCore.newTarget(prevTarget,obj.currTarget,obj.currFrame);
      end
      obj.syncPrevAxesVirtualLabels_();
      notify(obj, 'updatePrevAxesLabels');
      obj.labels2VizUpdate('dotrkres',true,'setlbls',false,'setprimarytgt',true);
    end

    function labelsUpdateNewFrameAndTarget(obj,prevFrm,prevTgt)
      if ~isempty(obj.lblCore)
        obj.lblCore.newFrameAndTarget(...
          prevFrm,obj.currFrame,...
          prevTgt,obj.currTarget);
      end
      obj.syncPrevAxesVirtualLabels_();
      notify(obj, 'updatePrevAxesLabels');
      obj.labels2VizUpdate('dotrkres',true,'setprimarytgt',true);
    end  % function

    function syncPrevAxesVirtualLabels_(obj)
      % Sync virtual prev-axes label positions with labeler state.
      % In FROZEN mode, set positions from frozen frame info.
      % In LASTSEEN mode, set positions from prevFrame labels.
      if obj.prevAxesMode == PrevAxesMode.FROZEN
        try
          obj.prevAxesSetFrozenLabels_(obj.prevAxesModeTargetSpec_);
        catch
          % do nothing
        end
      elseif ~isnan(obj.prevFrame) && ~isempty(obj.lblPrev_ptsH)
        obj.prevAxesSetLastseenLabels_(obj.currMovie, obj.prevFrame, obj.currTarget);
      else
        setPositionsOfLabelLinesAndTextsToNanBangBang(obj.lblPrev_ptsH, obj.lblPrev_ptsTxtH);
      end
    end  % function
        
  end  % methods (Access=private)
   
  %% GT mode
  methods
    function gtSetGTMode(obj,tf,varargin)
      validateattributes(tf,{'numeric' 'logical'},{'scalar'});
      
      warnChange = myparse(varargin,...
        'warnChange',false); % if true, throw a warning that GT mode is changing
      
      tf = logical(tf);
      if tf==obj.gtIsGTMode
        % none, value unchanged
      else
        if warnChange
          if tf
            warningNoTrace('Entering Ground-Truthing mode.');
          else
            warningNoTrace('Leaving Ground-Truthing mode.');
          end
        end
        obj.gtIsGTMode = tf;
        nMov = obj.nmoviesGTaware;
        if nMov==0
          obj.movieSetNoMovie();
        else
          IMOV = 1; % FUTURE: remember last/previous iMov in "other" gt mode
          obj.movieSetGUI(IMOV);
        end
        obj.syncPropsMfahl_() ;
        obj.notify('updateFrameTableComplete');
        obj.notify('gtIsGTModeChanged');
      end
    end

    function gtThrowErrIfInGTMode(obj)
      if obj.gtIsGTMode
        error('Labeler:gt','Unsupported when in GT mode.');
      end
    end

    function PROPS = gtGetSharedProps(obj)
      PROPS = Labeler.gtGetSharedPropsStc(obj.gtIsGTMode);
    end

%     function gtInitSuggestions(obj,gtSuggType,nSamp)
%       % Init/set GT suggestions using gtGenerateSuggestions
%       tblMFT = obj.gtGenerateSuggestions(gtSuggType,nSamp);
%       obj.gtSetUserSuggestions(tblMFT);
%     end

    function gtSetUserSuggestions(obj,tblMFT,varargin)
      % Set user-specified/defined GT suggestions
      %
      % tblMFT: .mov (MovieIndices), .frm, .iTgt. If [], default to all
      % labeled GT rows in proj
      
      sortcanonical = myparse(varargin,...
        'sortcanonical',false);
      
      if isequal(tblMFT,[])
        fprintf(1,'Setting to-label list for groundtruthing to all current groundtruth labels...\n');
        tblMFT = obj.labelGetMFTableLabeled('useTrain',0,'mftonly',true);
        tblMFT = unique(tblMFT,'rows');
        fprintf(1,'... found %d groundtruth labels.\n',height(tblMFT));
      end
      
      if ~istable(tblMFT) && ~all(tblfldscontains(tblMFT,MFTable.FLDSID))
        error('Specified table is not a valid Movie-Frame-Target table.');
      end
      
      tblMFT = tblMFT(:,MFTable.FLDSID);
      
      if ~isa(tblMFT.mov,'MovieIndex')
        warningNoTrace('Table .mov is numeric. Assuming positive indices into GT movie list (.movieFilesAllGT).');
        tblMFT.mov = MovieIndex(tblMFT.mov,true);
      end
      
      if isempty(tblMFT)
        % pass
      else
        [tf,tfGT] = tblMFT.mov.isConsistentSet();
        if ~(tf && tfGT)
          error('All MovieIndices in input table must reference GT movies.');
        end
        
        n0 = height(tblMFT);
        n1 = height(unique(tblMFT(:,MFTable.FLDSID)));
        if n0~=n1
          error('Input table appears to contain duplicate rows.');
        end
        
        if sortcanonical
          tblMFT2 = MFTable.sortCanonical(tblMFT);
          if ~isequal(tblMFT2,tblMFT)
            warningNoTrace('Sorting table into canonical row ordering.');
            tblMFT = tblMFT2;
          end
        else
          % UI requires sorting by movies; hopefully the movie sort leaves
          % row ordering otherwise undisturbed. This appears to be the case
          % in 2017a.
          %
          % See issue #201. User has a gtSuggestions table that is not fully
          % sorted by movie, but with a desired random row order within each
          % movie. A full/canonical sorting would be undesireable.
          tblMFT2 = sortrows(tblMFT,{'mov'},{'descend'}); % descend as gt movieindices are negative
          if ~isequal(tblMFT2,tblMFT)
            warningNoTrace('Sorting table by movie.');
            tblMFT = tblMFT2;
          end
        end
      end
      
      obj.gtSuggMFTable = tblMFT;
      obj.gtUpdateSuggMFTableLbledComplete();
      obj.gtTblRes = [];
      obj.notify('gtSuggUpdated');
      obj.notify('gtResUpdated');      
    end

    function gtUpdateSuggMFTableLbledComplete(obj,varargin)
      % update .gtUpdateSuggMFTableLbled from .gtSuggMFTable/.labeledposGT
      
      donotify = myparse(varargin,...
        'donotify',false); 
      
      tbl = obj.gtSuggMFTable;
      if isempty(tbl)
        obj.gtSuggMFTableLbled = false(0,1);
        if donotify
          obj.notify('gtSuggUpdated'); % use this event for full/general update
        end
        return;
      end
      
      tfAllTgtsLbled = obj.getIsLabeledGT(tbl);
%       lposCell = obj.labeledposGT;
%       fcn = @(zm,zf,zt) (nnz(isnan(lposCell{-zm}(:,:,zf,zt)))==0);
%       % a mft row is labeled if all pts are either labeled, or estocc, or
%       % fullocc (lpos will be inf which is not nan)
%       fprintf(2,'Todo: replace inefficient\n');
%       tfAllTgtsLbled = rowfun(fcn,tbl,...
%         'InputVariables',{'mov' 'frm' 'iTgt'},...
%         'OutputFormat','uni');
      szassert(tfAllTgtsLbled,[height(tbl) 1]);
      obj.gtSuggMFTableLbled = tfAllTgtsLbled;
      if donotify
        obj.notify('gtSuggUpdated'); % use this event for full/general update
      end
    end

    function gtUpdateSuggMFTableLbledIncremental(obj)
      % Assume that .labeledposGT and .gtSuggMFTableLbled differ at most in
      % currMov/currTarget/currFrame
      
      assert(obj.gtIsGTMode);
      % If not gtMode, currMovie/Frame/Target do not apply to GT
      % movies/labels. Maybe call gtUpdateSuggMFTableLbledComplete() in
      % this case.
      
      iMov = obj.currMovie;
      frm = obj.currFrame;
      iTgt = obj.currTarget;
      tblGT = obj.gtSuggMFTable;
      
      tfInTbl = tblGT.mov==(-iMov) & tblGT.frm==frm & tblGT.iTgt==iTgt;
      nRow = nnz(tfInTbl);
      if nRow>0
        assert(nRow==1);
        s = obj.labelsGT{iMov};
        [tf,p] = Labels.isLabeledFT(s,frm,iTgt);
        obj.gtSuggMFTableLbled(tfInTbl) = tf && nnz(isnan(p))==0;
        obj.notify('gtSuggMFTableLbledUpdated');
      end
    end

%     function tblMFT = gtGenerateSuggestions(obj,gtSuggType,nSamp)
%       assert(isa(gtSuggType,'GTSuggestionType'));
%       
%       % Start with full table (every frame), then sample
%       mfts = MFTSetEnum.AllMovAllTgtAllFrm;
%       tblMFT = mfts.getMFTable(obj);
%       tblMFT = gtSuggType.sampleMFTTable(tblMFT,nSamp);
%     end

    function [tf,idx] = gtCurrMovFrmTgtIsInGTSuggestions(obj)
      % Determine if current movie/frm/target is in gt suggestions.
      % 
      % tf: scalar logical
      % idx: if tf==true, row index into .gtSuggMFTable. If tf==false, idx
      %  is indeterminate.
      
      assert(obj.gtIsGTMode);
      mIdx = obj.currMovIdx;
      frm = obj.currFrame;
      iTgt = obj.currTarget;
      tblGT = obj.gtSuggMFTable;
      tf = mIdx==tblGT.mov & frm==tblGT.frm & iTgt==tblGT.iTgt;
      idx = find(tf);
      assert(isempty(idx) || isscalar(idx));
      tf = ~isempty(idx);
    end

    function nNewLbls = gtComputeNewLabelCount(obj)
      % Determine the number of non-suggested labels.

      tblMFTSugg = obj.gtSuggMFTable;
      mfts = MFTSet(MovieIndexSetVariable.AllGTMov,...
        FrameSetVariable.LabeledFrm,FrameDecimationFixed(1),...
        TargetSetVariable.AllTgts);    
      tblMFTLbld = mfts.getMFTable(obj);
      
      % [tfSuggAnyLbl,loc] = tblismember(tblMFTSugg,tblMFTLbld,MFTable.FLDSID);
      mftflds = MFTable.FLDSID;
      if obj.maIsMA  
        % remove tgt field for multi-animal projects
        mftflds(strcmp(mftflds,'iTgt')) = [];
      end
      tfLbldExtra = ~tblismember(tblMFTLbld,tblMFTSugg,mftflds);

      nNewLbls = nnz(tfLbldExtra);
    end  % function

    function tblMFT_SuggAndLbled = gtGetTblSuggAndLbled(obj,whichlabels)
      % Compile table of GT suggestions with their labels.

      if nargin < 2 || isempty(whichlabels),
        whichlabels = 'suggestonly' ;
      end
      assert(strcmp(whichlabels, 'all') || strcmp(whichlabels, 'suggestonly')) ;

      tblMFTSugg = obj.gtSuggMFTable;
      mfts = MFTSet(MovieIndexSetVariable.AllGTMov,...
        FrameSetVariable.LabeledFrm,FrameDecimationFixed(1),...
        TargetSetVariable.AllTgts);    
      tblMFTLbld = mfts.getMFTable(obj);
      
      % [tfSuggAnyLbl,loc] = tblismember(tblMFTSugg,tblMFTLbld,MFTable.FLDSID);
      mftflds = MFTable.FLDSID;
      if obj.maIsMA  
        % remove tgt field for multi-animal projects
        mftflds(strcmp(mftflds,'iTgt')) = [];
      end
      [tfSuggIsLbld,loc] = tblismember(tblMFTSugg,tblMFTLbld,mftflds);
      tfLbldExtra = ~tblismember(tblMFTLbld,tblMFTSugg,mftflds);

      nNewLbls = nnz(tfLbldExtra);
      if nNewLbls > 0 && strcmpi(whichlabels,'all')
        obj.gtSetUserSuggestions([]);
        tblMFTSugg = obj.gtSuggMFTable;
        [tfSuggIsLbld,loc] = tblismember(tblMFTSugg,tblMFTLbld,mftflds);
      end
      
      % tblMFTLbld includes rows where any pt/coord is labeled;
      % obj.gtSuggMFTableLbled is only true if all pts/coords labeled 
      tfSuggFullyLbled = obj.getIsLabeledGT(tblMFTSugg,true); % true asks for fully labeled
      assert(all(tfSuggIsLbld(tfSuggFullyLbled)));
      tfSuggPartiallyLbled = tfSuggIsLbld & ~tfSuggFullyLbled;
      tfSuggUnLbled = ~tfSuggIsLbld;
      
      nSuggUnlbled = nnz(tfSuggUnLbled);
      if nSuggUnlbled>0
        warningNoTrace('Labeler:gt',...
          '%d suggested GT frames have not been labeled.',nSuggUnlbled);
      end

      nSuggPartiallyLbled = nnz(tfSuggPartiallyLbled);
      if nSuggPartiallyLbled>0
        warningNoTrace('Labeler:gt',...
          '%d suggested GT frames have only been partially labeled.',nSuggPartiallyLbled);
      end

      % Labeled GT table, in order of tblMFTSugg
      tblMFT_SuggAndLbled = tblMFTLbld(loc(tfSuggIsLbld),:);
      if obj.maIsMA
        tblMFT_SuggAndLbled = MFTable.unsetTgt(tblMFT_SuggAndLbled );
      end
    end  % function

    function gtComputeGTPerformance(obj,varargin)
      % Front door entry point for computing gt performance
      
      % Deal with optional args
      [whichlabels,argsrest] = myparse_nocheck(varargin,'whichlabels','suggestonly'); % whether to use all labels ('all'), suggestonly ('suggestonly')
      assert(strcmp(whichlabels, 'all') || strcmp(whichlabels, 'suggestonly')) ;
      [useLabels2] = myparse(argsrest,...
                             'useLabels2',false); % if true, use labels2 "imported preds" instead of tracking

      % Make sure in GT mode
      if ~obj.gtIsGTMode
        error('Project is not in Ground-Truthing mode.');
      end

      % On to business...
      obj.pushBusyStatus('Compiling list of Ground Truth Labels frames and tracking them...');
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      tblMFT = obj.gtGetTblSuggAndLbled(whichlabels);

      % Either spawn the computation of the GT predictions, or import them and show the
      % results.
      if useLabels2 
        % Use imported predictions
        obj.showGTResults('lblTbl',tblMFT,'useLabels2',useLabels2);
      else
        % Tracking runs in a separate async process spawned by shell.
        % .showGTResults() gets called from a callback registered on the completion of
        % this async process.

        backend = obj.trackDLBackEnd;

        % If backend is AWS, abort.
        if backend.type == DLBackEnd.AWS ,
          error('Cannot use AWS backend to do GT computation.  Pick a different backend.')
        end
        
        % What exactly are we doing here?  -- ALT, 2025-01-14
        [movidx,~,newmov] = unique(tblMFT.mov);
        movidx_new = [];
        for ndx = 1:numel(movidx)
          mn = movidx(ndx).get();
          movidx_new(end+1) = mn;  %#ok<AGROW> 
        end
        movidx = movidx_new;
        tblMFT.mov = newmov;
        movfiles = obj.movieFilesAllFullGTaware(movidx,:);

        % What exactly are we doing here?  -- ALT, 2025-01-14
        if obj.hasTrx
          trxfiles = obj.trxFilesAllFullGTaware;
          trxfiles = trxfiles(movidx,:);
        else
          trxfiles = {};
        end

        % What exactly are we doing here?  -- ALT, 2025-01-14
        if obj.cropProjHasCrops
          cropInfo = obj.getMovieFilesAllCropInfoGTAware();
          croprois = cell([size(movfiles,1),obj.nview]);
          for i = 1:size(movfiles,1)
            for j = 1:obj.nview
              croprois{i,j} = cropInfo{movidx(i)}(j).roi;
            end
          end
        else
          croprois = {};
        end

        % What exactly are we doing here?  -- ALT, 2025-01-14
        caldata = obj.viewCalibrationDataGTaware;
        if ~isempty(caldata)
          if ~obj.viewCalProjWide
            caldata = caldata(movidx);
          end
        end

        % % Check that everything is set up for tracking
        % % (We added this mainly to spin-up the AWS backend when in use, during an
        % % abortive attempt to AWS GT working.)
        % [tfCanTrack,reason] = obj.trackCanTrack(tblMFT) ;
        % if ~tfCanTrack ,
        %   error('Setup for GT tracking failed: %s', reason) ;
        % end

        % Tell the tracker to spawn the tracking jobs
        tObj = obj.tracker;
        totrackinfo = ...
          ToTrackInfo('tblMFT',tblMFT,'movfiles',movfiles,...
                      'trxfiles',trxfiles,'views',1:obj.nview,'stages',1:tObj.getNumStages(),'croprois',croprois,...
                      'calibrationdata',caldata,'isma',obj.maIsMA,'isgtjob',true,...
                      'islistjob',true);
        tObj.trackList('totrackinfo',totrackinfo,'backend',backend,'isgt',true,argsrest{:});

        % Broadcast a notification about recent events
        obj.notify('didSpawnTrackingForGT') ;
      end  % if
    end  % function
    
    function showGTResults(obj,varargin)
      [gtResultTbl,tblLbl,useLabels2] = ...
        myparse(varargin,...
                'gtResultTbl',[],...
                'lblTbl',[],...
                'useLabels2',false);

      if isempty(tblLbl)
        if ~isempty(gtResultTbl),
          tblLbl = gtResultTbl(:,MFTable.FLDSID);
        else
          tblLbl = obj.gtGetTblSuggAndLbled();
        end
      end
        
      if useLabels2
        fprintf('Computing GT performance with %d GT rows.\n',...
                height(tblLbl));
        
        % wbObj = WaitBarWithCancel('Compiling Imported Predictions');
        % oc = onCleanup(@()delete(wbObj));
        obj.progressMeter_.arm('title', 'Compiling Imported Predictions') ;
        oc = onCleanup(@()(obj.disarmProgressMeter())) ;
        gtResultTbl = obj.labelGetMFTableLabeled('wbObj',obj.progressMeter_,...          
                                                 'useLabels2',true,...  % in GT mode, so this compiles labels2GT
                                                 'tblMFTrestrict',tblLbl);
        if obj.progressMeter_.wasCanceled
          warningNoTrace('Labeler property .gtTblRes not set.');
          return
        end
        
        gtResultTbl.pTrk = gtResultTbl.p; % .p is imported positions => imported tracking
        gtResultTbl(:,'p') = [];
      end

      obj.gtComputeGTPerformanceTable(tblLbl,gtResultTbl); % also sets obj.gtTblRes
      % obj.didSpawnTrackingForGT_ = [] ;  % reset this
      obj.notify('didComputeGTResults') ;
      obj.popBusyStatus();
    end  % function

    function tblGTres = gtComputeGTPerformanceTable(obj, ...
                                                    tblMFT_SuggAndLbled, ...
                                                    tblTrkRes, ...
                                                    varargin)
      % Compute GT performance 
      % 
      % tblMFT_SuggAndLbled: MFTable, no shape/label field (eg .p or
      % .pLbl). This will be added with .labelAddLabelsMFTable.
      % tblTrkRes: MFTable, predictions/tracking in field .pTrk. .pTrk is
      % in absolute coords if rois are used.
      %
      % At the moment, all rows of tblMFT_SuggAndLbled must be in tblTrkRes
      % (wrt MFTable.FLDSID). ie, there must be tracking results for all
      % suggested/labeled rows.
      %
      % All position-fields (.pTrk, .pLbl, .pTrx) in output/result are in 
      % absolute coords.
      % 
      % Assigns .gtTblRes.

%       pTrkOccThresh = myparse(varargin,...
%         'pTrkOccThresh',0.5 ... % threshold for predicted occlusions
%         );
      
      tblLblMovStr = tblMFT_SuggAndLbled;
      tblLblMovStr.mov = obj.getMovieFilesAllFullMovIdx(tblMFT_SuggAndLbled.mov(:,1));

      tblTrkResMovStr = tblTrkRes;
      tblTrkResMovStr.mov = obj.getMovieFilesAllFullMovIdx(tblTrkResMovStr.mov(:,1));

      [tf,loc] = tblismember(tblLblMovStr,tblTrkResMovStr,MFTable.FLDSID);
      if ~all(tf)
        warningNoTrace('Tracking/prediction results not present for %d GT rows. Results will be computed with those rows removed.',...
          nnz(~tf));
        tblMFT_SuggAndLbled = tblMFT_SuggAndLbled(tf,:);
        loc = loc(tf);
      end      
      tblTrkRes = tblTrkRes(loc,:);
      
      if obj.maIsMA
        maxn = obj.trackParams.ROOT.MultiAnimal.Track.max_n_animals;
      else
        maxn = 1;
      end

      tblMFT_SuggAndLbled = obj.labelAddLabelsMFTable(tblMFT_SuggAndLbled,'isma',obj.maIsMA,'maxanimals',maxn);

      if obj.maIsMA
        [err,fp,fn] = computeMAErr(tblTrkRes,tblMFT_SuggAndLbled,obj.tracker.sPrmAll.ROOT.MultiAnimal.multi_loss_mask);  % nframes x maxn x nkeypoints
        fp = sum(fp,2,'omitmissing');
        fn = sum(fn,2,'omitmissing');
      else
        pTrk = tblTrkRes.pTrk; % absolute coords
        pLbl = tblMFT_SuggAndLbled.p; % absolute coords
        nrow = size(pTrk,1);
        npts = obj.nLabelPoints;
        szassert(pTrk,[nrow 2*npts]);
        szassert(pLbl,[nrow 2*npts]);
        
        % L2 err matrix: [nrow x npt]
        pTrk = reshape(pTrk,[nrow npts 2]);
        pLbl = reshape(pLbl,[nrow npts 2]);
        err = sqrt(sum((pTrk-pLbl).^2,3));      
        tflblinf = any(isinf(pLbl),3); % [nrow x npts] fully-occ indicator mat; lbls currently coded as inf
        err(tflblinf) = nan; % treat fully-occ err as nan here
        fp = nan([nrow,1]);
        fn = nan([nrow,1]);
      end

      muerr = mean(err,ndims(err),'omitnan'); % and ignore in meanL2err  
      % in MA case, will be nframes x nanimals, take the mean again
      muerr = mean(muerr,2,'omitnan');
          
      % ctab for occlusion pred
%       % this is not adding value yet
%       tfTrkHasOcc = isfield(tblTrkRes,'pTrkocc');
%       if tfTrkHasOcc        
%         pTrkocctf = tblTrkRes.pTrkocc>=pTrkOccThresh;
%         szassert(pTrkocctf,size(tflblinf));
%       end        
      
      tblTmp = tblMFT_SuggAndLbled(:,{'p' 'pTS' 'tfocc' 'pTrx'});
      tblTmp.Properties.VariableNames = {'pLbl' 'pLblTS' 'tfoccLbl' 'pTrx'};
      if tblfldscontains(tblTrkRes,'pTrx')
        if ~obj.maIsMA
           assert(isequal(tblTrkRes.pTrx,tblTmp.pTrx),'Mismatch in .pTrx fields.');
        end
        tblTrkRes(:,'pTrx') = [];
      end
      tblGTres = [tblTrkRes tblTmp table(err,muerr,fp,fn,'VariableNames',{'L2err' 'meanL2err','FP','FN'})];
      
      obj.gtTblRes = tblGTres;
      obj.notify('gtResUpdated');
    end  % function

    function gtClearGTPerformanceTable(obj)
      obj.gtTblRes = [] ;
      obj.notify('gtResUpdated') ;
    end

    function [nextmft,loc] = gtNextUnlabeledMFT(obj)

      fldsid = MFTable.FLDSID;

      currmft = MFTable.allocateTable(fldsid,1);
      currmft.mov(1) = obj.currMovIdx;
      currmft.frm(1) = obj.currFrame;
      currmft.iTgt(1) = obj.currTarget;

      tblLbl = obj.gtSuggMFTable(~obj.gtSuggMFTableLbled,fldsid);
      isafter = MFTable.isAfter(tblLbl,currmft);
      if ~any(isafter),
        nextmft = MFTable.emptyTable(fldsid);
        loc = [];
        return;
      end
      loc = find(isafter,1);
      nextmft = tblLbl(loc,:);

    end

    function gtNextUnlabeledUI(obj)
      % Like pressing "Next Unlabeled" in GTManager.

      if ~obj.gtIsGTMode,
        warningNoTrace('Not in GT mode.');
        return;
      end

      nextmft = obj.gtNextUnlabeledMFT();
      if isempty(nextmft),
        msgbox('No more unlabeled frames in to-label list.','','modal');
        return;
      end

      iMov = nextmft.mov.get();
      if iMov~=obj.currMovie
        obj.movieSetGUI(iMov);
      end
      obj.setFrameAndTargetGUI(nextmft.frm,nextmft.iTgt);

    end
    

    function [iMov,iMovGT] = gtCommonMoviesRegGT(obj)
      % Find movies common to both regular and GT lists
      %
      % For multiview projs, movienames must match across all views
      % 
      % iMov: vector of positive ints, reg movie(set) indices
      % iMovGT: " gt movie(set) indices

      [iMov,iMovGT] = Labeler.identifyCommonMovSets(...
        obj.movieFilesAllFull,obj.movieFilesAllGTFull);
    end

    function fname = getDefaultFilenameExportGTResults(obj)
      gtstr = 'gtresults';
      rawname = getDefaultFilenameExport(obj,gtstr,'.mat');
      fname = FSPath.macroReplace(rawname,sMacro);
    end

    function tbl = gtLabeledFrameSummary(obj)
      % return/print summary of gt movies with number of labels
      
      imov = obj.gtSuggMFTable.mov.abs();
      imovun = unique(imov);
      tflbld = obj.gtSuggMFTableLbled;

      imovuncnt = arrayfun(@(x)nnz(x==imov),imovun);
      imovunlbledcnt = arrayfun(@(x)nnz(x==imov & tflbld),imovun);      
      tbl = table(imovun,imovunlbledcnt,imovuncnt,...
        'VariableNames',{'GT Movie Index' 'Labeled Frames' 'Total GT Frames'});
    end
  end
  methods (Static)
    function PROPS = gtGetSharedPropsStc(gt)
      PROPS = Labeler.PROPS_GTSHARED;
      if gt
        PROPS = PROPS.gt;
      else
        PROPS = PROPS.reg;
      end
    end
  end
  
  %% Susp 
  methods
    
    function suspInit(obj)
      obj.suspScore = cell(obj.nmovies,1);
      obj.suspSelectedMFT = MFTable.emptySusp;
      obj.suspComputeFcn = [];
    end
    
    function suspSetComputeFcn(obj,fcn)
      assert(isa(fcn,'function_handle'));      
      obj.suspInit();
      obj.suspComputeFcn = fcn;
    end
    
    function tfsucc = suspCompute(obj)
      % Populate .suspScore, .suspSelectedMFT by calling .suspComputeFcn
      
      fcn = obj.suspComputeFcn;
      if isempty(fcn)
        error('Labeler:susp','No suspiciousness function has been set.');
      end
      [suspscore,tblsusp,diagstr] = fcn(obj);
      if isempty(suspscore)
        % cancel/fail
        warningNoTrace('Labeler:susp','No suspicious scores computed.');
        tfsucc = false;
        return;
      end
      
      obj.suspVerifyScore(suspscore);
      if ~isempty(tblsusp)
        if ~istable(tblsusp) && all(ismember(MFTable.FLDSSUSP,...
                                  tblsusp.Properties.VariableNames'))
          error('Labeler:susp',...
            'Invalid ''tblsusp'' output from suspicisouness computation.');
        end
      end
      
      obj.suspScore = suspscore;
      obj.suspSelectedMFT = tblsusp;
      obj.suspDiag = diagstr;
      
      tfsucc = true;
    end
    
  end  % function
  
  methods (Hidden)
    
    function suspVerifyScore(obj,suspscore)
      nmov = obj.nmoviesGTaware;
      if ~(iscell(suspscore) && numel(suspscore)==nmov)
        error('Labeler:susp',...
          'Invalid ''suspscore'' output from suspicisouness computation.');
      end
      gt = obj.gtIsGTMode;
      for imov=1:nmov
        [nfrm,ntgt] = obj.getNFrmNTrx(gt,imov);
        if ~isequal(size(suspscore{imov}),[nfrm ntgt])
          error('Labeler:susp',...
            'Invalid ''suspscore'' output from suspicisouness computation.');
        end
      end
    end
  end

    
        
%     function setSuspScore(obj,ss)
%       assert(~obj.isMultiView);
%       
%       if isequal(ss,[])
%         % none; this is ok
%       else
%         nMov = obj.nmovies;
%         nTgt = obj.nTargets;
%         assert(iscell(ss) && isvector(ss) && numel(ss)==nMov);
%         for iMov = 1:nMov
%           ifo = obj.movieInfoAll{iMov,1}; 
%           assert(isequal(size(ss{iMov}),[ifo.nframes nTgt]),...
%             'Size mismatch for score for movie %d.',iMov);
%         end
%       end
%       
%       obj.suspScore = ss;
%     end
    
%     function updateCurrSusp(obj)
%       % Update .currSusp from .suspScore, currMovie, .currFrm, .currTarget
%       
%       tfDoSusp = ~isempty(obj.suspScore) && ...
%                   obj.hasMovie && ...
%                   obj.currMovie>0 && ...
%                   obj.currFrame>0;
%       if tfDoSusp
%         ss = obj.suspScore{obj.currMovie};
%         obj.currSusp = ss(obj.currFrame,obj.currTarget);       
%       else
%         obj.currSusp = [];
%       end
%       if ~isequal(obj.currSusp,[])
%         obj.currImHud.updateSusp(obj.currSusp);
%       end
%     end
  
%% PreProc
%  
% "Preprocessing" represents transformations taken on raw movie/image data
% in preparation for tracking: eg histeq, cropping, nbormasking, bgsubbing.
% The hope is that these transformations can normalize/clean up data to
% increase overall tracking performance. One imagines that many preproc
% steps apply equally well to any particular tracking algorithm; then there
% may be algorithm-specific preproc steps as well.
% 
% Preproc computations are governed by a set of parameters. Within the UI
% these are available under the Track->Configure Tracking Parameters menu.
% The user is also able to turn on/off preprocessing steps as they desire.
% 
% Conceptually Preproc is a linear pipeline accepting raw movie data as
% the input and producing images ready for consumption by the tracker. 
% Ideally the PreProc pipeline is the SOLE SOURCE of input data (besides
% parameters) for trackers.
% 
% APT caches preprocessed data for two reasons. It can be convenient from a
% performance standpoint, eg if one is iteratively retraining and gauging
% performance on a test set of frames. Another reason is that by caching
% data, users can conveniently browse training data, diagnostic metadata,
% etc.
%
% For CPR, the PP cache is in-memory. For DL trackers, the PP cache is on
% disk as the PP data/images ultimately need to be written to disk anyway
% for tracking.
% 
% When a preprocessing parameter is mutated, the pp cache must be cleared.
% We define an invariant so that any live data in the PP cache must have
% been generated with the current PP parameters. However, there is a
% question about whether a trained tracker should also be cleared, if a PP
% parameter is altered.
% 
% The conservative route is to always clear any trained tracker if any PP
% parameter is mutated. However, there may be cases where this is
% undesirable. Viewed as a black box, the tracker accepts images and
% produces annotations; the fact that it was trained on images produced in
% a certain way doesn't guarantee that a user might not want to apply it to
% images produced in a (possibly slightly) different way. For instance,
% after a tracker has already been trained, a background image for a movie
% might be incrementally improved. Using the old/trained tracker with data 
% PPed using the new background image with bgsub might be perfectly
% reasonable. There may be other interesting cases as well, eg train with
% no nbor mask, track with nbor mask, where "more noise" during training is
% desireable to improve generalization.
% 
% With preprocessing clearly separated from tracking, it should be fairly
% easy to enable these more complex scenarios in the future if desired.
% 
% Parameter mutation Summary
% - PreProc parameter mutated -> PP cache cleared; trained tracker cleared (maybe in future, something fancier)
% - Tracking parameter mutated -> PP cache untouched; trained tracker cleared
% 
% NOTE: During a retrain(), a snapshot of preprocParams should be taken and
% saved with the trained tracker. That way it is recorded/known precisely 
% how that tracker was generated.

  methods
    
    function preProcInit(obj)
      %obj.preProcParams = [];
      % obj.preProcH0 = [];
      obj.ppdbInit();
      obj.movieFilesAllHistEqLUT = cell(obj.nmovies,obj.nview);
      obj.movieFilesAllGTHistEqLUT = cell(obj.nmoviesGT,obj.nview);
    end
    
    function ppdbInit(obj)
      if isempty(obj.ppdb)
        obj.ppdb = PreProcDB();
      end
      obj.ppdb.init();
    end
    
%     function tfPPprmsChanged = preProcSetParams(obj,ppPrms) % THROWS
%       % ppPrms: OLD-style preproc params
%       
%       assert(isstruct(ppPrms));
% 
%       if ppPrms.histeq 
%         if ppPrms.BackSub.Use
%           error('Histogram Equalization and Background Subtraction cannot both be enabled.');
%         end
%         if ppPrms.NeighborMask.Use
%           error('Histogram Equalization and Neighbor Masking cannot both be enabled.');
%         end
%       end
%       
%       ppPrms0 = obj.preProcParams;
%       tfPPprmsChanged = ~isequaln(ppPrms0,ppPrms);
%       if tfPPprmsChanged
%         warningNoTrace('Preprocessing parameters altered; data cache cleared.');
%         obj.preProcInitData();
%         obj.ppdbInit(); % AL20190123: currently only ppPrms.TargetCrop affect ppdb
%         
%         bgPrms = ppPrms.BackSub;
%         mrs = obj.movieReader;
%         for i=1:numel(mrs)
%           mrs(i).open(mrs(i).filename,'bgType',bgPrms.BGType,...
%             'bgReadFcn',bgPrms.BGReadFcn); 
%           % mrs(i) should already be faithful to .forceGrayscale, 
%           % .movieInvert, cropInfo
%         end
%       end
%       obj.preProcParams = ppPrms;
%     end
    
    function preProcNonstandardParamChanged(obj)
      % Normally, preProcSetParams() handles changes to preproc-related
      % parameters. If a preproc parameter changes, then the output
      % variable tfPPprmsChanged is set to true, and the caller 
      % clears/updates the tracker as necessary.
      %
      % There are two nonstandard pre-preprocessing "parameters" that are 
      % set outside of preProcSetParams: .movieInvert, and
      % .movieFilesAll*CropInfo. If these properties are mutated, any
      % preprocessing data, trained tracker, and tracking results must also
      % be cleared.
      
      obj.ppdbInit();
      trackers = obj.trackerHistory_ ;
      for i=1:numel(trackers)
        if trackers{i}.hasBeenTrained()
          warningNoTrace('Trained tracker(s) and tracking results cleared.');
          break
        end
      end
      obj.deleteOldTrackers();
      obj.resetCurrentTracker() ;
    end
    
    function tblP = preProcCropLabelsToRoiIfNec(obj,tblP,varargin)
      % Add .roi column to table if appropriate/nec
      %
      % Preconds: One of these must hold
      %   - proifld and pabsfld are both not fields/cols
      %     - if roi is present as a field, its existing value is checked/confirmed
      %   - None of {roi, proifld, pabsfld} are present
      %
      % PostConditions:
      % If hasTrx, modify tblP as follows:
      %   - add .roi
      %   - add .pRoi (proifld), p relative to ROI
      %   - set .pAbs (pabsfld) to the original/absolute .p (pfld)
      %   - set .p (pfld) to be .pRoi (proifld)
      % If cropProjHasCrops, same as hasTrx.
      % Otherwise:
      %   - no .roi
      %   - set .pAbs (pabsfld) to be .p (pfld)
      
      [prmsTgtCrop,doRemoveOOB,pfld,pabsfld,proifld] = myparse(varargin,...
        'prmsTgtCrop',[],...
        'doRemoveOOB',true,...
        'pfld','p',...  % see desc above
        'pabsfld','pAbs',... % etc
        'proifld','pRoi'... % 
        );
      if isempty(prmsTgtCrop)
        if isempty(obj.trackParams)
          error('Please set tracking parameters.');
        else
          prmsTgtCrop = obj.trackParams.ROOT.MultiAnimal.TargetCrop;
        end
      end
      
      if obj.hasTrx || obj.cropProjHasCrops
        % at the end of this branch, all fields .roi, proifld, pabsfld must
        % be added
        
        tf2pflds = tblfldscontains(tblP,{proifld pabsfld});
        tfroi = tblfldscontains(tblP,'roi');
        %assert(all(tf2pflds) || ~any(tf2pflds));
        if ~any(tf2pflds)
          if tfroi
            % save existing .roi fld and confirm it matches the one that
            % will be added below
            roi0 = tblP.roi;
            tblP(:,'roi') = [];
          end
          if obj.hasTrx
            roiRadius = maGetTgtCropRad(prmsTgtCrop);
            tblP = obj.labelMFTableAddROITrx(tblP,roiRadius,...
              'rmOOB',doRemoveOOB,...
              'pfld',pfld,'proifld',proifld);
          else
            tblP = obj.labelMFTableAddROICrop(tblP,...
              'rmOOB',doRemoveOOB,...
              'pfld',pfld,'proifld',proifld);
          end
          if tfroi
            assert(isequaln(roi0,tblP.roi));
          end
          tblP.(pabsfld) = tblP.(pfld);
          tblP.(pfld) = tblP.(proifld);
        else % one of the two fields exists
          assert(all(tf2pflds) && tfroi);
        end
      else
        if tblfldscontains(tblP,pabsfld) % AL20190207 add this now some downstream clients want it
          assert(isequaln(tblP.(pfld),tblP.(pabsfld)));
        else
          tblP.(pabsfld) = tblP.(pfld);
        end
        % none; tblP.p is .pAbs. No .roi field.
      end
    end
    
    function tblP = preProcGetMFTableLbled(obj,varargin)
      % labelGetMFTableLabeled + preProcCropLabelsToRoiIfNec
      %
      % Get MFTable for all movies/labeledframes. Exclude partially-labeled 
      % frames.
      %
      % tblP: MFTable of labeled frames. Precise cols may vary. However:
      % - MFTable.FLDSFULL are guaranteed where .p is:
      %   * The absolute position for single-target trackers
      %   * The position relative to .roi for multi-target trackers
      % - .roi is guaranteed when .hasTrx or .cropProjHasCropInfo

      [wbObj,tblMFTrestrict,gtModeOK,prmsTgtCrop,doRemoveOOB,...
        treatInfPosAsOcc] = myparse(varargin,...
        'wbObj',[], ... % optional WaitBarWithCancel. If cancel:
                    ... % 1. obj const 
                    ... % 2. tblP indeterminate
        'tblMFTrestrict',[],... % see labelGetMFTableLabeld
        'gtModeOK',false,... % by default, this meth should not be called in GT mode
        'prmsTgtCrop',[],...
        'doRemoveOOB',true,...
        'treatInfPosAsOcc',true  ... % if true, treat inf labels as 
                                 ... % 'fully occluded'; if false, remove 
                                 ... % any rows with inf labels
        ); 
      tfWB = ~isempty(wbObj);
      if ~isempty(tblMFTrestrict)
        tblfldsassert(tblMFTrestrict,MFTable.FLDSID);
      end
      
      if obj.gtIsGTMode && ~gtModeOK
        error('Unsupported in GT mode.');
      end
      
      tblP = obj.labelGetMFTableLabeled('wbObj',wbObj,...
        'tblMFTrestrict',tblMFTrestrict);
      if tfWB && wbObj.isCancel
        % tblP indeterminate, return it anyway
        return;
      end
      if isempty(tblP),
        return;
      end
      
      % In tblP we can have:
      % * regular labels: .p is non-nan & non-inf; .tfocc is false
      % * estimated-occluded labels: .p is non-nan & non-inf; .tfocc is true
      % * fully-occ labels: .p is inf, .tfocc is false

      % For now we don't accept partially-labeled rows
%       tfnanrow = any(isnan(tblP.p),2);
%       nnanrow = nnz(tfnanrow);
%       if nnanrow>0
%         warningNoTrace('Labeler:nanData',...
%           'Not including %d partially-labeled rows.',nnanrow);
%       end
%       tblP = tblP(~tfnanrow,:);
        
      % Deal with full-occ rows in tblP. Do this here as otherwise the call 
      % to preProcCropLabelsToRoiIfNec will consider inf labels as "OOB"
      if treatInfPosAsOcc
        tfinf = isinf(tblP.p);        
        pAbsIsFld = any(strcmp(tblP.Properties.VariableNames,'pAbs'));
        if pAbsIsFld % probably .pAbs is put in below by preProcCropLabelsToRoiIfNec
          assert(isequal(tfinf,isinf(tblP.pAbs)));
        end
        tfinf2 = reshape(tfinf,height(tblP),[],2);
        assert(isequal(tfinf2(:,:,1),tfinf2(:,:,2))); % both x/y coords should be inf for fully-occ
        nfulloccpts = nnz(tfinf2(:,:,1));
        if nfulloccpts>0
          warningNoTrace('Utilizing %d fully-occluded landmarks.',nfulloccpts);
        end
      
        tblP.p(tfinf) = nan;
        if pAbsIsFld
          tblP.pAbs(tfinf) = nan;
        end
        tblP.tfocc(tfinf2(:,:,1)) = true;
      else
        tfinf = any(isinf(tblP.p),2);
        ninf = nnz(tfinf);
        if ninf>0
          warningNoTrace('Labeler:infData',...
            'Not including %d rows with fully-occluded labels.',ninf);
        end
        tblP = tblP(~tfinf,:);
      end
      
      tblP = obj.preProcCropLabelsToRoiIfNec(tblP,'prmsTgtCrop',prmsTgtCrop,...
        'doRemoveOOB',doRemoveOOB);
    end
    
    % Hist Eq Notes
    %
    % The Labeler has the ability/responsibility to compute typical image
    % histograms representative of all movies in the current project, for
    % each view. See movieEstimateImHist. 
    %
    % .preProcH0 stores this typical hgram for use in preprocessing. 
    % Conceptually, .preProcH0 is like .preProcParams in that it is a 
    % parameter (vector) governing how data is preprocessed. Instead of 
    % being user-set like other parameters, .preProcH0 is updated/learned 
    % from the project movies.
    %
    % .preProcH0 is set at retrain- (fulltraining-) time. It is not updated
    % during tracking, or during incremental trains. If users are adding
    % movies/labels, they should periodically retrain fully, to refresh 
    % .preProcH0 relative to the movies.
    %
    
%     function preProcUpdateH0IfNec(obj)
%       % Update obj.preProcH0
%       % Update .movieFilesAllHistEqLUT, .movieFilesAllGTHistEqLUT 
% 
%       % AL20180910: currently using frame-by-frame CLAHE
%       USECLAHE = true;
%       if USECLAHE
%         obj.preProcH0 = [];
%         return
%       end      
% 
% %       ppPrms = obj.preProcParams;
% %       if ppPrms.histeq
% %         nFrmSampH0 = ppPrms.histeqH0NumFrames;
% %         s = struct();
% %         [s.hgram,s.hgraminfo] = obj.movieEstimateImHist(...
% %           'nFrmPerMov',nFrmSampH0,'debugViz',false);
% %         
% %         data = obj.preProcData;
% %         if data.N>0 && ~isempty(obj.preProcH0) && ~isequal(data.H0,obj.preProcH0.hgram)
% %           assert(false,'.preProcData.H0 differs from .preProcH0');
% %         end
% %         if ~isequal(data.H0,s.hgram)
% %           obj.preProcInitData();
% %           % Note, currently we do not clear trained tracker/tracking
% %           % results, see above. This is caller's responsibility. Atm all
% %           % callers do a retrain or equivalent
% %         end
% %         obj.preProcH0 = s;
% %         
% %         wbObj = WaitBarWithCancel('Computing Histogram Matching LUTs',...
% %           'cancelDisabled',true);
% %         oc = onCleanup(@()delete(wbObj));
% %         obj.movieEstimateHistEqLUTs('nFrmPerMov',nFrmSampH0,...
% %           'wbObj',wbObj,'docheck',true);
% %       else
% %         assert(isempty(obj.preProcData.H0));
% %         
% %         % For now we don't force .preProcH0, .movieFilesAll*HistEq* to be
% %         % empty. User can compute them, then turn off HistEq, and the state
% %         % remains
% %         if false
% %           assert(isempty(obj.preProcH0));
% %           tf = cellfun(@isempty,obj.movieFilesAllHistEqLUT);
% %           tfGT = cellfun(@isempty,obj.movieFilesAllGTHistEqLUT);
% %           assert(all(tf(:)));
% %           assert(all(tfGT(:)));
% %         end
% %       end
%     end  % function
    
%     function [data,dataIdx,tblP,tblPReadFailed,tfReadFailed] = ...
%         preProcDataFetch(obj,tblP,varargin)
%       % dataUpdate, then retrieve
%       %
%       % Input args: See PreProcDataUpdate
%       %
%       % data: CPRData handle, equal to obj.preProcData
%       % dataIdx. data.I(dataIdx,:) gives the rows corresponding to tblP
%       %   (out); order preserved
%       % tblP (out): subset of tblP (input), rows for failed reads removed
%       % tblPReadFailed: subset of tblP (input) where reads failed
%       % tfReadFailed: indicator vec into tblP (input) for failed reads.
%       %   tblP (out) is guaranteed to correspond to tblP (in) with
%       %   tfReadFailed rows removed. (unless early/degenerate/empty return)
%       
%       % See preProcDataUpdateRaw re 'preProcParams' opt arg. When supplied,
%       % .preProcData is not updated.
%       
%       [wbObj,updateRowsMustMatch,prmpp] = myparse(varargin,...
%         'wbObj',[],... % WaitBarWithCancel. If cancel: obj unchanged, data and dataIdx are [].
%         'updateRowsMustMatch',false, ... % See preProcDataUpdateRaw
%         'preProcParams',[]...
%         );
%       isPreProcParamsIn = ~isempty(prmpp);
%       tfWB = ~isempty(wbObj);
%       
%       [tblPReadFailed,data] = obj.preProcDataUpdate(tblP,'wbObj',wbObj,...
%         'updateRowsMustMatch',updateRowsMustMatch,'preProcParams',prmpp);
%       if tfWB && wbObj.isCancel
%         data = [];
%         dataIdx = [];
%         tblP = [];
%         tblPReadFailed = [];
%         tfReadFailed = [];
%         return;
%       end
%       
%       if ~isPreProcParamsIn,
%         data = obj.preProcData;
%       end
%       tfReadFailed = tblismember(tblP,tblPReadFailed,MFTable.FLDSID);
%       tblP(tfReadFailed,:) = [];
%       [tf,dataIdx] = tblismember(tblP,data.MD,MFTable.FLDSID);
%       assert(all(tf));
%     end
    
%     function [tblPReadFailed,dataNew] = preProcDataUpdate(obj,tblP,varargin)
%       % Update .preProcData to include tblP
%       %
%       % tblP:
%       %   - MFTable.FLDSCORE: required.
%       %   - .roi: optional, USED WHEN PRESENT. (prob needs to be either
%       %   consistently there or not-there for a given obj or initData()
%       %   "session"
%       %   IMPORTANT: if .roi is present, .p (labels) are expected to be 
%       %   relative to the roi.
%       %   - .pTS: optional (if present, deleted)
%       
%       [wbObj,updateRowsMustMatch,prmpp] = myparse(varargin,...
%         'wbObj',[],... % WaitBarWithCancel. If cancel, obj unchanged.
%         'updateRowsMustMatch',false, ... % See preProcDataUpdateRaw
%         'preProcParams',[]...
%         );
%       
%       if any(strcmp('pTS',tblP.Properties.VariableNames))
%         % AL20170530: Not sure why we do this
%         tblP(:,'pTS') = [];
%       end
%       isPreProcParamsIn = ~isempty(prmpp);
%       if isPreProcParamsIn,
%         tblPnew = tblP;
%         tblPupdate = tblP([],:);
%       else
%         [tblPnew,tblPupdate] = obj.preProcData.tblPDiff(tblP);
%       end
%       [tblPReadFailed,dataNew] = obj.preProcDataUpdateRaw(tblPnew,tblPupdate,...
%         'wbObj',wbObj,'updateRowsMustMatch',updateRowsMustMatch,...
%         'preProcParams',prmpp);
%     end
    
%     function [tblPReadFailed,dataNew] = preProcDataUpdateRaw(obj,...
%         tblPnew,tblPupdate,varargin)
%       % Incremental data update
%       %
%       % * Rows appended and pGT/tfocc updated; but other information
%       % untouched
%       % * histeq (if enabled) uses .preProcH0. See "Hist Eq Notes" below.
%       % .preProcH0 is NOT updated here.
%       %
%       % QUESTION: why is pTS not updated?
%       %
%       % tblPNew: new rows. MFTable.FLDSCORE are required fields. .roi may 
%       %   be present and if so WILL BE USED to grab images and included in 
%       %   data/MD. Other fields are ignored.
%       %   IMPORTANT: if .roi is present, .p (labels) are expected to be 
%       %   relative to the roi.
%       %
%       % tblPupdate: updated rows (rows with updated pGT/tfocc).
%       %   MFTable.FLDSCORE fields are required. Only .pGT and .tfocc are 
%       %   otherwise used. Other fields ignored, INCLUDING eg .roi and 
%       %   .nNborMask. Ie, you cannot currently update the roi of a row in 
%       %   the cache (whose image has already been fetched)
%       %
%       %   
%       % tblPReadFailed: table of failed-to-read rows. Currently subset of
%       %   tblPnew. If non-empty, then .preProcData was not updated with 
%       %   these rows as requested.
%       %
%       % Updates .preProcData, .preProcDataTS
%       
%       % NOTE: when the preProcParams opt arg is [] (isPreProcParamsIn is 
%       % false), this is maybe a separate method, def distinct behavior. 
%       % When isPreProcParamsIn is true, .preProcData is not updated, etc.
%       
%       dataNew = [];
%       
%       [wbObj,updateRowsMustMatch,prmpp] = myparse(varargin,...
%         'wbObj',[], ... % Optional WaitBarWithCancel obj. If cancel, obj unchanged.
%         'updateRowsMustMatch',false, ... % if true, assert/check that tblPupdate matches current cache
%         'preProcParams',[]...
%         );
%       tfWB = ~isempty(wbObj);
%       
%       FLDSREQUIRED = MFTable.FLDSCORE;
%       FLDSALLOWED = [MFTable.FLDSCORE {'roi' 'nNborMask'}];
%       tblfldscontainsassert(tblPnew,FLDSREQUIRED);
%       tblfldscontainsassert(tblPupdate,FLDSREQUIRED);
%       
%       tblPReadFailed = tblPnew([],:);
%       
%       isPreProcParamsIn = ~isempty(prmpp);
%       if ~isPreProcParamsIn,
%         prmpp = obj.preProcParams;
%         if isempty(prmpp)
%           error('Please specify tracking parameters.');
%         end
%         dataCurr = obj.preProcData;
%       end
%       
%       USECLAHE = true;
% 
%       if prmpp.histeq
%         if ~USECLAHE && isPreProcParamsIn,
%           assert(dataCurr.N==0 || isequal(dataCurr.H0,obj.preProcH0.hgram));
%         end
%         assert(~prmpp.BackSub.Use,...
%           'Histogram Equalization and Background Subtraction cannot both be enabled.');
%         assert(~prmpp.NeighborMask.Use,...
%           'Histogram Equalization and Neighbor Masking cannot both be enabled.');
%       end
%       if ~isempty(prmpp.channelsFcn)
%         assert(obj.nview==1,...
%           'Channels preprocessing currently unsupported for multiview tracking.');
%       end
%       
%       %%% NEW ROWS read images + PP. Append to dataCurr. %%%
%       FLDSID = MFTable.FLDSID;
%       assert(isPreProcParamsIn||~any(tblismember(tblPnew,dataCurr.MD,FLDSID)));
%       
%       tblPNewConcrete = obj.mftTableConcretizeMov(tblPnew);
%       nNew = height(tblPnew);
%       if nNew>0
%         fprintf(1,'Adding %d new rows to data...\n',nNew);
% 
%         [I,nNborMask,didread] = CPRData.getFrames(tblPNewConcrete,...
%           'wbObj',wbObj,...
%           'forceGrayscale',obj.movieForceGrayscale,...
%           'preload',obj.movieReadPreLoadMovies,...
%           'movieInvert',obj.movieInvert,...
%           'roiPadVal',prmpp.TargetCrop.PadBkgd,...
%           'doBGsub',prmpp.BackSub.Use,...
%           'bgReadFcn',prmpp.BackSub.BGReadFcn,...
%           'bgType',prmpp.BackSub.BGType,...
%           'maskNeighbors',prmpp.NeighborMask.Use,...
%           'maskNeighborsMeth',prmpp.NeighborMask.SegmentMethod,...
%           'maskNeighborsEmpPDF',obj.fgEmpiricalPDF,...
%           'fgThresh',prmpp.NeighborMask.FGThresh,...
%           'trxCache',obj.trxCache);
%         if tfWB && wbObj.isCancel
%           % obj unchanged
%           return;
%         end
%         % Include only FLDSALLOWED in metadata to keep CPRData md
%         % consistent (so can be appended)
%         
%         didreadallviews = all(didread,2);
%         tblPReadFailed = tblPnew(~didreadallviews,:);
%         tblPnew(~didreadallviews,:) = [];
%         I(~didreadallviews,:) = [];
%         nNborMask(~didreadallviews,:) = [];
%         
%         % AL: a little worried if all reads fail -- might get a harderr
%         
%         tfColsAllowed = ismember(tblPnew.Properties.VariableNames,...
%           FLDSALLOWED);
%         tblPnewMD = tblPnew(:,tfColsAllowed);
%         tblPnewMD = [tblPnewMD table(nNborMask)];
%         
%         if prmpp.histeq
%           if USECLAHE
%             if tfWB
%               wbObj.startPeriod('Performing CLAHE','shownumden',true,...
%                 'denominator',numel(I));
%             end
%             for i=1:numel(I)
%               if tfWB
%                 wbObj.updateFracWithNumDen(i);
%               end
%               I{i} = adapthisteq(I{i});
%             end
%             if tfWB
%               wbObj.endPeriod();
%             end
%             dataNew = CPRData(I,tblPnewMD);            
%           else
%             J = obj.movieHistEqApplyLUTs(I,tblPnewMD.mov); 
%             dataNew = CPRData(J,tblPnewMD);
%             dataNew.H0 = obj.preProcH0.hgram;
% 
%             if ~isPreProcParamsIn && dataCurr.N==0
%               dataCurr.H0 = dataNew.H0;
%               % these need to match for append()
%             end
%           end
%         else
%           dataNew = CPRData(I,tblPnewMD);
%         end
%                 
%         if ~isempty(prmpp.channelsFcn)
%           feval(prmpp.channelsFcn,dataNew);
%           assert(~isempty(dataNew.IppInfo),...
%             'Preprocessing channelsFcn did not set .IppInfo.');
%           if ~isPreProcParamsIn && isempty(dataCurr.IppInfo)
%             assert(dataCurr.N==0,'Ippinfo can be empty only for empty/new data.');
%             dataCurr.IppInfo = dataNew.IppInfo;
%           end
%         end
%         
%         if ~isPreProcParamsIn,
%           dataCurr.append(dataNew);
%         end
%       end
%       
%       %%% EXISTING ROWS -- just update pGT and tfocc. Existing images are
%       %%% OK and already histeq'ed correctly
%       nUpdate = size(tblPupdate,1);
%       if ~isPreProcParamsIn && nUpdate>0 % AL 20160413 Shouldn't need to special-case, MATLAB 
%                    % table indexing API may not be polished
%         [tf,loc] = tblismember(tblPupdate,dataCurr.MD,FLDSID);
%         assert(all(tf));
%         if updateRowsMustMatch
%           assert(isequal(dataCurr.MD{loc,'tfocc'},tblPupdate.tfocc),...
%             'Unexpected discrepancy in preproc data cache: .tfocc field');
%           if tblfldscontains(tblPupdate,'roi')
%             assert(isequal(dataCurr.MD{loc,'roi'},tblPupdate.roi),...
%               'Unexpected discrepancy in preproc data cache: .roi field');
%           end
%           if tblfldscontains(tblPupdate,'nNborMask')
%             assert(isequal(dataCurr.MD{loc,'nNborMask'},tblPupdate.nNborMask),...
%               'Unexpected discrepancy in preproc data cache: .nNborMask field');
%           end
%           assert(isequaln(dataCurr.pGT(loc,:),tblPupdate.p),...
%             'Unexpected discrepancy in preproc data cache: .p field');
%         else
%           fprintf(1,'Updating labels for %d rows...\n',nUpdate);
%           dataCurr.MD{loc,'tfocc'} = tblPupdate.tfocc; % AL 20160413 throws if nUpdate==0
%           dataCurr.pGT(loc,:) = tblPupdate.p;
%           % Check .roi, .nNborMask?
%         end
%       end
%       
%       if ~isPreProcParamsIn,
%         if nUpdate>0 || nNew>0 % AL: if all reads fail, nNew>0 but no new rows were actually read
%           assert(obj.preProcData==dataCurr); % handles; not sure why this is asserted in this branch specifically
%           obj.preProcDataTS = now;
%         else
%           warningNoTrace('Nothing to update in data.');
%         end
%       end
%     end  % function
   
  end  % methods
  

  %% Tracker
  methods
    
    % function [tObj,iTrk] = trackGetTracker(obj, algoName)
    %   % Find a particular tracker
    %   %
    %   % algoName: char, to match LabelTracker.algorithmName
    %   %
    %   % tObj: either [], or scalar tracking object 
    %   % iTrk: either 0, or index into .trackersAll
    %   for iTrk=1:numel(obj.trackersAll)
    %     if strcmp(obj.trackersAll{iTrk}.algorithmName,algoName)
    %       tObj = obj.trackersAll{iTrk};
    %       return
    %     end
    %   end
    %   tObj = [];
    %   iTrk = 0;
    % end
  
    function trackMakeExistingTrackerCurrentGivenIndex(obj, iTrk)      
      % Validate the new value
      trackers = obj.trackerHistory_ ;
      tracker_count = numel(trackers) ;
      if is_index_in_range(iTrk, tracker_count)
        % all is well
      else
        error('APT:invalidPropertyValue', 'Invalid tracker index') ;
      end
      
      % If iTrk==1, do nothing, that's already the current tracker
      if iTrk==1
        return
      end
    
      % Want to do some stuff before the set, apparently
      oldTracker = trackers{1} ;
      oldTracker.deactivate() ;
      oldTracker.setHideViz(true);
   
      % Shuffle trackerHistory_ to bring iTrk to the front
      % Also delete any untrained trackers.
      trackersNewFirst = trackers(iTrk) ;  % singleton cell array
      trackersNewRestDraft = delete_elements(trackers, iTrk) ;
      isTrained = cellfun(@(tracker)(tracker.hasBeenTrained), trackersNewRestDraft) ;
      trackersNewRest = trackersNewRestDraft(isTrained) ;
      trackersNew = horzcat(trackersNewFirst, trackersNewRest) ;
      obj.trackerHistory_ = trackersNew ;

      % Activate the newly-selected tracker
      newCurrentTracker = trackersNew{1} ;
      if ~isempty(newCurrentTracker),
        newCurrentTracker.activate() ;
        newCurrentTracker.setHideViz(false);
      end
      
      % What is this doing, exactly?  -- ALT, 2025-02-05
      obj.labelingInit('labelMode',obj.labelMode);      

      % Update the timeline
      if ~isempty(newCurrentTracker)
        propList = newCurrentTracker.propList() ;
      else
        propList = [] ;
      end      
      obj.infoTimelineModel_.didChangeCurrentTracker(propList) ;
      obj.notify('updateTimelineProps');
      % obj.notify('updateTimelineSelection');

      % Send the notifications
      obj.notify('didSetCurrTracker') ;
      % obj.notify('update_menu_track_tracking_algorithm_quick') ;      
      obj.notify('update_menu_track_tracker_history') ;
      obj.notify('update_text_trackerinfo') ;
    end  % function

    function trackMakeBackupOfCurrentTrackerIfHasBeenTrained(obj)
      % Validate the new value
      trackers = obj.trackerHistory_ ;
      tracker_count = numel(trackers) ;
      
      % If tracker_count == 0, do nothing, although not clear how that would happen
      if tracker_count == 0
        return
      end
    
      % Get the tracker we're backing up
      originalTracker = trackers{1} ;
      if ~originalTracker.hasBeenTrained()
        % If untrained, exit now
        return
      end

      % Make the backup, which is like a copy but with the same obj.lObj
      backupTracker = originalTracker.twin() ;

      % If in debug mode, run some checks
      if obj.isInDebugMode
        if ~originalTracker.tfIsTwin(backupTracker) 
          error('Internal error: The backup tracker is not the twin of the original') ;
        end
        if ~backupTracker.tfIsTwin(originalTracker) 
          error('Internal error: The original tracker is not the twin of the backup') ;
        end
      end

      % Insert the backup into the list of trackers, just behind the original
      trackersRest = trackers(2:end) ;
      trackersNew = horzcat({originalTracker}, {backupTracker}, trackersRest) ;
      obj.trackerHistory_ = trackersNew ;
      
      % Send the notification
      obj.notify('update_menu_track_tracker_history') ;
    end  % function
    
    % function trackMakeNewTrackerGivenIndex(obj, tciIndex, varargin)
    %   % Make a new tracker, and make it current.  tciIndex should be a valid index
    %   % into obj.trackersAll and/or obj.trackersAllCreateInfo_.  The varargin should
    %   % contain the stage 1 and stage 2 constructor args if and only if tciIndex
    %   % indicates a custom two-stage tracker.
    % 
    %   % Validate the new value      
    %   tcis = obj.trackersAllCreateInfo_ ;
    %   template_count = numel(tcis) ;
    %   if is_index_in_range(tciIndex, template_count)
    %     % all is well
    %   else
    %     error('APT:invalidPropertyValue', 'Invalid tracker template index') ;
    %   end
    % 
    %   % Want to do some stuff before the set, apparently
    %   trackers = obj.trackerHistory_ ;
    %   if ~isempty(trackers) ,
    %     oldCurrentTracker = trackers{1} ;
    %     oldCurrentTracker.deactivate() ;
    %     oldCurrentTracker.setHideViz(true);
    %     if ~oldCurrentTracker.hasBeenTrained() ,
    %       % If the current model is untrained, don't keep it in the history
    %       delete(oldCurrentTracker) ;          
    %       trackers = trackers(2:end) ;
    %     end
    %   end
    % 
    %   % Create the new tracker
    %   rawTCI = tcis{tciIndex} ;
    %   tci = apt.fillInCustomStagesIfNeeded(rawTCI, varargin{:}) ;
    %   newTracker = LabelTracker.create(obj, tci) ;     
    % 
    %   % Filter untrained trackers out of trackers
    %   isTrained = cellfun(@(tracker)(tracker.hasBeenTrained), trackers) ;
    %   trainedTrackers = trackers(isTrained) ;
    % 
    %   % Put the new tracker at the front of the history
    %   trackersNew = horzcat({newTracker}, trainedTrackers) ;
    %   obj.trackerHistory_ = trackersNew ;
    % 
    %   % Activate the new tracker
    %   if ~isempty(newTracker),
    %     newTracker.activate() ;
    %   end
    % 
    %   % Turn the visualization back on for the new current tracker
    %   newTracker.setHideViz(false);
    % 
    %   % What is this doing, exactly?  -- ALT, 2025-02-05
    %   obj.labelingInit('labelMode',obj.labelMode);
    % 
    %   % Update the timeline
    %   if ~isempty(newTracker)
    %     propList = newTracker.propList() ;
    %   else
    %     propList = [] ;
    %   end
    %   obj.infoTimelineModel_.didChangeCurrentTracker(propList) ;
    %   obj.notify('updateTimelineAndFriends');
    % 
    %   % Send the needed notifications
    %   obj.notify('didSetCurrTracker') ;      
    %   % obj.notify('update_menu_track_tracking_algorithm_quick') ;
    %   obj.notify('update_menu_track_tracker_history') ;      
    %   obj.notify('update_text_trackerinfo') ;      
    % end  % function

    function t = trackGetCurrTrackerStageNetTypes(obj,trackercurr)
      % t = trackGetCurrTrackerStageNetTypes(obj,trackercurr)
      % returns the trnNetTypes for the current tracker. trackercurr
      % can be given as an optional input, otherwise obj.tracker is used.

      if nargin < 2,
        trackercurr = obj.tracker;
      end
      if isa(trackercurr,'DeepTrackerTopDown') || isa(trackercurr,'DeepTrackerTopDownCuston'),
        t = [trackercurr.stage1Tracker.trnNetType,trackercurr.trnNetType];
      elseif isa(trackercurr,'DeepTrackerBottomUp') || isa(trackercurr,'DeepTracker'),
        t = trackercurr.trnNetType;
      else
        t = [];
      end
    end  % function

    function trackMakeNewTrackerGivenNetTypes(obj, netTypes)
      % trackMakeNewTrackerGivenNetTypes(obj, nettypes)
      % Create a new tracker based on the input nettypes. 
      assert(isShortDLNetTypesRowArray(netTypes));
      tci = TrackerCreateInfo.fromNetTypes(netTypes, obj.maIsMA);
      newTracker = LabelTracker.create(obj, tci) ;
      obj.trackInsertNewTracker_(newTracker);
    end  % function

    function trackInsertNewTracker_(obj, newTracker)
      % Insert a new tracker.  Not meant to be called by external clients.

      % % Validate the new value      
      % tcis = obj.trackersAllCreateInfo_ ;
      % template_count = numel(tcis) ;
      % if is_index_in_range(tciIndex, template_count)
      %   % all is well
      % else
      %   error('APT:invalidPropertyValue', 'Invalid tracker template index') ;
      % end
      
      % Want to do some stuff before the set, apparently
      trackers = obj.trackerHistory_ ;
      if ~isempty(trackers) ,
        oldCurrentTracker = trackers{1} ;
        oldCurrentTracker.deactivate() ;
        oldCurrentTracker.setHideViz(true);
        if ~oldCurrentTracker.hasBeenTrained() ,
          % If the current model is untrained, don't keep it in the history
          delete(oldCurrentTracker) ;          
          trackers = trackers(2:end) ;
        end
      end

      % Filter untrained trackers out of trackers
      isTrained = cellfun(@(tracker)(tracker.hasBeenTrained), trackers) ;
      trainedTrackers = trackers(isTrained) ;

      % Put the new tracker at the front of the history
      trackersNew = horzcat({newTracker}, trainedTrackers) ;
      obj.trackerHistory_ = trackersNew ;

      % Activate the new tracker
      if ~isempty(newTracker),
        newTracker.activate() ;
      end
      
      % Turn the visualization back on for the new current tracker
      newTracker.setHideViz(false);

      % What is this doing, exactly?  -- ALT, 2025-02-05
      obj.labelingInit('labelMode',obj.labelMode);

      % Update the timeline
      if ~isempty(newTracker)
        propList = newTracker.propList() ;
      else
        propList = [] ;
      end
      obj.infoTimelineModel_.didChangeCurrentTracker(propList) ;
      obj.notify('updateTimelineProps');
      % obj.notify('updateTimelineSelection');
      
      % Send the needed notifications
      obj.notify('didSetCurrTracker') ;      
      % obj.notify('update_menu_track_tracking_algorithm_quick') ;
      obj.notify('update_menu_track_tracker_history') ;      
      obj.notify('update_text_trackerinfo') ;      
    end  % function

    % function trackMakeNewTrackerGivenAlgoName(obj, algoName, varargin)
    %   algorithmNameFromTciIndex = cellfun(@(tracker)(tracker.algorithmName), ...
    %                                       obj.trackersAll_, ...
    %                                       'UniformOutput', false) ;
    %   matchingIndices = find(strcmp(algoName, algorithmNameFromTciIndex)) ;
    %   if isempty(matchingIndices) ,
    %     error('No algorithm named %s among the available trackers', algoName) ;
    %   elseif isscalar(matchingIndices) ,
    %     % all is well
    %     tciIndex = matchingIndices ;
    %   else
    %     tciIndex = matchingIndices(1) ;
    %     warningNoTrace('More than one algorithm named %s among the available trackers, using first one, at index %d', algoName, tciIndex) ;
    %   end
    %   obj.trackMakeNewTrackerGivenIndex(tciIndex, varargin{:}) ;
    % end  % function

    function trackMakeExistingTrackerCurrentGivenAlgoName(obj, algoName)
      algorithmNameFromHistoryIndex = cellfun(@(tracker)(tracker.algorithmName), ...
                                              obj.trackerHistory_, ...
                                              'UniformOutput', false) ;
      matchingIndices = find(strcmp(algoName, algorithmNameFromHistoryIndex)) ;
      if isempty(matchingIndices) ,
        error('No algorithm named %s among the trackers in the history', algoName) ;
      elseif isscalar(matchingIndices) ,
        % all is well
        trackerIndex = matchingIndices ;
      else
        trackerIndex = matchingIndices(1) ;
        warningNoTrace('More than one algorithm named %s among the available trackers, using first one, at index %d', algoName, trackerIndex) ;
      end
      obj.trackMakeExistingTrackerCurrentGivenIndex(trackerIndex) ;
    end  % function

    function result = trackIsTrackerInHistoryByName(obj, algoName)
      algorithmNameFromHistoryIndex = cellfun(@(tracker)(tracker.algorithmName), ...
                                              obj.trackerHistory_, ...
                                              'UniformOutput', false) ;
      matchingIndices = find(strcmp(algoName, algorithmNameFromHistoryIndex)) ;  %#ok<EFIND>
      result = ~isempty(matchingIndices) ;
    end  % function

    function sPrm = setTrackNFramesParams(obj,sPrm)
      obj.trackNFramesSmall = sPrm.ROOT.Track.NFramesSmall;
      obj.trackNFramesLarge = sPrm.ROOT.Track.NFramesLarge;
      obj.trackNFramesNear = sPrm.ROOT.Track.NFramesNeighborhood;
      sPrm.ROOT.Track = rmfield(sPrm.ROOT.Track,{'NFramesSmall','NFramesLarge','NFramesNeighborhood'});
    end
    
    function trackSetTrainingParams(obj, sPrm, varargin)
      % Set all parameters:
      %  - preproc
      %  - cpr
      %  - common dl
      %  - specific dl
      % 
      % sPrm: scalar struct containing *NEW*-style params:
      % sPrm.ROOT.Track
      %          .CPR
      %          .DeepTrack
      
      obj.pushBusyStatus('Setting training parameters...');
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      
      [setall, istrack] = ...
        myparse(varargin, ...
                'all',false, ... % if true, sPrm can contain 'extra parameters' like fliplandmarks. no callsites currently
                'istrack',false ... % if true, this is being called by the trackSetTrackParams function
                ) ;
      sPrm = APTParameters.enforceConsistency(sPrm);

      [tfOK,msgs] = APTParameters.checkParams(sPrm);
      if ~tfOK,
        error('%s. ',msgs{:});
      end
      
      sPrm0 = obj.trackParams;  % original params
      tfPreprocesssingParamsChanged = ...
        xor(isempty(sPrm0),isempty(sPrm)) || ...
        ~APTParameters.isEqualPreProcParams(sPrm0,sPrm) ;
      sPrm = obj.setTrackNFramesParams(sPrm);      
      if setall,
        sPrm = obj.setExtraParams(sPrm);
      end
      
      obj.trackParams = sPrm;
      
      if tfPreprocesssingParamsChanged
        assert(~istrack);
        warningNoTrace('Preprocessing parameters altered; data cache cleared.');
        obj.ppdbInit(); % AL20190123: currently only ppPrms.TargetCrop affect ppdb
        
        bgPrms = sPrm.ROOT.ImageProcessing.BackSub;
        mrs = obj.movieReader;
        for i=1:numel(mrs)
          mrs(i).open(mrs(i).filename,...
                      'bgType',bgPrms.BGType,...
                      'bgReadFcn',bgPrms.BGReadFcn);
          % mrs(i) should already be faithful to .forceGrayscale,
          % .movieInvert, cropInfo
        end
        
        if obj.maIsMA && ~istrack,
          obj.lblCore.preProcParamsChanged();          
        end
      end
      
    end  % function trackSetParams
    
    function tPrm = trackGetTrackParams(obj)
      % Get current parameters related to tracking

      sPrmCurrent = obj.trackGetTrainingParams();
      sPrmCurrent = APTParameters.all2TrackParams(sPrmCurrent);
      % Start with default "new" parameter tree/specification
      tPrm = APTParameters.defaultTrackParamsTree();  % object of class TreeNode
      % Overlay our starting pt
      tPrm.structapply(sPrmCurrent);      
    end
    
    % function sPrmAll = trackSetTrackParamsCore_(obj,sPrmTrack,varargin)      
    %   sPrmAll = obj.trackGetTrainingParams();
    %   sPrmAll = APTParameters.setTrackParams(sPrmAll,sPrmTrack);
    % 
    %   obj.trackSetTrainingParams(sPrmAll,varargin{:},'istrack',true);
    % 
    %   % set all tracker parameters
    %   for i = 1:numel(obj.trackerHistory_),
    %     obj.trackerHistory_{i}.setTrackParams(sPrmTrack);
    %   end
    % end  % function
    
    function [sPrmDT,sPrmCPRold,ppPrms,trackNFramesSmall,trackNFramesLarge,...
        trackNFramesNear] = convertNew2OldParams(obj,sPrm) % obj CONST
      % Conversion routine
      % 
      % sPrm: scalar struct containing *NEW*-style params:
      % sPrm.ROOT.Track
      %          .CPR
      %          .DeepTrack
              
      sPrm = APTParameters.enforceConsistency(sPrm);
      
      sPrmDT = sPrm.ROOT.DeepTrack;
      sPrmPPandCPR = sPrm;
      sPrmPPandCPR.ROOT = rmfield(sPrmPPandCPR.ROOT,'DeepTrack'); 
      
      [sPrmPPandCPRold,trackNFramesSmall,trackNFramesLarge,...
        trackNFramesNear] = cprParamNew2Old(sPrmPPandCPR,obj.nPhysPoints,obj.nview);
      
      ppPrms = sPrmPPandCPRold.PreProc;
      sPrmCPRold = rmfield(sPrmPPandCPRold,'PreProc');
    end
    
    function sPrm = trackGetTrainingParams(obj,varargin)
      % Get all user-settable parameters, including preproc etc.
      %
      % Doesn't include APT-added params for DL backend.
      
%       [getall] = myparse(varargin,'all',false);
%       assert(~getall);

      sPrm = obj.trackParams;
      sPrm.ROOT.Track.NFramesSmall = obj.trackNFramesSmall;
      sPrm.ROOT.Track.NFramesLarge = obj.trackNFramesLarge;
      sPrm.ROOT.Track.NFramesNeighborhood = obj.trackNFramesNear;      
      if obj.maIsMA ,
        sPrm.ROOT.MultiAnimal.multi_crop_im_sz = obj.get_ma_crop_sz() ;
      end

%       if getall,
%         sPrm = obj.addExtraParams(sPrm);
%       end      
    end
    
    function be = trackGetDLBackend(obj)
      be = obj.trackDLBackEnd;
    end
    
    function trainIncremental(obj)
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      if ~obj.hasMovie
        error('Labeler:track','No movie.');
      end
      tObj.trainIncremental();
    end
        
    function train(obj, varargin)
      [tblMFTtrn, trainArgs, do_just_generate_db, do_call_apt_interface_dot_py] = myparse(...
        varargin,...
        'tblMFTtrn',[],... % table on which to train (cols MFTable.FLDSID only). defaults to all of obj.preProcGetMFTableLbled
        'trainArgs',{},... % args to pass to tracker.train()
        'do_just_generate_db', false, ...
        'do_call_apt_interface_dot_py', true ...
        );
      
      % Do a few checks
      tracker = obj.tracker;
      if isempty(tracker)
        error('Labeler:train','No tracker set.');
      end
      if ~obj.hasMovie
        error('Labeler:train','No movie.');
      end

      % If the current tracker has been trained at all, make a duplicate of it so
      % that we can go back.
      obj.trackMakeBackupOfCurrentTrackerIfHasBeenTrained() ;

      % Update the status
      obj.pushBusyStatus('Spawning training job...') ;
      oc = onCleanup(@()(obj.popBusyStatusAndSendUpdateNotification_()));

      % Update the 'status' on the console
      fprintf('Training started at %s...\n',datestr(now()));
      
      % Do something, for some reason.  Maybe vestigial?  -- ALT, 2025-01-24
      if ~isempty(tblMFTtrn)
        assert(strcmp(tracker.algorithmName,'cpr'));
        % assert this as we do not fetch tblMFTp to treatInfPosAsOcc
        tblMFTp = obj.preProcGetMFTableLbled('tblMFTrestrict',tblMFTtrn);
        trainArgs = [trainArgs(:)' {'tblPTrn' tblMFTp}];
      end           
      
      % % This is vestigial, can be removed.  -- ALT, 2025-01-24
      % if ~dontUpdateH0
      %   obj.preProcUpdateH0IfNec();
      % end

      % Spin up the backend in preparation for training
      backend = obj.trackDLBackEnd ;
      [isReady, reasonNotReady] = backend.ensureIsRunning() ;
      if ~isReady ,
        error('Labeler:unableToStartBackend', 'Unable to start backend: %s', reasonNotReady) ;
      end

      % Call the tracker to do the heavy lifting
      tracker.train(trainArgs{:}, ...
                    'do_just_generate_db', do_just_generate_db, ...
                    'do_call_apt_interface_dot_py', do_call_apt_interface_dot_py, ...
                    'projTempDir', obj.projTempDir);

      % Update the bg processing status string, if training is actually running
      if obj.bgTrnIsRunning ,
        algName = obj.tracker.algorithmName;
        backend_type_string = obj.trackDLBackEnd.prettyName();
        obj.backgroundProcessingStatusString_ = ...
          sprintf('%s training on %s (started %s)',algName,backend_type_string,datestr(now(),'HH:MM'));  %#ok<TNOW1,DATST>
      end      
    end  % function

    function result = bgTrnIsRunningFromTrackerIndex(obj)
      trackers = obj.trackerHistory_ ;
      result = cellfun(@(t)(t.bgTrnIsRunning), trackers) ;
    end  % function
    
    function [tfCanExport,reason] = trackCanExport(obj,varargin)
      
      tfCanExport = false;
      % is tracker and movie
      if isempty(obj.tracker),
        reason = 'The tracker has not been set.';
        return;
      end
      if ~obj.hasMovie,
        reason = 'There must be at least one movie in the project.';
        return;
      end
      
      if isempty(obj.trackParams)
        reason = 'Tracking parameters have not been set.';
        return;
      end
      
      % allow 'treatInfPosAsOcc' to default to false in these calls; we are
      % just checking number of labeled rows
      % this is probably overkill, but no other way to figure out how many
      % labels there are? 
      warnst = warning('off','Labeler:infData'); % ignore "Not including n rows with fully-occ labels"
      oc = onCleanup(@()warning(warnst));
      tblMFTp = obj.preProcGetMFTableLbled('gtModeOK',true);
    
      nlabels = size(tblMFTp,1);
      
      if nlabels < 2,
        tfCanExport = false;
        reason = 'There must be at least two labeled frames in the project.';
        return;
      end
      
      tfCanExport = true;
      reason = '';
      
    end

    function [tfCanTrain,reason] = trackCanTrain(obj,varargin)
      
      tfCanTrain = false;
      % is tracker and movie
      if isempty(obj.tracker),
        reason = 'The tracker has not been set.';
        return;
      end
      if ~obj.hasMovie,
        reason = 'There must be at least one movie in the project.';
        return;
      end
      
      if isempty(obj.trackParams)
        reason = 'Tracking parameters have not been set.';
        return;
      end
      
      % parameters set at project-level but not at tracker-level
      [tfCanTrain,reason] = obj.tracker.canTrain();
      if ~tfCanTrain,
        return;
      end
      
      % are labels
      [tblMFTtrn] = myparse(varargin,...
        'tblMFTtrn',[]... % (opt) table on which to train (cols MFTable.FLDSID only). defaults to all of obj.preProcGetMFTableLbled
        );
      
      % allow 'treatInfPosAsOcc' to default to false in these calls; we are
      % just checking number of labeled rows
      warnst = warning('off','Labeler:infData'); % ignore "Not including n rows with fully-occ labels"
      oc = onCleanup(@()warning(warnst));
      if ~isempty(tblMFTtrn)
        tblMFTp = obj.preProcGetMFTableLbled('tblMFTrestrict',tblMFTtrn);
      else
        tblMFTp = obj.preProcGetMFTableLbled();
      end
    
      nlabels = size(tblMFTp,1);
      
      if nlabels < 2,
        tfCanTrain = false;
        reason = 'There must be at least two labeled frames in the project.';
        return;
      end
      
      tfCanTrain = true;      
      
    end
    
    function [tfCanTrack,reason] = trackCanTrack(obj,tblMFT)
      tfCanTrack = false;
      if isempty(obj.tracker),
        reason = 'The tracker has not been set.';
        return;
      end      
      [tfCanTrack,reason] = obj.tracker.canTrack();      
      if ~tfCanTrack,
        return
      end      
      [tfCanTrack,reason] = PostProcess.canPostProcess(obj,tblMFT);
    end  % function
    
    function tfCanTrack = trackAllCanTrack(obj)
      tfCanTrack = cellfun(@(t)(t.canTrack), obj.trackerHistory_) ;
    end
    
    function [result, message] = doProjectAndMovieExist(obj)
      % Returns true iff a project exists and a movie is open.
      % If no project exists, returns false.
      % If a project exists but no movie is open, returns false.
      % Otherwise, returns true.
      % message gives info about why the result it what is.
      if obj.hasProject ,
        if obj.hasMovie ,
          result = true ;
          message = '' ;
        else
          result = false ;
          message = 'There is no movie open.' ;
        end
      else
        result = false ;
        message = 'There is no project open.' ;
      end
    end
    
    function track(obj, varargin)
      % When this method exits, update the views
      oc = onCleanup(@()(obj.notify('update'))) ;

      tm = obj.getTrackModeMFTSet() ;
      [okToProceed, message] = obj.doProjectAndMovieExist() ;
      if ~okToProceed ,
        error(message) ;
      end
      obj.pushBusyStatus('Preparing for tracking...');
      cleaner = onCleanup(@()(obj.popBusyStatus())) ;
      tblMFT = tm.getMFTable(obj,'istrack',true);
      if isempty(tblMFT) ,
        error('All frames already tracked.') ;
      end
      [tfCanTrack,reason] = obj.trackCanTrack(tblMFT);
      if ~tfCanTrack,
        error('Error tracking: %s', reason) ;
      end
      fprintf('Tracking started at %s...\n',datestr(now()));
      obj.trackCore_(tblMFT, varargin{:}) ;      

      % Update the background processing status string, if tracking really was
      % spawned.
      if obj.bgTrkIsRunning 
        algName = obj.tracker.algorithmName;
        backend_type_string = obj.trackDLBackEnd.prettyName() ;
        obj.backgroundProcessingStatusString_ = ...
          sprintf('%s tracking on %s (started %s)', algName, backend_type_string, datestr(now(),'HH:MM')) ;  %#ok<TNOW1,DATST>        
      end
    end

    function trackCore_(obj, mftset, varargin)
      % mftset: an MFTSet or table tblMFT

      % Let user know what's going on...
      obj.pushBusyStatus('Spawning tracking job...');
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      if obj.maIsMA
        args = horzcat({'trackType', apt.TrackType.detect}, varargin) ;
      else
        args = varargin ;
      end
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end

      if isa(mftset,'table'),
        tblMFT = mftset;
      else
        assert(isa(mftset,'MFTSet'));
        tblMFT = mftset.getMFTable(obj,'istrack',true);
      end

      % Which movies are we tracking?
      [movidx,~,newmov] = unique(tblMFT.mov);
      movidx_new = [];
      for ndx = 1:numel(movidx)
        mn = movidx.get();
        movidx_new(end+1) = mn;  %#ok<AGROW> 
      end
      movidx = movidx_new;
      tblMFT.mov = newmov;
      all_movfiles = obj.movieFilesAllFullGTaware;
      movfiles = all_movfiles(movidx,:);

      % get data associated with those movies
      if obj.hasTrx,
        trxfiles = obj.trxFilesAllFullGTaware;
        trxfiles = trxfiles(movidx,:);
      else
        trxfiles = {};
      end

      % Get crop ROIs
      movieset_count = size(movfiles,1) ;
      if obj.cropProjHasCrops,
        cropInfo = obj.getMovieFilesAllCropInfoGTAware();
        croprois = cell([movieset_count,obj.nview]);
        for i = 1:movieset_count,
          cropInfoThisMovie = cropInfo{movidx(i)} ;
          for j = 1:obj.nview,
            croprois{i,j} = cropInfoThisMovie(j).roi;
          end
        end
      else
        croprois = {};
      end

      % Get calibration data
      caldata = obj.viewCalibrationDataGTaware;
      if ~isempty(caldata)
        if ~obj.viewCalProjWide
          caldata = caldata(movidx);
        end
      end

      % Put all the info into a ToTrackInfo object
      totrackinfo = ...
        ToTrackInfo('tblMFT',tblMFT, ...
                    'movfiles',movfiles, ...
                    'trxfiles',trxfiles, ...
                    'views',1:obj.nview, ...
                    'stages',1:tObj.getNumStages(), ...
                    'croprois',croprois, ...
                    'calibrationdata',caldata) ;

      % Call the Tracker object to do the heavy lifting of tracking
      tObj.track('totrackinfo', totrackinfo, 'isexternal', false, args{:}, 'projTempDir', obj.projTempDir) ;

      % For template mode to see new tracking results
      obj.labelsUpdateNewFrame(true);
    end  % track() function
    
    function trackTbl(obj,tblMFT,varargin)
      assert(false,'This is not supported')
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      tObj.track(tblMFT,varargin{:});
      % For template mode to see new tracking results
      obj.labelsUpdateNewFrame(true);
      
      fprintf('Tracking complete at %s.\n',datestr(now));
    end
    
    function deleteOldTrackers(obj)
      obj.pushBusyStatus('Deleting old trained trackers and all tracking results...');
      oc = onCleanup(@()(obj.popBusyStatus())) ;      
      trackers = obj.trackerHistory_ ;
      cellfun(@delete, trackers(2:end)) ;
      currentTracker = trackers(1) ;  % singleton cell array
      obj.trackerHistory_ = currentTracker ;
      obj.notify('update_menu_track_tracker_history') ;
    end
    
    function resetCurrentTracker(obj)
      obj.pushBusyStatus('Resetting current trained tracker and all tracking results...');      
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      obj.notify('update_text_trackerinfo') ;
      obj.notify('update_menu_track_tracker_history') ;
    end
    
    function deleteCurrentTracker(obj)
      obj.pushBusyStatus('Deleting current tracker and all tracking results...');
      oc = onCleanup(@()(obj.popBusyStatus())) ;      
      trackers = obj.trackerHistory_ ;
      if numel(trackers) > 1 ,
        delete(trackers{1}) ;
        obj.trackerHistory_ = trackers(2:end) ;
      else
        tracker = obj.tracker ;
        tracker.init() ;
      end
      obj.notify('update_text_trackerinfo') ;
      obj.notify('update_menu_track_tracker_history') ;
    end  % function
    
    function [tfsucc,tblPCache,s] = trackCreateDeepTrackerStrippedLbl(obj, varargin)
      % For use with DeepTrackers. Create stripped lbl based on
      % .currTracker
      %
      % tfsucc: false if user canceled etc.
      % tblPCache: table of data cached in stripped lbl (eg training data, 
      %   or gt data)
      % s: scalar struct, stripped lbl struct
      
      [wbObj,ppdata,sPrmAll,shuffleRows,updateCacheOnly] = myparse(varargin,...
        'wbObj',[],...
        'ppdata',[],... % preproc data; or [] => auto-generate; or 'skip' => dont include ppdatacache in stripped lbl (eg for projs that use trnpacks)
        'sPrmAll',[],...
        'shuffleRows',true, ...
        'updateCacheOnly',false ... % if true, output args are 
                                ... % [tfsucc,tblPCache,ppdbICache] where ppdbICache 
                                ... % are indices into obj.ppdb for tblPCache
        );
      tfWB = ~isempty(wbObj);
      
      if ~obj.hasMovie
        % for NumChans see below
        error('Please select/open a movie.');
      end
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('There is no current tracker selected.');
      end
      
      isGT = obj.gtIsGTMode;
      if isGT
        descstr = 'gt';
      else
        descstr = 'training';
      end
      
      %
      % Determine the training set
      % 
      tfSkipPPData = strcmp(ppdata,'skip') || ...
                     isempty(ppdata) && tObj.trnNetMode.isTrnPack;
      tfSkipPPData = tfSkipPPData && tObj.lObj.maIsMA;
                   %MK 7 feb 22 -- Adding preprocessing for single animal 
      if tfSkipPPData
        assert(~updateCacheOnly);
        tblPCache = [];
      elseif isempty(ppdata),
        treatInfPosAsOcc = ...
          isa(tObj,'DeepTracker') && tObj.trnNetType.doesOccPred;
        tblPCache = obj.preProcGetMFTableLbled(...
          'wbObj',wbObj,...
          'gtModeOK',isGT,...
          'treatInfPosAsOcc',treatInfPosAsOcc ...
          );
        if tfWB && wbObj.isCancel
          tfsucc = false;
          tblPCache = [];
          s = [];
          return;
        end
        
        if isempty(tblPCache)
          error('No %s data available.',descstr);
        end
        
        if obj.hasTrx
          tblfldscontainsassert(tblPCache,[MFTable.FLDSCOREROI {'thetaTrx'}]);
        elseif obj.cropProjHasCrops
          tblfldscontainsassert(tblPCache,[MFTable.FLDSCOREROI]);
        else
          tblfldscontainsassert(tblPCache,MFTable.FLDSCORE);
        end
        
        prmsTgtCropTmp = tObj.sPrmAll.ROOT.MultiAnimal.TargetCrop;
%         if tObj.trnNetMode.isTrnPack
%           % Temp fix; prob should just skip adding imcache to stripped lbl
%           prmsTgtCropTmp.AlignUsingTrxTheta = false;
%         end
        [tblAddReadFailed,tfAU,locAU] = obj.ppdb.addAndUpdate(tblPCache,obj,...
          'wbObj',wbObj,'prmsTgtCrop',prmsTgtCropTmp);
        if tfWB && wbObj.isCancel
          tfsucc = false;
          tblPCache = [];
          s = [];
          return;
        end
        nMissedReads = height(tblAddReadFailed);
        if nMissedReads>0
          warningNoTrace('Removing %d %s rows, failed to read images.\n',...
            nMissedReads,descstr);
        end
        
        assert(all(locAU(~tfAU)==0));
        
        ppdbICache = locAU(tfAU); % row indices into obj.ppdb.dat for our training set
        tblPCache = obj.ppdb.dat.MD(ppdbICache,:);
        
        fprintf(1,'%s with %d rows.\n',descstr,numel(ppdbICache));
        fprintf(1,'%s data summary:\n',descstr);
        obj.ppdb.dat.summarize('mov',ppdbICache);        
      else
        % training set provided; note it may or may not include fully-occ
        % labels etc.
        tblPCache = ppdata.MD;
        ppdbICache = (1:ppdata.N)';
      end
      
      if updateCacheOnly
        tfsucc = true;
        % tblPCache set
        s = ppdbICache;
        return;
      end
      
      
      % 
      % Create the stripped lbl struct
      % 

      if isempty(ppdata),
        s = obj.projGetSaveStruct('forceIncDataCache',true,...
          'macroreplace',true,'massageCropProps',true);
      else
        s = obj.projGetSaveStruct('forceExcDataCache',true,...
          'macroreplace',true,'massageCropProps',true);
      end
      s.projectFile = obj.projectfile;

      nchan = arrayfun(@(x)x.getreadnchan,obj.movieReader);
      nchan = unique(nchan);
      if ~isscalar(nchan)
        error('Number of channels differs across views.');
      end
      s.cfg.NumChans = nchan; % see below, we change this again
      s.cfg.HasTrx = obj.hasTrx;
%       if nchan>1
%         warningNoTrace('Images have %d channels. Typically grayscale images are preferred; select View>Convert to grayscale.',nchan);
%       end
      
      % AL: moved above
      if ~isempty(ppdata) % includes tfSkipPPData
%         ppdbICache = true(ppdata.N,1);
      else
        % De-objectize .ppdb.dat (CPRData)
        ppdata = s.ppdb.dat;
      end
      
      if ~tfSkipPPData
        fprintf(1,'Stripped lbl preproc data cache: exporting %d/%d %s rows.\n',...
          numel(ppdbICache),ppdata.N,descstr);

        ppdataI = ppdata.I(ppdbICache,:);
        ppdataP = ppdata.pGT(ppdbICache,:);
        ppdataMD = ppdata.MD(ppdbICache,:);

        ppdataMD.mov = int32(ppdataMD.mov); % MovieIndex
        ppMDflds = tblflds(ppdataMD);
        s.preProcData_I = ppdataI;
        s.preProcData_P = ppdataP;
        for f=ppMDflds(:)',f=f{1}; %#ok<FXSET>
          sfld = ['preProcData_MD_' f];
          s.(sfld) = ppdataMD.(f);
        end

        if isfield(s,'ppdb'),
          s = rmfield(s,'ppdb');
        end
        if isfield(s,'preProcData'),
          s = rmfield(s,'preProcData');
        end

        % 20201120 randomize training rows
        if shuffleRows
          fldsPP = fieldnames(s);
          fldsPP = fldsPP(startsWith(fldsPP,'preProcData_'));
          nTrn = size(ppdataI,1);
          prand = obj.projDeterministicRandFcn(@()randperm(nTrn));
          fprintf(1,'Shuffling training rows. Your RNG seed is: %d\n',obj.projRngSeed);
          for f=fldsPP(:)',f=f{1}; %#ok<FXSET>
            v = s.(f);
            assert(ndims(v)==2 && size(v,1)==nTrn); %#ok<ISMAT>
            s.(f) = v(prand,:);
          end
        else
          prand = [];
        end
        s.preProcData_prand = prand;
        
        % check with Mayank, thought we wanted number of "underlying" chans
        % but DL is erring when pp data is grayscale but NumChans is 3
        s.cfg.NumChans = size(s.preProcData_I{1},3);
      end
      
      s.trackerClass = {'__UNUSED__' 'DeepTracker'};
      
      %
      % Final Massage
      % 

      % KB 20220517 - wanted to use this part of the code elsewhere, broke
      % into another function
      s.trackerData = ...
        DeepTracker.massageTrackerData(s.trackerData{1},...
                                       obj,...
                                       'sPrmAll',sPrmAll);
      
%       tdata = s.trackerData{s.currTracker};
%       tfTD = isfield(tdata,'stg2');      
%       if tfTD
%         tdata = [tdata.stg1; tdata.stg2];
%       end
%       netmodes = [tdata.trnNetMode];
%       assert(all(tfTD==[netmodes.isTwoStage]));
%       
%       for i=1:numel(tdata)
%         if ~isempty(sPrmAll)
%           tdata(i).sPrmAll = sPrmAll;
%         end
%         tdata(i).sPrmAll = obj.addExtraParams(tdata(i).sPrmAll,...
%           tdata(i).trnNetMode);
%         tdata(i).trnNetTypeString = char(tdata(i).trnNetType);
%       end
%       
%       if tfTD
%         s.trackerData = num2cell(tdata(:)');
%         
%         % stage 1 trackData; move Detect.DeepTrack to top-level
%         s.trackerData{1}.sPrmAll.ROOT.DeepTrack = ...
%           s.trackerData{1}.sPrmAll.ROOT.MultiAnimal.Detect.DeepTrack;
%         s.trackerData{1}.sPrmAll.ROOT.MultiAnimal.Detect = rmfield(...
%           s.trackerData{1}.sPrmAll.ROOT.MultiAnimal.Detect,'DeepTrack');
%       else
%         s.trackerData = {[] tdata};
%       end
%       % remove detect/DeepTrack from stage2
%       s.trackerData{2}.sPrmAll.ROOT.MultiAnimal.Detect = rmfield(...
%           s.trackerData{2}.sPrmAll.ROOT.MultiAnimal.Detect,'DeepTrack');        
      s.nLabels = ppdata.N;      
      
      tfsucc = true;
    end
    
    % See also Lbl.m for addnl stripped lbl meths
    
    function sPrmAll = addExtraParams(obj,sPrmAll,netmode)  % const
      % sPrmAll = addExtraParams(obj,sPrmAll)
      % sPrmAll = addExtraParams(obj,sPrmAll,netmode)
      %
      % if generating trnpack for use by backend, include netmode.
      % without netmode, backend-specific config params are not included.
      
      skel = obj.skeletonEdges;
      if ~isempty(skel),
        nedge = size(skel,1);
        skelstr = arrayfun(@(x)sprintf('%d %d',skel(x,1),skel(x,2)),1:nedge,'uni',0);
        skelstr = String.cellstr2CommaSepList(skelstr);
        sPrmAll.ROOT.DeepTrack.OpenPose.affinity_graph = skelstr;
        sPrmAll.ROOT.MultiAnimal.Detect.DeepTrack.OpenPose.affinity_graph = skelstr;
      else
        sPrmAll.ROOT.DeepTrack.OpenPose.affinity_graph = '';
        sPrmAll.ROOT.MultiAnimal.Detect.DeepTrack.OpenPose.affinity_graph = '';
      end
            
      % add landmark matches
      matches = obj.flipLandmarkMatches;
      nedge = size(matches,1);
      matchstr = arrayfun(@(x)sprintf('%d %d',matches(x,1),matches(x,2)),1:nedge,'uni',0);
      matchstr = String.cellstr2CommaSepList(matchstr);
      sPrmAll.ROOT.DeepTrack.DataAugmentation.flipLandmarkMatches = matchstr;
      sPrmAll.ROOT.MultiAnimal.Detect.DeepTrack.DataAugmentation.flipLandmarkMatches = matchstr;
      
      % ma stuff
      prmsTgtCrop = sPrmAll.ROOT.MultiAnimal.TargetCrop;
      r = maGetTgtCropRad(prmsTgtCrop);
      % actual radius that will be used by backend
      sPrmAll.ROOT.MultiAnimal.TargetCrop.Radius = r;
      tfBackEnd = exist('netmode','var');
      if tfBackEnd
        sPrmAll.ROOT.MultiAnimal.is_multi = netmode.is_multi;
        can_multi_crop_ims = netmode.multi_crop_ims;
        if sPrmAll.ROOT.MultiAnimal.multi_crop_ims && ~can_multi_crop_ims
          warningNoTrace('setting multi_crop_ims to False.');
          sPrmAll.ROOT.MultiAnimal.multi_crop_ims = false;
        end
        sPrmAll.ROOT.MultiAnimal.Detect.multi_only_ht = netmode.multi_only_ht;
        sPrmAll.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta = ...
          netmode.isHeadTail || netmode==DLNetMode.multiAnimalTDPoseTrx;
      end
      % headtail
      if ~isempty(obj.skelHead)
        iptHead = obj.skelHead;
      else
        iptHead = 0;
      end
      if ~isempty(obj.skelTail)
        iptTail = obj.skelTail;
      else
        iptTail = 0;
      end        
      sPrmAll.ROOT.MultiAnimal.Detect.ht_pts = [iptHead iptTail];
    end
    
    function sPrmAll = setExtraParams(obj,sPrmAll)
      % AL 20200409 sets .skeletonEdges and .setFliplandmarkMatches from 
      % sPrmAll fields.
      
      if structisfield(sPrmAll,'ROOT.DeepTrack.OpenPose.affinity_graph'),
        skelstr = sPrmAll.ROOT.DeepTrack.OpenPose.affinity_graph;
        skel = Labeler.hlpParseCommaSepGraph(skelstr);
        obj.skeletonEdges = skel;
        sPrmAll.ROOT.DeepTrack.OpenPose = rmfield(sPrmAll.ROOT.DeepTrack.OpenPose,'affinity_graph');
        if isempty(fieldnames(sPrmAll.ROOT.DeepTrack.OpenPose)),
          sPrmAll.ROOT.DeepTrack.OpenPose = '';
        end
      end

      % add landmark matches
      if structisfield(sPrmAll,'ROOT.DeepTrack.DataAugmentation.flipLandmarkMatches'),
        matchstr = sPrmAll.ROOT.DeepTrack.DataAugmentation.flipLandmarkMatches;
        matches = Labeler.hlpParseCommaSepGraph(matchstr);
        obj.setFlipLandmarkMatches(matches);
        sPrmAll.ROOT.DeepTrack.DataAugmentation = rmfield(sPrmAll.ROOT.DeepTrack.DataAugmentation,'flipLandmarkMatches');
        if isempty(fieldnames(sPrmAll.ROOT.DeepTrack.DataAugmentation)),
          sPrmAll.ROOT.DeepTrack.DataAugmentation = '';
        end
      end
    end
    
%     function trackCrossValidate(obj,varargin)
%       % Run k-fold crossvalidation. Results stored in .xvResults
%       
%       [kFold,initData,wbObj,tblMFgt,tblMFgtIsFinal,partTst,dontInitH0] = ...
%         myparse(varargin,...
%         'kfold',3,... % number of folds
%         'initData',false,... % OBSOLETE, you would never want this. if true, call .initData() between folds to minimize mem usage
%         'wbObj',[],... % (opt) WaitBarWithCancel
%         'tblMFgt',[],... % (opt), MFTable of data to consider. Defaults to all labeled rows. 
%                          % tblMFgt should only contain fields .mov, .frm, .iTgt. labels, rois, etc will be assembled from proj
%         'tblMFgtIsFinal',false,... % a bit silly, for APT developers only. Set to true if your tblMFgt is in final form.
%         'partTst',[],... % (opt) pre-defined training splits. If supplied, partTst must be a [height(tblMFgt) x kfold] logical. 
%                          % tblMFgt should be supplied. true values indicate test rows, false values indicate training rows.
%         'dontInitH0',true...
%       );        
%       
%       tfWB = ~isempty(wbObj);
%       tfTblMFgt = ~isempty(tblMFgt);      
%       tfPart = ~isempty(partTst);
%       
%       if obj.gtIsGTMode
%         error('Unsupported in GT mode.');
%       end
%       
%       if ~tfTblMFgt
%         % CPR required below; allow 'treatInfPosAsOcc' to default to false
%         tblMFgt = obj.preProcGetMFTableLbled();
%       elseif ~tblMFgtIsFinal        
%         tblMFgt0 = tblMFgt; % legacy checks below
%         % CPR required below; allow 'treatInfPosAsOcc' to default to false
%         tblMFgt = obj.preProcGetMFTableLbled('tblMFTrestrict',tblMFgt);
%         % Legacy checks/assert can remove at some pt
%         assert(height(tblMFgt0)==height(tblMFgt),...
%           'Specified ''tblMFgt'' contains unlabeled row(s).');
%         assert(isequal(tblMFgt(:,MFTable.FLDSID),tblMFgt0));
%         assert(isa(tblMFgt.mov,'MovieIndex'));
%       else
%         % tblMFgt supplied, and should have labels etc.
%       end
%       assert(isa(tblMFgt.mov,'MovieIndex'));
%       [~,gt] = tblMFgt.mov.get();
%       assert(~any(gt));
%       
%       if ~tfPart
%         movC = categorical(tblMFgt.mov);
%         tgtC = categorical(tblMFgt.iTgt);
%         grpC = movC.*tgtC;
%         cvPart = cvpartition(grpC,'kfold',kFold);
%         partTrn = arrayfun(@(x)cvPart.training(x),1:kFold,'uni',0);
%         partTst = arrayfun(@(x)cvPart.test(x),1:kFold,'uni',0);
%         partTrn = cat(2,partTrn{:});
%         partTst = cat(2,partTst{:});
%       else
%         partTrn = ~partTst;
%       end
%       assert(islogical(partTrn) && islogical(partTst));
%       n = height(tblMFgt);
%       szassert(partTrn,[n kFold]);
%       szassert(partTst,[n kFold]);
%       tmp = partTrn+partTst;
%       assert(all(tmp(:)==1),'Invalid cv splits specified.'); % partTrn==~partTst
%       assert(all(sum(partTst,2)==1),...
%         'Invalid cv splits specified; each row must be tested precisely once.');
%       
%       tObj = obj.tracker;
%       if isempty(tObj)
%         error('Labeler:tracker','No tracker is available for this project.');
%       end
%       if ~strcmp(tObj.algorithmName,'cpr')
%         % DeepTrackers do non-blocking/bg tracking
%         error('Only CPR tracking currently supported.');
%       end      
% 
%       if ~dontInitH0
%         obj.preProcUpdateH0IfNec();
%       end
%       
%       % Basically an initHook() here
%       if initData
%         obj.preProcInitData();
%         obj.ppdbInit();
%       end
%       tObj.trnDataInit(); % not strictly necessary as .retrain() should do it 
%       tObj.trnResInit(); % not strictly necessary as .retrain() should do it 
%       tObj.trackResInit();
%       tObj.vizInit();
%       tObj.asyncReset();
%       
%       npts = obj.nLabelPoints;
%       pTrkCell = cell(kFold,1);
%       dGTTrkCell = cell(kFold,1);
%       if tfWB
%         wbObj.startPeriod('Fold','shownumden',true,'denominator',kFold);
%       end
%       for iFold=1:kFold
%         if tfWB
%           wbObj.updateFracWithNumDen(iFold);
%         end
%         tblMFgtTrain = tblMFgt(partTrn(:,iFold),:);
%         tblMFgtTrack = tblMFgt(partTst(:,iFold),:);
%         fprintf(1,'Fold %d: nTrain=%d, nTest=%d.\n',iFold,...
%           height(tblMFgtTrain),height(tblMFgtTrack));
%         if tfWB
%           wbObj.startPeriod('Training','nobar',true);
%         end
%         tObj.retrain('tblPTrn',tblMFgtTrain,'wbObj',wbObj);
%         if tfWB
%           wbObj.endPeriod();
%         end
%         tObj.track(tblMFgtTrack,'wbObj',wbObj);        
%         tblTrkRes = tObj.getTrackingResultsTable(); % if wbObj.isCancel, partial tracking results
%         if initData
%           obj.preProcInitData();
%           obj.ppdbInit();
%         end
%         tObj.trnDataInit();
%         tObj.trnResInit();
%         tObj.trackResInit();
%         if tfWB && wbObj.isCancel
%           return;
%         end
%         
%         %assert(isequal(pTrkiPt(:)',1:npts));
%         assert(isequal(tblTrkRes(:,MFTable.FLDSID),...
%                        tblMFgtTrack(:,MFTable.FLDSID)));
%         if obj.hasTrx || obj.cropProjHasCrops
%           pGT = tblMFgtTrack.pAbs;
%         else
%           if tblfldscontains(tblMFgtTrack,'pAbs')
%             assert(isequal(tblMFgtTrack.p,tblMFgtTrack.pAbs));
%           end
%           pGT = tblMFgtTrack.p;
%         end
%         d = tblTrkRes.pTrk - pGT;
%         [ntst,Dtrk] = size(d);
%         assert(Dtrk==npts*2); % npts=nPhysPts*nview
%         d = reshape(d,ntst,npts,2);
%         d = sqrt(sum(d.^2,3)); % [ntst x npts]
%         
%         pTrkCell{iFold} = tblTrkRes;
%         dGTTrkCell{iFold} = d;
%       end
% 
%       % create output table
%       for iFold=1:kFold
%         tblFold = table(repmat(iFold,height(pTrkCell{iFold}),1),...
%           'VariableNames',{'fold'});
%         pTrkCell{iFold} = [tblFold pTrkCell{iFold}];
%       end
%       pTrkAll = cat(1,pTrkCell{:});
%       dGTTrkAll = cat(1,dGTTrkCell{:});
%       assert(isequal(height(pTrkAll),height(tblMFgt),size(dGTTrkAll,1)));
%       [tf,loc] = tblismember(tblMFgt,pTrkAll,MFTable.FLDSID);
%       assert(all(tf));
%       pTrkAll = pTrkAll(loc,:);
%       dGTTrkAll = dGTTrkAll(loc,:);
%       
%       if tblfldscontains(tblMFgt,'roi')
%         flds = MFTable.FLDSCOREROI;
%       else
%         flds = MFTable.FLDSCORE;
%       end
%       tblXVres = tblMFgt(:,flds);
%       if tblfldscontains(tblMFgt,'pAbs')
%         tblXVres.p = tblMFgt.pAbs;
%       end
%       tblXVres.pTrk = pTrkAll.pTrk;
%       tblXVres.dGTTrk = dGTTrkAll;
%       tblXVres = [pTrkAll(:,'fold') tblXVres];
%       
%       obj.xvResults = tblXVres;
%       obj.xvResultsTS = now;
%     end  % function
        
    function [tf,lposTrk,occTrk] = trackIsCurrMovFrmTracked(obj,iTgt)
      % tf: scalar logical, true if tracker has results/predictions for 
      %   currentMov/frm/iTgt 
      % lposTrk: [nptsx2] if tf is true, xy coords from tracker; otherwise
      %   indeterminate
      % occTrk: [npts] if tf is true, point is predicted as occluded
      
      tObj = obj.tracker;
      if isempty(tObj)
        tf = false;
        lposTrk = [];
        occTrk = [];
      else
        [tfhaspred,xy,occ] = tObj.getTrackingResultsCurrFrm();
        tf = tfhaspred(iTgt);
        szassert(xy,[obj.nLabelPoints 2 obj.nTargets]);        
        lposTrk = xy(:,:,iTgt);
        occTrk = occ(:,iTgt);
      end
    end
    
    % [tbl,I,tfReadFailed] = trackLabelMontageProcessData(obj,tbl)      
    % Process tbl data for montage plotting
    function [tbl,I,tfReadFailed] = trackLabelMontageProcessData(obj,tbl)      
      
      tbl = obj.preProcCropLabelsToRoiIfNec(tbl,...
        'doRemoveOOB',false,...
        'pfld','pLbl',...
        'pabsfld','pLblAbs',...
        'proifld','pLblRoi');
      tbl = obj.preProcCropLabelsToRoiIfNec(tbl,...
        'doRemoveOOB',false,...
        'pfld','pTrk',...
        'pabsfld','pTrkAbs',...
        'proifld','pTrkRoi');

      % tbl.pLbl/pTrk now in relative coords if appropriate
     
      % Create a table to call preProcDataFetch so we can use images in
      % preProc cache.      
      FLDSTMP = {'mov' 'frm' 'iTgt' 'tfoccLbl' 'pLblAbs'}; % MFTable.FLDSCORE
      tfROI = tblfldscontains(tbl,'roi');     
      if tfROI
        FLDSTMP = [FLDSTMP 'roi'];
      end
%       if tblisfield(tbl,'nNborMask')
%         % Why this is in the cache?
%         FLDSTMP = [FLDSTMP 'nNborMask'];
%       end
      tblCacheUpdate = tbl(:,FLDSTMP);
      tblCacheUpdate.Properties.VariableNames(4:5) = {'tfocc' 'pAbs'};
%       [ppdata,ppdataIdx,~,~,tfReadFailed] = ...
%         obj.preProcDataFetch(tblCacheUpdate,'updateRowsMustMatch',true);
      
      % computeOnly=true out of abundance of caution (GT rows)
      [~,ppdata,tfReadFailed] = obj.ppdb.add(tblCacheUpdate,obj,'computeOnly',true);
      
      nReadFailed = nnz(tfReadFailed);
      if nReadFailed>0
        warningNoTrace('Failed to read %d frames/images; these will not be included in montage.',...
          nReadFailed);
        % Would be better to include with "blank" image
      end
      
      I = ppdata.I;      
    end
    
    function trackLabelMontage(obj,tbl,errfld,varargin)
      
      [nr,nc,h,npts,nphyspts,nplot,frmlblclr,frmlblbgclr,readImgFcn] = ...
        myparse(varargin,...
        'nr',3,...
        'nc',4,...
        'hPlot',[],...
        'npts',obj.nLabelPoints,... % hack
        'nphyspts',obj.nPhysPoints,... % hack
        'nplot',height(tbl),... % show/include nplot worst rows
        'frmlblclr',[1 1 1], ...
        'frmlblbgclr',[0 0 0], ...
        'readImgFcn',@obj.trackLabelMontageProcessData ... 
        );
      
      if nplot>height(tbl)
        warningNoTrace('''nplot'' argument too large. Only %d GT rows are available.',height(tbl));
        nplot = height(tbl);
      end
      
      tbl = sortrows(tbl,{errfld},{'descend'});
      tbl = tbl(1:nplot,:);
      
      [tbl,I,tfReadFailed] = readImgFcn(tbl);

      tblPostRead = tbl(:,{'pLbl' 'pTrk' 'mov' 'frm' 'iTgt' errfld});
      tblPostRead(tfReadFailed,:) = [];
    
      if obj.hasTrx
        frmLblsAll = arrayfun(@(zm,zf,zt,ze)sprintf('mov/frm/tgt=%d/%d/%d,err=%.2f',zm,zf,zt,ze),...
          abs(tblPostRead.mov),tblPostRead.frm,tblPostRead.iTgt,tblPostRead.(errfld),'uni',0);        
      else
        frmLblsAll = arrayfun(@(zm,zf,ze)sprintf('mov/frm=%d/%d,err=%.2f',zm,zf,ze),...
          abs(tblPostRead.mov),tblPostRead.frm,tblPostRead.(errfld),'uni',0);
      end
      
      nrowsPlot = height(tblPostRead);
      startIdxs = 1:nr*nc:nrowsPlot;
      for i=1:numel(startIdxs)
        plotIdxs = startIdxs(i):min(startIdxs(i)+nr*nc-1,nrowsPlot);
        frmLblsThis = frmLblsAll(plotIdxs);
        for iView=1:obj.nview
          h(end+1,1) = figure('Name','Tracking Error Montage','windowstyle','docked'); %#ok<AGROW>
          pColIdx = (1:nphyspts)+(iView-1)*nphyspts;
          pColIdx = [pColIdx pColIdx+npts]; %#ok<AGROW>
          Shape.montage(I(:,iView),tblPostRead.pLbl(:,pColIdx),'fig',h(end),...
            'nr',nr,'nc',nc,'idxs',plotIdxs,...
            'framelbls',frmLblsThis,'framelblscolor',frmlblclr,...
            'framelblsbgcolor',frmlblbgclr,'p2',tblPostRead.pTrk(:,pColIdx),...
            'p2marker','+','titlestr','Tracking Montage, descending err (''+'' is tracked)');
        end
      end
    end
    
    function tv = createTrackingVisualizer(obj,ptsPlotInfoFld,gfxTagPfix)
      % Create TV appropriate to this proj
      %
      % gfxTagPfix: arbitrary id/prefix for graphics handle tags
      
      if obj.maIsMA
        tv = TrackingVisualizerTracklets(obj,ptsPlotInfoFld,gfxTagPfix);
      elseif obj.hasTrx
        tfadvanced = true; %RC.getpropdefault('optimizeImportedViz',false);
        if tfadvanced
          tv = TrackingVisualizerMTFast(obj,ptsPlotInfoFld,gfxTagPfix);
        else
          tv = TrackingVisualizerMT(obj,ptsPlotInfoFld,gfxTagPfix);
        end
      else
        tv = TrackingVisualizerMT(obj,ptsPlotInfoFld,gfxTagPfix);
      end
    end
  end

  methods (Static)
    function edges = hlpParseCommaSepGraph(str)
      % str: eg '1 2, 3 4'
      if isempty(str)
        edgesCell = cell(0,1);
      else
        edgesCell = strsplit(str,',');
      end
      nedge = numel(edgesCell);
      edges = nan(nedge,2);
      for i = 1:nedge,
        edges(i,:) = sscanf(edgesCell{i},'%d %d');
      end
    end

    function tblLbled = hlpTblLbled(tblLbled)
      tblLbled.mov = int32(tblLbled.mov);
      tblLbled = tblLbled(:,[MFTable.FLDSID {'p'}]);
      isLbled = true(height(tblLbled),1);
      tblLbled = [tblLbled table(isLbled)];
    end

    function trkErr = hlpTblErr(tblBig,fldLbled,fldTrked,fldpLbl,fldpTrk,npts)
      tf = tblBig.(fldLbled) & tblBig.(fldTrked);
      pLbl = tblBig.(fldpLbl)(tf,:);
      pTrk = tblBig.(fldpTrk)(tf,:);
      szassert(pLbl,size(pTrk));
      nTrkLbl = size(pLbl,1);
      dErr = pLbl-pTrk;
      dErr = reshape(dErr,nTrkLbl,npts,2);
      dErr = sqrt(sum(dErr.^2,3)); % [nTrkLbl x npts] L2 err
      dErr = mean(dErr,2); % [nTrkLblx1] L2 err, mean across pts      
      trkErr = nan(height(tblBig),1);
      trkErr(tf) = dErr;
    end
    
    function sPrm = trackGetParamsFromStruct(s)
      % Get all parameters:
      %  - preproc
      %  - cpr
      %  - common dl
      %  - specific dl
      %
      % sPrm: scalar struct containing NEW-style params:
      % sPrm.ROOT.Track
      %          .CPR
      %          .DeepTrack (if applicable)
      % Top-level fields .Track, .CPR, .DeepTrack may be missing if they
      % don't exist yet.
      
      % Future TODO: As in trackSetParams, currently this is hardcoded when
      % it ideally would just be a generic loop
      
      if isfield(s,'trackParams'),
        sPrm = s.trackParams;
        return;
      end
      
      prmCpr = [];
      for iTrk=1:numel(s.trackerData)
        if strcmp(s.trackerClass{iTrk}{1},'CPRLabelTracker') && ~isempty(s.trackerData{iTrk})
          prmCpr = s.trackerData{iTrk}.sPrm;
          break;
        end
      end
      
      prmPP = s.preProcParams;
      
      prmDLCommon = s.trackDLParams;
      
      prmDLSpecific = struct;
      for i = 1:numel(s.trackerData),
        if ~strcmp(s.trackerClass{i}{1},'DeepTracker') || isempty(s.trackerData{i}),
          continue;
        end
        prmField = APTParameters.getParamField(s.trackerData{i}.trnNetType);
        prmDLSpecific.(prmField) = s.trackerData{i}.sPrm;
      end
      
      prmTrack = struct;
      prmTrack.trackNFramesSmall = s.cfg.Track.PredictFrameStep;
      prmTrack.trackNFramesLarge = s.cfg.Track.PredictFrameStepBig;
      prmTrack.trackNFramesNear = s.cfg.Track.PredictNeighborhood;
      
      sPrm = Labeler.trackGetParamsHelper(prmCpr,prmPP,prmDLCommon,...
                                          prmDLSpecific,prmTrack);      
    end
    
    function sPrmAll = trackGetParamsHelper(prmCpr,prmPP,prmDLCommon,prmDLSpecific,obj)
      
      sPrmAll = APTParameters.defaultParamsStructAll;
      
%      assert(~xor(isempty(prmCpr),isempty(prmPP)));
      if ~isempty(prmCpr)
        sPrmAll = APTParameters.setCPRParams(sPrmAll,prmCpr);
      end
      if ~isempty(prmPP),
        sPrmAll = APTParameters.setPreProcParams(sPrmAll,prmPP);
      end
      if ~isempty(obj)
        sPrmAll = APTParameters.setNFramesTrackParams(sPrmAll,obj);
      end
      if ~isempty(prmDLCommon)
        sPrmAll = APTParameters.setTrackDLParams(sPrmAll,prmDLCommon);
      end
            
      % specific parameters
      fns = fieldnames(prmDLSpecific);
      for i = 1:numel(fns),
        sPrmAll = APTParameters.setDLSpecificParams(sPrmAll,fns{i},prmDLSpecific.(fns{i}));
      end
      
    end
    
  end
  methods
    function tblBig = trackGetBigLabeledTrackedTable_(obj)
      % Do the core work of generating the big target summary table.
      % tblBig: MFT table indcating isLbled, isTrked, trkErr, etc.
      
      progressMeter = obj.progressMeter_ ;
      assert(~isempty(progressMeter)) ;
      
      tblLbled = obj.labelGetMFTableLabeled('wbObj',progressMeter);
      if progressMeter.wasCanceled
        tblBig = [];
        return
      end
      tblLbled = Labeler.hlpTblLbled(tblLbled);
      
      tblLbled2 = obj.labelGetMFTableLabeled('wbObj',progressMeter,'useLabels2',true);
      if progressMeter.wasCanceled
        tblBig = [];
        return
      end

      tblLbled2 = Labeler.hlpTblLbled(tblLbled2);
      tblfldsassert(tblLbled2,[MFTable.FLDSID {'p' 'isLbled'}]);
      tblLbled2.Properties.VariableNames(end-1:end) = {'pImport' 'isImported'};
      
      % Sanity check
      nptsLabels = size(tblLbled.p,2)/2 ;
      nptsLabels2 = size(tblLbled2.pImport,2)/2 ;
      if nptsLabels ~= nptsLabels2 
        error('The number of keypoints in the labels (%d) does not match the number of keypoints in the imported labels (%d)', ...
              nptsLabels, ...
              nptsLabels2) ;
      end

      %npts = obj.nLabelPoints;

      tObj = obj.tracker;
      if ~isempty(tObj)
        tblTrked = tObj.getTrackingResultsTable();
      else
        tblTrked = [];
      end
      if ~isempty(tblTrked)
%         tblTrked = tObj.trkPMD(:,MFTable.FLDSID);
        tblTrked.mov = int32(tblTrked.mov); % from MovieIndex
%         pTrk = tblTrked.pTrk;
%         if isempty(pTrk)
%           % AL 20180620 Shouldn't be nec. 
%           % edge case, tracker not initting properly
%           pTrk = nan(0,npts*2);
%         end
        isTrked = true(height(tblTrked),1);
        tblTrked = [tblTrked table(isTrked)];     
      end

      if isempty(tblTrked)
        % 20171106 ML bug 2015b outerjoin empty input table
        % Seems fixed in 16b
        pTrk = nan(size(tblLbled.p));
        isTrked = false(size(tblLbled.isLbled));
        tblBig = [tblLbled table(pTrk,isTrked)];
      else
        tblBig = outerjoin(tblLbled,tblTrked,'Keys',MFTable.FLDSID,...
          'MergeKeys',true);
      end
      tblBig = outerjoin(tblBig,tblLbled2,'Keys',MFTable.FLDSID,'MergeKeys',true);
        
      % Compute tracking err (for rows both labeled and tracked)
      npts = obj.nLabelPoints;
      trkErr = Labeler.hlpTblErr(tblBig,'isLbled','isTrked','p','pTrk',npts);
      importedErr = Labeler.hlpTblErr(tblBig,'isLbled','isImported','p','pImport',npts); % treating as imported tracking
      tblBig = [tblBig table(trkErr) table(importedErr)];
      
      xvres = obj.xvResults;
      tfhasXV = ~isempty(xvres);
      if tfhasXV
        tblXVerr = rowfun(@(varargin)(mean(varargin{:}, 'omitnan')),xvres,'InputVariables',{'dGTTrk'},...
                          'OutputVariableNames',{'xvErr'});
        hasXV = true(height(xvres),1);
        tblXV = [xvres(:,MFTable.FLDSID) tblXVerr table(hasXV)];        
      else
        tblXV = table(nan(0,1),nan(0,1),nan(0,1),nan(0,1),false(0,1),...
          'VariableNames',{'mov' 'frm' 'iTgt' 'xvErr' 'hasXV'});
      end
      
      % 20171106 ML bug 2015b outerjoin empty input table
      tblBig = outerjoin(tblBig,tblXV,'Keys',MFTable.FLDSID,'MergeKeys',true);
      tblBig = tblBig(:,[MFTable.FLDSID {'trkErr' 'importedErr' 'xvErr' 'isLbled' 'isTrked' 'isImported' 'hasXV'}]);
      if tfhasXV && ~isequal(tblBig.isLbled,tblXV.hasXV)
        warningNoTrace('Cross-validation results appear out-of-date with respect to current set of labeled frames.');
      end
    end  % function
    
    function tblSumm = trackGetSummaryTable(obj,tblBig)
      % tblSumm: Big summary table, one row per (mov,tgt)
      
      assert(~obj.gtIsGTMode,'Currently unsupported in GT mode.');
      obj.pushBusyStatus('Computing big summary table...') ;
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      % generate tblSummBase
      if obj.hasTrx
        assert(obj.nview==1,'Currently unsupported for multiview projects.');
        % KB 20210626 - use cached trx file info
        %tfaf = obj.trxFilesAllFull;
        trxinfo = obj.trxInfoAll;
        dataacc = nan(0,4); % mov, tgt, trajlen, frm1
        for iMov=1:obj.nmovies
          nTgt = trxinfo{iMov,1}.ntgts;

%           tfile = tfaf{iMov,1};
%           tifo = obj.trxCache(tfile);
%           frm2trxI = tifo.frm2trx;
% 
%           nTgt = size(frm2trxI,2);
          for iTgt=1:nTgt
            trajlen = trxinfo{iMov,1}.endframes(iTgt)-trxinfo{iMov,1}.firstframes(iTgt)+1;
            frm1 = trxinfo{iMov,1}.firstframes(iTgt);

%             tflive = frm2trxI(:,iTgt);
%             sp = get_interval_ends(tflive);
%             if isempty(sp)
%               trajlen = 0;
%               frm1 = nan;
%             else
%               if numel(sp)>1
%                 warningNoTrace('Movie %d, target %d is live over non-consecutive frames.',...
%                   iMov,iTgt);
%               end
%               trajlen = nnz(tflive); % when numel(sp)>1, track is 
%               % non-consecutive and this won't strictly be trajlen
%               frm1 = sp(1);
%             end
            dataacc(end+1,:) = [iMov iTgt trajlen frm1]; %#ok<AGROW>
          end
        end
      
        % Contains every (mov,tgt)
        tblSummBase = array2table(dataacc,'VariableNames',{'mov' 'iTgt' 'trajlen' 'frm1'});
      else
        mov = (1:obj.nmovies)';
        iTgt = ones(size(mov));
        nfrms = cellfun(@(x)x.nframes,obj.movieInfoAll(:,1));
        trajlen = nfrms;
        frm1 = iTgt;
        tblSummBase = table(mov,iTgt,trajlen,frm1);
      end
      
      tblStats = rowfun(@Labeler.hlpTrackTgtStats,tblBig,...
        'InputVariables',{'isLbled' 'isTrked' 'isImported' 'hasXV' 'trkErr' 'importedErr' 'xvErr'},...
        'GroupingVariables',{'mov' 'iTgt'},...
        'OutputVariableNames',{'nFrmLbl' 'nFrmTrk' 'nFrmImported' ...
          'nFrmLblTrk' 'lblTrkMeanErr' ...
          'nFrmLblImported' 'lblImportedMeanErr' ...
          'nFrmXV' 'xvMeanErr'},...
        'NumOutputs',9);

      [tblSumm,iTblSummB,iTblStats] = outerjoin(tblSummBase,tblStats,...
        'Keys',{'mov' 'iTgt'},'MergeKeys',true);
      assert(~any(iTblSummB==0)); % tblSummBase should contain every (mov,tgt)
      tblSumm(:,{'GroupCount'}) = [];
      tfNoStats = iTblStats==0;
      % These next 4 sets replace NaNs from the join (due to "no
      % corresponding row in tblStats") with zeros
      tblSumm.nFrmLbl(tfNoStats) = 0;
      tblSumm.nFrmTrk(tfNoStats) = 0;
      tblSumm.nFrmImported(tfNoStats) = 0;
      tblSumm.nFrmLblTrk(tfNoStats) = 0;
      tblSumm.nFrmLblImported(tfNoStats) = 0;
      tblSumm.nFrmXV(tfNoStats) = 0;
    end  
  end  % methods

  methods (Static)
    function [nFrmLbl,nFrmTrk,nFrmImported,...
              nFrmLblTrk,lblTrkMeanErr,...
              nFrmLblImported,lblImportedMeanErr,...
              nFrmXV,xvMeanErr] = ...
      hlpTrackTgtStats(isLbled,isTrked,isImported,hasXV,...
                       trkErr,importedErr,xvErr)
        % nFrmLbl: number of labeled frames
        % nFrmTrk: number of tracked frames
        % nFrmLblTrk: " labeled&tracked frames
        % lblTrkMeanErr: L2 err, mean across pts, mean over labeled&tracked frames
        % nFrmXV: number of frames with XV res (should be same as nFrmLbl unless XV
        %   out-of-date)
        % xvMeanErr: L2 err, mean across pts, mean over xv frames
        
        nFrmLbl = nnz(isLbled);
        nFrmTrk = nnz(isTrked);
        nFrmImported = nnz(isImported);
        
        tfLbledTrked = isLbled & isTrked;
        nFrmLblTrk = nnz(tfLbledTrked);
        lblTrkMeanErr = mean(trkErr(tfLbledTrked), 'omitnan');
        
        tfLbledImported = isLbled & isImported;
        nFrmLblImported = nnz(tfLbledImported);
        lblImportedMeanErr = mean(importedErr(tfLbledImported), 'omitnan');        
        
        nFrmXV = nnz(hasXV);
        xvMeanErr = mean(xvErr(hasXV));
      end
      
 
%     function [success,fname] = trackSaveLoadAsHelper(obj,rcprop,uifcn,...
%         promptstr,rawMeth)
%       % rcprop: Name of RC property for guessing path
%       % uifcn: either 'uiputfile' or 'uigetfile'
%       % promptstr: used in uiputfile
%       % rawMeth: track*Raw method to call when a file is specified
%       
%       % Guess a path/location for save/load
%       lastFile = obj.rcGetProp(rcprop);
%       if isempty(lastFile)
%         projFile = obj.projectfile;
%         if ~isempty(projFile)
%           savepath = fileparts(projFile);
%         else
%           savepath = pwd;
%         end
%       else
%         savepath = fileparts(lastFile);
%       end
%       
%       filterspec = fullfile(savepath,'*.mat');
%       [fname,pth] = feval(uifcn,filterspec,promptstr);
%       if isequal(fname,0)
%         fname = [];
%         success = false;
%       else
%         fname = fullfile(pth,fname);
%         success = true;
%         obj.(rawMeth)(fname);
%       end
%     end
     
  end
  
  methods % TrackRes
    
    function trackResInit(obj)
      obj.trkResIDs = cell(0,1);
      obj.trkRes = cell(obj.nmovies,obj.nview,0);
      obj.trkResGT = cell(obj.nmoviesGT,obj.nview,0);
      for i=1:numel(obj.trkResViz)
        delete(obj.trkResViz{i});
      end
      obj.trkResViz = cell(0,1);
    end
    
    function [tf,iTrkRes] = trackResFindID(obj,id)
      iTrkRes = find(strcmp(obj.trkResIDs,id));
      assert(isempty(iTrkRes) || isscalar(iTrkRes));
      tf = ~isempty(iTrkRes);      
    end
    
    function tv = trackResFindTV(obj,id) % throws
      % Return TrackingVisualizer object for id, or err
      [tf,iTR] = obj.trackResFindID(id);
      if ~tf
        error('Tracking results ''%s'' not found in project.',id);
      end
      tv = obj.trkResViz{iTR};
    end
    
    function iTrkRes = trackResEnsureID(obj,id)
      iTrkRes = find(strcmp(obj.trkResIDs,id));
      if isempty(iTrkRes)
        handleTagPfix = ['handletag_' id];
        tv = TrackingVisualizer(obj,handleTagPfix);
        tv.vizInit();
        
        fprintf(1,'Adding new tracking results set ''%s''.\n',id);
        obj.trkResIDs{end+1,1} = id;
        obj.trkRes(:,:,end+1) = {[]};
        obj.trkResGT(:,:,end+1) = {[]};
        obj.trkResViz{end+1,1} = tv;
        iTrkRes = numel(obj.trkResIDs);
      end
    end
    
    function trackResRmID(obj,id)
      iTR = find(strcmp(obj.trkResIDs,id));
      if isempty(iTR)
        error('Tracking results ID ''%s'' not found in project',id);
      end

      assert(isscalar(iTR));
      obj.trkResIDs(iTR,:) = [];
      obj.trkRes(:,:,iTR) = [];
      obj.trkResGT(:,:,iTR) = [];
      tv = obj.trkResViz(iTR);
      delete(tv);
      obj.trkResViz(iTR,:) = [];
    end
    
    function trackResAddCurrMov(obj,id,trkfiles)
      if ~obj.hasMovie
        error('No movie is open.');
      end
      mIdx = obj.currMovIdx;
      obj.trackResAdd(id,mIdx,trkfiles);
    end
    
    function trackResAdd(obj,id,mIdx,trkfiles)
      % id: trackRes ID
      % mIdx: [n] MovieIndex vector
      % trkfiles: [nxnview] cell of trkfile objects or fullpaths

      assert(isa(mIdx,'MovieIndex'));
      n = numel(mIdx);
      if ischar(trkfiles)
        trkfiles = cellstr(trkfiles);
      end
      assert(iscell(trkfiles));
      szassert(trkfiles,[n obj.nview]);
      
      iTR = obj.trackResEnsureID(id);
      [iMovs,gt] = mIdx.get();
      if gt
        TRFLD = 'trkResGT';
      else
        TRFLD = 'trkRes';
      end
      
      if iscellstr(trkfiles) || isstring(trkfiles)
        trkfileobjs = cellfun(@TrkFile.load,trkfiles,'uni',0);
      else
        assert(isa(trkfiles{1},'TrkFile'));
        trkfileobjs = trkfiles;
      end
      
      for iMov = iMovs(:)'
        v = obj.(TRFLD){iMov,1,iTR};
        if ~isempty(v)
          warningNoTrace('Overwriting existing tracking results for ''%s'' at movie index %d.',...
            id,iMov);
        end
      end
      obj.(TRFLD)(iMovs,:,iTR) = trkfileobjs;      
    end
    
    function hlpTrackResSetViz(obj,tvMeth,id,varargin)
      if isempty(id)
        cellfun(@(x)x.(tvMeth)(varargin{:}),obj.trkResViz);
      else
        tv = obj.trackResFindTV(id); % throws
        tv.(tvMeth)(varargin{:});
      end
    end
    
    function trackResSetHideViz(obj,id,tf)
      obj.hlpTrackResSetViz('setHideViz',id,tf);
    end
    
    function trackResSetHideTextLbls(obj,id,tf)
      obj.hlpTrackResSetViz('setHideTextLbls',id,tf);
    end    
    
    function trackResSetMarkerCosmetics(obj,id,varargin)
      obj.hlpTrackResSetViz('setMarkerCosmetics',id,varargin);
    end

    function trackResSetTextCosmetics(obj,id,varargin)
      obj.hlpTrackResSetViz('setTextCosmetics',id,varargin);
    end
  end  % methods
   
  %% Video
  methods    
    function [tfsucc,xy] = videoCenterOnCurrTargetPointHelp(obj)
      % get (x,y) for current movieCenterOnTargetIPt
      
      tfsucc = true;
      f = obj.currFrame;
      itgt = obj.currTarget;
      ipt = obj.movieCenterOnTargetIpt;
      
      s = obj.labelsCurrMovie;
      if ~isempty(s)
        [tf,p] = Labels.isLabeledFT(s,f,itgt);
        if tf
          xy = reshape(p,s.npts,2);
          xy = xy(ipt,:);
          if all(~isnan(xy))
            return
          end
        end
      end
      
      tracker = obj.tracker;
      if ~isempty(tracker)
        [tfhaspred,xy] = tracker.getTrackingResultsCurrFrm();
        if tfhaspred
          return
        end
      end
      
      s = obj.labels2CurrMovie;
      if ~isempty(s)
        [tf,p] = Labels.isLabeledFT(s,f,itgt);
        if tf
          xy = reshape(p,s.npts,2);
          xy = xy(ipt,:);
          if all(~isnan(xy))
            return
          end
        end
      end
      
      tfsucc = false;
      xy = [];
    end  % function    
    
    function unsetdrag(obj)
      obj.drag = false;
      obj.drag_pt = [];
    end    
  end  % methods
  
  %% Crop
  methods    
    function cropSetCropMode(obj, tf)
      if ~obj.hasMovie ,
        error('Can''t do that without a movie') ;
      end      
      if obj.hasTrx && tf
        error('User-specied cropping is unsupported for projects with trx.');
      end
      obj.pushBusyStatus('Switching crop mode...');      
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      obj.cropCheckCropSizeConsistency();
      if obj.cropIsCropMode,
        obj.syncCropInfoToCurrMov();
      end
      obj.cropIsCropMode = tf;
      obj.notify('cropIsCropModeChanged');
    end
    
    function syncCropInfoToCurrMov(obj)      
      iMov = obj.currMovie;
      if obj.gtIsGTMode,
        cropInfo = obj.movieFilesAllGTCropInfo{iMov};
      else
        cropInfo = obj.movieFilesAllCropInfo{iMov};
      end
      if ~isempty(cropInfo), 
        for iView=1:obj.nview,
            obj.movieReader(iView).setCropInfo(cropInfo(iView));
        end
      end
    end
    
    function [whDefault,whMaxAllowed] = cropComputeDfltWidthHeight(obj)
      % whDefault: [nview x 2]. cols: width, height
      %   See defns at top of CropInfo.m.
      % whMaxAllowed: [nview x 2]. maximum widthHeight that will fit in all
      %   images, per view
      
      movInfos = [obj.movieInfoAll; obj.movieInfoAllGT];
      nrall = cellfun(@(x)x.info.nr,movInfos); % [(nmov+nmovGT) x nview]
      ncall = cellfun(@(x)x.info.nc,movInfos); % etc
      nrmin = min(nrall,[],1); % [1xnview]
      ncmin = min(ncall,[],1); % [1xnview]
      widthMax = ncmin(:)-1; % col1..col<ncmin>. Minus 1 for posn vs roi 
      heightMax = nrmin(:)-1; % etc
      whMaxAllowed = [widthMax heightMax];
      whDefault = whMaxAllowed/2;
    end
    
    function wh = cropInitCropsAllMovies(obj)
      % Create/initialize default crops for all movies (including GT) in
      % all views
      %
      % wh: [nviewx2] widthHeight of default crop size used. See defns at 
      % top of CropInfo.m.      
      wh = obj.cropComputeDfltWidthHeight;
      obj.cropInitCropsGen(wh,'movieInfoAll','movieFilesAllCropInfo');
      obj.cropInitCropsGen(wh,'movieInfoAllGT','movieFilesAllGTCropInfo');
      obj.preProcNonstandardParamChanged();
      obj.notify('cropCropsChanged');
    end

    function cropInitCropsGen(obj,widthHeight,fldMIA,fldMFACI,varargin)
      % Init crops for certain movies
      % 
      % widthHeight: [nviewx2]
      
      iMov = myparse(varargin,...
        'iMov','__undef__'); % if supplied, indices into .(fldMIA), .(fldMFACI). latter will be initialized

      movInfoAll = obj.(fldMIA);
      [nmov,nvw] = size(movInfoAll);
      szassert(widthHeight,[nvw 2]);

      if strcmp(iMov,'__undef__')
        iMov = 1:nmov;
      else
        iMov = iMov(:)';
      end
      
      for ivw=1:nvw
        whview = widthHeight(ivw,:);
        for i=iMov
          ifo = movInfoAll{i,ivw}.info;
          xyCtr = (1+[ifo.nc ifo.nr])/2;
          posnCtrd = [xyCtr whview];
          obj.(fldMFACI){i}(ivw) = CropInfo.CropInfoCentered(posnCtrd);
        end
      end
    end
      
    function [tfhascrop,roi] = cropGetCropCurrMovie(obj)
      % Current crop per GT mode
      %
      %
      % tfhascrop: scalar logical.
      % roi: [nview x 4]. Applies only when tfhascrop==true; otherwise
      %   indeterminate. roi(ivw,:) is [xlo xhi ylo yhi]. 
      
      iMov = obj.currMovie;
      if iMov==0
        tfhascrop = false;
        roi = [];
      else
        [tfhascrop,roi] = obj.cropGetCropMovieIdx(iMov);
      end      
    end
    
    function [tfhascrop,roi] = cropGetCropMovieIdx(obj,iMov)
      % Current crop per GT mode
      %
      %
      % tfhascrop: scalar logical.
      % roi: [nview x 4]. Applies only when tfhascrop==true; otherwise
      %   indeterminate. roi(ivw,:) is [xlo xhi ylo yhi]. 
      
      PROPS = obj.gtGetSharedProps();
      cropInfo = obj.(PROPS.MFACI){iMov};
      if isempty(cropInfo)
        tfhascrop = false;
        roi = [];
      else
        tfhascrop = true;
        roi = cat(1,cropInfo.roi);
        szassert(roi,[obj.nview 4]);
      end
    end
    
    function cropSetNewRoiCurrMov(obj,iview,roi)
      % Set new crop/roi for current movie (GT-aware).
      %
      % If the crop size has changed, update crops sizes for all movies.
      % 
      % roi: [1x4]. [xlo xhi ylo yhi]. See defns at top of CropInfo.m
      
      iMov = obj.currMovie;
      if iMov==0
        error('No movie selected.');
      end
      
      obj.cropSetNewRoi(iMov,iview,roi);
    end
      
    function cropSetNewRoi(obj,iMov,iview,roi)
      % Set new crop/roi for input movie iMov (GT-aware).
      %
      % If the crop size has changed, update crops sizes for all movies.
      % 
      % roi: [1x4]. [xlo xhi ylo yhi]. See defns at top of CropInfo.m
      % KB added 20200504
      
      movIfo = obj.movieInfoAllGTaware{iMov,iview}.info;
      imnc = movIfo.nc;
      imnr = movIfo.nr;
      tfproper = CropInfo.roiIsProper(roi,imnc,imnr);
      if ~tfproper
        error('ROI extends outside of video. Video image is %d x %d.\n',...
          imnc,imnr);
      end
      
      if ~obj.cropProjHasCrops
        obj.cropInitCropsAllMovies();
        fprintf(1,'Default crop initialized for all movies.\n');
      end
      
      PROPS = obj.gtGetSharedProps();
      roi0 = obj.(PROPS.MFACI){iMov}(iview).roi;
      posn0 = CropInfo.roi2RectPos(roi0);
      posn = CropInfo.roi2RectPos(roi);
      widthHeight0 = posn0(3:4);
      widthHeight = posn(3:4);
      tfSzChanged = ~isequal(widthHeight,widthHeight0);
      tfProceedSet = true;
      if tfSzChanged
        tfOKSz = obj.cropCheckValidCropSize(iview,widthHeight);
        if tfOKSz 
          obj.cropSetSizeAllCrops(iview,widthHeight);
          %warningNoTrace('Crop sizes in all movies altered.');
        else
          warningNoTrace('New roi/crop size is too big for one or more movies in project. ROI not set.');
          tfProceedSet = false;
        end
      end
      if tfProceedSet
        obj.(PROPS.MFACI){iMov}(iview).roi = roi;
      end
      
      % KB 20200113: changing roi for movies with labels will require
      % preproc data to be cleaned out. For now, we are clearing out
      % trackers all together in this case. 
      
      if ~obj.gtIsGTMode && obj.labelPosMovieHasLabels(iMov),
        obj.preProcNonstandardParamChanged();
      end

     
%       if ~obj.gtIsGTMode && obj.labelPosMovieHasLabels(iMov),
%         % if this movie has labels, retraining might be necessary
%         % set timestamp for all labels in this movie to now
%         obj.reportLabelChange();
%       end
%       
%       % actually in some codepaths nothing changed, but shouldn't hurt
%       if tfSzChanged,
%         obj.preProcNonstandardParamChanged();
%       end
      obj.notify('cropCropsChanged'); 
    end
    
    function reportLabelChange(obj)      
      obj.labeledposNeedsSave = true;
      obj.lastLabelChangeTS = now;
    end
    
    function tfOKSz = cropCheckValidCropSize(obj,iview,widthHeight)
      [~,whMaxAllowed] = obj.cropComputeDfltWidthHeight();
      wh = whMaxAllowed(iview,:);
      tfOKSz = all(widthHeight<=wh);
    end
    
    function cropSetSizeAllCrops(obj,iview,widthHeight)
      % Sets crop size for all movies (GT included) for iview. Keeps crops
      % centered as possible
      
      obj.cropSetSizeAllCropsHlp(iview,widthHeight,'movieInfoAll',...
        'movieFilesAllCropInfo');
      obj.cropSetSizeAllCropsHlp(iview,widthHeight,'movieInfoAllGT',...
        'movieFilesAllGTCropInfo');
      obj.preProcNonstandardParamChanged();
      obj.notify('cropCropsChanged'); 
    end

    function cropSetSizeAllCropsHlp(obj,iview,widthHeight,fldMIA,fldMFACI)
      movInfoAll = obj.(fldMIA);
      cropInfos = obj.(fldMFACI);
      for i=1:numel(cropInfos)
        posn = CropInfo.roi2RectPos(cropInfos{i}(iview).roi);
        posn = CropInfo.rectPosResize(posn,widthHeight);
        roi = CropInfo.rectPos2roi(posn);
        
        ifo = movInfoAll{i,iview}.info;
        [roi,tfchanged] = CropInfo.roiSmartFit(roi,ifo.nc,ifo.nr);
        if tfchanged
          warningNoTrace('ROI center moved to stay within movie frame.');
        end
        obj.(fldMFACI){i}(iview).roi = roi;
      end
    end
    
    function cropClearAllCrops(obj)
      % Clear crops for all movies/views, including GT.
      %
      % Note, this is different than cropInitCropsAllMovies. That method
      % creates/sets default crops. This method obliterates all crop infos
      % so that .cropProjHasCrops returns false;
      
      if obj.cropProjHasCrops
        obj.preProcNonstandardParamChanged();
      end
      obj.movieFilesAllCropInfo(:) = {CropInfo.empty(0,0)};
      obj.movieFilesAllGTCropInfo(:) = {CropInfo.empty(0,0)};
      obj.notify('cropCropsChanged'); 
    end
    
    function wh = cropGetCurrentCropWidthHeightOrDefault(obj)
      % If obj.cropProjHasCropInfo is true, get current crop width/height.
      % Otherwise, get default widthHeight.
      %
      % Assumes that the proj has at least one regular movie.
      %
      % wh: [nview x 2]
      
      ci = obj.movieFilesAllCropInfo{1}; 
      if ~isempty(ci)
        roi = cat(1,ci.roi);
        posn = CropInfo.roi2RectPos(roi);
        wh = posn(:,3:4);
      else
        wh = obj.cropComputeDfltWidthHeight();
      end
      szassert(wh,[obj.nview 2]);
    end
    
    function cropCheckCropSizeConsistency(obj)
      % Crop size integrity check. Like a big assert.
      
      if obj.cropProjHasCrops
        cInfoAll = [obj.movieFilesAllCropInfo; obj.movieFilesAllGTCropInfo];
        roisAll = cellfun(@(x)cat(1,x.roi),cInfoAll,'uni',0);
        posnAll = cellfun(@CropInfo.roi2RectPos,roisAll,'uni',0); % posnAll{i} is [nview x 4]
        whAll = cellfun(@(x)x(:,3:4),posnAll,'uni',0);
        whAll = cat(3,whAll{:});
        nCIAll = numel(cInfoAll);
        szassert(whAll,[obj.nview 2 nCIAll]); % ivw, w/h, cropInfo
        assert(isequal(repmat(whAll(:,:,1),1,1,nCIAll),whAll),...
          'Unexpected inconsistency crop sizes.')
      end      
    end
    
%     function rois = cropGetAllRois(obj)
%       % Get all rois per current GT mode
%       %
%       % rois: [nmovGTaware x 4 x nview]
%       
%       if ~obj.cropProjHasCrops
%         error('Project does not have crops defined.');
%       end
%       
%       cInfoAll = obj.movieFilesAllCropInfoGTaware;
%       rois = cellfun(@(x)cat(1,x.roi),cInfoAll,'uni',0);
%       rois = cat(3,rois{:}); % nview, {xlo/xhi/ylo/yhi}, nmov
%       rois = permute(rois,[3 2 1]);    
%     end 
    
%     function hFig = cropMontage(obj,varargin)
%       % Create crop montages for all views. Per current GT state.
%       %
%       % hFig: [nfig] figure handles
%       
%       [type,imov,nr,nc,plotlabelcolor,figargs] = myparse(varargin,...
%         'type','wide',... either 'wide' or 'cropped'. wide shows rois in context of full im. 
%         'imov',[],... % show crops for these movs. defaults to 1:nmoviesGTaware
%         'nr',9,... % number of rows in montage
%         'nc',10,... % etc
%         'plotlabelcolor',[1 1 0],...
%         'figargs',{'WindowStyle','docked'});
%       
%       if obj.hasTrx
%         error('Unsupported for projects with trx.');
%       end
%       if ~obj.cropProjHasCrops
%         error('Project does not have crops defined.');
%       end
%       
%       if isempty(imov)
%         imov = 1:obj.nmoviesGTaware;
%       end
%       
%       switch lower(type)
%         case 'wide', tfWide = true;
%         case 'cropped', tfWide = false;
%         otherwise, assert(false);
%       end
%       
%       % get MFTable to pull first frame of each mov
%       mov = obj.movieFilesAllFullGTaware(imov,:);
%       nmov = size(mov,1);
%       if nmov==0
%         error('No movies.');
%       end
%       frm = ones(nmov,1);
%       iTgt = ones(nmov,1);
%       tblMFT = table(mov,frm,iTgt);
%       wbObj = WaitBarWithCancel('Montage');
%       oc = onCleanup(@()delete(wbObj));      
%       I1 = CPRData.getFrames(tblMFT,...
%         'movieInvert',obj.movieInvert,...
%         'wbObj',wbObj);
%       I1 = cellfun(@DataAugMontage.convertIm2Double,I1,'uni',0);
% 
%       roisAll = obj.cropGetAllRois; 
%       roisAll = roisAll(imov,:,:);
%       
%       nvw = obj.nview;
%       if ~tfWide
%         for iimov=1:nmov
%           for ivw=1:nvw
%             roi = roisAll(iimov,:,ivw);
%             I1{iimov,ivw} = I1{iimov,ivw}(roi(3):roi(4),roi(1):roi(2));
%           end
%         end
%       end
% 
%       nplotperbatch = nr*nc;
%       nbatch = ceil(nmov/nplotperbatch);
%       szassert(I1,[nmov nvw]);
%       szassert(roisAll,[nmov 4 nvw]);
%       hFig = gobjects(0,1);
%       for ivw=1:nvw
%         roi1 = roisAll(1,:,ivw);
%         pos1 = CropInfo.roi2RectPos(roi1);
%         wh = pos1(3:4)+1;
%         
%         if tfWide
%           imsz = cellfun(@size,I1(:,ivw),'uni',0);
%           imsz = cat(1,imsz{:}); % will err if some ims are color etc
%           imszUn = unique(imsz,'rows');
%           tfImsHeterogeneousSz = size(imszUn,1)>1;
%         end
%         
%         for ibatch=1:nbatch
%           iimovs = (1:nplotperbatch) + (ibatch-1)*nplotperbatch;
%           iimovs(iimovs>nmov) = [];
%           figstr = sprintf('movs %d->%d. view %d.',...
%             iimovs(1),iimovs(end),ivw);
%           titlestr = sprintf('movs %d->%d. view %d. [w h]: %s',...
%             iimovs(1),iimovs(end),ivw,mat2str(wh));
%           
%           hFig(end+1,1) = figure(figargs{:}); %#ok<AGROW>
%           hFig(end).Name = figstr;
%           
%           if tfWide
%             Shape.montage(I1(:,ivw),nan(nmov,2),...
%               'fig',hFig(end),...
%               'nr',nr,'nc',nc,'idxs',iimovs,...
%               'rois',roisAll(:,:,ivw),...
%               'imsHeterogeneousSz',tfImsHeterogeneousSz,...
%               'framelbls',arrayfun(@num2str,imov(iimovs),'uni',0),...
%               'framelblscolor',plotlabelcolor,...
%               'titlestr',titlestr);
%           else
%             Shape.montage(I1(:,ivw),nan(nmov,2),...
%               'fig',hFig(end),...
%               'nr',nr,'nc',nc,'idxs',iimovs,...
%               'framelbls',arrayfun(@num2str,imov(iimovs),'uni',0),...
%               'framelblscolor',plotlabelcolor,...
%               'titlestr',titlestr);
%           end
%         end
%       end        
%     end
            
  end  % methods
 
  
  %% Navigation
  methods
    
    function navPrefsUI(obj)
      NavPrefs(obj);
    end
    
    function setFrameGUI(obj,frm,varargin)
      % Set movie frame, maintaining current movie/target.
      %
      % CTRL-C note: This is fairly ctrl-c safe; a ctrl-c break may leave
      % obj state a little askew but it should be cosmetic and another
      % (full/completed) setFrameGUI() call should fix things up. We could
      % prob make it even more Ctrl-C safe with onCleanup-plus-a-flag.
      
      debugtiming = false;
      if debugtiming,
        setframetic = tic;
        starttime = setframetic;   
      end
      [tfforcereadmovie,tfforcelabelupdate,updateLabels,updateTables,...
       updateTrajs,changeTgtsIfNec] = ...
        myparse(varargin,...
                'tfforcereadmovie',false,...
                'tfforcelabelupdate',false,...
                'updateLabels',true,...
                'updateTables',true,...
                'updateTrajs',true,...
                'changeTgtsIfNec',false... % if true, will alter the current target if it is not live in frm
                );
      
      if debugtiming,
        fprintf('setFrame %d, parse inputs took %f seconds\n',frm,toc(setframetic)); setframetic = tic;
      end
      
      if obj.hasTrx
        assert(~obj.isMultiView,'MultiView labeling not supported with trx.');
        frm2trxThisFrm = obj.frm2trx(frm,:);
        iTgt = obj.currTarget;
        if ~frm2trxThisFrm(1,iTgt)
          if changeTgtsIfNec
            iTgtsLive = find(frm2trxThisFrm);
            if isempty(iTgtsLive)
              error('Labeler:target','No targets live in frame %d.',frm);              
            else
              iTgtsLiveDist = abs(iTgtsLive-iTgt);
              itmp = argmin(iTgtsLiveDist);
              iTgtNew = iTgtsLive(itmp);
              warningNoTrace('Target %d is not live in frame %d. Changing to target %d.\n',...
                             iTgt,frm,iTgtNew);
              obj.setFrameAndTargetGUI(frm,iTgtNew);
              return;
            end
          else
            error('Labeler:target','Target %d not live in frame %d.',...
                  iTgt,frm);
          end
        end
      elseif obj.maIsMA
        % do nothing
      end
      
      if debugtiming,
        fprintf('setFrame %d, trx stuff took %f seconds\n',frm,toc(setframetic)); setframetic = tic;
      end
      
      % Remainder nearly identical to setFrameAndTarget()
      try
        obj.setCurrentAndPreviousFrameData_(frm,tfforcereadmovie);
      catch ME
        warning(ME.identifier,'Could not set previous frame:\n%s',getReport(ME));
      end
      
      if debugtiming,
        fprintf('setFrame %d, setcurrprevframe took %f seconds\n',frm,toc(setframetic)); setframetic = tic;
      end
      
      obj.notify('updateTargetCentrationAndZoom') ;
      
      if debugtiming,
        fprintf('setFrame %d, center and rotate took %f seconds\n',frm,toc(setframetic)); setframetic = tic;
      end
      
      if updateLabels
        obj.labelsUpdateNewFrame(tfforcelabelupdate);
      end
      
      if debugtiming,
        fprintf('setFrame %d, updatelabels took %f seconds\n',frm,toc(setframetic)); setframetic = tic;
      end
      
      if updateTables
        obj.notify('updateTrxTable');
      end
      
      if updateTrajs
        obj.notify('updateTrxSetShowFalse') ;
      end
      
      if debugtiming,
        fprintf('setFrame %d, update showtrx took %f seconds\n',frm,toc(setframetic));
      end

      if debugtiming,
        fprintf('setFrame to %d took %f seconds\n',frm,toc(starttime));
      end
      
    end  % function setFrameGUI
    
%     function setTargetID(obj,tgtID)
%       % Set target ID, maintaining current movie/frame.
%       
%       iTgt = obj.trxIdPlusPlus2Idx(tgtID+1);
%       assert(~isnan(iTgt),'Invalid target ID: %d.');
%       obj.setTarget(iTgt);
%     end

    
    function setTarget(obj,iTgt,varargin)
      % Set target index, maintaining current movie/frameframe.
      % iTgt: INDEX into obj.trx
      
      obj.pushBusyStatus(sprintf('Switching to target %d...',iTgt));
      oc = onCleanup(@()(obj.popBusyStatus())) ;

      if obj.hasTrx
        frm = obj.currFrame;
        if ~obj.frm2trx(frm,iTgt)
          error('Labeler:target',...
            'Target idx %d is not live at current frame (%d).',iTgt,frm);
        end
      end
      
      prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      if obj.hasTrx || obj.maIsMA
        obj.labelsUpdateNewTarget(prevTarget);
        obj.notify('updateTargetCentrationAndZoom') ;
      end
%       obj.updateCurrSusp();
      if obj.hasTrx
        obj.updateShowTrx();
      end
    end
    
    function setTargetMA(obj,iTgt)
      % "raw". maybe shldnt be a sep meth
      obj.currTarget = iTgt;
    end
        
    function setFrameAndTargetGUI(obj,frm,iTgt,tfforce)
      % Set to new frame and target for current movie.
      % Prefer setFrameGUI() or setTarget() if possible to
      % provide better continuity wrt labeling etc.
     
%       validateattributes(iTgt,{'numeric'},...
%         {'positive' 'integer' '<=' obj.nTargets});

      % changed this to default to NOT forcing
      if nargin < 4,
        tfforce = false;
      end

      if ~obj.isinit && obj.hasTrx && ~obj.frm2trx(frm,iTgt)
        error('Labeler:target',...
          'Target idx %d is not live at current frame (%d).',iTgt,frm);
      end

      try
        obj.setCurrentAndPreviousFrameData_(frm,tfforce);
      catch ME
        warning(ME.identifier,'Could not set previous frame: %s', ME.message);
      end

      if isnan(iTgt)
        return;
      end

      prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      
      obj.notify('updateTargetCentrationAndZoom') ;
      
      if ~obj.isinit
        obj.labelsUpdateNewFrameAndTarget(obj.prevFrame,prevTarget);
        obj.notify('updateTrxTable');
        obj.updateShowTrx();  % All this does is send a notification, and only in some cases
      end
    end  % function setFrameAndTargetGUI
  end

  methods
%     function frameUpNextLbled(obj,tfback,varargin)
%       % call obj.setFrameGUI() on next labeled frame. 
%       % 
%       % tfback: optional. if true, seek backwards.
%             
%       if ~obj.hasMovie || obj.currMovie==0
%         return;
%       end
%       
%       lpos = myparse(varargin,...
%         'lpos','__UNSET__'); % optional, provide "big" lpos array to use instead of .labeledposCurrMovie
%       
%       if tfback
%         df = -1;
%       else
%         df = 1;
%       end
%       
%       if strcmp(lpos,'__UNSET__')
%         lpos = obj.labeledposCurrMovie;
%       elseif isempty(lpos)
%         % edge case
%         return;
%       else
%         szassert(lpos,[obj.nLabelPoints 2 obj.nframes obj.nTargets]);
%       end
%         
%       [tffound,f] = Labeler.seekBigLpos(lpos,obj.currFrame,df,...
%         obj.currTarget);
%       if tffound
%         obj.setFrameProtected(f);
%       end
%     end
    
    function [x,y,th] = currentTargetLoc(obj,varargin)
      % Return current target loc, or movie center if no target
      
      nowarn = myparse(varargin,...
        'nowarn',false ... % If true, don't warn about incompatible .currFrame/.currTrx. 
          ... % The incompatibility really shouldn't happen but it's an edge case
          ... % and not worth the trouble to fix for now.
        );
      
      assert(~obj.isMultiView,'Not supported for MultiView.');
      
      if obj.hasTrx
        cfrm = obj.currFrame;
        ctrx = obj.currTrx;


        if cfrm < ctrx.firstframe || cfrm > ctrx.endframe
          if ~nowarn
            warningNoTrace('Labeler:target',...
              'No track for current target at frame %d.',cfrm);
          end
          movroictr = obj.movieroictr; % [1x2]
          x = round(movroictr(1));
          y = round(movroictr(2));
          th = 0;
        else
          i = cfrm - ctrx.firstframe + 1;
          x = ctrx.x(i);
          y = ctrx.y(i);
          th = ctrx.theta(i);
        end
      elseif obj.maIsMA
        cfrm = obj.currFrame;
        imov = obj.currMovie;
        itgt = obj.currTarget;
        if itgt==0
          x=NaN; y=NaN; th=NaN;
          return 
        end
        lpos = obj.labelsGTaware;
        s = lpos{imov};        
        [tf,p] = Labels.isLabeledFT(s,cfrm,itgt);
        if tf 
          p = reshape(p,obj.nLabelPoints,2);
          pmu = mean(p,1);
          x = pmu(1);
          y = pmu(2);        
          
          if ~isempty(obj.skelHead)
            phead = p(obj.skelHead,:);
            if ~isempty(obj.skelTail)
              ptail = p(obj.skelTail,:);
            else
              ptail = pmu;
              warningNoTrace('No tail point defined; using centroid.');
            end
            vmuhead = phead-ptail;
            th = atan2(vmuhead(2),vmuhead(1));
          else
            th = 0;
          end
        else
          movroictr = obj.movieroictr; % [1x2]
          x = round(movroictr(1));
          y = round(movroictr(2));
          th = 0;
        end
      else
        movroictr = obj.movieroictr; % [1x2]
        x = round(movroictr(1));
        y = round(movroictr(2));
        th = 0;
      end
    end
    
    function [x,y,th] = targetLoc(obj,iMov,iTgt,frm)
      % Added by KB 20181010
      % Return current target center and orientation for input movie, target, and frame
      
      assert(obj.hasTrx,'This should not be called when videos do not have trajectories');
      
      if iMov == obj.currMovie && iTgt == obj.currTarget,
        ctrx = obj.currTrx;
      else
        trxfname = obj.trxFilesAllFullGTaware{iMov,1};
        movIfo = obj.movieInfoAllGTaware{iMov};
        [s.trx,s.frm2trx] = obj.getTrx(trxfname,movIfo.nframes);
        ctrx = s.trx(iTgt);
      end
      cfrm = frm;

      assert(cfrm >= ctrx.firstframe && cfrm <= ctrx.endframe);
      i = cfrm - ctrx.firstframe + 1;
      x = double(ctrx.x(i));
      y = double(ctrx.y(i));
      th = double(ctrx.theta(i));
    end

    function result = get.selectedFrames(obj)
      result = find(obj.infoTimelineModel.isSelectedFromFrameIndex) ;
    end

    % function set.selectedFrames(obj, newValue)
    %   if isempty(newValue)
    %     obj.selectedFrames_ = [] ;        
    %   elseif ~obj.hasMovie
    %     error('Labeler:noMovie',...
    %           'Cannot set selected frames when no movie is loaded.');
    %   else
    %     validateattributes(newValue,{'numeric'},{'integer' 'vector' '>=' 1 '<=' obj.nframes}) ;
    %     obj.selectedFrames_ = newValue ;  
    %   end
    %   obj.notify('updateTimelineAndFriends');
    % end
    
    
    function updateMovieFilesAllHaveLbls(obj)
      fcnNumLbledRows = @Labels.numLbls;
      obj.movieFilesAllHaveLbls = cellfun(fcnNumLbledRows,obj.labels);
      obj.movieFilesAllGTHaveLbls = cellfun(fcnNumLbledRows,obj.labelsGT);
    end

    function setCurrentAndPreviousFrameData_(obj, frameIndex, tfforce)
      % helper for setFrameGUI(), setFrameAndTargetGUI()

      currFrameOriginal = obj.currFrame ;
      % imcurr = controller.image_curr;
      if isempty(obj.currIm) || isempty(obj.currIm{1})
        currIm1Original = 0 ;
        currImRoi1Original = [ 1 1 1 1 ] ;
      else
        currIm1Original = obj.currIm{1} ;
        currImRoi1Original = obj.currImRoi{1} ;
      end

      frameCount = min([obj.movieReader(:).nframes]);
      if frameIndex > frameCount
        frameIndex = frameCount ;
      end

      if obj.currFrame~=frameIndex || tfforce
        tfCropMode = obj.cropIsCropMode;        
        for iView=1:obj.nview
          if tfCropMode
            [obj.currIm{iView},~,obj.currImRoi{iView}] = ...
              obj.movieReader(iView).readframe(frameIndex,...
                                               'doBGsub',obj.movieViewBGsubbed,'docrop',false);                   
          else
            [obj.currIm{iView},~,obj.currImRoi{iView}] = ...
              obj.movieReader(iView).readframe(frameIndex,...
                                               'doBGsub',obj.movieViewBGsubbed,'docrop',true);                  
          end          
        end
        obj.notify('updateCurrImagesAllViews') ;

        obj.currFrame = frameIndex;
        
        if ~isempty(obj.tracker),
          obj.tracker.newLabelerFrame();
        end
      end
      
      obj.prevFrame = currFrameOriginal;
      currIm1Nr = size(obj.currIm{1},1);
      currIm1Nc = size(obj.currIm{1},2);
      if ~isequal([size(currIm1Original,1) size(currIm1Original,2)],...
                  [currIm1Nr currIm1Nc])
        % In this scenario we do not use currIm1Orig b/c axes_prev and 
        % axes_curr are linked and that can force the axes into 'manual'
        % XLimMode and so on. Generally it is disruptive to view-handling.
        % Ideally maybe we would prefer to catch/handle this in view code,
        % but there is no convenient hook between setting the image CData
        % and display.
        %
        % Maybe more importantly, the only time the sizes will disagree are 
        % in edge cases eg when a project is loaded or changed. In this 
        % case currIm1Orig is not going to represent anything meaningful 
        % anyway.
        % obj.prevIm = struct('CData',zeros(currIm1Nr,currIm1Nc),...
        %                     'XData',1:currIm1Nc,'YData',1:currIm1Nr);
        obj.prevIm = zeros(currIm1Nr,currIm1Nc) ;
        obj.prevImRoi = [ 1 currIm1Nc 1 currIm1Nr ] ;
      else
        obj.prevIm = currIm1Original ;
        obj.prevImRoi = currImRoi1Original ;
      end
      % obj.prevAxesImFrmUpdate(tfforce) ;      
      obj.notify('updatePrevAxesImage') ;
    end  % function
  end
  
  %% PrevAxes
  methods

    function result = get.prevAxesMode(obj)
      result = obj.prevAxesMode_;
    end  % function

    function result = get.prevAxesModeTargetSpec(obj)
      result = obj.prevAxesModeTargetSpec_;
    end  % function

    function isvalid = isPrevAxesModeInfoSet(obj)
      % Returns true iff obj.prevAxesModeTargetSpec has valid identity fields set.
      isvalid = obj.prevAxesModeTargetSpec.isTargetSet() ;
    end
    
    function spec = computePrevAxesTargetSpec_(obj)
      % Compute a fresh, fully-populated PrevAxesTargetSpec from the current
      % Labeler state.  Reads identity fields and dxlim/dylim from
      % obj.prevAxesModeTargetSpec_.  Does not mutate obj.
      existingSpec = obj.prevAxesModeTargetSpec_;
      spec = PrevAxesTargetSpec();
      spec.iMov = existingSpec.iMov;
      spec.frm = existingSpec.frm;
      spec.iTgt = existingSpec.iTgt;
      spec.gtmode = existingSpec.gtmode;
      if ~spec.isTargetSet() || ~obj.hasMovie
        return
      end

      % Read the image and rotation info from the movie
      [im, isrotated, xdata, ydata, A, tform] = ...
        obj.readTargetImageFromMovie(spec.iMov, spec.frm, spec.iTgt, 1, obj.prevAxesYDir_);
      spec.im = im;
      spec.isrotated = isrotated;
      spec.xdata = xdata;
      spec.ydata = ydata;
      spec.A = A;
      spec.tform = tform;

      % Compute xlim, ylim, and prevAxesProps from the label positions
      axesCurrProps = obj.currAxesProps_;
      prevAxesSize = obj.prevAxesSizeInPixels_;
      viewi = 1;
      ptidx = (obj.labeledposIPt2View == viewi);
      [~, poscurr, ~] = ...
        obj.labelPosIsLabeled(spec.frm, ...
                              spec.iTgt, ...
                              'iMov', spec.iMov, ...
                              'gtmode', spec.gtmode);
      poscurr = poscurr(ptidx, :);
      if obj.hasTrx
        poscurr = [poscurr, ones(size(poscurr, 1), 1)] * A;
        poscurr = poscurr(:, 1:2);
      end

      minpos = min(poscurr, [], 1);
      maxpos = max(poscurr, [], 1);
      centerpos = (minpos + maxpos) / 2;
      borderfrac = .5;
      r = max(1, (maxpos - minpos) / 2 * (1 + borderfrac));
      xlim = centerpos(1) + [-1 1] * r(1);
      ylim = centerpos(2) + [-1 1] * r(2);

      % Adjust aspect ratio to match the prev-axes widget
      axw = prevAxesSize(1);
      axh = prevAxesSize(2);
      axszratio = axw / axh;
      limratio = diff(xlim) / diff(ylim);
      if axszratio > limratio
        extendratio = axszratio / limratio;
        xlim = centerpos(1) + [-1 1] * r(1) * extendratio;
      elseif axszratio < limratio
        extendratio = limratio / axszratio;
        ylim = centerpos(2) + [-1 1] * r(2) * extendratio;
      end

      % Apply any existing pan offsets from the existing spec
      if ~isempty(existingSpec.dxlim)
        dxlim = existingSpec.dxlim;
        dylim = existingSpec.dylim;
        xlim0 = xlim;
        ylim0 = ylim;
        xlim = xlim + dxlim;
        ylim = ylim + dylim;
        % make sure all parts are visible
        if minpos(1) < xlim(1) || minpos(2) < ylim(1) || maxpos(1) > xlim(2) || maxpos(2) < ylim(2)
          dxlim = [0 0];
          dylim = [0 0];
          xlim = xlim0;
          ylim = ylim0;
        else
          % xlim/ylim and dxlim/dylim are good
        end
      else
        dxlim = [0 0];
        dylim = [0 0];
      end
      xlim = fixLim(xlim);
      ylim = fixLim(ylim);

      % Package up prevAxesProps
      prevAxesProps = struct('XLim', xlim, ...
                             'YLim', ylim, ...
                             'XDir', axesCurrProps.XDir, ...
                             'YDir', axesCurrProps.YDir, ...
                             'CameraViewAngleMode', 'auto');

      % Set the remaining spec fields
      spec = setprop(spec, 'xlim', xlim, 'ylim', ylim, 'dxlim', dxlim, 'dylim', dylim, 'prevAxesProps', prevAxesProps);
    end  % function
    
    function [im,isrotated,xdata,ydata,A,tform] = readTargetImageFromMovie(obj,mov,frm,tgt,viewi,prevAxesYDir)
      % Get the image (and associated data) for the reference image pane.  Does not
      % mutate obj.
      if ~exist('prevAxesYDir', 'var')
        prevAxesYDir = 'reverse';
      end
      isrotated = false;
      % if (int32(mov) == obj.currMovie) && (gtmode==obj.gtIsGTMode)
      %   mr = obj.movieReader(viewi) ;
      % else
      mr = MovieReader();
      mr.openForLabeler(obj,MovieIndex(mov),viewi);
      % end
      [im,~,imRoi] = ...
        mr.readframe(frm,...
                     'doBGsub',obj.movieViewBGsubbed,...
                     'docrop',~obj.cropIsCropMode);

      % to do: figure out [~,~what to do when there are multiple views
      if ~obj.hasTrx,
        xdata = imRoi(1:2);
        ydata = imRoi(3:4);
        A = [];
        tform = [];
      else
        if strcmpi(prevAxesYDir,'normal'),
          pi2sign = -1;
        else
          pi2sign = 1;
        end
      
        [x,y,th] = obj.targetLoc(abs(mov),tgt,frm);
        if isnan(th),
          th = -pi/2;
        end
        A = [1,0,0;0,1,0;-x,-y,1]*[cos(th+pi2sign*pi/2),-sin(th+pi2sign*pi/2),0;sin(th+pi2sign*pi/2),cos(th+pi2sign*pi/2),0;0,0,1];
        tform = maketform('affine',A);  %#ok<MTFA1> 
        [im,xdata,ydata] = imtransform(im,tform,'bicubic');  %#ok<DIMTRNS> 
        isrotated = true;
      end
    end  % function

    function [allims,allpos] = cropTargetImageFromMovie(obj,mov,frm,iTgt,p)
      % [allims,allpos] = cropTargetImageFromMovie(obj,mov,frm,iTgt,p)
      % called by LabelerController.createGTResultFigures_ to crop out an
      % image around label p. 
      % allims are the cropped images around p and allpos are the labels in
      % the coordinate system of the cropped images

      nviews = obj.nview;
      npts = numel(p)/2;
      nphyspt = npts/nviews;
      p = reshape(p,npts,2);
      allims = cell(1,nviews);
      allpos = zeros([nphyspt,2,nviews]);
      for view = 1:nviews
        curl = p( ((view-1)*nphyspt+1):view*nphyspt,:);
        [im,isrotated,xdata,ydata,A] = obj.readTargetImageFromMovie(mov,frm,iTgt,view);
        if isrotated
          curl = [curl,ones(nphyspt,1)]*A;
          curl = curl(:,1:2);
          curl(:,1) = curl(:,1)-xdata(1)+1;
          curl(:,2) = curl(:,2)-ydata(1)+1;
        end
        minpos = min(curl,[],1);
        maxpos = max(curl,[],1);
        centerpos = (minpos+maxpos)/2;
        % border defined by borderfrac
        r = max(1,(maxpos-minpos));
        xlim = round(centerpos(1)+[-1,1]*r(1));
        ylim = round(centerpos(2)+[-1,1]*r(2));
        xlim = min(size(im,2),max(1,xlim));
        ylim = min(size(im,1),max(1,ylim));
        im = im(ylim(1):ylim(2),xlim(1):xlim(2),:);
        curl(:,1) = curl(:,1)-xlim(1)+1;
        curl(:,2) = curl(:,2)-ylim(1)+1;
        allpos(:,:,view) = curl;
        allims{view} = im;
      end  % for

    end
      

  end  % methods

  methods  % (Access=private)
    function prevAxesSetFrozenLabels_(obj, spec)
      % Set prev-axes label positions for FROZEN mode from a PrevAxesTargetSpec.

      persistent tfWarningThrownAlready

      [~, lpos0, lpostag2] = obj.labelPosIsLabeled(spec.frm, spec.iTgt, 'iMov', spec.iMov, 'gtmode', spec.gtmode);
      if spec.isrotated
        lpos2 = [lpos0, ones(size(lpos0, 1), 1)] * spec.A;
        lpos2 = lpos2(:, 1:2);
      else
        lpos2 = lpos0;
      end

      ipts = 1:obj.nPhysPoints;
      txtOffset = obj.labelPointsPlotInfo.TextOffset;
      lpos3 = apt.patch_lpos(lpos2);
      assignLabelCoordsHandlingOcclusionBangBang(obj.lblPrev_ptsH(ipts), ...
                                                 obj.lblPrev_ptsTxtH(ipts), ...
                                                 lpos3(ipts, :), ...
                                                 txtOffset);
      if any(lpostag2(ipts))
        if isempty(tfWarningThrownAlready)
          warningNoTrace('Labeler:labelsPrev', ...
                         'Label tags in previous frame not visualized.');
          tfWarningThrownAlready = true;
        end
      end
    end  % function

    function prevAxesSetLastseenLabels_(obj, iMov, frm, iTgt)
      % Set prev-axes label positions for LASTSEEN mode.

      persistent tfWarningThrownAlready

      if isempty(frm)
        lpos2 = nan(obj.nLabelPoints, 2);
        lpostag2 = false(obj.nLabelPoints, 1);
      else
        [~, lpos2, lpostag2] = obj.labelPosIsLabeled(frm, iTgt, 'iMov', iMov);
      end

      ipts = 1:obj.nPhysPoints;
      txtOffset = obj.labelPointsPlotInfo.TextOffset;
      lpos3 = apt.patch_lpos(lpos2);
      assignLabelCoordsHandlingOcclusionBangBang(obj.lblPrev_ptsH(ipts), ...
                                                 obj.lblPrev_ptsTxtH(ipts), ...
                                                 lpos3(ipts, :), ...
                                                 txtOffset);
      if any(lpostag2(ipts))
        if isempty(tfWarningThrownAlready)
          warningNoTrace('Labeler:labelsPrev', ...
                         'Label tags in previous frame not visualized.');
          tfWarningThrownAlready = true;
        end
      end
    end  % function
  end  % methods
  
  %% Labels2/OtherTarget labels
  
  % Labels2 notes.
  % We are using TrkFiles as the contents of .labels2{iMov}. In contrast
  % to .labels{iMov}, which are are sparse user annotations, imported
  % tracking are of the tracklet type (segments of contiguous/dense results
  % etc).
  %
  % Per the TrackletTest performance test, either the tracklet or
  % dense/full TrkFile format can perform better for eg accessing "all 
  % tracking results for given frame (across all tgts)", depending on the 
  % sparsity of tracklet data. For essentially dense data like AR/Flybub, 
  % the  array/full format performs a little quicker. For larger numbers of 
  % sparse tracklets, tracklets perform better. Unless the sparsity reaches 
  % extreme levels (at might be reached for user annotations), the "Labels" 
  % format never performs well.
  %
  % For now the TrkFiles in .labels2 are always in tracklet format. One can
  % imagine switching to dense as a performance optimization.
   
  
  methods
    
% AL 201806 no callers atm
%     function labels2BulkSet(obj,lpos)
%       assert(numel(lpos)==numel(obj.labeledpos2));
%       for i=1:numel(lpos)
%         assert(isequal(size(lpos{i}),size(obj.labeledpos2{i})))
%         obj.labeledpos2{i} = lpos{i};
%       end
%       if ~obj.gtIsGTMode
%         obj.labels2VizUpdate();
%       end
%     end

    function [tf,lpos2] = labels2IsCurrMovFrmLbled(obj,iTgt)
      % tf: scalar logical, true if tracker has results/predictions for 
      %   currentMov/frm/iTgt 
      % lpos2: [nptsx2] landmark coords
     
      PROPS = obj.gtGetSharedProps();
      iMov = obj.currMovie;
      frm = obj.currFrame;
      trk = obj.(PROPS.LBL2){iMov};
      [tf,lpos2] = trk.getPTrkFT(frm,iTgt,'collapse',true);
    end
    
    % 20210524 no callers atm
%     function labels2SetCurrMovie(obj,lpos)
%       % Works in both reg/GT mode
%       PROPS = obj.gtGetSharedProps();
%       iMov = obj.currMovie;      
%       assert(isequal(size(lpos),size(obj.(PROPS.LPOS){iMov})));
%       obj.(PROPS.LPOS2){iMov} = lpos;
%     end
    
    function labels2Clear(obj)
      % Operates based on current reg/GT mode
      PROPS = obj.gtGetSharedProps();

      % resetting rather than clearing
      nlblpts = obj.nLabelPoints;
      for i = 1:obj.nmoviesGTaware,
        trxinfo = obj.(PROPS.TIA){i};
        nfrms = obj.(PROPS.MIA){i}.nframes;
        if obj.maIsMA
          tfo = TrkFile(nlblpts,zeros(0,1));
          tfo.initFrm2Tlt(nfrms);
          obj.(PROPS.LBL2){i,1} = tfo;
        else
          tfo = TrkFile(nlblpts,1:trxinfo.ntgts);
          tfo.initFrm2Tlt(nfrms);
          obj.(PROPS.LBL2){i,1} = tfo;
        end
      end

      % lbl2 = obj.(PROPLBL2);
      % cellfun(@(x)x.clearTracklet(),lbl2);
      obj.labels2TrkVizInit();
      obj.labels2VizUpdate();
      obj.notify('dataImported');
    end
    
    function labels2ImportTrk(obj,iMovs,trkfiles)
      % Works per current GT mode
      %PROPS = obj.gtGetSharedProps;
      mIdx = MovieIndex(iMovs,obj.gtIsGTMode);
      obj.labelImportTrkGeneric(mIdx,trkfiles,'LBL2');
      obj.labels2TrkVizInit();
      obj.labels2VizUpdate();
      obj.labels2VizShowHideUpdate();
      obj.notify('dataImported');
      obj.rcSaveProp('lastTrkFileImported',trkfiles{end});
    end
    
    function colors = Set2PointColors(obj,colors)
      colors = colors(obj.labeledposIPt2Set,:);
    end
    
    function colors = LabelPointColors(obj,idx)
      colors = obj.Set2PointColors(obj.labelPointsPlotInfo.Colors);
      if nargin > 1,
        colors = colors(idx,:);
      end      
    end
    
    function colors = PredictPointColors(obj)
      colors = obj.Set2PointColors(obj.predPointsPlotInfo.Colors);
    end
    
    function labels2TrkVizInit(obj,varargin)
      % Initialize trkViz for .labeledpos2, .trkRes*
%       
%       vizNtrxMax = myparse(varargin,...
%         'vizNtrxMax',20 ...
%         );
      
      tv = obj.labeledpos2trkViz;
      if ~isempty(tv)
        tv.delete();
        obj.labeledpos2trkViz = [];
      end
      
      iMov = obj.currMovie;
      if iMov == 0 % "no movie"
        return
      end
      
      trk = obj.labels2GTaware{iMov};
      if ~trk.hasdata()
        % no imported labels for this mov
        return;
      end
        
      % We only create/init a tv if there are actually imported results for 
      % the current movie. Otherwise, obj.labeledpos2trkViz will be [] 
      % which optimizes browse speed.
      tv = obj.createTrackingVisualizer('impPointsPlotInfo','labeledpos2');      
      if ~isempty(obj.trackParams) && isfield(obj.trackParams.ROOT.MultiAnimal,'Track')
        maxNanimals = obj.trackParams.ROOT.MultiAnimal.Track.max_n_animals;
        maxNanimals = max(ceil(maxNanimals*1.5),10);
      else
        maxNanimals = 20;
      end
      % ntgtmax spec only used by tracklets currently
      tv.vizInit('ntgtmax',maxNanimals);
      tv.trkInit(trk);
      obj.labeledpos2trkViz = tv;
    end
    
    function trkResVizInit(obj)
      for i=1:numel(obj.trkResViz)
        tv = obj.trkResViz{i};
        if isempty(tv.lObj)
          tv.postLoadInit(obj);
        else
          tv.vizInit('postload',true);
        end
      end
    end
    
    function labels2VizUpdate(obj,varargin)
      % update trkres from lObj.labeledpos
      
      [dotrkres,setlbls,setprimarytgt] = myparse(varargin,...
        'dotrkres',false,...
        'setlbls',true,...
        'setprimarytgt',false ...
        );
            
      iTgt = obj.currTarget;
      tv = obj.labeledpos2trkViz;
      if isempty(tv)
        return;
      end
      
      if setlbls
        iMov = obj.currMovie;
        frm = obj.currFrame;
        if obj.maIsMA || obj.hasTrx
          tv.newFrame(frm);
        else
          % nonMA: either SA, or SA-trx
          trk = obj.labels2GTaware{iMov};
          [~,xy,tfocc] = trk.getPTrkFrame(frm,'collapse',true);
          tv.updateTrackRes(xy,tfocc);
        end
      end
      if setprimarytgt
        tv.updatePrimary(iTgt);        
      end
      
      if dotrkres && ~obj.maIsMA
        %assert(~obj.maIsMA,'Unsupported for multi-animal projects.');
        trkres = obj.trkResGTaware;
        trvs = obj.trkResViz;
        nTR = numel(trvs);
        for iTR=1:nTR
          if ~isempty(trkres{iMov,1,iTR})
            tObjsAll = trkres(iMov,:,iTR);
            
            trkP = cell(obj.nview,1);
            tfHasRes = true;
            for ivw=1:obj.nview
              tObj = tObjsAll{ivw};
              ifrm = find(frm==tObj.pTrkFrm);
              iitgt = find(iTgt==tObj.pTrkiTgt);
              if ~isempty(ifrm) && ~isempty(iitgt)
                trkP{ivw} = tObj.pTrk(:,:,ifrm,iitgt);
              else
                tfHasRes = false;
                break;
              end
            end
            if tfHasRes            
              trkP = cat(1,trkP{:}); % [nlabelpoints x 2 x nfrm x ntgt]
              trv = trvs{iTR};
              trv.updateTrackRes(trkP);
                  % % From DeepTracker.getTrackingResultsCurrFrm
                  % AL20160502: When changing movies, order of updates to
                  % % lObj.currMovie and lObj.currFrame is unspecified. currMovie can
                  % % be updated first, resulting in an OOB currFrame; protect against
                  % % this.
                  % frm = min(frm,size(xyPCM,3));
                  % xy = squeeze(xyPCM(:,:,frm,:)); % [npt x d x ntgt]
            end
          end
        end
      end
    end
    
    function labels2VizShowHideUpdate(obj)
      tfHide = obj.labels2Hide;
      txtprops = obj.impPointsPlotInfo.TextProps;
      tfHideTxt = strcmp(txtprops.Visible,'off');
      tv = obj.labeledpos2trkViz;
      if ~isempty(tv)
        tv.setAllShowHide(tfHide,tfHideTxt,obj.labels2ShowCurrTargetOnly,obj.showSkeleton);
      end
    end
    
    function labels2VizShow(obj)
      obj.labels2Hide = false;
      obj.labels2VizShowHideUpdate();
    end
    
    function labels2VizHide(obj)
      obj.labels2Hide = true;
      obj.labels2VizShowHideUpdate();
    end
    
    function labels2VizToggle(obj)
      if obj.labels2Hide
        obj.labels2VizShow();
      else
        obj.labels2VizHide();
      end
    end

    function labels2VizSetShowCurrTargetOnly(obj,tf)
      obj.labels2ShowCurrTargetOnly = tf;
      obj.labels2VizShowHideUpdate();
    end
     
  end 
  
%   %% Emp PDF
%   methods
%     function updateFGEmpiricalPDF(obj,varargin)
%       
%       if ~obj.hasTrx
%         error('Method only supported for projects with trx.');
%       end
%       if obj.gtIsGTMode
%         error('Method is not supported in GT mode.');
%       end
%       if obj.cropProjHasCrops
%         % in general projs with trx would never use crop info 
%         error('Method unsupported for projects with cropping info.');
%       end
%       if obj.nview>1
%         error('Method is not supported for multiple views.');
%       end
%       tObj = obj.tracker;
%       if isempty(tObj)
%         error('Method only supported for projects with trackers.');
%       end
%       
%       prmPP = obj.preProcParams;
%       prmBackSub = prmPP.BackSub;
%       prmNborMask = prmPP.NeighborMask;
%       if isempty(prmBackSub.BGType) || isempty(prmBackSub.BGReadFcn)
%         error(strcatg('Computing the empirical foreground PDF requires a background type and ', ...
%                            'background read function to be defined in the tracking parameters.'));
%       end
%       if ~prmNborMask.Use
%         warningNoTrace('Neighbor masking is currently not turned on in your tracking parameters.');
%       end
%       
%       % Start with all labeled rows. Prefer these b/c user apparently cares
%       % more about these frames
%       wbObj = WaitBarWithCancel('Empirical Foreground PDF');
%       oc = onCleanup(@()delete(wbObj));
%       tblMFTlbled = obj.labelGetMFTableLabeled('wbObj',wbObj);
%       if wbObj.isCancel
%         return;
%       end
%       assert(false,'Unsupported'); % eg radius ref here is out of date
%       roiRadius = prmPP.TargetCrop.Radius;
%       tblMFTlbled = obj.labelMFTableAddROITrx(tblMFTlbled,roiRadius);
% 
%       amu = mean(tblMFTlbled.aTrx);
%       bmu = mean(tblMFTlbled.bTrx);
%       %fprintf('amu/bmu: %.4f/%.4f\n',amu,bmu);
%       
%       % get stuff we will need for movies: movieReaders, bgimages, etc
%       movieStuff = cell(obj.nmovies,1);
%       iMovsLbled = unique(tblMFTlbled.mov);
%       for iMov=iMovsLbled(:)'        
%         s = struct();
%         
%         mr = MovieReader();
%         obj.movieMovieReaderOpen(mr,MovieIndex(iMov),1);
%         s.movRdr = mr;
%         
%         trxfname = obj.trxFilesAllFull{iMov,1};
%         movIfo = obj.movieInfoAll{iMov};
%         [s.trx,s.frm2trx] = obj.getTrx(trxfname,movIfo.nframes);
%                 
%         movieStuff{iMov} = s;
%       end    
%       
%       hFigViz = figure; %#ok<NASGU>
%       ax = axes;
%     
%       xroictr = -roiRadius:roiRadius;
%       yroictr = -roiRadius:roiRadius;
%       [xgrid,ygrid] = meshgrid(xroictr,yroictr); % xgrid, ygrid give coords for each pixel where (0,0) is the central pixel (at target)
%       pdfRoiAcc = zeros(2*roiRadius+1,2*roiRadius+1);
%       nAcc = 0;
%       n = height(tblMFTlbled);
%       for i=1:n
%         trow = tblMFTlbled(i,:);
%         iMov = trow.mov;
%         frm = trow.frm;
%         iTgt = trow.iTgt;
%         
%         sMovStuff = movieStuff{iMov};
%         
%         [tflive,trxxs,trxys,trxths,trxas,trxbs] = ...
%           PxAssign.getTrxStuffAtFrm(sMovStuff.trx,frm);
%         assert(isequal(tflive(:),sMovStuff.frm2trx(frm,:)'));
% 
%         % Skip roi if it contains more than 1 trxcenter.
%         roi = trow.roi; % [xlo xhi ylo yhi]
%         roixlo = roi(1);
%         roixhi = roi(2);
%         roiylo = roi(3);
%         roiyhi = roi(4);
%         tfCtrInRoi = roixlo<=trxxs & trxxs<=roixhi & roiylo<=trxys & trxys<=roiyhi;
%         if nnz(tfCtrInRoi)>1
%           continue;
%         end
%       
%         % In addition run CC pxAssign and keep only the central CC to get
%         % rid of any objects at the periphery
%         imdiff = sMovStuff.movRdr.readframe(frm,'doBGsub',true);
%         assert(isa(imdiff,'double'));
%         imbwl = PxAssign.asgnCCcore(imdiff,sMovStuff.trx,frm,prmNborMask.FGThresh);
%         xTgtCtrRound = round(trxxs(iTgt));
%         yTgtCtrRound = round(trxys(iTgt));
%         ccKeep = imbwl(yTgtCtrRound,xTgtCtrRound);
%         if ccKeep==0
%           warningNoTrace('Unexpected non-foreground pixel for (mov,frm,tgt)=(%d,%d,%d) at (r,c)=(%d,%d).',...
%             iMov,frm,iTgt,yTgtCtrRound,xTgtCtrRound);
%         else
%           imfgUse = zeros(size(imbwl));
%           imfgUse(imbwl==ccKeep) = 1;                    
%           imfgUseRoi = padgrab(imfgUse,0,roiylo,roiyhi,roixlo,roixhi);
%           
%           th = trxths(iTgt);
%           a = trxas(iTgt);
%           b = trxbs(iTgt);
%           imforebwcanon = readpdf(imfgUseRoi,xgrid,ygrid,xgrid,ygrid,0,0,-th);
%           xfac = a/amu;
%           yfac = b/bmu;
%           imforebwcanonscale = interp2(xgrid,ygrid,imforebwcanon,...
%             xgrid*xfac,ygrid*yfac,'linear',0);
%           
%           imshow(imforebwcanonscale,'Parent',ax);
%           tstr = sprintf('row %d/%d',i,n);
%           title(tstr,'fontweight','bold','interpreter','none');
%           drawnow;
%           
%           pdfRoi = imforebwcanonscale/sum(imforebwcanonscale(:)); % each row equally weighted
%           pdfRoiAcc = pdfRoiAcc + pdfRoi;
%           nAcc = nAcc + 1;
%         end
%       end
%       
%       fgpdf = pdfRoiAcc/nAcc;
%       
%       imshow(fgpdf,[],'xdata',xroictr,'ydata',yroictr);
%       colorbar;
%       tstr = sprintf('N=%d, amu=%.3f, bmu=%.3f. FGThresh=%.2f',...
%         nAcc,amu,bmu,prmNborMask.FGThresh);
%       title(tstr,'fontweight','bold','interpreter','none');
%       
%       obj.fgEmpiricalPDF = struct(...
%         'amu',amu,'bmu',bmu,...
%         'xpdfctr',xroictr,'ypdfctr',yroictr,...        
%         'fgpdf',fgpdf,...
%         'n',nAcc,...
%         'roiRadius',roiRadius,...
%         'prmBackSub',prmBackSub,...
%         'prmNborMask',prmNborMask);
%     end
%   end
  
  
  %% Util
  methods
    
    function tblConcrete = mftTableConcretizeMov(obj,tbl)
      % tbl: MFTable where .mov is MovieIndex array
      %
      % tblConcrete: Same table where .mov is [NxNview] cellstr; column 
      %   .trxFile is also added when appropriate
      
      assert(isa(tbl,'table'));

      n = height(tbl);
      tblConcrete = tbl;
      mIdx = tblConcrete.mov;
      assert(isa(mIdx,'MovieIndex'));
      [iMovAbs,tfGTRow] = mIdx.get();
      tfRegRow = ~tfGTRow;
      assert(~any(iMovAbs==0));
      movStr = cell(n,obj.nview);
      movStr(tfRegRow,:) = obj.movieFilesAllFull(iMovAbs(tfRegRow),:); % [NxnView] 
      movStr(tfGTRow,:) = obj.movieFilesAllGTFull(iMovAbs(tfGTRow),:);
      tblConcrete.mov = movStr;

      if obj.hasTrx && obj.nview==1
        trxFile = repmat({''},n,1);
        trxFile(tfRegRow) = obj.trxFilesAllFull(iMovAbs(tfRegRow),:);
        trxFile(tfGTRow) = obj.trxFilesAllGTFull(iMovAbs(tfGTRow),:);
        tblConcrete = [tblConcrete table(trxFile)];
      end
    end
    
    function initVirtualPrevAxesLabelPointViz_(obj, plotIfo)
      markerPVs = plotIfo.MarkerProps;
      textPVs = plotIfo.TextProps;

      % Extra plot params
      allowedPlotParams = {'HitTest' 'PickableParts'};
      plotIfoFields = fieldnames(plotIfo);
      ism = ismember(cellfun(@lower, allowedPlotParams, 'Uni', 0), ...
                     cellfun(@lower, plotIfoFields, 'Uni', 0));

      npts = obj.nLabelPoints;
      obj.lblPrev_ptsH = VirtualLine.empty(0, 1);
      obj.lblPrev_ptsTxtH = VirtualText.empty(0, 1);

      for i = 1:npts
        vl = VirtualLine();
        set(vl, markerPVs);
        vl.Color = plotIfo.Colors(i, :);
        vl.UserData = i;
        vl.Tag = sprintf('Labeler_lblPrev_ptsH_%d', i);
        for j = find(ism)
          vl.(allowedPlotParams{j}) = plotIfo.(allowedPlotParams{j});
        end
        obj.lblPrev_ptsH(i, 1) = vl;

        vt = VirtualText();
        set(vt, textPVs);
        vt.Color = plotIfo.Colors(i, :);
        vt.String = num2str(i);
        vt.PickableParts = 'none';
        vt.Tag = sprintf('Labeler_lblPrev_ptsTxtH_%d', i);
        obj.lblPrev_ptsTxtH(i, 1) = vt;
      end
    end
   
    function pushBusyStatus(obj, new_raw_status_string)
      % Called to indicate the Labeler is doing something.
      % If there's a controller, it will show the status string in the lower left,
      % and the mouse pointer will be a watch/hourglass/whatever.  We keep a stack
      % of the status strings to we can pop them off as nested tasks complete.
      obj.howBusy_ = obj.howBusy_ + 1 ;
      obj.rawStatusStringStack_ = horzcat(obj.rawStatusStringStack_, {new_raw_status_string}) ;
      obj.notify('updateStatusAndPointer') ;     
    end

    function popBusyStatus(obj)
      % Called to indicate the Labeler is done doing something.  If this call
      % indicates that the Labeler has finished its last nested task, and there's a
      % controller attached, this will cause the main window to show the idle status
      % message in the lower left, and the mouse pointer will go back to being an
      % arrow.
      obj.howBusy_ = max(0, obj.howBusy_ - 1) ;
      obj.rawStatusStringStack_ = obj.rawStatusStringStack_(1:obj.howBusy_) ;
      obj.notify('updateStatusAndPointer') ;      
    end

    function result = get.isStatusBusy(obj)
      result = (obj.howBusy_ > 0) ;
    end

    function result = get.rawStatusString(obj)
      if obj.howBusy_ == 0
        result = obj.rawClearStatusString_ ;
      else
        result = obj.rawStatusStringStack_{end} ;
      end
    end

    % function result = get.didSpawnTrackingForGT(obj)
    %   result = obj.didSpawnTrackingForGT_ ;
    % end

    function setRawClearStatusString_(obj, new_value)
      % This should go away eventually, once a few more functions in LabelerGUI get
      % folded into Labeler.
      obj.rawClearStatusString_ = new_value ;
    end
    
    function v = allMovIdx(obj)      
      v = MovieIndex(1:obj.nmoviesGTaware,obj.gtIsGTMode);      
    end
    
  end  % methods
  
  methods (Static)
    
    function [iMov1,iMov2] = identifyCommonMovSets(movset1,movset2)
      % Find common rows (moviesets) in two sets of moviesets
      %
      % movset1: [n1 x nview] cellstr of fullpaths. Call 
      %   FSPath.standardPath and FSPath.platformizePath on these first
      % movset2: [n2 x nview] "
      %
      % IMPORTANT: movset1 is allowed to have duplicate rows/moviesets,
      % but dupkicate rows/movsets in movset2 will be "lost".
      % 
      % iMov1: [ncommon x 1] index vector into movset1. Could be empty
      % iMov2: [ncommon x 1] " movset2 "
      %
      % If ncommon>0, then movset1(iMov1,:) is equal to movset2(iMov2,:)
      
      assert(size(movset1,2)==size(movset2,2));
      nvw = size(movset1,2);
      
      for ivw=nvw:-1:1
        [tf(:,ivw),loc(:,ivw)] = ismember(movset1(:,ivw),movset2(:,ivw));
      end
      
      tf = all(tf,2);
      iMov1 = find(tf);      
      nCommon = numel(iMov1);
      iMov2 = zeros(nCommon,1);
      for i=1:nCommon
        locUn = unique(loc(iMov1(i),:));
        if isscalar(locUn)
          iMov2(i) = locUn;
        else
          % warningNoTrace('Inconsistency in movielists detected across views.');
          % iMov2(i) initted to 0
        end
      end
      
      tfRm = iMov2==0;
      iMov1(tfRm,:) = [];
      iMov2(tfRm,:) = [];
     end    

  end  % methods (Static)

  methods
    function value = get.doesNeedSave(obj)
      value = obj.doesNeedSave_ ;
    end

    function setDoesNeedSave(obj, doesNeedSave, why)
      % Can't have a normal setter b/c of the why string.
      if islogical(doesNeedSave) && isscalar(doesNeedSave) ,
        obj.doesNeedSave_ = doesNeedSave ;
        if doesNeedSave ,
          if ~exist('why', 'var') || isempty(why) ,
            why = 'Save needed' ;
          end
          info = obj.projFSInfo ;
          if isempty(info) ,
            if isempty(obj.projectfile) ,
              % This indicates a new project, just created.
              obj.rawClearStatusString_ = sprintf('%s, not yet saved.', why) ;
            else
              obj.rawClearStatusString_ = sprintf('%s since $PROJECTNAME saved.', why) ;
            end
          else
            obj.rawClearStatusString_ = sprintf('%s since $PROJECTNAME %s at %s', why, info.action, datestr(info.timestamp,16)) ;
          end
          obj.notify('updateStatusAndPointer') ;
        end        
      else
        error('APT:invalidValue', 'Illegal value for doesNeedSave') ;
      end

      obj.notify('updateDoesNeedSave') ;
    end
    
    function value = get_backend_property(obj, property_name)
      backend = obj.trackDLBackEnd ;
      value = backend.(property_name) ;
    end

    function set_backend_property(obj, property_name, new_value)
      backend = obj.trackDLBackEnd ;
      backend.(property_name) = new_value ;  % this can throw if value is invalid
      if strcmp(property_name, 'type') ,
        obj.notify('didSetTrackDLBackEnd') ;
      end
      obj.setDoesNeedSave(true, 'Changed backend parameter') ;  % this is a public method, will send update notification
    end

    function set.projname(obj, newValue)
      obj.projname = newValue ;
      str = sprintf('Project $PROJECTNAME created (unsaved) at %s',datestr(now(),16));
      obj.setRawClearStatusString_(str) ;      
      obj.notify('didSetProjectName') ;
    end

    function set.projFSInfo(obj, newValue)
      obj.projFSInfo = newValue ;
      info = obj.projFSInfo ;
      if ~isempty(info)
        str = sprintf('Project $PROJECTNAME %s at %s',info.action,datestr(info.timestamp,16)) ;
        obj.setRawClearStatusString_(str) ;
      end
      obj.notify('didSetProjFSInfo') ;
    end

    function set.movieFilesAll(obj, newValue)
      obj.movieFilesAll = newValue ;
      obj.notify('didSetMovieFilesAll') ;
    end

    function set.movieFilesAllGT(obj, newValue)
      obj.movieFilesAllGT = newValue ;
      obj.notify('didSetMovieFilesAllGT') ;
    end

    function set.movieFilesAllHaveLbls(obj, newValue)
      if ~isequal(newValue, obj.movieFilesAllHaveLbls) ,
        obj.movieFilesAllHaveLbls = newValue ;
        obj.notify('didSetMovieFilesAllHaveLbls') ;
      end
    end

    function set.movieFilesAllGTHaveLbls(obj, newValue)
      if ~isequal(newValue, obj.movieFilesAllGTHaveLbls) ,
        obj.movieFilesAllGTHaveLbls = newValue ;
        obj.notify('didSetMovieFilesAllGTHaveLbls') ;
      end
    end

    function set.trxFilesAll(obj, newValue)
      obj.trxFilesAll = newValue ;
      obj.notify('didSetTrxFilesAll') ;
    end

    function set.trxFilesAllGT(obj, newValue)
      obj.trxFilesAllGT = newValue ;
      obj.notify('didSetTrxFilesAllGT') ;
    end

    function set.showTrx(obj, newValue)
      % Note that setShowTrx() also exists, seems like maybe what clients are
      % intended to call?  -- ALT, 2023-05-15
      obj.showTrx = newValue ;
      obj.notify('didSetShowTrx') ;
    end
    
    function set.showTrxCurrTargetOnly(obj, newValue)
      % Note that setShowTrxCurrTargetOnly() also exists, seems like maybe what clients are
      % intended to call?  -- ALT, 2023-05-15
      obj.showTrxCurrTargetOnly = newValue ;
      obj.notify('didSetShowTrxCurrTargetOnly') ;
    end
    
    function set.showOccludedBox(obj, newValue)
      obj.showOccludedBox = newValue ;
      obj.notify('didSetShowOccludedBox') ;
    end

    function set.showSkeleton(obj, newValue)
      obj.showSkeleton = newValue ;
      obj.notify('didSetShowSkeleton') ;
    end

    function set.showMaRoi(obj, newValue)
      obj.showMaRoi = newValue ;
      obj.notify('didSetShowMaRoi') ;
    end

    function set.showMaRoiAux(obj, newValue)
      obj.showMaRoiAux = newValue ;
      obj.notify('didSetShowMaRoiAux') ;
    end

    function set.labelMode(obj, newValue)
      obj.labelMode = newValue ;
      obj.notify('didSetLabelMode') ;
    end

    function set.labels2Hide(obj, newValue)
      obj.labels2Hide = newValue ;
      obj.notify('didSetLabels2Hide') ;
    end

    function set.labels2ShowCurrTargetOnly(obj, newValue)
      obj.labels2ShowCurrTargetOnly = newValue ;
      obj.notify('didSetLabels2ShowCurrTargetOnly') ;
    end

    function set.labeledposNeedsSave(obj, newValue)
      obj.labeledposNeedsSave = newValue ;
      obj.setDoesNeedSave(newValue, 'Unsaved labels') ;
    end

    function set.lastLabelChangeTS(obj, newValue)
      obj.lastLabelChangeTS = newValue ;
      obj.notify('didSetLastLabelChangeTS') ;
    end

    function set.lblCore(obj, newValue)
      obj.lblCore = newValue ;
      obj.notify('didSetLblCore') ;
    end

    % function result = get.trackersAll(obj)
    %   result = obj.trackersAll_ ;
    % end
    % 
    % function result = get.trackersAllCreateInfo(obj)
    %   result = obj.trackersAllCreateInfo_ ;
    % end

    % function set.currTracker(obj, newValue)
    %   % Want to do some stuff before the set, apparently
    %   if ~obj.isinit ,
    %     oldTracker = obj.tracker ;
    %     if ~isempty(oldTracker) ,
    %       oldTracker.deactivate() ;
    %     end
    %   end
    % 
    %   % Set the value
    %   obj.currTracker = newValue ;
    % 
    %   % Activate the newly-selected tracker
    %   newTracker = obj.tracker ;
    %   if ~isempty(newTracker),
    %     newTracker.activate();
    %   end
    % 
    %   % Send the notification
    %   obj.notify('didSetCurrTracker') ;
    % end

    function set.currTarget(obj, newValue)
      obj.currTarget = newValue ;
      sendMaybe(obj.tracker, 'newLabelerTarget')
      obj.notify('didSetCurrTarget') ;
    end

    function set.trackModeIdx(obj, newValue)
      obj.trackModeIdx = newValue ;
      obj.notify('didSetTrackModeIdx') ;
    end
      
    function set.trackDLBackEnd(obj, newValue)
      obj.trackDLBackEnd = newValue ;
      obj.notify('didSetTrackDLBackEnd') ;
    end
      
    function set.trackNFramesSmall(obj, newValue)
      obj.trackNFramesSmall = newValue ;
      obj.notify('didSetTrackNFramesSmall') ;
    end
      
    function set.trackNFramesLarge(obj, newValue)
      obj.trackNFramesLarge = newValue ;
      obj.notify('didSetTrackNFramesLarge') ;
    end

    function set.trackNFramesNear(obj, newValue)
      obj.trackNFramesNear = newValue ;
      obj.notify('didSetTrackNFramesNear') ;
    end

    function set.trackParams(obj, newValue)
      obj.trackParams = newValue ;
      obj.notify('didSetTrackParams') ;
    end

    function result = getMftInfoStruct(obj)
      % Get some info needed for pretty-printing of MFT info in the UI (or
      % something).  We used to just pass the whole labeler object, but that seems
      % inelegant.  Now we just pass a struct with the handful of values we need.
      result = struct() ;
      result.nTargets = obj.nTargets ;
      result.trackNFramesNear = obj.trackNFramesNear ;
      result.trackNFramesLarge = obj.trackNFramesLarge ;
      result.trackNFramesSmall = obj.trackNFramesSmall ;
      result.nmoviesGTaware = obj.nmoviesGTaware ;
      result.gtIsGTMode = obj.gtIsGTMode ;
      result.currMovIdx = obj.currMovIdx ;
      result.moviesSelected = obj.moviesSelected ;
      result.nmovies = obj.nmovies ;
      result.nmoviesGT = obj.nmoviesGT ;
      result.hasMovie = obj.hasMovie ;
    end  % function
 
    function trainAugOnly(obj)
      % This function exists just to hide the Labeler internals a bit more.
      tracker = obj.tracker ;
      tracker.train('augOnly', true) ;
    end  % function

    function mftset = getTrackModeMFTSet(obj)
      % Get the current tracking mode as an MFTSetEnum
      idx = obj.trackModeIdx ;
      mftset = MFTSetEnum.TrackingMenuTrx(idx) ;
    end  % function

    function result = get.bgTrnIsRunning(obj)        
      % Whether training is running.  Technically, only checks whether the
      % background process that polls for training progress is running.
      if isempty(obj.tracker) ,
        result = false ;
      else
        result = obj.tracker.bgTrnIsRunning ;
      end
    end  % function

    function result = get.bgTrkIsRunning(obj)        
      % Whether tracking is running.  Technically, only checks whether the
      % background process that polls for tracking progress is running.
      if isempty(obj.tracker) ,
        result = false ;
      else
        result = obj.tracker.bgTrkIsRunning ;
      end
    end  % function

    function result = get.lastTrainEndCause(obj)        
      if isempty(obj.tracker) ,
        result = EndCause.undefined ;
      else
        result = obj.tracker.lastTrainEndCause ;
      end
    end  % function
    
    function result = get.lastTrackEndCause(obj)        
      if isempty(obj.tracker) ,
        result = EndCause.undefined ;
      else
        result = obj.tracker.lastTrackEndCause ;
      end
    end  % function
    
    function result = get.silent(obj)        
      result = obj.silent_ ;
    end  % function
    
    function set.silent(obj, newValue)        
      obj.silent_ = newValue ;
      % tracker = obj.tracker;
      % if ~isempty(tracker) ,        
      %   tracker.skip_dlgs = newValue ;
      % end      
    end  % function
    
    function result = get.progressMeter(obj) 
      result = obj.progressMeter_ ;
    end

    function set.currFrame(obj, newValue)
      obj.currFrame = newValue ;
      obj.infoTimelineModel_.didSetCurrFrame(newValue) ;
      sendMaybe(obj.tracker, 'newLabelerFrame') ;
      obj.notify('updateAfterCurrentFrameSet') ;
    end    

    function setPropertiesToFireCallbacksToInitializeUI_(obj)
      % These properties need their callbacks fired to properly init UI.  (For
      % now...)
      propsNeedInit = {
        'labelMode'
        'suspScore'
        'showTrx'
        'showTrxCurrTargetOnly' %  'showPredTxtLbl'
        'trackNFramesSmall' % trackNFramesLarge, trackNframesNear currently share same callback
        'trackModeIdx'
        'movieCenterOnTarget'
        'movieForceGrayscale'
        'movieInvert'
        'showOccludedBox'
        } ;
      propCount = numel(propsNeedInit) ;
      for i = 1 : propCount ,
        propName = propsNeedInit{i} ;
        obj.(propName) = obj.(propName) ;
      end
    end  % function
    
    % function didUpdateTrackerInfo(obj)
    %   % Notify listeners that trackerInfo was updated in obj.tracker.
    %   % Called by the current tracker when this happens.
    %   obj.notify('update_text_trackerinfo') ;
    % end
    
    function needRefreshTrackMonitorViz(obj)
      obj.notify('refreshTrackMonitorViz') ;
    end

    function didReceivePollResultsRetrograde(obj, track_or_train)
      if strcmp(track_or_train, 'train') ,
        obj.notify('updateTrainMonitorViz') ;
      elseif strcmp(track_or_train, 'track') ,
        obj.notify('updateTrackMonitorViz') ;
      else
        error('Internal error: %s should be ''track'' or ''train''', track_or_train) ;
      end
    end

    % function didReceiveTrackingPollResults_(obj)
    %   obj.notify('updateTrackMonitorViz') ;
    % end
    % 
    % function didReceiveTrainingPollResults_(obj)
    %   obj.notify('updateTrainMonitorViz') ;
    % end    

    function needRefreshTrainMonitorViz(obj)
      obj.notify('refreshTrainMonitorViz') ;
    end

    function result = get.backend(obj)
      result = obj.trackDLBackEnd ;
    end

    function killAndClearRegisteredJobs(obj, train_or_track)
      obj.trackDLBackEnd.killAndClearRegisteredJobs(train_or_track) ;
    end    
    
    % function raiseTrainingStoppedDialog_(obj) 
    %   obj.notify('raiseTrainingStoppedDialog') ;      
    % end

    function abortTraining(obj)
      % Abort the in-progress training.  Called when the user presses the "Stop
      % training" button during training.
      sendMaybe(obj.tracker, 'abortTraining') ;
    end
    
    function abortTracking(obj)
      % Abort the in-progress tracking.  Called when the user presses the "Stop
      % tracking" button during tracking.
      sendMaybe(obj.tracker, 'abortTracking') ;
    end
    
    function doNotify(obj, eventName)
      % Used by child objects to fire events from the Labeler
      % dbstack
      % fprintf('About to call tracker.doNotify(''%s'')\n', eventName) ;
      obj.notify(eventName) ;
    end

    % function initializeTrackersAllAndFriends_(obj)
    %   % Create initial values for a few Labeler props, including trackersAll.
    % 
    %   % Forcibly clear out any old stuff
    %   cellfun(@delete, obj.trackersAll_) ;
    % 
    %   % Create new templates, trackers
    %   trackersCreateInfo = ...
    %     LabelTracker.getAllTrackersCreateInfo(obj.maIsMA) ;  % 1 x number-of-trackers
    %   tAll = cellfun(@(createInfo)(LabelTracker.create(obj, createInfo)), ...
    %                  trackersCreateInfo, ...
    %                  'UniformOutput', false) ;  % 1 x number-of-trackers
    %   obj.trackersAllCreateInfo_ = trackersCreateInfo ;
    %   obj.trackersAll_ = tAll ;
    %   %obj.notify('update_menu_track_tracking_algorithm') ;
    % end

    function result = get.trackerHistory(obj)
      result = obj.trackerHistory_ ;
    end
  end  % methods

  methods (Static)
    function [maposenets,mabboxnets,saposenets] = getAllTrackerTypes()
      % [maposenets,mabboxnets,saposenets] = getAllTrackerTypes(obj)
      % returns all deep learning nettypes. 
      % All trackers are found with enumeration('DLNetType'), and they are
      % segreagated into multi-animal-pose networks (maposenets), multi-animal
      % bounding-box networks (mabboxnets), and single-animal posenets (saposenets).
      % Parsing is somewhat based on the names of the trackers, so this is pretty
      % delicate. 

      dlnets = enumeration('DLNetType') ;
      isma = [dlnets.isMultiAnimal] ;
      saposenets = dlnets(~isma) ;
      
      is_bbox = false(1,numel(dlnets)) ;
      for dndx = 1:numel(dlnets)          
        is_bbox(dndx) = dlnets(dndx).isMultiAnimal && startsWith(char(dlnets(dndx)),'detect_') ;
      end  % for
      
      maposenets = dlnets(isma & ~is_bbox) ;
      mabboxnets = dlnets(isma & is_bbox) ;

      dokeep = cellfun(@isempty,regexp({maposenets.displayString},'Deprecated','once'));
      maposenets = maposenets(dokeep);
      
      dokeep = cellfun(@isempty,regexp({mabboxnets.displayString},'Deprecated','once'));
      mabboxnets = mabboxnets(dokeep);

      dokeep = cellfun(@isempty,regexp({saposenets.displayString},'Deprecated','once'));
      saposenets = saposenets(dokeep);
    end
  end  % methods (Static)

  methods
    % function result = doesCurrentTrackerMatchFromTrackersAllIndex(obj)
    %   % Returns a logical row array specifying whether the current tracker matches
    %   % each of the trackers in obj.trackersAll
    %   currentTrackerAlgoName = obj.tracker.algorithmName ;
    %   trackersAll = obj.trackersAll_ ;
    %   algoNameFromTrackersAllIndex = cellfun(@(t)(t.algorithmName), trackersAll, 'UniformOutput', false) ;
    %   result =strcmp(currentTrackerAlgoName, algoNameFromTrackersAllIndex) ;
    % end
    
    function hlpApplyCosmetics(obj,colorSpecs,mrkrSpecs,skelSpecs)
      obj.updateLandmarkColors(colorSpecs);
      obj.updateLandmarkCosmetics(mrkrSpecs);
      obj.updateSkeletonCosmetics(skelSpecs);
    end

    function gtToggleGTMode(obj)
      gt = obj.gtIsGTMode;
      gtNew = ~gt;
      if gtNew
        statusMessage = 'Switching to Ground Truth Mode...' ;
      else
        statusMessage = 'Switching back to Labeling Mode...' ;
      end
      obj.pushBusyStatus(statusMessage);
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      obj.gtSetGTMode(gtNew);
    end

    function setTrackingParameters(obj, sPrmTrack)
      obj.pushBusyStatus('Setting tracking parameters...');
      on = onCleanup(@()(obj.popBusyStatus())) ;
      if ~isempty(sPrmTrack),
        sPrmAll = obj.trackGetTrainingParams();
        sPrmAllNew = APTParameters.setTrackParams(sPrmAll,sPrmTrack);        
        obj.trackSetTrainingParams(sPrmAllNew,'istrack',true);        
        % set all tracker parameters
        for i = 1:numel(obj.trackerHistory_),
          obj.trackerHistory_{i}.setTrackParams(sPrmTrack);
        end
        obj.setDoesNeedSave(true, 'Parameters changed') ;
      end
    end  % function

    function projRemoveOtherTempDirs(obj, todelete)
      obj.pushBusyStatus('Deleting temp directories...');
      oc = onCleanup(@()(obj.popBusyStatus())) ;      
      ndelete = 0;
      for i = 1:numel(todelete),
        [success, msg, ~] = rmdir(todelete{i},'s');
        if success,
          ndelete = ndelete + 1;
          fprintf('Cleared temp directory: %s\n',todelete{i});
        else
          warning('Could not clear the temp directory %s: %s',todelete{i},msg);
        end
      end
      success = (ndelete == numel(todelete)) ;
      if ~success ,
        error('Unable to clear all temporary directories.  See console for details.') ;
      end
    end    
    
    function clearTrackingResults(obj)
      obj.pushBusyStatus('Clearing tracking results...') ;
      oc = onCleanup(@()(obj.popBusyStatus())) ;
      tracker = obj.tracker ;
      if ~isempty(tracker) 
        tracker.clearTrackingResults() ;
      end
    end  % function    
    
    function replaceMovieAndTrxPathPrefixes(obj, oldPrefix, newPrefix)
      % For all the movie and trx file paths, does macro-substitution, then replaces
      % any paths that start with oldPrefix with newPrefix.  Used in testing code to
      % translate from Linux-style PRFS paths to Windows-style.
      movieFilesAllLiteral = FSPath.macroReplace(obj.movieFilesAll, obj.projMacros) ;
      movieFilesAllGTLiteral = FSPath.macroReplace(obj.movieFilesAllGT, obj.projMacros) ;
      trxFilesAllLiteral = FSPath.macroReplace(obj.trxFilesAll, obj.projMacros) ;
      trxFilesAllGTLiteral = FSPath.macroReplace(obj.trxFilesAllGT, obj.projMacros) ;
      movieFilesAllNew = replace_prefix_path(movieFilesAllLiteral, oldPrefix, newPrefix) ;
      movieFilesAllGTNew = replace_prefix_path(movieFilesAllGTLiteral, oldPrefix, newPrefix) ;
      trxFilesAllNew = replace_prefix_path(trxFilesAllLiteral, oldPrefix, newPrefix) ;
      trxFilesAllGTNew = replace_prefix_path(trxFilesAllGTLiteral, oldPrefix, newPrefix) ;
      % Actually update the state
      obj.movieFilesAll = movieFilesAllNew ;
      obj.movieFilesAllGT = movieFilesAllGTNew ;
      obj.trxFilesAll = trxFilesAllNew ;
      obj.trxFilesAllGT = trxFilesAllGTNew ;      
    end  % function

    function trackBatch(obj, toTrackRaw, varargin)
      % Batch Tracking: Track the movies specifies by toTrackRaw.
      % toTrackRaw seems to be essentially a struct that contains sufficient
      % information to construct a ToTrackInfo object from it.
    
      % Parse optional arguments
      [trackType, leftoverArgs] = ...
        myparse_nocheck(varargin,...
                        'trackType', apt.TrackType.track) ;

      % Check arguments
      assert(~isempty(toTrackRaw));
      assert(isa(trackType, 'apt.TrackType'));
      
      % Make a ToTrackInfo object from toTrackRaw
      toTrack = tidyToTrackStructForBatchTracking(toTrackRaw) ;
      totrackinfo = ...
        ToTrackInfo('movfiles',toTrack.movfiles,...
                    'trxfiles',toTrack.trxfiles,...
                    'trkfiles',toTrack.trkfiles,...
                    'views',1:obj.nview,...
                    'stages',1:obj.tracker.getNumStages(),...
                    'croprois',toTrack.cropRois,...
                    'calibrationfiles',calibrationfiles,...
                    'frm0',f0s,...
                    'frm1',f1s,...
                    'trxids',toTrack.targets);
      
      % Call obj.tracker.track to do the real tracking
      obj.tracker.track('totrackinfo',totrackinfo, ...
                        'trackType', trackType, ...
                        'isexternal',true, ...
                        leftoverArgs{:});
    end  % function

    function [didLaunchSucceed, instanceID] = launchNewAWSInstance(obj)
      [didLaunchSucceed, instanceID] = obj.backend.launchNewAWSInstance() ;
    end

    % function trainingDidStart(obj)
    %   % Normally called from children of Labeler to inform it that training has
    %   % just started.
    %   algName = obj.tracker.algorithmName;
    %   backend_type_string = obj.trackDLBackEnd.prettyName();
    %   obj.backgroundProcessingStatusString_ = ...
    %     sprintf('%s training on %s (started %s)',algName,backend_type_string,datestr(now(),'HH:MM'));  %#ok<TNOW1,DATST>
    %   obj.notify('trainStart') ;
    % end

    % function trackingDidStart(obj)
    %   % Normally called from children of Labeler to inform it that training has
    %   % just started.
    %   algName = obj.tracker.algorithmName;
    %   backend_type_string = obj.trackDLBackEnd.prettyName();
    %   obj.backgroundProcessingStatusString_ = ...
    %     sprintf('%s training on %s (started %s)',algName,backend_type_string,datestr(now(),'HH:MM'));  %#ok<TNOW1,DATST>
    %   obj.notify('trainStart') ;
    % end

    function trainingEndedRetrograde(obj, endCause, pollingResultOrEmpty)
      % Normally called from children of Labeler to inform it that training has
      % just ended.
      if endCause == EndCause.complete 
        obj.setDoesNeedSave(true, 'Tracker trained') ;
      elseif endCause == EndCause.error
        obj.printErrorInfo_('train', pollingResultOrEmpty)
      end
      obj.notify('trainEnd') ;  % With a controller present, this will causes any needed dialogs to be raised
    end
    
    function trackingEndedRetrograde(obj, endCause, pollingResultOrEmpty)
      % Normally called from children of Labeler to inform it that tracking has
      % just ended.
      if endCause == EndCause.complete
        obj.infoTimelineModel_.invalidateTraceCache() ;
        obj.setDoesNeedSave(true, 'New frames tracked') ;
      elseif endCause == EndCause.error
        obj.printErrorInfo_('track', pollingResultOrEmpty)
      end
      obj.notify('trackEnd') ;  % With a controller present, this will causes any needed dialogs to be raised
    end
    
    function printErrorInfo_(obj, train_or_track, pollingResultOrEmpty)
      % Produce an error message on the console
      fprintf('Error occurred during %sing:\n', train_or_track) ;
      if isempty(pollingResultOrEmpty) ,
        fprintf('Something went very wrong.  No other information is available.\n') ;
      else
        pollingResult = pollingResultOrEmpty ;
        errorFileIndexMaybe = find(pollingResult.errFileExists, 1) ; 
        if isempty(errorFileIndexMaybe) ,
          fprintf('One of the background jobs exited, for unknown reasons.  No error file was produced.\n') ;
        else
          errorFileIndex = errorFileIndexMaybe ;
          errFile = pollingResult.errFile{errorFileIndex} ;
          doesErrorFileExist = obj.backend.tfDoesCacheFileExist(errFile) ;
          if doesErrorFileExist ,
            fprintf('\n### %s\n\n',errFile);
            errContents = obj.backend.cacheFileContents(errFile) ;
            disp(errContents);
          else
            fprintf('One of the background jobs exited, for unknown reasons.  An error file allegedly existed, but was not found.\n') ;
          end      
        end      
      end
    end  % function

    function result= get.backgroundProcessingStatusString(obj)
      result = obj.backgroundProcessingStatusString_ ;
    end

    function set.backgroundProcessingStatusString(obj, str)
      obj.backgroundProcessingStatusString_ = str ;
      obj.notify('update') ;
    end
  end  % methods

  methods (Static)
    function result = defaultCfgFilePath()
      result = fullfile(APT.Root, 'matlab', 'config.default.yaml') ;
    end  % function
  end  % methods (Static)

  methods
    function updateTrainingMonitorRetrograde(obj)
      % Called by children to generate a notification
      obj.notify('updateTrainingMonitor') ;
    end  % function

    function updateTrackingMonitorRetrograde(obj)
      % Called by children to generate a notification
      obj.notify('updateTrackingMonitor') ;
    end  % function

    function pushBusyStatusRetrograde(obj, new_raw_status_string)
      % Called by children when they want to update the busy status
      obj.pushBusyStatus(new_raw_status_string) ;
    end

    function popBusyStatusRetrograde(obj)
      % Called by children when they want to update the busy status
      obj.popBusyStatus() ;
    end

    function testBackendConfig(obj)
      obj.backend.testBackendConfig(obj) ;
    end    

    function setTimelineSelectMode(obj, newValue)
      if ~obj.doProjectAndMovieExist()
        return
      end
      itm = obj.infoTimelineModel_ ;
      % oldValue = itm.selectOn ;
      itm.setSelectMode(newValue, obj.currFrame) ;
      % newValue = itm.selectOn ;
      % if oldValue && ~newValue  % if .selectOn was true and is now false
      %   % selectedFrames = bouts2frames(itm.selectGetSelectionAsBouts());
      %   selectedFrames = find(itm.isSelectedFromFrameIndex) ;
      %   obj.selectedFrames_ = selectedFrames ;
      % end
      % obj.notify('updateTimelineProps');
      obj.notify('updateTimelineSelection');
    end

    function data = getTimelineDataForCurrentMovieAndTarget(obj)
      % Get timeline data for current movie/target
      data = obj.infoTimelineModel.getTimelineDataForCurrentMovieAndTarget(obj) ;
    end

    function tf = hasTimelinePrediction(obj)
      % Check if timeline has prediction data available
      itm = obj.infoTimelineModel ;
      tf = ismember('Predictions',itm.proptypes) && isvalid(obj.tracker);
      if tf,
        pcode = itm.props_tracker(1);
        data = obj.tracker.getPropValues(pcode);
        tf = ~isempty(data) && any(~isnan(data(:)));
      end
    end

    function setCurPropTypePredictionDefault(obj)
      % Set timeline to show prediction results if currently on default and predictions are available
      if obj.infoTimelineModel.isdefault && obj.hasTimelinePrediction()
        itm = obj.infoTimelineModel ;
        proptypei = find(strcmpi(itm.proptypes,'Predictions'),1);
        if itm.hasPredictionConfidence(),
          propi = numel(itm.props)+1;
        else
          propi = 1;
        end
        obj.setTimelineCurrentPropertyType(proptypei,propi);
      end
    end

    function data = getIsLabeledCurrMovTgt(obj)
      % Get is-labeled data for current movie/target
      % Returns: [nptsxnfrm] logical array indicating which points are labeled
      
      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      
      if isempty(iMov) || iMov==0 || ~obj.hasMovie
        data = nan(obj.nLabelPoints,1);
      else
        s = obj.labelsGTaware{iMov};       
        [p,~] = Labels.getLabelsT_full(s,iTgt,obj.nframes);
        xy = reshape(p,obj.nLabelPoints,2,obj.nframes);
        data = reshape(all(~isnan(xy),2),obj.nLabelPoints,obj.nframes);
      end
    end

    function tflbledDisp = getLabeledTgts(obj, maxTgts)
      % Get labeled targets for current movie, limited to maxTgts
      % maxTgts: maximum number of targets to display
      % Returns: [nframes x maxTgts] logical array
      
      iMov = obj.currMovie;
      if iMov==0
        tflbledDisp = nan;
        return;
      end
      tflbledDisp = obj.labelPosLabeledTgts(iMov);
      ntgtsmax = size(tflbledDisp,2);
      ntgtDisp = maxTgts;
      if ntgtsmax>=ntgtDisp 
        tflbledDisp = tflbledDisp(:,1:ntgtDisp);
      else
        tflbledDisp(:,ntgtsmax+1:ntgtDisp) = false;
      end
    end

    function addCustomTimelineFeatureGivenFileName(obj, fileName)
      % Add custom timeline feature from a .mat file
      % file: full path to .mat file containing variable 'x' with feature data
      %
      % Throws error if file cannot be loaded or doesn't contain required variable
      
      obj.infoTimelineModel.addCustomFeatureGivenFileName(fileName);
      obj.notify('updateTimelineTraces') ;
      obj.notify('updateTimelineProps');
      % obj.notify('updateTimelineSelection');
    end

    function clearBoutInTimeline(obj)
      frameIndex = obj.currFrame ;
      obj.infoTimelineModel_.clearBout(frameIndex) ;
      % obj.notify('updateTimelineProps');
      obj.notify('updateTimelineSelection');
    end    

    function clearSelectedFrames(obj)
      % obj.selectedFrames_ = [] ;
      obj.infoTimelineModel_.clearSelection(obj.nframes) ;
      % obj.notify('updateTimelineProps');
      obj.notify('updateTimelineSelection');
    end

    function result = get.infoTimelineModel(obj)
      result = obj.infoTimelineModel_ ;
    end

    function result = isCurrentFrameSelected(obj)
      currFrame = obj.currFrame ;
      isSelectedFromFrameIndex = obj.infoTimelineModel_.isSelectedFromFrameIndex ;
      if 1<=currFrame && currFrame<=numel(isSelectedFromFrameIndex)
        result = isSelectedFromFrameIndex(currFrame) ;
      else
        result = false ;
      end
    end

    function result = areAnyFramesSelected(obj)      
      isSelectedFromFrameIndex = obj.infoTimelineModel_.isSelectedFromFrameIndex ;
      result = any(isSelectedFromFrameIndex) ;
    end

    function setTimelineFramesInView(obj, nframes)
      validateattributes(nframes,{'numeric'},{'nonnegative' 'integer'});
      obj.projPrefs.InfoTimelines.FrameRadius = round(nframes/2);      
      obj.notify('update') ;
    end

    % function setTimelineCurrentPropertyTypeToDefault(obj)
    %   iproptype = 1 ;
    %   iprop = 1 ;
    %   itm = obj.infoTimelineModel ;
    %   itm.curproptype = iproptype;
    %   if iprop ~= itm.curprop,
    %     itm.curprop = iprop;
    %   end
    %   itm.isdefault = true ;
    %   obj.notify('updateTimelineTraces');
    %   obj.notify('updateTimelineLandmarkColors');
    % end
    
    function setTimelineCurrentPropertyType(obj, iproptype, iprop)
      % iproptype, iprop assumed to be consistent already.
      itm = obj.infoTimelineModel ;
      itm.setCurrentPropertyType(iproptype, iprop) ;
      obj.notify('updateTimelineTraces');
      obj.notify('updateTimelineLandmarkColors');
      obj.notify('updateTimelineProps');
      % obj.notify('updateTimelineSelection');
    end  % function    

    function popBusyStatusAndSendUpdateNotification_(obj)
      % Utility method, often called via onCleanup() in another method,
      % to clear the busy status and notify the controller it needs to update.
      obj.popBusyStatus() ;
      obj.notify('update') ;
    end  % function        

    function muckAbout_(obj)  %#ok<MANU>
      % Used while debugging to set private properties
      nop() ;
    end  % function

    function syncPropsMfahl_(obj)
      [nTgts,nPts,nRois] = obj.labelPosLabeledFramesStats();
      tfFrm = nTgts>0 | nPts>0 | nRois>0;

      nTgtsLbledFrms = nTgts(tfFrm);

      nTgtsTot = sum(nTgtsLbledFrms);

      if obj.hasMovie
        PROPS = obj.gtGetSharedProps();
        obj.(PROPS.MFAHL)(obj.currMovie) = nTgtsTot;
      end
    end  % function
    
    function setCachedAxesProperties(obj, prevAxesYDir, currAxesProps, prevAxesSizeInPixels)
      obj.prevAxesYDir_ = prevAxesYDir;
      obj.currAxesProps_ = currAxesProps;
      obj.prevAxesSizeInPixels_ = prevAxesSizeInPixels;
    end  % function
    
    function setPrevAxesModeTarget(obj)
      % Set the target animal for the 'sidekick' axes to the current labeled target.
      % This also switches the model to PrevAxesMode.FROZEN. This is what happens
      % when you click the "Freeze" button in the GUI.

      % Check for sanity
      if ~obj.hasMovie
        return
      end

      % Ask the controller to populate some of our fields with
      % values that only it knows.
      obj.notify('downdateCachedAxesProperties') ;

      % This implicitly sets to FROZEN mode
      obj.prevAxesMode_ = PrevAxesMode.FROZEN ;

      % Set the identity fields, clear pan offsets
      spec = PrevAxesTargetSpec();
      spec.iMov = obj.currMovie;
      spec.frm = obj.currFrame;
      spec.iTgt = obj.currTarget;
      spec.gtmode = obj.gtIsGTMode;
      spec.dxlim = [0 0];
      spec.dylim = [0 0];
      obj.prevAxesModeTargetSpec_ = spec;

      % Compute the full spec (image, limits, etc.)
      spec = obj.computePrevAxesTargetSpec_();
      obj.prevAxesModeTargetSpec_ = spec ;

      % Set up the virtual line and text objects
      obj.prevAxesSetFrozenLabels_(spec);

      % Fire an event to update the GUI
      obj.notify('updatePrevAxes') ;
    end

    function prevAxesMovieRemap_(obj, mIdxOrig2New)
      if ~obj.isPrevAxesModeInfoSet()
        return
      end
      newIdx = mIdxOrig2New(obj.prevAxesModeTargetSpec.iMov);
      if newIdx == 0
        obj.clearPrevAxesModeTarget();
      else
        obj.prevAxesModeTargetSpec_.iMov = newIdx;
        obj.restorePrevAxesMode();
      end
    end  % function

    function clearPrevAxesModeTarget(obj)
      obj.prevAxesModeTargetSpec_ = PrevAxesTargetSpec() ;
      obj.restorePrevAxesMode();
    end  % function

    function restorePrevAxesMode(obj)
      % Set the prevAxesMode to what is already is,
      % as a way of restoring the internal cache of the prev_axes image, lines, and
      % texts to what they should be.  This method is a hack, and should eventually
      % not be needed and go away.
      obj.setPrevAxesMode(obj.prevAxesMode_) ;
    end

    function setPrevAxesMode(obj, mode)
      % Set the mode for the 'sidekick' axes to pamode.

      % Check the mode is valid
      assert(isa(mode, 'PrevAxesMode')) ;

      % Set the relevant prop to the passed arg
      obj.prevAxesMode_ = mode ;

      % If frozen mode, rebuild the spec and labels
      if mode == PrevAxesMode.FROZEN && obj.hasMovie
        % Ask the controller to populate some of our fields with
        % values that only it knows.
        obj.notify('downdateCachedAxesProperties') ;

        % Recompute the full spec from the existing identity + dxlim/dylim
        spec = obj.computePrevAxesTargetSpec_();
        obj.prevAxesModeTargetSpec_ = spec ;

        % Set the virtual lines/texts to what they should be
        obj.prevAxesSetFrozenLabels_(spec);
      end

      % Fire an event to update the GUI
      obj.notify('updatePrevAxes') ;
    end
    
    function setPrevAxesLimits(obj, newxlim, newylim)
      assert(obj.prevAxesMode == PrevAxesMode.FROZEN) ;
      dx = newxlim - obj.prevAxesModeTargetSpec_.prevAxesProps.XLim;
      dy = newylim - obj.prevAxesModeTargetSpec_.prevAxesProps.YLim;
      obj.prevAxesModeTargetSpec_.prevAxesProps.XLim = newxlim;
      obj.prevAxesModeTargetSpec_.prevAxesProps.YLim = newylim;
      obj.prevAxesModeTargetSpec_.dxlim = obj.prevAxesModeTargetSpec_.dxlim + dx;
      obj.prevAxesModeTargetSpec_.dylim = obj.prevAxesModeTargetSpec_.dylim + dy;
    end  % function

    function setPrevAxesDirections(obj, xdir, ydir)
      obj.prevAxesModeTargetSpec_.prevAxesProps.XDir = xdir;
      obj.prevAxesModeTargetSpec_.prevAxesProps.YDir = ydir;
      obj.restorePrevAxesMode();
    end  % function

  end  % methods
end  % classdef
