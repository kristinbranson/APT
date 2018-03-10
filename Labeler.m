classdef Labeler < handle
% Bransonlab Animal Video Labeler/Tracker

  properties (Constant,Hidden)
    VERSION = '3.1';
    DEFAULT_LBLFILENAME = '%s.lbl';
    DEFAULT_CFG_FILENAME = 'config.default.yaml';
    
    % non-config props
    SAVEPROPS = { ...
      'VERSION' 'projname' ...
      'movieFilesAll' 'movieInfoAll' 'trxFilesAll' 'projMacros'...
      'movieFilesAllGT' 'movieInfoAllGT' 'trxFilesAllGT' ...
      'viewCalibrationData' 'viewCalProjWide' ...
      'viewCalibrationDataGT' ...
      'labeledpos' 'labeledpostag' 'labeledposTS' 'labeledposMarked' 'labeledpos2' ...
      'labeledposGT' 'labeledpostagGT' 'labeledposTSGT'...
      'currMovie' 'currFrame' 'currTarget' ...
      'gtIsGTMode' 'gtSuggMFTable' 'gtTblRes' ...
      'labelTemplate' ...
      'trackModeIdx' ...
      'suspScore' 'suspSelectedMFT' 'suspComputeFcn' ...
      'preProcParams' 'preProcH0' ...
      'xvResults' 'xvResultsTS' ...
      'fgEmpiricalPDF'};
    SAVEPROPS_LPOS = {...
      'labeledpos' 'nan'
      'labeledpos2' 'nan'
      'labeledposGT' 'nan'
      'labeledposTS' 'ts'
      'labeledposTSGT' 'ts'
      'labeledpostag' 'log'
      'labeledposMarked' 'log'
      'labeledpostagGT' 'log'};
    
    SAVEBUTNOTLOADPROPS = { ...
       'VERSION' 'currFrame' 'currMovie' 'currTarget'};
    
    NEIGHBORING_FRAME_MAXRADIUS = 10;
  end
  properties
    % Don't compute as a Constant prop, requires APT path initialization
    % before loading class
    NEIGHBORING_FRAME_OFFSETS;
  end
  properties (Constant,Hidden)    
    PROPS_GTSHARED = struct('reg',...
      struct('MFA','movieFilesAll',...
             'MFAF','movieFilesAllFull',...
             'MFAHL','movieFilesAllHaveLbls',...
             'MIA','movieInfoAll',...
             'TFA','trxFilesAll',...
             'TFAF','trxFilesAllFull',...
             'LPOS','labeledpos',...
             'LPOSTS','labeledposTS',...
             'LPOSTAG','labeledpostag',...
             'VCD','viewCalibrationData'),'gt',...
      struct('MFA','movieFilesAllGT',...
             'MFAF','movieFilesAllGTFull',...
             'MFAHL','movieFilesAllGTHaveLbls',...
             'MIA','movieInfoAllGT',...
             'TFA','trxFilesAllGT',...
             'TFAF','trxFilesAllGTFull',...
             'LPOS','labeledposGT',...
             'LPOSTS','labeledposTSGT',...
             'LPOSTAG','labeledpostagGT',...
             'VCD','viewCalibrationDataGT'));
  end
  
  events
    newProject
    projLoaded
    newMovie
      
    % This event is thrown immediately before .currMovie is updated (if 
    % necessary). Listeners should not rely on the value of .currMovie at
    % event time. currMovie will be subsequently updated (in the usual way) 
    % if necessary. 
    %
    % The EventData for this event is a MoviesRemappedEventData which
    % provides details on the old->new movie idx mapping.
    movieRemoved
    
    % EventData is a MoviesRemappedEventData
    moviesReordered
  end
      
  
  %% Project
  properties (SetObservable)
    projname              % init: PN
    projFSInfo;           % filesystem info
  end
  properties (SetAccess=private)
    projMacros = struct(); % scalar struct, filesys macros. init: PN
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
    % property within the cfg structure. Since the user-facing prop is
    % SetObservable, UI changes can be triggered immediately.
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
    projPrefs; % init: C
    
  end
  properties (Dependent)
    hasProject            % scalar logical
    projectfile;          % Full path to current project 
    projectroot;          % Parent dir of projectfile, if it exists
  end

  %% Movie/Video
  % Originally "Movie" referred to high-level data units/elements, eg
  % things added, removed, managed by the MovieManager etc; while "Video"
  % referred to visual details, display, playback, etc. But I sort of
  % forgot and mixed them up so that Movie sometimes applies to the latter.
  properties
    nview; % number of views. init: C
    viewNames % [nview] cellstr. init: C
    
    % States of viewCalProjWide/viewCalData:
    % .viewCalProjWide=[], .vCD=any. Here .vcPW is uninitted and .vCD is unset/immaterial.
    % .viewCalProjWide=true, .vCD=<scalar Calrig obj>. Scalar Calrig obj apples to all movies, including GT
    % .viewCalProjWide=false, .vCD=[nMovSet] cell array of calRigs. .vCD
    % applies element-wise to movies. .vCD{i} can be empty indicating unset
    % calibration object for that movie.
    viewCalProjWide % [], true, or false. init: PN
    viewCalibrationData % Opaque calibration 'useradata' for multiview. init: PN
    viewCalibrationDataGT % etc. 
    
    movieReader = []; % [1xnview] MovieReader objects. init: C
    movieInfoAll = {}; % cell-of-structs, same size as movieFilesAll
    movieInfoAllGT = {}; % same as .movieInfoAll but for GT mode
    movieDontAskRmMovieWithLabels = false; % If true, won't warn about removing-movies-with-labels    
  end
  properties (Dependent)
    movieInfoAllGTaware; 
    viewCalibrationDataGTaware % Either viewCalData or viewCalDataGT
    viewCalibrationDataCurrent % view calibration data applicable to current movie (gt aware)
  end
  properties (SetObservable)
    movieFilesAll = {}; % [nmovset x nview] column cellstr, full paths to movies; can include macros 
    movieFilesAllGT = {}; % same as .movieFilesAll but for GT mode
  end
  properties (SetObservable,AbortSet)
    movieFilesAllHaveLbls = false(0,1); % [nmovsetx1] logical. 
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
        
    movieFilesAllGTHaveLbls = false(0,1); % etc
  end
  properties (SetObservable)
    moviename; % short 'pretty' name, cosmetic purposes only. For multiview, primary movie name.
    movieCenterOnTarget = false; % scalar logical.
    movieRotateTargetUp = false;
    movieForceGrayscale = false; % scalar logical. In future could make [1xnview].
    movieFrameStepBig; % scalar positive int
    movieShiftArrowNavMode; % scalar ShiftArrowMovieNavMode
    moviePlaySegRadius; % scalar int
    moviePlayFPS; 
    movieInvert; % [1xnview] logical. If true, movie should be inverted when read. This is to compensate for codec issues where movies can be read inverted on platform A wrt platform B
    
    movieIsPlaying = false;
  end
  properties (SetAccess=private) 
    movieCache = []; % containers.Map. Transient. Keys: standardized
      % fullpath, el of .movieFilesAllFull or .movieFilesAllGTFull. 
      % vals: MovieReader objects with that movie open
  end  
  properties (Dependent)
    isMultiView;
    movieFilesAllFull; % like movieFilesAll, but macro-replaced and platformized
    movieFilesAllGTFull; % etc
    movieFilesAllFullGTaware;
    movieFilesAllHaveLblsGTaware;
    hasMovie;
    moviefile;
    nframes;
    movienr; % [nview]
    movienc; % [nview]
    nmovies;
    nmoviesGT;
    nmoviesGTaware;
    moviesSelected; % [nSel] vector of movie indices currently selected in MovieManager
  end
  
  %% Trx
  properties (SetObservable)
    trxFilesAll = {};  % column cellstr, full paths to trxs. Same size as movieFilesAll.
    trxFilesAllGT = {}; % etc. Same size as movieFilesAllGT.
  end
  properties (SetAccess=private)
    trxCache = [];            % containers.Map. Keys: fullpath. vals: lazy-loaded structs with fields: .trx and .frm2trx
    trx = [];                 % trx object
    frm2trx = [];             % nFrm x nTrx logical. frm2trx(iFrm,iTrx) is true if trx iTrx is live on frame iFrm (for current movie)
  end
  properties (Dependent,SetObservable)
    targetZoomRadiusDefault;
  end
  properties (Dependent)
    trxFilesAllFull % like .movieFilesAllFull, but for .trxFilesAll
    trxFilesAllGTFull % etc
    trxFilesAllFullGTaware
    hasTrx
    currTrx
    nTrx
    nTargets % nTrx, or 1 if no Trx
  end
  
  %% ShowTrx
  properties (SetObservable)
    showTrx;                  % true to show trajectories
    showTrxCurrTargetOnly;    % if true, plot only current target
    showTrxIDLbl;             % true to show id label 
  end
  properties
    hTraj;                    % nTrx x 1 vector of line handles
    hTrx;                     % nTrx x 1 vector of line handles
    hTrxTxt;                  % nTrx x 1 vector of text handles
    showTrxPreNFrm = 15;      % number of preceding frames to show in traj
    showTrxPostNFrm = 5;      % number of following frames to show in traj
  end
  
  %% Labeling
  properties (SetObservable)
    labelMode;            % scalar LabelMode. init: C
    labeledpos;           % column cell vec with .nmovies elements. labeledpos{iMov} is npts x 2 x nFrm(iMov) x nTrx(iMov) double array; labeledpos{1}(:,1,:,:) is X-coord, labeledpos{1}(:,2,:,:) is Y-coord. init: PN
    % Multiview. Right now all 3d pts must live in all views, eg
    % .nLabelPoints=nView*NumLabelPoints. first dim of labeledpos is
    % ordered as {pt1vw1,pt2vw1,...ptNvw1,pt1vw2,...ptNvwK}
    labeledposTS;         % labeledposTS{iMov} is nptsxnFrm(iMov)xnTrx(iMov). It is the last time .labeledpos or .labeledpostag was touched. init: PN
    labeledposMarked;     % labeledposMarked{iMov} is a nptsxnFrm(iMov)xnTrx(iMov) logical array. Elements are set to true when the corresponding pts have their labels set; users can set elements to false at random. init: PN
    labeledpostag;        % column cell vec with .nmovies elements. labeledpostag{iMov} is npts x nFrm(iMov) x nTrx(iMov) logical indicating *occludedness*. ("tag" for legacy reasons) init: PN
    
    labeledpos2;          % identical size/shape with labeledpos. aux labels (eg predicted, 2nd set, etc). init: PN
    labels2Hide;      % scalar logical
    
    labeledposGT          % like .labeledpos
    labeledposTSGT        % like .labeledposTS
    labeledpostagGT       % like .labeledpostag
  end
  properties % make public setaccess
    labelPointsPlotInfo;  % struct containing cosmetic info for labelPoints. init: C
  end
  properties (SetAccess=private)
    nLabelPoints;         % scalar integer. This is the total number of 2D labeled points across all views. Contrast with nPhysPoints. init: C
    labelTemplate;    
    
    labeledposIPtSetMap;  % [nptsets x nview] 3d 'point set' identifications. labeledposIPtSetMap(iSet,:) gives
                          % point indices for set iSet in various views. init: C
    labeledposSetNames;   % [nptsets] cellstr names labeling rows of .labeledposIPtSetMap.
                          % NOTE: arguably the "point names" should be. init: C
    labeledposIPt2View;   % [npts] vector of indices into 1:obj.nview. Convenience prop, derived from .labeledposIPtSetMap. init: C
    labeledposIPt2Set;    % [npts] vector of set indices for each point. Convenience prop. init: C
  end
  properties (SetObservable)
    labeledposNeedsSave;  % scalar logical, .labeledpos has been touched since last save. Currently does NOT account for labeledpostag
  end
  properties (Dependent)
    labeledposGTaware;
    labeledposTSGTaware;
    labeledpostagGTaware;
    labeledposCurrMovie;
    labeledpostagCurrMovie;
    
    nPhysPoints; % number of physical/3D points
  end
  properties (SetObservable)
    lblCore; % init: L
  end
  properties
    labeledpos2_ptsH;     % [npts]
    labeledpos2_ptsTxtH;  % [npts]    
    lblOtherTgts_ptsH;    % [npts]
  end
  
  properties
    fgEmpiricalPDF % struct containing empirical FG pdf and metadata
  end
  
  %% GT mode
  properties (SetObservable,SetAccess=private)
    gtIsGTMode % scalar logical
  end
  properties
    gtSuggMFTable % [nGTSugg x ncol] MFTable for suggested frames to label. .mov values are MovieIndexes
    gtSuggMFTableLbled % [nGTSuggx1] logical flags indicating whether rows of .gtSuggMFTable were gt-labeled

    gtTblRes % [nGTcomp x ncol] table, or []. Most recent GT performance results. 
      % gtTblRes(:,MFTable.FLDSID) need not match
      % gtSuggMFTable(:,MFTable.FLDSID) because eg GT performance can be 
      % computed even if some suggested frames are not be labeled.
  end
  properties (Dependent)
    gtNumSugg % height(gtSuggMFTable)
  end
  events
    % Instead of making gtSuggMFTable* SetObservable, we use these events.
    % The two variables are coupled (number of rows must be equal,
    % so updating them is a nonatomic (2-step) process. Listeners directly
    % listening to property sets will sometimes see inconsistent state.
    
    gtIsGTModeChanged 
    gtSuggUpdated % general update occurred of gtSuggMFTable*
    gtSuggMFTableLbledUpdated % incremental update of gtSuggMFTableLbled occurred
    gtResUpdated % update of GT performance results occurred
  end
  
  
  %% Suspiciousness
  properties (SetObservable,SetAccess=private)
    suspScore; % column cell vec same size as labeledpos. suspScore{iMov} is nFrm(iMov) x nTrx(iMov)
    suspSelectedMFT; % MFT table of selected suspicous frames.
    suspComputeFcn; 
    % Function with sig [score,tblMFT,diagstr]=fcn(labelerObj) that 
    % computes suspScore, suspSelectedMFT.
    % See .suspScore for required size/dims of suspScore and contents.
    % diagstr is arbitrary diagnostic info (assumed char for now).
    
    suspDiag; % Transient "userdata", diagnostic output from suspComputeFcn
    
%     currSusp; % suspScore for current mov/frm/tgt. Can be [] indicating 'N/A'
    %     suspNotes; % column cell vec same size as labeledpos. suspNotes{iMov} is a nFrm x nTrx column cellstr
  end
  
  %% PreProc
  properties
    preProcParams % struct
    preProcH0 % image hist used in current preProcData. Conceptually, this is a preProcParam that APT updates from movies
    preProcData % for CPR, a CPRData; for DL trackers, likely a tracker-specific object pointing to data on disk
    preProcDataTS % scalar timestamp  
  end
  
  %% Tracking
  properties (SetObservable)    
    tracker % LabelTracker object. init: PLPN
    trackModeIdx % index into MFTSetEnum.TrackingMenu* for current trackmode. 
     %Note MFTSetEnum.TrackingMenuNoTrx==MFTSetEnum.TrackingMenuTrx(1:K).
     %Values of trackModeIdx 1..K apply to either the NoTrx or Trx cases; 
     %larger values apply only the Trx case.
    trackNFramesSmall % small/fine frame increment for tracking. init: C
    trackNFramesLarge % big/coarse ". init: C
    trackNFramesNear % neighborhood radius. init: C
  end
  
  %% CrossValidation
  properties
    xvResults % table of most recent cross-validation results. This table
      % has a row for every labeled frame present at the time xvalidation 
      % was run. So it should be fairly explicit if/when it is out-of-date 
      % relative to the project (eg when labels are added or removed)
    xvResultsTS % timestamp for xvResults
  end
  
  %% Prev
  properties
    prevIm = []; % single array of image data ('primary' view only)
    prevAxesMode; % scalar PrevAxesMode
    prevAxesModeInfo; % "userdata" for .prevAxesMode
    lblPrev_ptsH; % [npts] gobjects. init: L
    lblPrev_ptsTxtH; % [npts] etc. init: L
  end
  
  %% Misc
  properties (SetObservable, AbortSet)
    prevFrame = nan;      % last previously VISITED frame
    currTarget = nan;     % always 1 if proj doesn't have trx
    
    currImHud; % scalar AxisHUD object TODO: move to LabelerGUI. init: C
  end
  properties (SetObservable)
    keyPressHandlers; % [nhandlerx1] cell array of LabelerKeyEventHandlers.
  end
  properties (AbortSet)
    currMovie; % idx into .movieFilesAll (row index, when obj.multiView is true), or .movieFilesAllGT when .gtIsGTmode is on
    % Don't observe this, listen to 'newMovie'
  end
  properties (Dependent)
    currMovIdx; % scalar MovieIndex
  end
  properties (SetObservable)
    currFrame = 1; % current frame
  end
  properties 
    currIm = [];            % [nview] cell vec of image data. init: C
    selectedFrames = [];    % vector of frames currently selected frames; typically t0:t1
    hFig; % handle to main LabelerGUI figure
  end
  properties (SetAccess=private)
    isinit = false;         % scalar logical; true during initialization, when some invariants not respected
  end
  properties (Dependent)
    gdata; % handles structure for LabelerGUI
  end

  
  %% Prop access
  methods % dependent prop getters
    function v = get.viewCalibrationDataGTaware(obj)
      if obj.gtIsGTMode
        v = obj.viewCalibrationDataGT;
      else
        v = obj.viewCalibrationData;
      end      
    end
    function v = get.viewCalibrationDataCurrent(obj)
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
    function v = get.isMultiView(obj)
      v = obj.nview>1;
    end
    function v = get.movieFilesAllFull(obj)
      % See also .projLocalizePath()
      sMacro = obj.projMacros;
%       if ~isfield(sMacro,'projdir')
%         % This conditional allows user to explictly specify project root
%         % Seems questionable
%         sMacro.projdir = obj.projectroot;
%       end
      v = FSPath.fullyLocalizeStandardize(obj.movieFilesAll,sMacro);
      FSPath.warnUnreplacedMacros(v);
    end
    function v = get.movieFilesAllGTFull(obj)
      sMacro = obj.projMacros;
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
    function v = getMovieFilesAllFullMovIdx(obj,mIdx)
      % mIdx: MovieIndex vector
      % v: [numel(mIdx)xnview] movieFilesAllFull/GT 
      
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
    function v = get.trxFilesAllFull(obj)
      v = Labeler.trxFilesLocalize(obj.trxFilesAll,obj.movieFilesAllFull);
    end    
    function v = get.trxFilesAllGTFull(obj)
      v = Labeler.trxFilesLocalize(obj.trxFilesAllGT,obj.movieFilesAllGTFull);
    end
    function v = get.trxFilesAllFullGTaware(obj)
      if obj.gtIsGTMode
        v = obj.trxFilesAllGTFull;
      else
        v = obj.trxFilesAllFull;
      end
    end
    function v = getTrxFilesAllFullMovIdx(obj,mIdx)
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.trxFilesAllGTFull(iMov,:);
      else
        v = obj.trxFilesAllFull(iMov,:);
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
    function v = get.movienr(obj)
      mr = obj.movieReader;
      if mr(1).isOpen
        v = [mr.nr]';
      else
        v = nan(obj.nview,1);
      end
    end
    function v = get.movienc(obj)
      mr = obj.movieReader;
      if mr(1).isOpen
        v = [mr.nc]';          
      else
        v = nan(obj.nview,1);
      end
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
    function v = get.moviesSelected(obj) %#%GUIREQ
      % Find MovieManager in LabelerGUI
      handles = obj.gdata;
      if isfield(handles,'movieMgr')
        mmc = handles.movieMgr;
      else
        mmc = [];
      end
      if ~isempty(mmc) && isvalid(mmc)
        v = mmc.getSelectedMovies();
      else
        error('Labeler:getMoviesSelected',...
          'Cannot access Movie Manager. Make sure your desired movies are selected in the Movie Manager.');
      end
    end
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
%     function v = get.currTrxID(obj)
%       if obj.hasTrx
%         v = obj.trx(obj.currTarget).id;
%       else
%         v = nan;
%       end
%     end
    function v = get.nTrx(obj)
      v = numel(obj.trx);
    end
    function v = getnTrxMovIdx(obj,mIdx)
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      PROPS = obj.gtGetSharedPropsStc(gt);
      tfaf = obj.(PROPS.TFAF){iMov,1};
      if isempty(tfaf)
        v = 1;
      else
        nfrm = obj.(PROPS.MIA){iMov,1}.nframes;
        trxI = obj.getTrx(tfaf,nfrm);
        v = numel(trxI);
      end
    end
    function v = get.nTargets(obj)
      if obj.hasTrx
        v = obj.nTrx;
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
      v = size(obj.movieFilesAll,2)>0;
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
    function v = getLabeledPosMovIdx(obj,mIdx)
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.labeledposGT{iMov};
      else 
        v = obj.labeledpos{iMov};
      end
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
    function v = get.labeledposCurrMovie(obj)
      if obj.currMovie==0
        v = [];
      elseif obj.gtIsGTMode
        v = obj.labeledposGT{obj.currMovie};
      else
        v = obj.labeledpos{obj.currMovie};
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
    function v = get.nPhysPoints(obj)
      v = size(obj.labeledposIPtSetMap,1);
    end
    function v = get.currMovIdx(obj)
      v = MovieIndex(obj.currMovie,obj.gtIsGTMode);
    end
    function v = get.gdata(obj)
      v = guidata(obj.hFig);
    end
    function v = get.gtNumSugg(obj)
      v = height(obj.gtSuggMFTable);
    end
  end
  
  methods % prop access
    % CONSIDER get rid of setter, use listeners
    function set.labeledpos(obj,v)
      obj.labeledpos = v;
      if ~obj.isinit %#ok<MCSUP> 
        obj.updateTrxTable();
        obj.updateFrameTableIncremental(); 
      end
    end
    function set.labeledposGT(obj,v)
      obj.labeledposGT = v;
      if ~obj.isinit %#ok<MCSUP> 
        obj.updateTrxTable();
        obj.updateFrameTableIncremental();
        obj.gtUpdateSuggMFTableLbledIncremental();
      end
    end
    function set.movieForceGrayscale(obj,v)
      assert(isscalar(v) && islogical(v));
      [obj.movieReader.forceGrayscale] = deal(v); %#ok<MCSUP>
      obj.movieForceGrayscale = v;
    end
    function set.movieInvert(obj,v)
      assert(islogical(v) && numel(v)==obj.nview); %#ok<MCSUP>
      mrs = obj.movieReader; %#ok<MCSUP>
      for i=1:obj.nview %#ok<MCSUP>
        mrs(i).flipVert = v(i);
      end
      obj.movieInvert = v;
    end
    function set.movieCenterOnTarget(obj,v)
      obj.movieCenterOnTarget = v;
      if ~v && obj.movieRotateTargetUp %#ok<MCSUP>
        obj.movieRotateTargetUp = false; %#ok<MCSUP>
      end
      if v
        if obj.hasTrx %#ok<MCSUP>
          obj.videoCenterOnCurrTarget();
        elseif ~obj.isinit %#ok<MCSUP>
          warningNoTrace('Labeler:trx',...
            'The current movie does not have an associated trx file. Property ''movieCenterOnTarget'' will have no effect.');
        end
      end
    end
    function set.movieRotateTargetUp(obj,v)
      if v && ~obj.movieCenterOnTarget %#ok<MCSUP>
        %warningNoTrace('Labeler:prop','Setting .movieCenterOnTarget to true.');
        obj.movieCenterOnTarget = true; %#ok<MCSUP>
      end
      obj.movieRotateTargetUp = v;
      if obj.hasTrx && obj.movieCenterOnTarget %#ok<MCSUP>
        obj.videoCenterOnCurrTarget();
      end
      if v
        if ~obj.hasTrx && ~obj.isinit %#ok<MCSUP>
          warningNoTrace('Labeler:trx',...
            'The current movie does not have an associated trx file. Property ''movieRotateTargetUp'' will have no effect.');
        end
      end
    end
    function set.targetZoomRadiusDefault(obj,v)
      obj.projPrefs.Trx.ZoomFactorDefault = v;
    end
  end
  
  %% Ctor/Dtor
  methods 
  
    function obj = Labeler(varargin)
      % lObj = Labeler();
      
      if exist('moveMenuItemAfter','file')==0 || ...
         exist('ReadYaml','file')==0
       fprintf('Configuring your path ...\n');
       APT.setpath;
      end
      obj.NEIGHBORING_FRAME_OFFSETS = ...
                  neighborIndices(Labeler.NEIGHBORING_FRAME_MAXRADIUS);

      obj.hFig = LabelerGUI(obj);
    end
     
    function delete(obj)
      if isvalid(obj.hFig)
        close(obj.hFig);
        obj.hFig = [];
      end
    end
    
  end
  
        
  %% Configurations
  methods (Hidden)

  % Property init legend
  % IFC: property initted during initFromConfig()
  % PNPL: property initted during projectNew() or projLoad()
  % L: property initted during labelingInit()
  % (todo) TI: property initted during trackingInit()
  %
  % There are only two ways to start working on a project.
  % 1. New/blank project: initFromConfig(), then projNew().
  % 2. Existing project: projLoad(), which is (initFromConfig(), then
  % property-initialization-equivalent-to-projNew().)
  
    function initFromConfig(obj,cfg)
      % Note: Config must be modernized
    
      % Views
      obj.nview = cfg.NumViews;
      if isempty(cfg.ViewNames)
        obj.viewNames = arrayfun(@(x)sprintf('view%d',x),1:obj.nview,'uni',0);
      else
        if numel(cfg.ViewNames)~=obj.nview
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
      obj.labeledposSetNames = setnames;
      
      obj.labelMode = LabelMode.(cfg.LabelMode);
      
      lpp = cfg.LabelPointsPlot;
      % Some convenience mods to .LabelPointsPlot
      if (~isfield(lpp,'Colors') || size(lpp.Colors,1)~=npts) && isfield(lpp,'ColorMapName') 
        lpp.Colors = feval(lpp.ColorMapName,npts);
      end
      if ~isfield(lpp,'ColorsSets') || size(lpp.ColorsSets,1)~=nSet
        if isfield(lpp,'ColorMapName')
          cmapName = lpp.ColorMapName;
        else
          cmapName = 'parula';
        end
        lpp.ColorsSets = feval(cmapName,nSet);
      end      
      obj.labelPointsPlotInfo = lpp;
            
      obj.trackNFramesSmall = cfg.Track.PredictFrameStep;
      obj.trackNFramesLarge = cfg.Track.PredictFrameStepBig;
      obj.trackNFramesNear = cfg.Track.PredictNeighborhood;
      obj.trackModeIdx = 1;
      cfg.Track = rmfield(cfg.Track,...
        {'PredictFrameStep' 'PredictFrameStepBig' 'PredictNeighborhood'});
                  
      arrayfun(@delete,obj.movieReader);
      obj.movieReader = [];
      for i=obj.nview:-1:1
        mr(1,i) = MovieReader;
      end
      obj.movieReader = mr;
      obj.currIm = cell(obj.nview,1);
      delete(obj.currImHud);
      gd = obj.gdata;
      obj.currImHud = AxisHUD(gd.axes_curr.Parent); 
      %obj.movieSetNoMovie();
      
      obj.movieForceGrayscale = logical(cfg.Movie.ForceGrayScale);
      obj.movieFrameStepBig = cfg.Movie.FrameStepBig;
      obj.movieShiftArrowNavMode = ShiftArrowMovieNavMode.(cfg.Movie.ShiftArrowNavMode);
      obj.moviePlaySegRadius = cfg.Movie.PlaySegmentRadius;
      obj.moviePlayFPS = cfg.Movie.PlayFPS;
           
      fldsRm = intersect(fieldnames(cfg),...
        {'NumViews' 'ViewNames' 'NumLabelPoints' 'LabelPointNames' ...
        'LabelMode' 'LabelPointsPlot' 'ProjectName' 'Movie'});
      obj.projPrefs = rmfield(cfg,fldsRm);
      % A few minor subprops of projPrefs have explicit props

      obj.notify('newProject');

      % order important: this needs to occur after 'newProject' event so
      % that figs are set up. (names get changed)
      movInvert = ViewConfig.getMovieInvert(cfg.View);
      obj.movieInvert = movInvert;
      obj.movieCenterOnTarget = cfg.View(1).CenterOnTarget;
      obj.movieRotateTargetUp = cfg.View(1).RotateTargetUp;
 
      % maybe useful to clear/reinit and shouldn't hurt
      obj.movieCache = containers.Map(); 
      
      obj.preProcInit();
      
      % For unclear reasons, creation of new tracker occurs downstream in
      % projLoad() or projNew()
      if ~isempty(obj.tracker)
        delete(obj.tracker);
        obj.tracker = [];
      end
      
      obj.showTrx = cfg.Trx.ShowTrx;
      obj.showTrxCurrTargetOnly = cfg.Trx.ShowTrxCurrentTargetOnly;
      obj.showTrxIDLbl = cfg.Trx.ShowTrxIDLbl;
      
      obj.labels2Hide = false;

      % New projs must start with LASTSEEN as there is nothing to freeze
      % yet. projLoad() will further set any loaded info
      obj.setPrevAxesMode(PrevAxesMode.LASTSEEN,[]);
      
      % maybe useful to clear/reinit and shouldn't hurt
      obj.trxCache = containers.Map();
      
      RC.saveprop('lastProjectConfig',obj.getCurrentConfig());
    end
    
    function cfg = getCurrentConfig(obj)
      % cfg is modernized

      cfg = obj.projPrefs;
      
      cfg.NumViews = obj.nview;
      cfg.ViewNames = obj.viewNames;
      cfg.NumLabelPoints = obj.nPhysPoints;
      cfg.LabelPointNames = obj.labeledposSetNames;
      cfg.LabelMode = char(obj.labelMode);

      % View stuff: read off current state of axes
      gd = obj.gdata;
      viewCfg = ViewConfig.readCfgOffViews(gd.figs_all,gd.axes_all,gd.axes_prev);
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
        'PlaySegmentRadius',obj.moviePlaySegRadius,...
        'PlayFPS',obj.moviePlayFPS);

      cfg.LabelPointsPlot = obj.labelPointsPlotInfo;
      cfg.Trx.ShowTrx = obj.showTrx;
      cfg.Trx.ShowTrxCurrentTargetOnly = obj.showTrxCurrTargetOnly;
      cfg.Trx.ShowTrxIDLbl = obj.showTrxIDLbl;
      cfg.Track.PredictFrameStep = obj.trackNFramesSmall;
      cfg.Track.PredictFrameStepBig = obj.trackNFramesLarge;
      cfg.Track.PredictNeighborhood = obj.trackNFramesNear;
      
      cfg.PrevAxes.Mode = char(obj.prevAxesMode);
      cfg.PrevAxes.ModeInfo = obj.prevAxesModeInfo;
    end
    
  end
    
  methods (Static)
    
    function cfg = cfgGetLastProjectConfigNoView
      cfgBase = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
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
    
    function cfg = cfgModernize(cfg)
      % Bring a cfg up-to-date with latest by adding in any new fields from
      % config.default.yaml.
      
      cfgBase = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      
      cfg = structoverlay(cfgBase,cfg,'dontWarnUnrecog',true,...
        'allowedUnrecogFlds',{'Colors' 'ColorsSets'});
      view = augmentOrTruncateVector(cfg.View,cfg.NumViews);
      cfg.View = view(:);
    end
    
    function cfg = cfgDefaultOrder(cfg)
      % Reorder fields of cfg struct to default order
      
      cfg0 = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      flds0 = fieldnames(cfg0);
      flds = fieldnames(cfg);
      flds0 = flds0(ismember(flds0,flds)); 
      fldsExtra = setdiff(flds,flds0);
      cfg = orderfields(cfg,[flds0(:);fldsExtra(:)]);
    end
    
  end
  
  %% Project/Lbl files
  methods
   
    function projNew(obj,name)
      % Create new project based on current configuration
      
      if exist('name','var')==0
        resp = inputdlg('Project name:','New Project');
        if isempty(resp)
          return;
        end
        name = resp{1};
      end

      obj.isinit = true;

      obj.projname = name;
      obj.projFSInfo = [];
      obj.movieFilesAll = cell(0,obj.nview);
      obj.movieFilesAllGT = cell(0,obj.nview);
      obj.movieFilesAllHaveLbls = false(0,1);
      obj.movieFilesAllGTHaveLbls = false(0,1);
      obj.movieInfoAll = cell(0,obj.nview);
      obj.movieInfoAllGT = cell(0,obj.nview);
      obj.trxFilesAll = cell(0,obj.nview);
      obj.trxFilesAllGT = cell(0,obj.nview);
      obj.projMacros = struct();
      obj.viewCalProjWide = [];
      obj.viewCalibrationData = [];
      obj.viewCalibrationDataGT = [];
      obj.labelTemplate = []; % order important here
      obj.movieSetNoMovie(); % order important here
      obj.labeledpos = cell(0,1);
      obj.labeledposGT = cell(0,1);
      obj.labeledposTS = cell(0,1);
      obj.labeledposTSGT = cell(0,1);
      obj.labeledposMarked = cell(0,1);
      obj.labeledpostag = cell(0,1);
      obj.labeledpostagGT = cell(0,1);
      obj.labeledpos2 = cell(0,1);
      obj.gtIsGTMode = false;
      obj.gtSuggMFTable = MFTable.emptyTable(MFTable.FLDSID);
      obj.gtSuggMFTableLbled = false(0,1);
      obj.gtTblRes = [];
      
      obj.isinit = false;
      
      obj.updateFrameTableComplete();  
      obj.labeledposNeedsSave = false;
      
      if ~isempty(obj.tracker)
        % the old tracker might be from a loaded proj etc and might not
        % match prefs.
        delete(obj.tracker);
        obj.tracker = [];
      end
      trkPrefs = obj.projPrefs.Track;
      if trkPrefs.Enable
        obj.tracker = feval(trkPrefs.Type,obj);
        obj.tracker.init();
      end

      props = obj.gdata.propsNeedInit;
      for p = props(:)', p=p{1}; %#ok<FXSET>
        obj.(p) = obj.(p);
      end
      obj.notify('gtIsGTModeChanged');
    end
      
    function projSaveRaw(obj,fname)
      s = obj.projGetSaveStruct();
      save(fname,'-mat','-struct','s');

      obj.labeledposNeedsSave = false;
      obj.projFSInfo = ProjectFSInfo('saved',fname);

      RC.saveprop('lastLblFile',fname);
    end
        
    function [success,lblfname] = projSaveAs(obj)
      % Saves a .lbl file, prompting user for filename.

      if ~isempty(obj.projectfile)
        filterspec = obj.projectfile;
      else
        % Guess a path/location for save
        lastLblFile = RC.getprop('lastLblFile');
        if isempty(lastLblFile)
          if obj.hasMovie
            savepath = fileparts(obj.moviefile);
          else
            savepath = pwd;          
          end
        else
          savepath = fileparts(lastLblFile);
        end

        if ~isempty(obj.projname)
          projfile = sprintf(obj.DEFAULT_LBLFILENAME,obj.projname);
        else
          projfile = sprintf(obj.DEFAULT_LBLFILENAME,'APTProject');
        end
        filterspec = fullfile(savepath,projfile);
      end
      
      [lblfname,pth] = uiputfile(filterspec,'Save label file');
      if isequal(lblfname,0)
        lblfname = [];
        success = false;
      else
        lblfname = fullfile(pth,lblfname);
        success = true;
        obj.projSaveRaw(lblfname);
      end      
    end
    
    function [success,lblfname] = projSaveSmart(obj)
      % Try to save to current project; if there is no project, do a saveas
      lblfname = obj.projectfile;
      if isempty(lblfname)
        [success,lblfname] = obj.projSaveAs();
      else
        success = true;
        obj.projSaveRaw(lblfname);
      end
    end
    
    function s = projGetSaveStruct(obj,varargin)
      sparsify = myparse(varargin,...
        'sparsify',true);
      
      s = struct();
      s.cfg = obj.getCurrentConfig();
      
      if sparsify
        lposProps = obj.SAVEPROPS_LPOS(:,1);
        lposPropsType = obj.SAVEPROPS_LPOS(:,2);
        
        for f=obj.SAVEPROPS, f=f{1}; %#ok<FXSET>
          iLpos = find(strcmp(f,lposProps));
          if isempty(iLpos)
            s.(f) = obj.(f);
          else
            lpostype = lposPropsType{iLpos};
            xarrFull = obj.(f);
            assert(iscell(xarrFull));
            xarrSprs = cellfun(@(x)SparseLabelArray.create(x,lpostype),...
              xarrFull,'uni',0);
            s.(f) = xarrSprs;
          end
        end
      else
        for f=obj.SAVEPROPS, f=f{1}; %#ok<FXSET>
          s.(f) = obj.(f);
        end
      end
      
      switch obj.labelMode
        case LabelMode.TEMPLATE
          s.labelTemplate = obj.lblCore.getTemplate();
      end
      
      tObj = obj.tracker;
      if isempty(tObj)
        s.trackerClass = '';
        s.trackerData = [];
      else
        s.trackerClass = class(tObj);
        s.trackerData = tObj.getSaveToken();
      end        
    end
    
    function currMovInfo = projLoad(obj,fname,varargin)
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
            
      nomovie = myparse(varargin,...
        'nomovie',false ... % If true, call movieSetNoMovie() instead of movieSet(currMovie)
        );
      
      currMovInfo = [];
      
      if exist('fname','var')==0
        lastLblFile = RC.getprop('lastLblFile');
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
            
      s = load(fname,'-mat');
      if ~all(isfield(s,{'VERSION' 'labeledpos'}))
        error('Labeler:load','Unexpected contents in Label file.');
      end
      RC.saveprop('lastLblFile',fname);

      s = Labeler.lblModernize(s);
      
      obj.isinit = true;

      obj.initFromConfig(s.cfg);
      
      % From here to the end of this method is a parallel initialization to
      % projNew()

      LOADPROPS = Labeler.SAVEPROPS(~ismember(Labeler.SAVEPROPS,...
                                              Labeler.SAVEBUTNOTLOADPROPS));
      lposProps = obj.SAVEPROPS_LPOS(:,1);
      for f=LOADPROPS(:)',f=f{1}; %#ok<FXSET>
        if isfield(s,f)          
          if ~any(strcmp(f,lposProps))
            obj.(f) = s.(f);
          else
            val = s.(f);
            assert(iscell(val));
            for iMov=1:numel(val)
              x = val{iMov};
              if isstruct(x)
                xfull = SparseLabelArray.full(x);
                val{iMov} = xfull;
              end
            end
            obj.(f) = val;
          end
        else
          warningNoTrace('Labeler:load','Missing load field ''%s''.',f);
          %obj.(f) = [];
        end
      end
     
      fcnAnyNonNan = @(x)any(~isnan(x(:)));
      obj.movieFilesAllHaveLbls = cellfun(fcnAnyNonNan,obj.labeledpos);
      obj.movieFilesAllGTHaveLbls = cellfun(fcnAnyNonNan,obj.labeledposGT);      
      obj.gtUpdateSuggMFTableLbledComplete();
      obj.isinit = false;
      
      % need this before setting movie so that .projectroot exists
      obj.projFSInfo = ProjectFSInfo('loaded',fname);

      % Tracker.
      if isempty(obj.tracker)
        tClsOld = '';
      else
        tClsOld = class(obj.tracker);
        delete(obj.tracker);
        obj.tracker = [];
      end
      % obj.tracker is always empty now
      if ~isempty(s.trackerClass)
        tCls = s.trackerClass;
        if exist(tCls,'class')==0
          error('Labeler:projLoad',...
            'Project tracker class ''%s'' cannot be found.',tCls);
        end
        if ~isempty(tClsOld) && ~strcmp(tClsOld,tCls)
          warning('Labeler:projLoad',...
            'Project tracker class ''%s'' will differ from current tracker class ''%s''.',...
            tCls,tClsOld);
        end
          
        tObjNew = feval(tCls,obj);
        tObjNew.init();
        obj.tracker = tObjNew;
      end
      if ~isempty(obj.tracker)
        fprintf(1,'Loading tracker info: %s.\n',tCls);
        obj.tracker.loadSaveToken(s.trackerData);
      end

      if obj.nmoviesGTaware==0 || s.currMovie==0 || nomovie
        obj.movieSetNoMovie();
      else
        [tfok,badfile] = obj.movieCheckFilesExistSimple(s.currMovie,s.gtIsGTMode);
        if ~tfok
          currMovInfo.iMov = s.currMovie;
          currMovInfo.badfile = badfile;
          obj.movieSetNoMovie();
        else
          obj.movieSet(s.currMovie);
          obj.setFrameAndTarget(s.currFrame,s.currTarget);
        end
      end
      
%       % Needs to occur after tracker has been set up so that labelCore can
%       % communicate with tracker if necessary (in particular, Template Mode 
%       % <-> Hide Predictions)
%       obj.labelingInit();

      obj.labeledposNeedsSave = false;
%       obj.suspScore = obj.suspScore;
            
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI
      
      if obj.currMovie>0
        obj.labelsUpdateNewFrame(true);
      end
      
      % This needs to occur after .labeledpos etc has been set
      pamode = PrevAxesMode.(s.cfg.PrevAxes.Mode);
      obj.setPrevAxesMode(pamode,s.cfg.PrevAxes.ModeInfo);
      
      props = obj.gdata.propsNeedInit;
      for p = props(:)', p=p{1}; %#ok<FXSET>
        obj.(p) = obj.(p);
      end
      
      obj.notify('projLoaded');
      obj.notify('gtIsGTModeChanged');
      obj.notify('gtSuggUpdated');
      obj.notify('gtResUpdated');
    end
    
    function projImport(obj,fname)
      % 'Import' the project fname, MERGING movies/labels into the current project.
          
      assert(false,'Unsupported');
      
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
        
      s = load(fname,'-mat');
      if s.nLabelPoints~=obj.nLabelPoints
        error('Labeler:projImport','Project %s uses nLabelPoints=%d instead of %d for the current project.',...
          fname,s.nLabelPoints,obj.nLabelPoints);
      end
      
      assert(~obj.isMultiView && iscolumn(s.movieFilesAll));
      
      if isfield(s,'projMacros') && ~isfield(s.projMacros,'projdir')
        s.projMacros.projdir = fileparts(fname);
      else
        s.projMacros = struct();
      end
      
      nMov = size(s.movieFilesAll,1);
      for iMov = 1:nMov
        movfile = s.movieFilesAll{iMov,1};
        movfileFull = Labeler.platformize(FSPath.macroReplace(movfile,s.projMacros));
        movifo = s.movieInfoAll{iMov,1};
        trxfl = s.trxFilesAll{iMov,1};
        lpos = s.labeledpos{iMov};
        lposTS = s.labeledposTS{iMov};
        lpostag = s.labeledpostag{iMov};
        if isempty(s.suspScore)
          suspscr = [];
        else
          suspscr = s.suspScore{iMov};
        end
        
        if exist(movfileFull,'file')==0 || ~isempty(trxfl)&&exist(trxfl,'file')==0
          warning('Labeler:projImport',...
            'Missing movie/trxfile for movie ''%s''. Not importing this movie.',...
            movfileFull);
          continue;
        end
           
        obj.movieFilesAll{end+1,1} = movfileFull;
        obj.movieFilesAllHaveLbls(end+1,1) = any(~isnan(lpos(:)));
        obj.movieInfoAll{end+1,1} = movifo;
        obj.trxFilesAll{end+1,1} = trxfl;
        obj.labeledpos{end+1,1} = lpos;
        obj.labeledposTS{end+1,1} = lposTS;
        obj.labeledposMarked{end+1,1} = false(size(lposTS));
        obj.labeledpostag{end+1,1} = lpostag;
        obj.labeledpos2{end+1,1} = s.labeledpos2{iMov};
        if ~isempty(obj.suspScore)
          obj.suspScore{end+1,1} = suspscr;
        end
%         if ~isempty(obj.suspNotes)
%           obj.suspNotes{end+1,1} = [];
%         end
      end

      obj.labeledposNeedsSave = true;
      obj.projFSInfo = ProjectFSInfo('imported',fname);
      
      % XXX prob would need .preProcInit() here
      
      if ~isempty(obj.tracker)
        warning('Labeler:projImport','Re-initting tracker.');
        obj.tracker.init();
      end
    end
    
    function projAssignProjNameFromProjFileIfAppropriate(obj)
      if isempty(obj.projname) && ~isempty(obj.projectfile)
        [~,fnameS] = fileparts(obj.projectfile);
        obj.projname = fnameS;
      end
    end
    
  end
  
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
    
    function projMacroSetUI(obj)
      % Set any/all current macros with inputdlg
      
      s = obj.projMacros;
      macros = fieldnames(s);
      macrosdisp = cellfun(@(x)['$' x],macros,'uni',0);
      vals = struct2cell(s);
      nmacros = numel(macros);
      INPUTBOXWIDTH = 100;
      resp = inputdlgWithBrowse(macrosdisp,'Project macros',...
        repmat([1 INPUTBOXWIDTH],nmacros,1),vals);
      if ~isempty(resp)
        assert(isequal(numel(macros),numel(vals),numel(resp)));
        for i=1:numel(macros)
          try
            obj.projMacroSet(macros{i},resp{i});
          catch ME
            warningNoTrace('Labeler:macro','Cannot set macro ''%s'': %s',...
              macrosdisp{i},ME.message);
          end
        end
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
      m = obj.projMacros;
      flds = fieldnames(m);
      vals = struct2cell(m);
      s = cellfun(@(x,y)sprintf('%s -> %s',x,y),flds,vals,'uni',0);
    end    
    
    function p = projLocalizePath(obj,p)
      p = FSPath.fullyLocalizeStandardizeChar(p,obj.projMacros);
    end
    
    function projNewImStack(obj,ims,varargin)
      % DEVELOPMENT ONLY
      %
      % Same as projNew, but initialize project to have a single 'movie'
      % consisting of an image stack. 
      %
      % Optional PVs. Let N=numel(ims).
      % - xyGT. [Nxnptx2], gt labels. 
      % - xyTstT. [Nxnptx2xRT], CPR test results (replicats)
      % - xyTstTRed. [Nxnptx2], CPR test results (selected/final).
      % - tstITst. [K] indices into 1:N. If provided, xyTstT and xyTstTRed
      %   should have K rows. xyTstTITst specify frames to which tracking 
      %   results apply.
      %
      % If xyGT/xyTstT/xyTstTRed provided, they are viewed with
      % LabelCoreCPRView. 
      %
      % TODO: Prob should set pGT onto .labeledpos, and let
      % LabelCoreCPRView handle replicates etc.
      
      assert(false,'Unsupported');
      
      [xyGT,xyTstT,xyTstTRed,tstITst] = myparse(varargin,...
        'xyGT',[],...
        'xyTstT',[],...
        'xyTstTRed',[],...
        'tstITst',[]);

      assert(false,'Unsupported: todo gt');
      assert(~obj.isMultiView);
      
      obj.projNew('IMSTACK__DEVONLY');

      mr = MovieReaderImStack;
      mr.open(ims);
      obj.movieReader = mr;
      movieInfo = struct();
      movieInfo.nframes = mr.nframes;
      
      obj.movieFilesAll{end+1,1} = '__IMSTACK__';
      obj.movieFilesAllHaveLbls(end+1,1) = false; % note, this refers to .labeledpos
      obj.movieInfoAll{end+1,1} = movieInfo;
      obj.trxFilesAll{end+1,1} = '__IMSTACK__';
      obj.currMovie = 1; % HACK
      obj.currTarget = 1;
%       obj.labeledpos{end+1,1} = [];
%       obj.labeledpostag{end+1,1} = [];
      
      N = numel(ims);
      tfGT = ~isempty(xyGT);
      if tfGT
        [Ntmp,npt,d] = size(xyGT); % npt equal nLabelPoint?
        assert(Ntmp==N && d==2);
      end
      
      tfTst = ~isempty(xyTstT);
      tfITst = ~isempty(tstITst);
      if tfTst
        sz1 = size(xyTstT);
        sz2 = size(xyTstTRed);
        RT = size(xyTstT,4);
        if tfITst
          k = numel(tstITst);
          assert(isequal([k npt d],sz1(1:3),sz2));
          xyTstTPad = nan(N,npt,d,RT);
          xyTstTRedPad = nan(N,npt,d);
          xyTstTPad(tstITst,:,:,:) = xyTstT;
          xyTstTRedPad(tstITst,:,:) = xyTstTRed;
          
          xyTstT = xyTstTPad;
          xyTstTRed = xyTstTRedPad;
        else
          assert(isequal([N npt d],sz1(1:3),sz2));          
        end
      else
        xyTstT = nan(N,npt,d,1);
        xyTstTRed = nan(N,npt,d);
      end
      
      if tfGT
        lc = LabelCoreCPRView(obj);
        lc.setPs(xyGT,xyTstT,xyTstTRed);
        delete(obj.lblCore);
        obj.lblCore = lc;
        lc.init(obj.nLabelPoints,obj.labelPointsPlotInfo);
        obj.setFrame(1);
      end
    end
            
  end
  
  methods (Static)
    
    function s = lblModernize(s)
      % s: struct, .lbl contents
      
      if ~isfield(s,'labeledposTS')
        nMov = numel(s.labeledpos);
        s.labeledposTS = cell(nMov,1);
        for iMov = 1:nMov
          lpos = s.labeledpos{iMov};
          [npts,~,nfrm,ntgt] = size(lpos);
          s.labeledposTS{iMov} = -inf(npts,nfrm,ntgt);
        end
        
        warningNoTrace('Label timestamps added (all set to -inf).');
      end
      
      if ~isfield(s,'labeledpos2')
        s.labeledpos2 = cellfun(@(x)nan(size(x)),s.labeledpos,'uni',0);
      end
      
      % 20160622
      if ~isfield(s,'nview') && ~isfield(s,'cfg')
        s.nview = 1;
      end
      if ~isfield(s,'viewNames') && ~isfield(s,'cfg')
        s.viewNames = {'1'};
      end

      % 20160629
      if isfield(s,'trackerClass')
        assert(isfield(s,'trackerData'));
      else
        if isfield(s,'CPRLabelTracker')
          s.trackerClass = 'CPRLabelTracker';
          s.trackerData = s.CPRLabelTracker;
        elseif isfield(s,'Interpolator')
          s.trackerClass = 'Interpolator';
          s.trackerData = s.Interpolator;
        else
          s.trackerClass = '';
          s.trackerData = [];
        end
      end
      
      % 20160707
      if ~isfield(s,'labeledposMarked')
        s.labeledposMarked = cellfun(@(x)false(size(x)),s.labeledposTS,'uni',0);
      end

      % 20160822 Modernize legacy projects that don't have a .cfg prop. 
      % Create a cfg from the lbl contents and fill in any missing fields 
      % with the current pref.yaml.
      if ~isfield(s,'cfg')
        % Create a config out what is in s. The large majority of config
        % info is not present in s; all other fields start from defaults.
        
        % first deal with multiview new def of NumLabelPoints
        nPointsReal = s.nLabelPoints/s.nview;
        assert(round(nPointsReal)==nPointsReal);
        s.nLabelPoints = nPointsReal;
        
        ptNames = arrayfun(@(x)sprintf('point%d',x),1:s.nLabelPoints,'uni',0);
        ptNames = ptNames(:);
        cfg = struct(...
          'NumViews',s.nview,...
          'ViewNames',{s.viewNames},...
          'NumLabelPoints',s.nLabelPoints,...
          'LabelPointNames',{ptNames},...
          'LabelMode',char(s.labelMode),...
          'LabelPointsPlot',s.labelPointsPlotInfo);
        fldsRm = {'nview' 'viewNames' 'nLabelPoints' 'labelMode' 'labelPointsPlotInfo'};
        s = rmfield(s,fldsRm);

        cfgbase = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
        if exist('pref.yaml','file')>0
          cfg1 = ReadYaml('pref.yaml');
          cfgbase = structoverlay(cfgbase,cfg1,'dontWarnUnrecog',true);
        end
        cfg = structoverlay(cfgbase,cfg,'dontWarnUnrecog',true);
        s.cfg = cfg;
      end      
      s.cfg = Labeler.cfgModernize(s.cfg);
      
      % 20160816
      if isfield(s,'minv')
        assert(numel(s.minv)==numel(s.maxv));
        
        nminv = numel(s.minv);
        if nminv~=s.cfg.NumViews
          s.minv = repmat(s.minv(1),s.cfg.NumViews,1);
          s.maxv = repmat(s.maxv(1),s.cfg.NumViews,1);
        end
        
        % 20160927
        assert(isequal(numel(s.minv),numel(s.maxv),s.cfg.NumViews));
        for iView=1:s.cfg.NumViews
          s.cfg.View(iView).CLim.Min = s.minv(iView);
          s.cfg.View(iView).CLim.Max = s.maxv(iView);
        end
        s = rmfield(s,{'minv' 'maxv'});
      end
      
      % 20160927
      if isfield(s,'movieForceGrayscale')
        s.cfg.Movie.ForceGrayScale = s.movieForceGrayscale;
        s = rmfield(s,'movieForceGrayscale');
      end
      
      % 20161213
      if ~isfield(s,'viewCalProjWide')
        if ~isempty(s.viewCalibrationData)
          % Prior to today, all viewCalibrationDatas were always proj-wide
          s.viewCalProjWide = true;
          assert(isscalar(s.viewCalibrationData));
        else
          s.viewCalProjWide = [];
        end
      end
      
      % 20170808
      if ~isfield(s,'trackModeIdx')
        s.trackModeIdx = 1;
      end
      
      % 20170829
      GTPROPS = {
        'movieFilesAllGT'
        'movieInfoAllGT'
        'trxFilesAllGT'
        'labeledposGT'
        'labeledposTSGT'
        'labeledpostagGT'
        'viewCalibrationDataGT'
        'gtIsGTMode'
        'gtSuggMFTable'
        'gtTblRes'
      };
      tfGTProps = isfield(s,GTPROPS);
      allGTPresent = all(tfGTProps);
      noGTPresent = ~any(tfGTProps);
      assert(allGTPresent || noGTPresent);
      if noGTPresent
        nview = s.cfg.NumViews;
        s.movieFilesAllGT = cell(0,nview);
        s.movieInfoAllGT = cell(0,nview);
        s.trxFilesAllGT = cell(0,nview);
        s.labeledposGT = cell(0,1);
        s.labeledposTSGT = cell(0,1);
        s.labeledpostagGT = cell(0,1);
        if isscalar(s.viewCalProjWide) && s.viewCalProjWide
          s.viewCalibrationDataGT = [];
        else
          s.viewCalibrationDataGT = cell(0,1);
        end
        s.gtIsGTMode = false;
        s.gtSuggMFTable = MFTable.emptyTable(MFTable.FLDSID);
        s.gtTblRes = [];
      end

      % 20170922
      if ~isfield(s,'suspSelectedMFT')
        s.suspSelectedMFT = [];
      end
      if ~isfield(s,'suspComputeFcn')
        s.suspComputeFcn = [];
      end
      
      % 20171102
      if ~isfield(s,'xvResults')
        s.xvResults = [];
        s.xvResultsTS = [];
      end
      
      % 20171110
      for f={'labeledpostag' 'labeledpostagGT'},f=f{1}; %#ok<FXSET>
        val = s.(f);
        for i=1:numel(val)
          if iscell(val{i})
            val{i} = strcmp(val{i},'occ');
          end
        end
        s.(f) = val;
      end
      
      % 20180309 Preproc params
      % If preproc params are present in trackerData, move them to s and 
      % remove from trackerData
      tfTrackerDataHasPPParams = ~isempty(s.trackerData) && ...
        ~isempty(s.trackerData.sPrm) && ...
        isfield(s.trackerData.sPrm,'PreProc');
      if isfield(s,'preProcParams')
        assert(isfield(s,'preProcH0'));
        assert(~tfTrackerDataHasPPParams);
      else
        if tfTrackerDataHasPPParams      
          ppPrm = s.trackerData.sPrm.PreProc;
          s.trackerData.sPrm = rmfield(s.trackerData.sPrm,'PreProc');
        else
          ppPrm = struct();
        end
        
        % 201803 NborMask. This default comes before the structoverlay
        % below b/c it is a "nonstandard default". Before adding GMM-EM
        % nborMask defaulted to Conn. Comp, so there theoretically could be
        % legacy projects with trained trackers based on conn.comp.
        if isfield(ppPrm,'NeighborMask') && ...
           ~isfield(ppPrm.NeighborMask,'SegmentMethod')
          ppPrm.NeighborMask.SegmentMethod = 'Conn. Comp';
        end
                
        s.preProcParams = ppPrm;
        s.preProcH0 = [];
      end
      
      ppPrm0 = CPRLabelTracker.readDefaultParams();
      ppPrm0 = ppPrm0.PreProc;
      [s.preProcParams,ppPrm0used] = structoverlay(ppPrm0,s.preProcParams,...
        'dontWarnUnrecog',true);
      if ~isempty(ppPrm0used)
        fprintf('Using default preprocessing parameters for: %s.\n',...
          String.cellstr2CommaSepList(ppPrm0used));
      end
      
    end  
    
  end 
  
  %% Movie
  methods
    
    function movieAdd(obj,moviefile,trxfile,varargin)
      % Add movie/trx to end of movie/trx list.
      %
      % moviefile: string or cellstr (can have macros)
      % trxfile: (optional) string or cellstr 
      
      assert(~obj.isMultiView,'Unsupported for multiview labeling.');
      
      [offerMacroization,gt] = myparse(varargin,...
        'offerMacroization',~isdeployed, ... % If true, look for matches with existing macros
        'gt',obj.gtIsGTMode ... % If true, add moviefile/trxfile to GT lists. Could be a separate method, but there is a lot of shared code/logic.
        );
      
      PROPS = Labeler.gtGetSharedPropsStc(gt);
      
      if exist('trxfile','var')==0 || isequal(trxfile,[])
        if ischar(moviefile)
          trxfile = '';
        elseif iscellstr(moviefile)
          trxfile = repmat({''},size(moviefile));
        else
          error('Labeler:movieAdd',...
            '''Moviefile'' must be a char or cellstr.');
        end
      end
      moviefile = cellstr(moviefile);
      trxfile = cellstr(trxfile);
      szassert(moviefile,size(trxfile));
      nMov = numel(moviefile);
      
      mr = MovieReader();
      for iMov = 1:nMov
        movFile = moviefile{iMov};
        tFile = trxfile{iMov};
        
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
          [tfMatch,tFileMacroized] = FSPath.tryTrxfileMacroization(...
            tFile,fileparts(movFileFull));
          if tfMatch
            tFile = tFileMacroized;
          end
        end
      
        movfilefull = obj.projLocalizePath(movFile);
        assert(exist(movfilefull,'file')>0,'Cannot find file ''%s''.',movfilefull);
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
        
        tFileFull = Labeler.trxFilesLocalize(tFile,movfilefull);
        if ~(isempty(tFileFull) || exist(tFileFull,'file')>0)
          FSPath.throwErrFileNotFoundMacroAware(tFile,tFileFull,'trxfile');
        end

        mr.open(movfilefull);
        ifo = struct();
        ifo.nframes = mr.nframes;
        ifo.info = mr.info;
        mr.close();
        
        if ~isempty(tFileFull)
          tmptrx = obj.getTrx(tFileFull,ifo.nframes);
          nTgt = numel(tmptrx);
        else
          nTgt = 1;
        end
        
        nlblpts = obj.nLabelPoints;
        nfrms = ifo.nframes;
        obj.(PROPS.MFA){end+1,1} = movFile;
        obj.(PROPS.MFAHL)(end+1,1) = false;
        obj.(PROPS.MIA){end+1,1} = ifo;
        obj.(PROPS.TFA){end+1,1} = tFile;
        obj.(PROPS.LPOS){end+1,1} = nan(nlblpts,2,nfrms,nTgt);
        obj.(PROPS.LPOSTS){end+1,1} = -inf(nlblpts,nfrms,nTgt);
        obj.(PROPS.LPOSTAG){end+1,1} = false(nlblpts,nfrms,nTgt);
        if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
          obj.(PROPS.VCD){end+1,1} = [];
        end
        if ~gt
          obj.labeledposMarked{end+1,1} = false(nlblpts,nfrms,nTgt);
          obj.labeledpos2{end+1,1} = nan(nlblpts,2,nfrms,nTgt);
        end
      end
    end
    
    function movieAddBatchFile(obj,bfile)
      % Read movies from batch file
      
      if exist(bfile,'file')==0
        error('Labeler:movieAddBatchFile','Cannot find file ''%s''.',bfile);
      end
      movs = importdata(bfile);
      try
        movs = regexp(movs,',','split');
        movs = cat(1,movs{:});
      catch ME
        error('Labeler:batchfile',...
          'Error reading file %s: %s',bfile,ME.message);
      end
      if size(movs,2)~=obj.nview
        error('Labeler:batchfile',...
          'Expected file %s to have %d column(s), one for each view.',...
          bfile,obj.nview);
      end
      if ~iscellstr(movs)
        error('Labeler:movieAddBatchFile',...
          'Could not parse file ''%s'' for filenames.',bfile);
      end
      nMovSetImport = size(movs,1);
      if obj.nview==1
        fprintf('Importing %d movies from file ''%s''.\n',nMovSetImport,bfile);
        obj.movieAdd(movs,[],'offerMacroization',false);
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
      
      moviefilesfull = cellfun(@(x)obj.projLocalizePath(x),moviefiles,'uni',0);
      cellfun(@(x)assert(exist(x,'file')>0,'Cannot find file ''%s''.',x),moviefilesfull);
      tfMFeq = arrayfun(@(x)strcmp(moviefiles{x},obj.(PROPS.MFA)(:,x)),...
        1:obj.nview,'uni',0);
      tfMFFeq = arrayfun(@(x)strcmp(moviefilesfull{x},obj.(PROPS.MFAF)(:,x)),...
        1:obj.nview,'uni',0);
      tfMFeq = cat(2,tfMFeq{:}); % [nmoviesetxnview], true when moviefiles matches movieFilesAll
      tfMFFeq = cat(2,tfMFFeq{:}); % [nmoviesetxnview], true when movfilefull matches movieFilesAllFull
      iAllViewsMatch = find(all(tfMFeq,2));
      if ~isempty(iAllViewsMatch)
        error('Labeler:dupmov',...
          'Movieset matches current movieset %d in project.',iAllViewsMatch(1));
      end
      for iView=1:obj.nview
        iMFmatches = find(tfMFeq(:,iView));
        iMFFmatches = find(tfMFFeq(:,iView));
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
      for iView = 1:obj.nview
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
      obj.(PROPS.MFAHL)(end+1,1) = false;
      obj.(PROPS.MIA)(end+1,:) = ifos;
      obj.(PROPS.TFA)(end+1,:) = repmat({''},1,obj.nview);
      obj.(PROPS.LPOS){end+1,1} = nan(nLblPts,2,nFrms,nTgt);
      obj.(PROPS.LPOSTS){end+1,1} = -inf(nLblPts,nFrms,nTgt);
      obj.(PROPS.LPOSTAG){end+1,1} = false(nLblPts,nFrms,nTgt);
      if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
        obj.(PROPS.VCD){end+1,1} = [];
      end
      if ~gt
        obj.labeledposMarked{end+1,1} = false(nLblPts,nFrms,nTgt);
        obj.labeledpos2{end+1,1} = nan(nLblPts,2,nFrms,nTgt);
      end
      
      % This clause does not occur in movieAdd(), b/c movieAdd is called
      % from UI functions which do this for the user. Currently movieSetAdd
      % does not have any UI so do it here.
      if ~obj.hasMovie && obj.nmoviesGTaware>0
        obj.movieSet(1,'isFirstMovie',true);
      end
    end

%     function tfSucc = movieRmName(obj,movName)
%       % movName: compared to .movieFilesAll (macros UNreplaced)
%       assert(~obj.isMultiView,'Unsupported for multiview projects.');
%       iMov = find(strcmp(movName,obj.movieFilesAll));
%       if isscalar(iMov)
%         tfSucc = obj.movieRm(iMov);
%       end
%     end

    function tfSucc = movieRm(obj,iMov,varargin)
      % tfSucc: true if movie removed, false otherwise
      
      [gt] = myparse(varargin,...
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
      haslbls2 = haslbls2(iMov);
      assert(haslbls1==haslbls2);
      if haslbls1 && ~obj.movieDontAskRmMovieWithLabels
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
      
      if tfProceedRm
        PROPS = Labeler.gtGetSharedPropsStc(gt);
        nMovOrigReg = obj.nmovies;
        nMovOrigGT = obj.nmoviesGT;
        
        obj.(PROPS.MFA)(iMov,:) = [];
        obj.(PROPS.MFAHL)(iMov,:) = [];
        obj.(PROPS.MIA)(iMov,:) = [];
        obj.(PROPS.TFA)(iMov,:) = [];
        
        tfOrig = obj.isinit;
        obj.isinit = true; % AL20160808. we do not want set.labeledpos side effects, listeners etc.
        obj.(PROPS.LPOS)(iMov,:) = []; % should never throw with .isinit==true
        obj.(PROPS.LPOSTS)(iMov,:) = [];
        obj.(PROPS.LPOSTAG)(iMov,:) = [];
        if ~gt
          obj.labeledposMarked(iMov,:) = [];
          obj.labeledpos2(iMov,:) = [];
        end
        if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
          szassert(obj.(PROPS.VCD),[nMovOrig 1]);
          obj.(PROPS.VCD)(iMov,:) = [];
        end
        obj.isinit = tfOrig;
        
        if gt
          movIdx = MovieIndex(-iMov);
        else
          movIdx = MovieIndex(iMov);
        end
        edata = MoviesRemappedEventData.movieRemovedEventData(...
          movIdx,nMovOrigReg,nMovOrigGT);
        obj.preProcData.movieRemap(edata.mIdxOrig2New);
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
        notify(obj,'movieRemoved',edata);
        
        if obj.currMovie>iMov && gt==obj.gtIsGTMode
          obj.movieSet(obj.currMovie-1);
        end
      end
      
      tfSucc = tfProceedRm;
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
        'trxFilesAll'};
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
      FLDS2 = {...
        'labeledpos' 'labeledposTS' 'labeledposMarked' 'labeledpostag' ...
        'labeledpos2'};
      for f=FLDS2,f=f{1}; %#ok<FXSET>
        obj.(f) = obj.(f)(p,:);
      end
      
      obj.isinit = tfOrig;
      
      edata = MoviesRemappedEventData.moviesReorderedEventData(...
        p,nmov,obj.nmoviesGT);
      obj.preProcData.movieRemap(edata.mIdxOrig2New);
      notify(obj,'moviesReordered',edata);

      if ~obj.gtIsGTMode
        iMovNew = find(p==iMov0);
        obj.movieSet(iMovNew); %#ok<FNDSB>
      end
    end
    
    function movieFilesMacroize(obj,str,macro)
      % Replace a string with a macro throughout .movieFilesAll and *gt. A 
      % project macro is also added for macro->string.
      %
      % str: a string 
      % macro: macro which will replace all matches of string (macro should
      % NOT include leading $)
      
      if isfield(obj.projMacros,macro) 
        currVal = obj.projMacros.(macro);
        if ~strcmp(currVal,str)
          error('Labeler:macro',...
            'Project macro ''%s'' is currently defined as ''%s''.',...
            macro,currVal);
        end
      end
        
      strpat = regexprep(str,'\\','\\\\');      
      mfa0 = obj.movieFilesAll;
      mfagt0 = obj.movieFilesAllGT;
      if ispc
        mfa1 = regexprep(mfa0,strpat,['$' macro],'ignorecase');
        mfagt1 = regexprep(mfagt0,strpat,['$' macro],'ignorecase');
      else
        mfa1 = regexprep(mfa0,strpat,['$' macro]);
        mfagt1 = regexprep(mfagt0,strpat,['$' macro]);
      end
      obj.movieFilesAll = mfa1;
      obj.movieFilesAllGT = mfagt1;
      
      if ~isfield(obj.projMacros,macro) 
        obj.projMacroAdd(macro,str);
      end
    end
    
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
    
    function [tfok,badfile] = movieCheckFilesExistSimple(obj,iMov,gt) % obj const
      % tfok: true if movie/trxfiles for iMov all exist, false otherwise.
      % badfile: if ~tfok, badfile contains a file that could not be found.

      PROPS = Labeler.gtGetSharedPropsStc(gt);
      
      if ~all(cellfun(@isempty,obj.(PROPS.TFA)(iMov,:)))
        assert(~obj.isMultiView,...
          'Multiview labeling with targets unsupported.');
      end
      
      for iView = 1:obj.nview
        movfileFull = obj.(PROPS.MFAF){iMov,iView};
        trxFileFull = obj.(PROPS.TFAF){iMov,iView};
        if exist(movfileFull,'file')==0
          tfok = false;
          badfile = movfileFull;
          return;
        elseif ~isempty(trxFileFull) && exist(trxFileFull,'file')==0
          tfok = false;
          badfile = trxFileFull;
          return;
        end
      end
      
      tfok = true;
      badfile = [];
    end
    
    function tfsuccess = movieCheckFilesExist(obj,iMov) % NOT obj const
      % Helper function for movieSet(), check that movie/trxfiles exist
      %
      % tfsuccess: false indicates user canceled or similar. True indicates
      % that i) obj.movieFilesAllFull(iMov,:) all exist; ii) if obj.hasTrx,
      % obj.trxFilesAllFull(iMov,:) all exist. User can update these fields
      % by browsing, updating macros etc.
      %
      % This function can harderror.
      %
      % This function is NOT obj const -- users can browse to 
      % movies/trxfiles, macro-related state can be mutated etc.
      
      tfsuccess = false;
      
      PROPS = obj.gtGetSharedProps();
      
      if ~all(cellfun(@isempty,obj.(PROPS.TFA)(iMov,:)))
        assert(~obj.isMultiView,...
          'Multiview labeling with targets unsupported.');
      end
                
      for iView = 1:obj.nview
        movfile = obj.(PROPS.MFA){iMov,iView};
        movfileFull = obj.(PROPS.MFAF){iMov,iView};
        
        if exist(movfileFull,'file')==0
          qstr = FSPath.errStrFileNotFoundMacroAware(movfile,...
            movfileFull,'movie');
          qtitle = 'Movie not found';
          if isdeployed
            error(qstr);
          end
          
          if FSPath.hasAnyMacro(movfile)
            qargs = {'Redefine macros','Browse to movie','Cancel','Cancel'};
          else
            qargs = {'Browse to movie','Cancel','Cancel'};
          end           
          resp = questdlg(qstr,qtitle,qargs{:});
          if isempty(resp)
            resp = 'Cancel';
          end
          switch resp
            case 'Cancel'
              return;
            case 'Redefine macros'
              obj.projMacroSetUI();
              movfileFull = obj.(PROPS.MFAF){iMov,iView};
              if exist(movfileFull,'file')==0
                emsg = FSPath.errStrFileNotFoundMacroAware(movfile,...
                  movfileFull,'movie');
                FSPath.errDlgFileNotFound(emsg);
                return;
              end
            case 'Browse to movie'
              pathguess = FSPath.maxExistingBasePath(movfileFull);
              if isempty(pathguess)
                pathguess = RC.getprop('lbl_lastmovie');
              end
              if isempty(pathguess)
                pathguess = pwd;
              end
              promptstr = sprintf('Select movie for %s',movfileFull);
              [newmovfile,newmovpath] = uigetfile('*.*',promptstr,pathguess);
              if isequal(newmovfile,0)
                return; % Cancel
              end
              movfileFull = fullfile(newmovpath,newmovfile);
              if exist(movfileFull,'file')==0
                emsg = FSPath.errStrFileNotFound(movfileFull,'movie');
                FSPath.errDlgFileNotFound(emsg);
                return;
              end
              
              % If possible, offer macroized movFile
              [tfCancel,macro,movfileMacroized] = ...
                FSPath.offerMacroization(obj.projMacros,{movfileFull});
              if tfCancel
                return;
              end
              tfMacroize = ~isempty(macro);
              if tfMacroize
                assert(isscalar(movfileMacroized));
                obj.(PROPS.MFA){iMov,iView} = movfileMacroized{1};
                movfileFull = obj.(PROPS.MFAF){iMov,iView};
              else
                obj.(PROPS.MFA){iMov,iView} = movfileFull;
              end
          end
          
          % At this point, either we have i) harderrored, ii)
          % early-returned with tfsuccess=false, or iii) movfileFull is set
          assert(exist(movfileFull,'file')>0);          
        end

        % trxfile
        %movfile = obj.(PROPS.MFA){iMov,iView};
        assert(strcmp(movfileFull,obj.(PROPS.MFAF){iMov,iView}));
        trxFile = obj.(PROPS.TFA){iMov,iView};
        trxFileFull = obj.(PROPS.TFAF){iMov,iView};
        tfTrx = ~isempty(trxFile);
        if tfTrx
          if exist(trxFileFull,'file')==0
            qstr = FSPath.errStrFileNotFoundMacroAware(trxFile,...
              trxFileFull,'trxfile');
            resp = questdlg(qstr,'Trxfile not found',...
              'Browse to trxfile','Cancel','Cancel');
            if isempty(resp)
              resp = 'Cancel';
            end
            switch resp
              case 'Browse to trxfile'
                % none
              case 'Cancel'
                return;
            end
            
            movfilepath = fileparts(movfileFull);
            promptstr = sprintf('Select trx file for %s',movfileFull);
            [newtrxfile,newtrxfilepath] = uigetfile('*.mat',promptstr,...
              movfilepath);
            if isequal(newtrxfile,0)
              return;
            end
            trxFile = fullfile(newtrxfilepath,newtrxfile);
            if exist(trxFile,'file')==0
              emsg = FSPath.errStrFileNotFound(trxFile,'trxfile');
              FSPath.errDlgFileNotFound(emsg);
              return;
            end
            [tfMatch,trxFileMacroized] = FSPath.tryTrxfileMacroization( ...
              trxFile,movfilepath);
            if tfMatch
              trxFile = trxFileMacroized;
            end
            obj.(PROPS.TFA){iMov,iView} = trxFile;
          end
          RC.saveprop('lbl_lasttrxfile',trxFile);
        end
      end
      
      % For multiview projs a user could theoretically alter macros in 
      % such a way as to incrementally locate files, breaking previously
      % found files
      for iView = 1:obj.nview
        movfile = obj.(PROPS.MFA){iMov,iView};
        movfileFull = obj.(PROPS.MFAF){iMov,iView};
        tfile = obj.(PROPS.TFA){iMov,iView};
        tfileFull = obj.(PROPS.TFAF){iMov,iView};
        if exist(movfileFull,'file')==0
          FSPath.throwErrFileNotFoundMacroAware(movfile,movfileFull,'movie');
        end
        if ~isempty(tfileFull) && exist(tfileFull,'file')==0
          FSPath.throwErrFileNotFoundMacroAware(tfile,tfileFull,'trxfile');
        end
      end
      
      tfsuccess = true;
    end
    
    function tfsuccess = movieSet(obj,iMov,varargin)
      % iMov: If multivew, movieSet index (row index into .movieFilesAll)      
      
      %#%MVOK
      
      assert(any(iMov==1:obj.nmoviesGTaware),...
                    'Invalid movie index ''%d''.',iMov);
      
      [isFirstMovie,noLabelingInit] = myparse(varargin,...
        'isFirstMovie',false,... % passing true for the first time a movie is added to a proj helps the UI
        'noLabelingInit',false); % DELETE OPTION
      
      tfsuccess = obj.movieCheckFilesExist(iMov); % throws
      if ~tfsuccess
        return;
      end
      
      movsAllFull = obj.movieFilesAllFullGTaware;
      for iView=1:obj.nview
        mov = movsAllFull{iMov,iView};
        obj.movieReader(iView).open(mov);
        RC.saveprop('lbl_lastmovie',mov);
        if iView==1
          obj.moviename = FSPath.twoLevelFilename(obj.moviefile);
        end
      end
      
      isInitOrig = obj.isinit;
      obj.isinit = true; % Initialization hell, invariants momentarily broken
      obj.currMovie = iMov;
      
      % for fun debugging
      %       obj.gdata.axes_all.addlistener('XLimMode','PreSet',@(s,e)lclTrace('preset'));
      %       obj.gdata.axes_all.addlistener('XLimMode','PostSet',@(s,e)lclTrace('postset'));
      obj.setFrameAndTarget(1,1);
      
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
        
      obj.isinit = isInitOrig; % end Initialization hell      

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
      assert(~isempty(obj.labeledpos{iMov}));
      assert(~isempty(obj.labeledposTS{iMov}));
      assert(~isempty(obj.labeledposMarked{iMov}));
      assert(~isempty(obj.labeledpostag{iMov}));
      assert(~isempty(obj.labeledpos2{iMov}));
      
      % KB 20161213: moved this up here so that we could redo in initHook
      obj.labelsMiscInit();
      if ~noLabelingInit
        obj.labelingInit();
      end
      
      edata = NewMovieEventData(isFirstMovie);
      notify(obj,'newMovie',edata);
      
      % Not a huge fan of this, maybe move to UI
      obj.updateFrameTableComplete();
      
      % Proj/Movie/LblCore initialization can maybe be improved
      % Call setFrame again now that lblCore is set up
      if obj.hasTrx
        obj.setFrameAndTarget(obj.currTrx.firstframe,obj.currTarget);
      else
        obj.setFrameAndTarget(1,1);
      end
    end
    
    function movieSetNoMovie(obj,varargin)
      % Set .currMov to 0
      
      noLabelingInit = myparse(varargin,... 
        'noLabelingInit',false); % DELETE OPTION
           
          % Stripped cut+paste form movieSet() for reference 20170714
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
      
      obj.labelsMiscInit();
      if ~noLabelingInit
        obj.labelingInit();
      end
      edata = NewMovieEventData(false);
      notify(obj,'newMovie',edata);
      obj.updateFrameTableComplete();

      % Set state equivalent to obj.setFrameAndTarget();
      gd = obj.gdata;
      imsall = gd.images_all;
      for iView=1:obj.nview
        obj.currIm{iView} = 0;
        set(imsall(iView),'CData',0);
      end
      obj.prevIm = 0;
      imprev = gd.image_prev;
      set(imprev,'CData',0);              
      obj.currTarget = 1;
      obj.currFrame = 1;
      obj.prevFrame = 1;
      
%       obj.currSusp = [];
    end
    
    function H0 = movieEstimateImHist(obj,nFrmSamp)
      % Estimate the typical image histogram H0 of movies in the project.
      %
      % nFrmSamp: number of frames to sample. Currently the typical image
      % histogram is computed with all frames weighted equally, even if
      % some movies have far more frames than others. An alternative would
      % be eg equal-weighting by movie. nFrmSamp is only an
      % estimate/target, the actual number of frames sampled may differ.
      
      % Always operates on regular (non-GT) movies
      
      assert(obj.nview==1,'Not supported for multiview.');
            
      nfrmsAll = cellfun(@(x)x.nframes,obj.movieInfoAll);
      nfrmsTotInProj = sum(nfrmsAll);
      dfSamp = ceil(nfrmsTotInProj/nFrmSamp);
      
      wbObj = WaitBarWithCancel('Histogram Equalization','cancelDisabled',true);
      oc = onCleanup(@()delete(wbObj));

      I = cell(0,1);
      mr = MovieReader;
      mr.forceGrayscale = true;
      iSamp = 0;
      wbObj.startPeriod('Reading data','shownumden',true,'denominator',nFrmSamp);
      for iMov = 1:obj.nmovies
        mov = obj.movieFilesAllFull{iMov};
        mr.open(mov);
        for f = 1:dfSamp:mr.nframes
          wbObj.updateFracWithNumDen(iSamp);
          iSamp = iSamp+1;
          I{end+1,1} = mr.readframe(f); %#ok<AGROW>
          %fprintf('Read movie %d, frame %d\n',iMov,f);
        end
      end
      wbObj.endPeriod();      
      
      H0 = typicalImHist(I,'wbObj',wbObj);
    end
        
    function movRdr = getMovieReader(obj,movname)
      movRdr = Labeler.getMovieReaderCacheStc(obj.movieCache,movname);
    end
  end
  methods (Static)
    function movRdr = getMovieReaderCacheStc(movCache,movfullpath)
      % Get movieReader for movname from .movieCache; load from filesys if
      % nec
      %
      % movCache: containers.Map
      % filename: fullpath to movie
      %
      % movRdr: scalar MovieReader
      
      if movCache.isKey(movfullpath)
        movRdr = movCache(movfullpath);
      else
        if exist(movfullpath,'file')==0
          error('Labeler:file','Cannot find movie ''%s''.',movfullpath);
        end
        movRdr = MovieReader;
        try
          movRdr.open(movfullpath);
        catch ME
          error('Could not open movie ''%s'': %s',movfullpath,ME.message);
        end
        movCache(movfullpath) = movRdr; %#ok<NASGU>
        RC.saveprop('lbl_lastmovie',movfullpath);        
      end
    end
    
  end
  
  %% Trx
  methods

    % TrxCache notes
    % To avoid repeated loading of trx data from filesystem, we cache
    % previously seen/loaded trx data in .trxCache. The cache is never
    % updated as it is assumed that trxfiles on disk do not mutate over the
    % course of a single APT session.
    
    function [trx,frm2trx] = getTrx(obj,filename,nfrm)
      % Get trx data for iMov/iView from .trxCache; load from filesys if
      % necessary      
      [trx,frm2trx] = Labeler.getTrxCacheStc(obj.trxCache,filename,nfrm);
    end
    
    function clearTrxCache(obj)
      % Forcibly clear .trxCache
      obj.trxCache = containers.Map();
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
      
      if trxCache.isKey(filename)
        s = trxCache(filename);
        trx = s.trx;
        frm2trx = s.frm2trx;
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
        maxID = max([obj.trx.id]);
      else
        maxID = -1;
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
      
      obj.currImHud.updateReadoutFields('hasTgt',obj.hasTrx);
      obj.initShowTrx();
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
    
    function [tfok,tblBig] = hlpTargetsTableUIgetBigTable(obj)
      wbObj = WaitBarWithCancel('Target Summary Table');
      centerOnParentFigure(wbObj.hWB,obj.hFig);
      oc = onCleanup(@()delete(wbObj));      
      tblBig = obj.trackGetBigLabeledTrackedTable('wbObj',wbObj);
      tfok = ~wbObj.isCancel;
      % if ~tfok, tblBig indeterminate
    end
    function hlpTargetsTableUIupdate(obj,navTbl)
      [tfok,tblBig] = obj.hlpTargetsTableUIgetBigTable();
      if tfok
        navTbl.setData(obj.trackGetSummaryTable(tblBig));
      end
    end
    function targetsTableUI(obj)
      [tfok,tblBig] = obj.hlpTargetsTableUIgetBigTable();
      if ~tfok
        return;
      end
      
      tblSumm = obj.trackGetSummaryTable(tblBig);
      hF = figure('Name','Target Summary (click row to navigate)',...
        'MenuBar','none','Visible','off');
      hF.Position(3:4) = [1280 500];
      centerfig(hF,obj.hFig);
      hPnl = uipanel('Parent',hF,'Position',[0 .08 1 .92]);
      BTNWIDTH = 100;
      DXY = 4;
      btnHeight = hPnl.Position(2)*hF.Position(4)-2*DXY;
      btnPos = [hF.Position(3)-BTNWIDTH-DXY DXY BTNWIDTH btnHeight];      
      hBtn = uicontrol('Style','pushbutton','Parent',hF,...
        'Position',btnPos,'String','Update',...
        'fontsize',12);
      FLDINFO = {
        'mov' 'Movie' 'integer' 30
        'iTgt' 'Target' 'integer' 30
        'trajlen' 'Traj. Length' 'integer' 45
        'frm1' 'Start Frm' 'integer' 30
        'nFrmLbl' '# Frms Lbled' 'integer' 60
        'nFrmTrk' '# Frms Trked' 'integer' 60
        'nFrmImported' '# Frms Imported' 'integer' 90
        'nFrmLblTrk' '# Frms Lbled&Trked' 'integer' 120
        'lblTrkMeanErr' 'Track Err' 'float' 60
        'nFrmLblImported' '# Frms Lbled&Imported' 'integer'  120
        'lblImportedMeanErr' 'Imported Err' 'float' 60
        'nFrmXV' '# Frms XV' 'integer' 40
        'xvMeanErr' 'XV Err' 'float' 40};
      tblfldsassert(tblSumm,FLDINFO(:,1));
      nt = NavigationTable(hPnl,[0 0 1 1],...
        @(row,rowdata)obj.setMFT(rowdata.mov,rowdata.frm1,rowdata.iTgt),...
        'ColumnName',FLDINFO(:,2)',...
        'ColumnFormat',FLDINFO(:,3)',...
        'ColumnPreferredWidth',cell2mat(FLDINFO(:,4)'));
%      jt = nt.jtable;
      nt.setData(tblSumm);
%      cr.setHorizontalAlignment(javax.swing.JLabel.CENTER);
%      h = jt.JTable.getTableHeader;
%      h.setPreferredSize(java.awt.Dimension(225,22));
%      jt.JTable.repaint;
      
      hF.UserData = nt;
      hBtn.Callback = @(s,e)obj.hlpTargetsTableUIupdate(nt);
      hF.Units = 'normalized';
      hBtn.Units = 'normalized';
      hF.Visible = 'on';

      % See LabelerGUI/addDepHandle
      handles = obj.gdata;
      handles.depHandles(end+1,1) = hF;
      guidata(obj.hFig,handles);
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
  methods % showTrx
    
    function initShowTrx(obj)
      deleteValidHandles(obj.hTraj);
      deleteValidHandles(obj.hTrx);
%       deleteValidHandles(obj.hTrxEll);
      deleteValidHandles(obj.hTrxTxt);
      obj.hTraj = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrx = matlab.graphics.primitive.Line.empty(0,1);
%       obj.hTrxEll = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrxTxt = matlab.graphics.primitive.Text.empty(0,1);
      
      ax = obj.gdata.axes_curr;
      pref = obj.projPrefs.Trx;
      for i = 1:obj.nTrx
        obj.hTraj(i,1) = line(...
          'parent',ax,...
          'xdata',nan, ...
          'ydata',nan, ...
          'color',pref.TrajColor,...
          'linestyle',pref.TrajLineStyle, ...
          'linewidth',pref.TrajLineWidth, ...
          'HitTest','off');

        obj.hTrx(i,1) = plot(ax,...
          nan,nan,pref.TrxMarker);
        set(obj.hTrx(i,1),'HitTest','off',...
          'Color',pref.TrajColor,...
          'MarkerSize',pref.TrxMarkerSize,...
          'LineWidth',pref.TrxLineWidth);
        
%         obj.hTrxEll(i,1) = plot(ax,nan,nan,'-');
%         set(obj.hTrxEll(i,1),'HitTest','off',...
%           'Color',pref.TrajColor);
        
%         id = find(obj.trxIdPlusPlus2Idx==i)-1;
        obj.hTrxTxt(i,1) = text(nan,nan,num2str(i),'Parent',ax,...
          'Color',pref.TrajColor,...
          'Fontsize',pref.TrxIDLblFontSize,...
          'Fontweight',pref.TrxIDLblFontWeight,...
          'PickableParts','none');
      end
    end
    
    function setShowTrx(obj,tf)
      assert(isscalar(tf) && islogical(tf));
      obj.showTrx = tf;
      obj.updateShowTrx();
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
      % Update .hTrx, .hTraj based on .trx, .showTrx*, .currFrame
      
      if ~obj.hasTrx
        return;
      end
      
      t = obj.currFrame;
      trxAll = obj.trx;
      nPre = obj.showTrxPreNFrm;
      nPst = obj.showTrxPostNFrm;
      pref = obj.projPrefs.Trx;
      
      if obj.showTrx        
        if obj.showTrxCurrTargetOnly
          tfShow = false(obj.nTrx,1);
          tfShow(obj.currTarget) = true;
        else
          tfShow = true(obj.nTrx,1);
        end
      else
        tfShow = false(obj.nTrx,1);
      end
      
%       tfShowEll = isscalar(obj.showTrxEll) && obj.showTrxEll ...
%         && all(isfield(trxAll,{'a' 'b' 'x' 'y' 'theta'}));
   
      % update coords/positions
      for iTrx = 1:obj.nTrx
        if tfShow(iTrx)
          trxCurr = trxAll(iTrx);
          t0 = trxCurr.firstframe;
          t1 = trxCurr.endframe;
          tTraj = max(t-nPre,t0):min(t+nPst,t1); % could be empty array
          iTraj = tTraj + trxCurr.off;
          xTraj = trxCurr.x(iTraj);
          yTraj = trxCurr.y(iTraj);
          if iTrx==obj.currTarget
            color = pref.TrajColorCurrent;
          else
            color = pref.TrajColor;
          end
          set(obj.hTraj(iTrx),'XData',xTraj,'YData',yTraj,'Color',color);

          if t0<=t && t<=t1
            idx = t+trxCurr.off;
            xTrx = trxCurr.x(idx);
            yTrx = trxCurr.y(idx);
          else
            xTrx = nan;
            yTrx = nan;
          end
          set(obj.hTrx(iTrx),'XData',xTrx,'YData',yTrx,'Color',color);
          
          if obj.showTrxIDLbl
            dx = pref.TrxIDLblOffset;
            set(obj.hTrxTxt(iTrx),'Position',[xTrx+dx yTrx+dx 1],...
              'Color',color);
          end
          
%           if tfShowEll && t0<=t && t<=t1
%             ellipsedraw(2*trxCurr.a(idx),2*trxCurr.b(idx),...
%               trxCurr.x(idx),trxCurr.y(idx),trxCurr.theta(idx),'-',...
%               'hEllipse',obj.hTrxEll(iTrx),'noseLine',true);
%           end
        end
      end
      set(obj.hTraj(tfShow),'Visible','on');
      set(obj.hTraj(~tfShow),'Visible','off');
      set(obj.hTrx(tfShow),'Visible','on');
      set(obj.hTrx(~tfShow),'Visible','off');
      if obj.showTrxIDLbl
        set(obj.hTrxTxt(tfShow),'Visible','on');
        set(obj.hTrxTxt(~tfShow),'Visible','off');
      else
        set(obj.hTrxTxt,'Visible','off');
      end
%       if tfShowEll
%         set(obj.hTrxEll(tfShow),'Visible','on');
%         set(obj.hTrxEll(~tfShow),'Visible','off');
%       else
%         set(obj.hTrxEll,'Visible','off');
%       end
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
      
      lblmode = myparse(varargin,...
        'labelMode',[]); % if true, force a call to labelsUpdateNewFrame(true) at end of call. Poorly named option.
      tfLblModeChange = ~isempty(lblmode);
      if tfLblModeChange
        assert(isa(lblmode,'LabelMode'));
      else
        lblmode = obj.labelMode;
      end
     
      nPts = obj.nLabelPoints;
      lblPtsPlotInfo = obj.labelPointsPlotInfo;
      template = obj.labelTemplate;
      
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
      obj.lblCore = LabelCore.createSafe(obj,lblmode);
      obj.lblCore.init(nPts,lblPtsPlotInfo);
      if hideLabelsPrev
        obj.lblCore.labelsHide();
      end
      if isprop(obj.lblCore,'streamlined') && exist('streamlinedPrev','var')>0
        obj.lblCore.streamlined = streamlinedPrev;
      end
      
      % labelmode-specific inits
      switch lblmode
        case LabelMode.TEMPLATE
          if ~isempty(template)
            obj.lblCore.setTemplate(template);
          end
      end
      if obj.lblCore.supportsCalibration
        vcd = obj.viewCalibrationDataCurrent;
        if isempty(vcd)
          warningNoTrace('Labeler:labelingInit',...
            'No calibration data loaded for calibrated labeling.');
        else
          obj.lblCore.projectionSetCalRig(vcd);
        end
      end
      obj.labelMode = lblmode;
      
      obj.genericInitLabelPointViz('lblPrev_ptsH','lblPrev_ptsTxtH',...
          obj.gdata.axes_prev,lblPtsPlotInfo);
          
      if tfLblModeChange
        % sometimes labelcore need this kick to get properly set up
        obj.labelsUpdateNewFrame(true);
      end
    end
    
    %%% labelpos
        
    function labelPosClear(obj)
      % Clear all labels AND TAGS for current movie/frame/target
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      
      PROPS = obj.gtGetSharedProps();
      x = obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTgt);
      if all(isnan(x(:)))
        % none; short-circuit set to avoid triggering .labeledposNeedsSave
      else        
        obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTgt) = nan;
        obj.labeledposNeedsSave = true;
      end
      
      obj.(PROPS.LPOSTS){iMov}(:,iFrm,iTgt) = now();
      obj.(PROPS.LPOSTAG){iMov}(:,iFrm,iTgt) = false;
      if ~obj.gtIsGTMode
        obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
      end
    end
    
    function labelPosClearI(obj,iPt)
      % Clear labels and tags for current movie/frame/target, point iPt
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      
      PROPS = obj.gtGetSharedProps();
      xy = obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt);
      if all(isnan(xy))
        % none; short-circuit set to avoid triggering .labeledposNeedsSave
      else
        obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt) = nan;
        obj.labeledposNeedsSave = true;
      end
      
      obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = now();
      obj.(PROPS.LPOSTAG){iMov}(iPt,iFrm,iTgt) = false;
      if ~obj.gtIsGTMode
        obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
      end
    end
    
    function [tf,lpos,lpostag] = labelPosIsLabeled(obj,iFrm,iTrx)
      % For current movie. Labeled includes fullyOccluded
      %
      % tf: scalar logical
      % lpos: [nptsx2] xy coords for iFrm/iTrx
      % lpostag: [npts] logical array 
      
      iMov = obj.currMovie;
      PROPS = obj.gtGetSharedProps();
      lpos = obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTrx);
      tfnan = isnan(lpos);
      tf = any(~tfnan(:));
      if nargout>=3
        lpostag = obj.(PROPS.LPOSTAG){iMov}(:,iFrm,iTrx);
      end
    end 
    
    function tf = labelPosIsLabeledMov(obj,iMov)
      % iMov: movie index (row index into .movieFilesAll)
      %
      % tf: [nframes-for-iMov], true if any point labeled in that mov/frame

      %#%MVOK
      
      ifo = obj.movieInfoAll{iMov,1};
      nf = ifo.nframes;
      lpos = obj.labeledpos{iMov};
      lposnnan = ~isnan(lpos);
      
      tf = arrayfun(@(x)nnz(lposnnan(:,:,x,:))>0,(1:nf)');
    end
    
    function tf = labelPosIsOccluded(obj,iFrm,iTrx)
      % Here Occluded refers to "pure occluded"
      % For current movie.
      % iFrm, iTrx: optional, defaults to current
      % Note: it is permitted to call eg LabelPosSet with inf coords
      % indicating occluded
      
      iMov = obj.currMovie;
      if exist('iFrm','var')==0
        iFrm = obj.currFrame;
      end
      if exist('iTrx','var')==0
        iTrx = obj.currTarget;
      end
      PROPS = obj.gtGetSharedProps();
      lpos = obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTrx);
      tf = isinf(lpos(:,1));
    end
    
    function labelPosSet(obj,xy)
      % Set labelpos for current movie/frame/target
            
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTgt) = xy;
      obj.(PROPS.LPOSTS){iMov}(:,iFrm,iTgt) = now();
      if ~obj.gtIsGTMode
      obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
      end
      obj.labeledposNeedsSave = true;
    end
        
    function labelPosSetI(obj,xy,iPt)
      % Set labelpos for current movie/frame/target, point iPt
      
      assert(~any(isnan(xy(:))));
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt) = xy;
      obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = now();
      if ~obj.gtIsGTMode
      obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
      end
      obj.labeledposNeedsSave = true;
    end
    
    function labelPosClearFramesI(obj,frms,iPt)
      xy = nan(2,1);
      obj.labelPosSetFramesI(frms,xy,iPt);      
    end
    
    function labelPosSetFramesI(obj,frms,xy,iPt)
      % Set labelpos for current movie/target to a single (constant) point
      % across multiple frames
      %
      % frms: vector of frames
      
      assert(isvector(frms));
      assert(numel(xy)==2);
      assert(isscalar(iPt));
      
      obj.trxCheckFramesLiveErr(frms);

      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(iPt,1,frms,iTgt) = xy(1);
      obj.(PROPS.LPOS){iMov}(iPt,2,frms,iTgt) = xy(2);
      obj.updateFrameTableComplete(); % above sets mutate .labeledpos{obj.currMovie} in more than just .currFrame
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      end
      
      obj.(PROPS.LPOSTS){iMov}(iPt,frms,iTgt) = now();
      if ~obj.gtIsGTMode
      obj.labeledposMarked{iMov}(iPt,frms,iTgt) = true;
      end

      obj.labeledposNeedsSave = true;
    end
    
    function labelPosSetFromLabeledPos2(obj)
      % copy .labeledpos2 to .labeledpos for current movie/frame/target
      
      assert(~obj.gtIsGTMode);
      
      iMov = obj.currMovie;
      if iMov>0
        frm = obj.currFrame;
        iTgt = obj.currTarget;
        lpos = obj.labeledpos2{iMov}(:,:,frm,iTgt);
        obj.labelPosSet(lpos);
      else
        warning('labeler:noMovie','No movie.');
      end
    end        
    
    function labelPosBulkImport(obj,xy)
      % Set ALL labels for current movie/target
      %
      % xy: [nptx2xnfrm]
      
      assert(~obj.gtIsGTMode);
      
      iMov = obj.currMovie;
      lposOld = obj.labeledpos{iMov};
      szassert(xy,size(lposOld));
      obj.labeledpos{iMov} = xy;
      obj.labeledposTS{iMov}(:) = now();
      obj.labeledposMarked{iMov}(:) = true; % not sure of right treatment
      
      obj.updateFrameTableComplete();
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      end
      obj.labeledposNeedsSave = true;
    end
    
    function labelPosSetUnmarkedFramesMovieFramesUnmarked(obj,xy,iMov,frms)
      % Set all unmarked labels for given movie, frames. Newly-labeled 
      % points are NOT marked in .labeledposmark
      %
      % xy: [nptsx2xnumel(frms)xntgts]
      % iMov: scalar movie index
      % frms: frames for iMov; labels 3rd dim of xy
      
      assert(~obj.gtIsGTMode);
      
      npts = obj.nLabelPoints;
      ntgts = obj.nTargets;
      nfrmsSpec = numel(frms);
      assert(size(xy,1)==npts);
      assert(size(xy,2)==2);
      assert(size(xy,3)==nfrmsSpec);
      assert(size(xy,4)==ntgts);
      validateattributes(iMov,{'numeric'},{'scalar' 'positive' 'integer' '<=' obj.nmovies});
      nfrmsMov = obj.movieInfoAll{iMov,1}.nframes;
      validateattributes(frms,{'numeric'},{'vector' 'positive' 'integer' '<=' nfrmsMov});    
      
      lposmarked = obj.labeledposMarked{iMov};      
      tfFrmSpec = false(npts,nfrmsMov,ntgts);
      tfFrmSpec(:,frms,:) = true;
      tfSet = tfFrmSpec & ~lposmarked;
      tfSet = reshape(tfSet,[npts 1 nfrmsMov ntgts]);
      tfLPosSet = repmat(tfSet,[1 2]); % [npts x 2 x nfrmsMov x ntgts]
      tfXYSet = ~lposmarked(:,frms,:); % [npts x nfrmsSpec x ntgts]
      tfXYSet = reshape(tfXYSet,[npts 1 nfrmsSpec ntgts]);
      tfXYSet = repmat(tfXYSet,[1 2]); % [npts x 2 x nfrmsSpec x ntgts]
      obj.labeledpos{iMov}(tfLPosSet) = xy(tfXYSet);
      
      obj.updateFrameTableComplete();
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      end
      obj.labeledposNeedsSave = true;  
      
%       for iTgt = 1:ntgts
%       for iFrm = 1:nfrmsSpec
%         f = frms(iFrm);
%         tfset = ~lposmarked(:,f,iTgt); % [npts x 1]
%         tfset = repmat(tfset,[1 2]); % [npts x 2]
%         lposFrmTgt = lpos(:,:,f,iTgt);
%         lposFrmTgt(tfset) = xy(:,:,iFrm,iTgt);
%         lpos(:,:,f,iTgt) = lposFrmTgt;
%       end
%       end
%       obj.labeledpos{iMov} = lpos;
    end
    
    function labelPosSetUnmarked(obj)
      % Clear .labeledposMarked for current movie/frame/target
      
      assert(~obj.gtIsGTMode);
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledposMarked{iMov}(:,iFrm,iTgt) = false;
    end
    
    function labelPosSetAllMarked(obj,val)
      % Clear .labeledposMarked for current movie, all frames/targets

      assert(~obj.gtIsGTMode);
      obj.labeledposMarked{iMov}(:) = val;
    end
        
    function labelPosSetOccludedI(obj,iPt)
      % Occluded is "pure occluded" here
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt) = inf;
      obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = now();
      if ~obj.gtIsGTMode
      obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
      end
      obj.labeledposNeedsSave = true;
    end
        
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
      obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = now();
      obj.(PROPS.LPOSTAG){iMov}(iPt,iFrm,iTgt) = true;
    end
    
    function labelPosTagClearI(obj,iPt)
      % iPt: can be vector
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      if obj.gtIsGTMode
        obj.labeledposTSGT{iMov}(iPt,iFrm,iTgt) = now();
        obj.labeledpostagGT{iMov}(iPt,iFrm,iTgt) = false;
      else
        obj.labeledposTS{iMov}(iPt,iFrm,iTgt) = now();
        obj.labeledpostag{iMov}(iPt,iFrm,iTgt) = false;
      end
    end
    
    function labelPosTagSetFramesI(obj,iPt,frms)
      % Set tag for current movie/target, given pt/frames

      obj.trxCheckFramesLiveErr(frms);
      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOSTAG){iMov}(iPt,frms,iTgt) = true;
    end
    
    function labelPosTagClearFramesI(obj,iPt,frms)
      % Clear tags for current movie/target, given pt/frames
      
      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOSTAG){iMov}(iPt,frms,iTgt) = false;
    end
    
    function [tfneighbor,iFrm0,lpos0] = ...
                          labelPosLabeledNeighbor(obj,iFrm,iTrx) % obj const
      % tfneighbor: if true, a labeled neighboring frame was found
      % iFrm0: index of labeled neighboring frame, relevant only if
      %   tfneighbor is true
      % lpos0: labels at iFrm0, relevant only if tfneighbor is true
      %
      % This method looks for a frame "near" iFrm for target iTrx that is
      % labeled. This could be iFrm itself if it is labeled. If a
      % neighboring frame is found, iFrm0 is not guaranteed to be the
      % closest or any particular neighboring frame although this will tend
      % to be true.      
      
      iMov = obj.currMovie;
      PROPS = obj.gtGetSharedProps();
      lposTrx = obj.(PROPS.LPOS){iMov}(:,:,:,iTrx);
      assert(isrow(obj.NEIGHBORING_FRAME_OFFSETS));
      for dFrm = obj.NEIGHBORING_FRAME_OFFSETS
        iFrm0 = iFrm + dFrm;
        iFrm0 = max(iFrm0,1);
        iFrm0 = min(iFrm0,obj.nframes);
        lpos0 = lposTrx(:,:,iFrm0);
        if ~isnan(lpos0(1))
          tfneighbor = true;
          return;
        end
      end
      
      tfneighbor = false;
      iFrm0 = nan;
      lpos0 = [];      
    end
    
    function [nTgts,nPts] = labelPosLabeledFramesStats(obj,frms) % obj const
      % Get stats re labeled frames in the current movie.
      % 
      % frms: vector of frame indices to consider. Defaults to
      %   1:obj.nframes.
      %
      % nTgts: numel(frms)-by-1 vector indicating number of targets labeled
      %   for each frame in consideration
      % nPts: numel(frms)-by-1 vector indicating number of points labeled 
      %   for each frame in consideration, across all targets
      
      if exist('frms','var')==0
        if isnan(obj.nframes)
          frms = [];
        else
          frms = 1:obj.nframes;
        end
        tfWaitBar = true;
      else
        tfWaitBar = false;
      end
      
      if ~obj.hasMovie || obj.currMovie==0 % invariants temporarily broken
        nTgts = nan(numel(frms),1);
        nPts = nan(numel(frms),1);
        return;
      end
      
      nf = numel(frms);
      ntgts = obj.nTargets;
      lpos = obj.labeledposCurrMovie;
      tflpos = ~isnan(lpos); % true->labeled (either regular or occluded)      
      
      nTgts = zeros(nf,1);
      nPts = zeros(nf,1);
      if tfWaitBar
        hWB = waitbar(0,'Updating frame table');
        centerOnParentFigure(hWB,obj.gdata.figure);
        ocp = onCleanup(@()delete(hWB));
      end
      for i = 1:nf
        if tfWaitBar && mod(i,1000)==0
          waitbar(i/nf,hWB);
        end
        f = frms(i);
        
        % don't squeeze() here it's expensive        
        tmpNTgts = 0;
        tmpNPts = 0;
        for iTgt = 1:ntgts
          z = sum(tflpos(:,1,f,iTgt));
          tmpNPts = tmpNPts+z;
          tfTgtLabeled = (z>0);
          if tfTgtLabeled
            tmpNTgts = tmpNTgts+1;
          end
        end
        nTgts(i) = tmpNTgts;
        nPts(i) = tmpNPts;        
      end
    end
    
    function tf = labelPosMovieHasLabels(obj,iMov,varargin)
      gt = myparse(varargin,'gt',obj.gtIsGTMode);
      if ~gt
        lpos = obj.labeledpos{iMov};
      else
        lpos = obj.labeledposGT{iMov};
      end
      tf = any(~isnan(lpos(:)));
    end

  end
  
  methods (Static)
    function trkfile = genTrkFileName(rawname,sMacro,movfile)
      % Generate a trkfilename from rawname by macro-replacing.      
      [sMacro.movdir,sMacro.movfile] = fileparts(movfile);
      trkfile = FSPath.macroReplace(rawname,sMacro);
      if ~(numel(rawname)>=4 && strcmp(rawname(end-3:end),'.trk'))
        trkfile = [trkfile '.trk'];
      end
    end
    function [tfok,trkfiles] = checkTrkFileNamesExportUI(trkfiles)
      % Check/confirm trkfile names for export. If any trkfiles exist, ask 
      % whether overwriting is ok; alternatively trkfiles may be 
      % modified/uniqueified using datetimestamps.
      %
      % trkfiles (input): cellstr of proposed trkfile names (full paths).
      % Can be an array.
      %
      % tfok: if true, trkfiles (output) is valid, and user has said it is 
      % ok to write to those files even if it is an overwrite.
      % trkfiles (output): cellstr, same size as trkfiles. .trk filenames
      % that are okay to write/overwrite to. Will match input if possible.
      
      tfexist = cellfun(@(x)exist(x,'file')>0,trkfiles(:));
      tfok = true;
      if any(tfexist)
        iExist = find(tfexist,1);
        queststr = sprintf('One or more .trk files already exist, eg: %s.',trkfiles{iExist});
        if isdeployed
          btn = 'Add datetime to filenames';
          warning('Labeler:trkFileNamesForExport',...
            'One or more .trk files already exist. Adding datetime to trk filenames.');
        else
          btn = questdlg(queststr,'Files exist','Overwrite','Add datetime to filenames',...
            'Cancel','Add datetime to filenames');
        end
        if isempty(btn)
          btn = 'Cancel';
        end
        switch btn
          case 'Overwrite'
            % none; use trkfiles as-is
          case 'Add datetime to filenames'
            nowstr = datestr(now,'yyyymmddTHHMMSS');
            [trkP,trkF] = cellfun(@fileparts,trkfiles,'uni',0);            
            trkfiles = cellfun(@(x,y)fullfile(x,[y '_' nowstr '.trk']),trkP,trkF,'uni',0);
          otherwise
            tfok = false;
            trkfiles = [];
        end
      end
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
        basename = [basename '_labels'];
      end
      
      rawname = fullfile('$movdir',basename);
    end
    
    function [tfok,rawtrkname] = getExportTrkRawnameUI(obj,varargin)
      % Prompt the user to get a raw/base trkfilename.
      %
      % varargin: see defaultExportTrkRawname
      % 
      % tfok: user canceled or similar
      % rawtrkname: use only if tfok==true
      
      rawtrkname = inputdlg('Enter name/pattern for trkfile(s) to be exported. Available macros: $movdir, $movfile, $projdir, $projfile, $projname.',...
        'Export Trk File',1,{obj.defaultExportTrkRawname(varargin{:})});
      tfok = ~isempty(rawtrkname);
      if tfok
        rawtrkname = rawtrkname{1};
      end
    end
    
    function fname = getDefaultFilenameExportLabelTable(obj)
      if ~isempty(obj.projectfile)
        rawname = '$projdir/$projfile_labels.mat';
      elseif ~isempty(obj.projname)
        rawname = '$projdir/$projname_labels.mat';
      else
        rawname = '$projdir/labels.mat';
      end
      sMacro = obj.baseTrkFileMacros();
      fname = FSPath.macroReplace(rawname,sMacro);
    end
        
    function [tfok,trkfiles] = getTrkFileNamesForExportUI(obj,movfiles,rawname)
      % Concretize a raw trkfilename, then check for conflicts etc.
      
      sMacro = obj.baseTrkFileMacros();
      trkfiles = cellfun(@(x)Labeler.genTrkFileName(rawname,sMacro,x),...
        movfiles,'uni',0);
      [tfok,trkfiles] = Labeler.checkTrkFileNamesExportUI(trkfiles);
    end
    
    function [tfok,trkfiles] = resolveTrkfilesVsTrkRawname(obj,iMovs,...
        trkfiles,rawname,defaultRawNameArgs)
      % Ugly, input arg helper. Methods that export a trkfile must have
      % either i) the trkfilenames directly supplied, ii) a raw/base 
      % trkname supplied, or iii) nothing supplied.
      %
      % If i), check the sizes.
      % If ii), generate the trkfilenames from the rawname.
      % If iii), first generate the rawname, then generate the
      % trkfilenames.
      %
      % Cases ii) and iii), are also UI/prompt if there are
      % existing/conflicting filenames already on disk.
      %
      % defaultRawNameArgs: cell of PVs to pass to defaultExportTrkRawname.
      % 
      % tfok: scalar, if true then trkfiles is usable; if false then user
      %   canceled or similar.
      % trkfiles: [iMovs] cellstr, trkfiles (full paths) to export to
      % 
      % This call can also throw.
      
      PROPS = obj.gtGetSharedProps();
      
      movfiles = obj.(PROPS.MFAF)(iMovs,:);
      if isempty(trkfiles)
        if isempty(rawname)
          rawname = obj.defaultExportTrkRawname(defaultRawNameArgs{:});
        end
        [tfok,trkfiles] = obj.getTrkFileNamesForExportUI(movfiles,rawname);
        if ~tfok
          return;
        end
      end
      
      nMov = numel(iMovs);
      nView = obj.nview;
      if size(trkfiles,1)~=nMov
        error('Labeler:argSize',...
          'Numbers of movies and trkfiles supplied must be equal.');
      end
      if size(trkfiles,2)~=nView
        error('Labeler:argSize',...
          'Number of columns in trkfiles (%d) must equal number of views in project (%d).',...
          size(trkfiles,2),nView);
      end
      
      tfok = true;
    end
  end
  
  methods
	
    function [trkfilesCommon,kwCommon,trkfilesAll] = ...
                      getTrkFileNamesForImport(obj,movfiles)
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
    end
    
    function labelExportTrk(obj,iMovs,varargin)
      % Export label data to trk files.
      %
      % iMov: optional, indices into (rows of) .movieFilesAllGTaware to 
      %   export. Defaults to 1:obj.nmoviesGTaware.
      
      [trkfiles,rawtrkname] = myparse(varargin,...
        'trkfiles',[],... % [nMov nView] cellstr, fullpaths to trkfilenames to export to
        'rawtrkname',[]... % string, rawname to apply over iMovs to generate trkfiles
        );
      
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmoviesGTaware;
      end
      
      [tfok,trkfiles] = obj.resolveTrkfilesVsTrkRawname(iMovs,trkfiles,...
        rawtrkname,{'labels' true});
      if ~tfok
        return;
      end
        
      nMov = numel(iMovs);
      nView = obj.nview;
      nPhysPts = obj.nPhysPoints;
      for i=1:nMov
        iMvSet = iMovs(i);
        lposFull = obj.labeledposGTaware{iMvSet};
        lposTSFull = obj.labeledposTSGTaware{iMvSet};
        lposTagFull = obj.labeledpostagGTaware{iMvSet};
        
        for iView=1:nView
          iPt = (1:nPhysPts) + (iView-1)*nPhysPts;
          if nView==1
            assert(nPhysPts==size(lposFull,1));
          else
            tmp = find(obj.labeledposIPt2View==iView);
            assert(isequal(tmp(:),iPt(:)));
          end
          trkfile = TrkFile(lposFull(iPt,:,:,:),...
            'pTrkTS',lposTSFull(iPt,:,:),...
            'pTrkTag',lposTagFull(iPt,:,:));
          trkfile.save(trkfiles{i,iView});
          fprintf('Saved trkfile: %s\n',trkfiles{i,iView});
        end
      end
      msgbox(sprintf('Results for %d moviesets exported.',nMov),'Export complete');
    end
    
    function labelImportTrkGeneric(obj,iMovSets,trkfiles,lposFld,...
                                            lposTSFld,lposTagFld)
      % iMovStes: [N] vector of movie set indices
      % trkfiles: [Nxnview] cellstr of trk filenames
      % lpos*Fld: property names for labeledpos, labeledposTS,
      % labeledposTag. Can be empty to not set that prop.
      
      assert(~obj.gtIsGTMode);
      
      nMovSets = numel(iMovSets);
      szassert(trkfiles,[nMovSets obj.nview]);
      nPhysPts = obj.nPhysPoints;   
      tfMV = obj.isMultiView;
      nView = obj.nview;
      
      for i=1:nMovSets
        iMov = iMovSets(i);
        lpos = nan(size(obj.labeledpos{iMov}));
        lposTS = -inf(size(obj.labeledposTS{iMov}));
        lpostag = false(size(obj.labeledpostag{iMov}));
        assert(size(lpos,1)==nPhysPts*nView);
        
        if tfMV
          fprintf('MovieSet %d...\n',iMov);
        end
        for iVw = 1:nView
          tfile = trkfiles{i,iVw};
          s = load(tfile,'-mat');
          s = TrkFile.modernizeStruct(s);
          
          if isfield(s,'pTrkiPt')
            iPt = s.pTrkiPt;
          else
            iPt = 1:size(s.pTrk,1);
          end
          tfInBounds = 1<=iPt & iPt<=nPhysPts;
          if any(~tfInBounds)
            if tfMV
              error('Labeler:trkImport',...
                'View %d: trkfile contains information for more points than exist in project (number physical points=%d).',...
                iVw,nPhysPts);
            else
              error('Labeler:trkImport',...
                'Trkfile contains information for more points than exist in project (number of points=%d).',...
                nPhysPts);
            end
          end
          if nnz(tfInBounds)<nPhysPts
            if tfMV
              warningNoTrace('Labeler:trkImport',...
                'View %d: trkfile does not contain labels for all points in project (number physical points=%d).',...
                iVw,nPhysPts);              
            else
               warningNoTrace('Labeler:trkImport',...
                 'Trkfile does not contain information for all points in project (number of points=%d).',...
                 nPhysPts);
            end
          end
          
          if isfield(s,'pTrkFrm')
            frmsTrk = s.pTrkFrm;
          else
            frmsTrk = 1:size(s.pTrk,3);
          end
          
          nfrmLpos = size(lpos,3);
          tfInBounds = 1<=frmsTrk & frmsTrk<=nfrmLpos;
          if any(~tfInBounds)
            warningNoTrace('Labeler:trkImport',...
              'Trkfile contains information for frames beyond end of movie (number of frames=%d). Ignoring additional frames.',...
              nfrmLpos);
          end
          if nnz(tfInBounds)<nfrmLpos
            warningNoTrace('Labeler:trkImport',...
              'Trkfile does not contain information for all frames in movie. Frames missing from Trkfile will be unlabeled.');
          end
          frmsTrkIB = frmsTrk(tfInBounds);
          
          if isfield(s,'pTrkiTgt')
            iTgt = s.pTrkiTgt;
          else
            iTgt = 1;
          end
          assert(size(s.pTrk,4)==numel(iTgt));
          nTgtProj = size(lpos,4);
          tfiTgtIB = 1<=iTgt & iTgt<=nTgtProj;
          if any(~tfiTgtIB)
            warningNoTrace('Labeler:trkImport',...
              'Trkfile contains information for targets not present in movie. Ignoring extra targets.');
          end
          if nnz(tfiTgtIB)<nTgtProj
            warningNoTrace('Labeler:trkImport',...
              'Trkfile does not contain information for all targets in movie.');
          end
          iTgtsIB = iTgt(tfiTgtIB);
          
          fprintf(1,'... Loaded %d frames for %d points, %d targets from trk file: %s.\n',...
            numel(frmsTrkIB),numel(iPt),numel(iTgtsIB),tfile);

          iPt = iPt + (iVw-1)*nPhysPts;
          lpos(iPt,:,frmsTrkIB,iTgtsIB) = s.pTrk(:,:,tfInBounds,tfiTgtIB);
          lposTS(iPt,frmsTrkIB,iTgtsIB) = s.pTrkTS(:,tfInBounds,tfiTgtIB);
          lpostag(iPt,frmsTrkIB,iTgtsIB) = s.pTrkTag(:,tfInBounds,tfiTgtIB);
        end

        obj.(lposFld){iMov} = lpos;
        if ~isempty(lposTSFld)
          obj.(lposTSFld){iMov} = lposTS;
        end
        if ~isempty(lposTagFld)
          obj.(lposTagFld){iMov} = lpostag;
        end
      end      
    end
    
    function labelImportTrk(obj,iMovs,trkfiles)
      % Import label data from trk files.
      %
      % iMovs: [nMovie]. Movie(set) indices for which to import.
      % trkfiles: [nMoviexnview] cellstr. full filenames to trk files
      %   corresponding to iMov.
      
      assert(~obj.gtIsGTMode);
      
      obj.labelImportTrkGeneric(iMovs,trkfiles,'labeledpos',...
          'labeledposTS','labeledpostag');
      
      obj.movieFilesAllHaveLbls(iMovs) = ...
        cellfun(@(x)any(~isnan(x(:))),obj.labeledpos(iMovs));
      
      obj.updateFrameTableComplete();
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      end
      
      %obj.labeledposNeedsSave = true; AL 20160609: don't touch this for
      %now, since what we are importing is already in the .trk file.
      obj.labelsUpdateNewFrame(true);
      
      RC.saveprop('lastTrkFileImported',trkfiles{end});
    end
    
    function [tfsucc,trkfilesUse] = labelImportTrkFindTrkFilesPrompt(obj,movfiles)
      % Find trkfiles present for given movies. Prompt user to pick a set
      % if more than one exists.
      %
      % movfiles: [nTrials x nview] cellstr
      %
      % tfsucc: if true, trkfilesUse is valid; if false, trkfilesUse is
      % intedeterminate
      % trkfilesUse: cellstr, same size as movfiles. Full paths to trkfiles
      % present/selected for import
      
      [trkfilesCommon,kwCommon] = obj.getTrkFileNamesForImport(movfiles);
      nCommon = numel(kwCommon);
      
      tfsucc = false;
      trkfilesUse = [];
      switch nCommon
        case 0
          warningNoTrace('Labeler:labelImportTrkPrompt',...
            'No consistently-named trk files found across %d given movies.',numel(movfiles));
          return;
        case 1
          trkfilesUseIdx = 1;
        otherwise
          msg = sprintf('Multiple consistently-named trkfiles found. Select trkfile pattern to import.');
          uiwait(msgbox(msg,'Multiple trkfiles found','modal'));
          trkfileExamples = trkfilesCommon{1};
          for i=1:numel(trkfileExamples)
            [~,trkfileExamples{i}] = myfileparts(trkfileExamples{i});
          end
          [sel,ok] = listdlg(...
            'Name','Select trkfiles',...
            'Promptstring','Select a trkfile (pattern) to import.',...
            'SelectionMode','single',...
            'listsize',[300 300],...
            'liststring',trkfileExamples);
          if ok
            trkfilesUseIdx = sel;
          else
            return;
          end
      end
      trkfilesUse = cellfun(@(x)x{trkfilesUseIdx},trkfilesCommon,'uni',0);
      tfsucc = true;
    end
    
    function labelImportTrkPromptGeneric(obj,iMovs,importFcn)
      assert(~obj.gtIsGTMode);
      
      movfiles = obj.movieFilesAllFull(iMovs,:);
      [tfsucc,trkfilesUse] = obj.labelImportTrkFindTrkFilesPrompt(movfiles);
      if tfsucc
        feval(importFcn,obj,iMovs,trkfilesUse);
      else
        if isscalar(iMovs) && obj.nview==1
          % In this case (single movie, single view) failure can occur if 
          % no trkfile is found alongside movie, or if user cancels during
          % a prompt.
          
          lastTrkFileImported = RC.getprop('lastTrkFileImported');
          if isempty(lastTrkFileImported)
            lastTrkFileImported = pwd;
          end
          [fname,pth] = uigetfile('*.trk','Import trkfile',lastTrkFileImported);
          if isequal(fname,0)
            return;
          end
          trkfile = fullfile(pth,fname);
          feval(importFcn,obj,iMovs,{trkfile});
        end
      end
      
    end
	
    function labelImportTrkPrompt(obj,iMovs)
      % Import label data from trk files, prompting if necessary to specify
      % which trk files to import.
      %
      % iMovs: [nMovie]. Optional, movie(set) indices to import.
      %
      % labelImportTrkPrompt will look for trk files with common keywords
      % (consistent naming) in .movieFilesAllFull(iMovs). If there is
      % precisely one consistent trkfile pattern, it will import those
      % trkfiles. Otherwise it will ask the user which trk files to import.
      
      assert(~obj.gtIsGTMode);

      if exist('iMovs','var')==0
        iMovs = 1:obj.nmovies;
      end
      obj.labelImportTrkPromptGeneric(iMovs,'labelImportTrk');
    end
    
    function labelMakeLabelMovie(obj,fname,varargin)
      % Make a movie of all labeled frames for current movie
      %
      % fname: output filename, movie to be created.
      % optional pvs:
      % - framerate. defaults to 10.
      
      framerate = myparse(varargin,'framerate',10);
      
      if ~obj.hasMovie
        error('Labeler:noMovie','No movie currently open.');
      end
      if exist(fname,'file')>0
        error('Labeler:movie','Output movie ''%s'' already exists. For safety reasons, this movie will not be overwritten. Please specify a new output moviename.',...
          fname);
      end
      
      nTgts = obj.labelPosLabeledFramesStats();
      frmsLbled = find(nTgts>0);
      nFrmsLbled = numel(frmsLbled);
      if nFrmsLbled==0
        msgbox('Current movie has no labeled frames.');
        return;
      end
            
      ax = obj.gdata.axes_curr;
      vr = VideoWriter(fname);      
      vr.FrameRate = framerate;

      vr.open();
      try
        hTxt = text(230,10,'','parent',obj.gdata.axes_curr,'Color','white','fontsize',24);
        hWB = waitbar(0,'Writing video');
        for i = 1:nFrmsLbled
          f = frmsLbled(i);
          obj.setFrame(f);
          hTxt.String = sprintf('%04d',f);
          tmpFrame = getframe(ax);
          vr.writeVideo(tmpFrame);
          waitbar(i/nFrmsLbled,hWB,sprintf('Wrote frame %d\n',f));
        end
      catch ME
        vr.close();
        delete(hTxt);
        ME.rethrow();
      end
      vr.close();
      delete(hTxt);
      delete(hWB);
    end   
    
    %#%GTOK
    function tblMF = labelGetMFTableLabeled(obj,varargin)
      % Compile mov/frm/tgt MFTable; include all labeled frames/tgts. 
      %
      % Does not include GT frames.
      %
      % tblMF: See MFTable.FLDSFULLTRX.
      
      [wbObj,useLabels2] = myparse(varargin,...
        'wbObj',[], ... % optional WaitBarWithCancel. If cancel:
                   ... % 1. obj logically const (eg except for obj.trxCache)
                   ... % 2. tblMF (output) indeterminate
        'useLabels2',false... % if true, use labels2 instead of labels
        ); 
      tfWB = ~isempty(wbObj);

      if obj.gtIsGTMode
        error('Labeler:gt','Not supported in GT mode.');
      end
      
      if useLabels2
        mfts = MFTSetEnum.AllMovAllLabeled2;
      else
        mfts = MFTSetEnum.AllMovAllLabeled;
      end
      tblMF = mfts.getMFTable(obj);
      if obj.hasTrx
        argsTrx = {'trxFilesAllFull',obj.trxFilesAllFull,'trxCache',obj.trxCache};
      else
        argsTrx = {};
      end
      if useLabels2
        lpos = obj.labeledpos2;
        lpostag = cellfun(@(x)false(size(x)),obj.labeledpostag,'uni',0);
        lposTS = cellfun(@(x)-inf(size(x)),obj.labeledposTS,'uni',0);
      else
        lpos = obj.labeledpos;
        lpostag = obj.labeledpostag;
        lposTS = obj.labeledposTS;
      end
      
      tblMF = Labeler.labelAddLabelsMFTableStc(tblMF,lpos,lpostag,lposTS,...
          argsTrx{:},'wbObj',wbObj);
      if tfWB && wbObj.isCancel
        % tblMF (return) indeterminate
        return;
      end
    end
    
    %#%GTOK
    function tblMF = labelGetMFTableCurrMovFrmTgt(obj)
      % Get MFTable for current movie/frame/target (single-row table)
      %
      % tblMF: See MFTable.FLDSFULLTRX.
                  
      if obj.gtIsGTMode
        % Easy to support in GT mode, just unnec for now
        error('Labeler:gt','Not supported in GT mode.');
      end
      
      iMov = obj.currMovie;
      frm = obj.currFrame;
      iTgt = obj.currTarget;
      lposFrmTgt = obj.labeledpos{iMov}(:,:,frm,iTgt);
      lpostagFrmTgt = obj.labeledpostag{iMov}(:,frm,iTgt);
      lposTSFrmTgt = obj.labeledposTS{iMov}(:,frm,iTgt);      

      mov = iMov;
      p = Shape.xy2vec(lposFrmTgt); % absolute position
      pTS = lposTSFrmTgt';
      tfocc = lpostagFrmTgt';
      if obj.hasTrx
        assert(~obj.isMultiView,'Unsupported for multiview.');
        assert(obj.frm2trx(frm,iTgt));
        trxCurr = obj.trx(iTgt);
        xtrxs = trxCurr.x(frm+trxCurr.off);
        ytrxs = trxCurr.y(frm+trxCurr.off);
        sclrassert(xtrxs); % legacy check
        sclrassert(ytrxs);
        pTrx = [xtrxs ytrxs];
      else
        pTrx = [nan nan];
      end
      
      tblMF = table(mov,frm,iTgt,p,pTS,tfocc,pTrx);
    end
    
    %#%GTOK
    function tblMF = labelMFTableAddROI(obj,tblMF,roiRadius)
      % Add .pRoi and .roi to tblMF
      %
      % tblMF.pRoi: Just like tblMF.p, but relative to tblMF.roi (p==1 => 
      %   first row/col of ROI)
      % tblMF.roi: [nrow x 2*2*nview]. Raster order {lo,hi},{x,y},view
      
      tblfldscontainsassert(tblMF,MFTable.FLDSFULLTRX);
      tblfldsdonotcontainassert(tblMF,{'pRoi' 'roi'});
      
      nphyspts = obj.nPhysPoints;
      nrow = height(tblMF);
      p = tblMF.p;
      pTrx = tblMF.pTrx;
      
      tfRmRow = false(nrow,1);
      pRoi = nan(size(p));
      roi = nan(nrow,4*obj.nview);
      for i=1:nrow
        xy = Shape.vec2xy(p(i,:));
        xyTrx = Shape.vec2xy(pTrx(i,:));
        [roiCurr,tfOOBview,xyROIcurr] = ...
          Shape.xyAndTrx2ROI(xy,xyTrx,nphyspts,roiRadius);
        if any(tfOOBview)
          warningNoTrace('CPRLabelTracker:oob',...
            'Movie(set) %d, frame %d, target %d: shape out of bounds of target ROI. Not including row.',...
            tblMF.mov(i),tblMF.frm(i),tblMF.iTgt(i));
          tfRmRow(i) = true;
        else
          pRoi(i,:) = Shape.xy2vec(xyROIcurr);
          roi(i,:) = roiCurr;
        end
      end
      
      tblMF = [tblMF table(pRoi,roi)];
      tblMF(tfRmRow,:) = [];
    end
    
    function tblMF = labelAddLabelsMFTable(obj,tblMF)
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
      tblMF = Labeler.labelAddLabelsMFTableStc(tblMF,...
        obj.(PROPS.LPOS),obj.(PROPS.LPOSTAG),obj.(PROPS.LPOSTS),...
        'trxFilesAllFull',tfaf,'trxCache',obj.trxCache);
    end
  end
  
  methods (Static)
    
%     function tblMF = labelGetMFTableLabeledStc(lpos,lpostag,lposTS,...
%         trxFilesAllFull,trxCache)
%       % Compile MFtable, by default for all labeled mov/frm/tgts
%       %
%       % tblMF: [NTrl rows] MFTable, one row per labeled movie/frame/target.
%       %   MULTIVIEW NOTE: tbl.p* is the 2d/projected label positions, ie
%       %   each shape has nLabelPoints*nView*2 coords, raster order is 1. pt
%       %   index, 2. view index, 3. coord index (x vs y)
%       %   
%       %   Fields: {'mov' 'frm' 'iTgt' 'p' 'pTS' 'tfocc' 'pTrx'}
%       %   Here 'p' is 'pAbs' or absolute position
%                   
%       s = structconstruct(MFTable.FLDSFULLTRX,[0 1]);
%       
%       nMov = size(lpos,1);
%       nView = size(trxFilesAllFull,2);
%       szassert(lpos,[nMov 1]);
%       szassert(lpostag,[nMov 1]);
%       szassert(lposTS,[nMov 1]);
%       szassert(trxFilesAllFull,[nMov nView]);
%       
%       for iMov = 1:nMov
%         lposI = lpos{iMov};
%         lpostagI = lpostag{iMov};
%         lposTSI = lposTS{iMov};
%         [npts,d,nfrms,ntgts] = size(lposI);
%         assert(d==2);
%         szassert(lpostagI,[npts nfrms ntgts]);
%         szassert(lposTSI,[npts nfrms ntgts]);
%         
%         [trxI,~,frm2trxTotAnd] = Labeler.getTrxCacheAcrossViewsStc(...
%                                   trxCache,trxFilesAllFull(iMov,:),nfrms);        
%         cellfun(@(x)assert(numel(x)==ntgts),trxI);        
%         for f=1:nfrms
%           for iTgt=1:ntgts
%             lposIFrmTgt = lposI(:,:,f,iTgt);
%             % read if any point (in any view) is labeled for this
%             % (frame,target)
%             tfReadTgt = any(~isnan(lposIFrmTgt(:)));
%             if tfReadTgt
%               assert(frm2trxTotAnd(f,iTgt),'Labeled target is not live.');
%               lpostagIFrmTgt = lpostagI(:,f,iTgt);
%               lposTSIFrmTgt = lposTSI(:,f,iTgt);
%               xtrxs = cellfun(@(xx)xx(iTgt).x(f+xx(iTgt).off),trxI);
%               ytrxs = cellfun(@(xx)xx(iTgt).y(f+xx(iTgt).off),trxI);
%               
%               s(end+1,1).mov = iMov; %#ok<AGROW> 
%               s(end).frm = f;
%               s(end).iTgt = iTgt;
%               s(end).p = Shape.xy2vec(lposIFrmTgt);
%               s(end).pTS = lposTSIFrmTgt';
%               s(end).tfocc = strcmp(lpostagIFrmTgt','occ');
%               s(end).pTrx = [xtrxs(:)' ytrxs(:)'];
%             end
%           end
%         end
%       end
%       tblMF = struct2table(s,'AsArray',true);      
%     end
    
    function tblMF = labelAddLabelsMFTableStc(tblMF,lpos,lpostag,lposTS,...
        varargin)
      % Add label/trx information to an MFTable
      %
      % tblMF (input): MFTable with flds MFTable.FLDSID. tblMF.mov are 
      %   MovieIndices. tblMF.mov.get() are indices into lpos,lpostag,lposTS.
      % lpos...lposTS: as in labelGetMFTableLabeledStc
      %
      % tblMF (output): Same rows as tblMF, but with addnl label-related
      %   fields as in labelGetMFTableLabeledStc
      
      [trxFilesAllFull,trxCache,wbObj] = myparse(varargin,...
        'trxFilesAllFull',[],... % cellstr, indexed by tblMV.mov. if supplied, tblMF will contain .pTrx field
        'trxCache',[],... % must be supplied if trxFilesAllFull is supplied
        'wbObj',[]... % optional WaitBarWithCancel. If cancel, tblMF (output) indeterminate
        );      
      tfWB = ~isempty(wbObj);
      
      assert(istable(tblMF));
      tblfldscontainsassert(tblMF,MFTable.FLDSID);
      nMov = size(lpos,1);
      szassert(lpos,[nMov 1]);
      szassert(lpostag,[nMov 1]);
      szassert(lposTS,[nMov 1]);
      
      tfTrx = ~isempty(trxFilesAllFull);
      if tfTrx
        nView = size(trxFilesAllFull,2);
        szassert(trxFilesAllFull,[nMov nView]);
        tfTfafEmpty = cellfun(@isempty,trxFilesAllFull);
        % Currently, projects allowed to have some movs with trxfiles and
        % some without.
        assert(all( all(tfTfafEmpty,2) | all(~tfTfafEmpty,2) ),...
          'Unexpected trxFilesAllFull specification.');
        tfMovHasTrx = all(~tfTfafEmpty,2); % tfTfafMovEmpty(i) indicates whether movie i has trxfiles
      else
        nView = 1;
      end
  
      nrow = height(tblMF);
      
      if tfWB
        wbObj.startPeriod('Compiling labels','shownumden',true,...
          'denominator',nrow);
        oc = onCleanup(@()wbObj.endPeriod);
      end
      
      % Maybe Optimize: group movies together

      npts = size(lpos{1},1);
      
      pAcc = nan(0,npts*2);
      pTSAcc = -inf(0,npts);
      tfoccAcc = false(0,npts);
      pTrxAcc = nan(0,nView*2); % xv1 xv2 ... xvk yv1 yv2 ... yvk
      thetaTrxAcc = nan(0,nView);
      aTrxAcc = nan(0,nView);
      bTrxAcc = nan(0,nView);
      tfInvalid = false(nrow,1); % flags for invalid rows of tblMF encountered
      for irow=1:nrow
        if tfWB
          tfCancel = wbObj.updateFracWithNumDen(irow);
          if tfCancel
            return;
          end
        end
        
        tblrow = tblMF(irow,:);
        iMov = tblrow.mov.get;
        frm = tblrow.frm;
        iTgt = tblrow.iTgt;

        lposI = lpos{iMov};
        lpostagI = lpostag{iMov};
        lposTSI = lposTS{iMov};
        [npts,d,nfrms,ntgts] = size(lposI);
        assert(d==2);
        szassert(lpostagI,[npts nfrms ntgts]);
        szassert(lposTSI,[npts nfrms ntgts]);

        if frm<1 || frm>nfrms
          tfInvalid(irow) = true;
          continue;
        end
        
        if tfTrx && tfMovHasTrx(iMov)
          [trxI,~,frm2trxTotAnd] = Labeler.getTrxCacheAcrossViewsStc(...
            trxCache,trxFilesAllFull(iMov,:),nfrms);          
          tgtLiveInFrm = frm2trxTotAnd(frm,iTgt);
          if ~tgtLiveInFrm
            tfInvalid(irow) = true;
            continue;
          end
        else
          assert(iTgt==1);
        end
 
        lposIFrmTgt = lposI(:,:,frm,iTgt);
        lpostagIFrmTgt = lpostagI(:,frm,iTgt);
        lposTSIFrmTgt = lposTSI(:,frm,iTgt);
        pAcc(end+1,:) = Shape.xy2vec(lposIFrmTgt); %#ok<AGROW>
        pTSAcc(end+1,:) = lposTSIFrmTgt'; %#ok<AGROW>
        tfoccAcc(end+1,:) = lpostagIFrmTgt'; %#ok<AGROW>

        if tfTrx && tfMovHasTrx(iMov)
          xtrxs = cellfun(@(xx)xx(iTgt).x(frm+xx(iTgt).off),trxI);
          ytrxs = cellfun(@(xx)xx(iTgt).y(frm+xx(iTgt).off),trxI);
          pTrxAcc(end+1,:) = [xtrxs(:)' ytrxs(:)']; %#ok<AGROW>
          thetas = cellfun(@(xx)xx(iTgt).theta(frm+xx(iTgt).off),trxI);
          thetaTrxAcc(end+1,:) = thetas(:)'; %#ok<AGROW>

          as = cellfun(@(xx)xx(iTgt).a(frm+xx(iTgt).off),trxI);
          bs = cellfun(@(xx)xx(iTgt).b(frm+xx(iTgt).off),trxI);
          aTrxAcc(end+1,:) = as(:)'; %#ok<AGROW>
          bTrxAcc(end+1,:) = bs(:)'; %#ok<AGROW>
        else
          pTrxAcc(end+1,:) = nan; %#ok<AGROW> % singleton exp
          thetaTrxAcc(end+1,:) = nan; %#ok<AGROW> % singleton exp
          aTrxAcc(end+1,:) = nan; %#ok<AGROW>
          bTrxAcc(end+1,:) = nan; %#ok<AGROW>
        end
      end
      
      if any(tfInvalid)
        warningNoTrace('Removed %d invalid rows of MFTable.',nnz(tfInvalid));
      end
      tblMF = tblMF(~tfInvalid,:);
      tLbl = table(pAcc,pTSAcc,tfoccAcc,pTrxAcc,thetaTrxAcc,aTrxAcc,bTrxAcc,...
        'VariableNames',{'p' 'pTS' 'tfocc' 'pTrx' 'thetaTrx' 'aTrx' 'bTrx'});
      tblMF = [tblMF tLbl];
    end
    
%     % Legacy meth. labelGetMFTableLabeledStc is new method but assumes
%     % .hasTrx
%     %#3DOK
%     function [I,tbl] = lblCompileContentsRaw(...
%         movieNames,lposes,lpostags,iMovs,frms,varargin)
%       % Read moviefiles with landmark labels
%       %
%       % movieNames: [NxnView] cellstr of movienames
%       % lposes: [N] cell array of labeledpos arrays [npts x 2 x nfrms x ntgts]. 
%       %   For multiview, npts=nView*NumLabelPoints.
%       % lpostags: [N] cell array of labeledpostags [npts x nfrms x ntgts]
%       % iMovs. [M] (row) indices into movieNames to read.
%       % frms. [M] cell array. frms{i} is a vector of frames to read for
%       % movie iMovs(i). frms{i} may also be:
%       %     * 'all' indicating "all frames" 
%       %     * 'lbl' indicating "all labeled frames" (currently includes partially-labeled)
%       %
%       % I: [NtrlxnView] cell vec of images
%       % tbl: [NTrl rows] labels/metadata MFTable.
%       %   MULTIVIEW NOTE: tbl.p is the 2d/projected label positions, ie
%       %   each shape has nLabelPoints*nView*2 coords, raster order is 1. pt
%       %   index, 2. view index, 3. coord index (x vs y)
%       %
%       % Optional PVs:
%       % - hWaitBar. Waitbar object
%       % - noImg. logical scalar default false. If true, all elements of I
%       % will be empty.
%       % - lposTS. [N] cell array of labeledposTS arrays [nptsxnfrms]
%       % - movieNamesID. [NxnView] Like movieNames (input arg). Use these
%       % names in tbl instead of movieNames. The point is that movieNames
%       % may be macro-replaced, platformized, etc; otoh in the MD table we
%       % might want macros unreplaced, a standard format etc.
%       % - tblMovArray. Scalar logical, defaults to false. Only relevant for
%       % multiview data. If true, use array of movies in tbl.mov. Otherwise, 
%       % use single compactified string ID.
%       
%       [hWB,noImg,lposTS,movieNamesID,tblMovArray] = myparse(varargin,...
%         'hWaitBar',[],...
%         'noImg',false,...
%         'lposTS',[],...
%         'movieNamesID',[],...
%         'tblMovArray',false);
%       assert(numel(iMovs)==numel(frms));
%       for i = 1:numel(frms)
%         val = frms{i};
%         assert(isnumeric(val) && isvector(val) || ismember(val,{'all' 'lbl'}));
%       end
%       
%       tfWB = ~isempty(hWB);
%       
%       assert(iscellstr(movieNames));
%       [N,nView] = size(movieNames);
%       assert(iscell(lposes) && iscell(lpostags));
%       assert(isequal(N,numel(lposes),numel(lpostags)));
%       tfLposTS = ~isempty(lposTS);
%       if tfLposTS
%         assert(numel(lposTS)==N);
%       end
%       for i=1:N
%         assert(size(lposes{i},1)==size(lpostags{i},1) && ...
%                size(lposes{i},3)==size(lpostags{i},2));
%         if tfLposTS
%           assert(isequal(size(lposTS{i}),size(lpostags{i})));
%         end
%       end
%       
%       if ~isempty(movieNamesID)
%         assert(iscellstr(movieNamesID));
%         szassert(movieNamesID,size(movieNames)); 
%       else
%         movieNamesID = movieNames;
%       end
%       
%       for iVw=nView:-1:1
%         mr(iVw) = MovieReader();
%       end
% 
%       I = [];
%       % Here, for multiview, mov are for the first movie in each set
%       s = struct('mov',cell(0,1),'frm',[],'p',[],'tfocc',[]);
%       
%       nMov = numel(iMovs);
%       fprintf('Reading %d movies.\n',nMov);
%       if nView>1
%         fprintf('nView=%d.\n',nView);
%       end
%       for i = 1:nMov
%         iMovSet = iMovs(i);
%         lpos = lposes{iMovSet}; % npts x 2 x nframes
%         lpostag = lpostags{iMovSet};
% 
%         [npts,d,nFrmAll] = size(lpos);
%         assert(d==2);
%         if isempty(lpos)
%           assert(isempty(lpostag));
%           lpostag = cell(npts,nFrmAll); % edge case: when lpos/lpostag are [], uninitted/degenerate case
%         end
%         szassert(lpostag,[npts nFrmAll]);
%         D = d*npts;
%         % Ordering of d is: {x1,x2,x3,...xN,y1,..yN} which for multiview is
%         % {xp1v1,xp2v1,...xpnv1,xp1v2,...xpnvk,yp1v1,...}. In other words,
%         % in decreasing raster order we have 1. pt index, 2. view index, 3.
%         % coord index (x vs y)
%         
%         for iVw=1:nView
%           movfull = movieNames{iMovSet,iVw};
%           mr(iVw).open(movfull);
%         end
%         
%         movID = MFTable.formMultiMovieID(movieNamesID(iMovSet,:));
%         
%         % find labeled/tagged frames (considering ALL frames for this
%         % movie)
%         tfLbled = arrayfun(@(x)nnz(~isnan(lpos(:,:,x)))>0,(1:nFrmAll)');
%         frmsLbled = find(tfLbled);
%         tftagged = ~cellfun(@isempty,lpostag); % [nptxnfrm]
%         ntagged = sum(tftagged,1);
%         frmsTagged = find(ntagged);
%         assert(all(ismember(frmsTagged,frmsLbled)));
% 
%         frms2Read = frms{i};
%         if strcmp(frms2Read,'all')
%           frms2Read = 1:nFrmAll;
%         elseif strcmp(frms2Read,'lbl')
%           frms2Read = frmsLbled;
%         end
%         nFrmRead = numel(frms2Read);
%         
%         ITmp = cell(nFrmRead,nView);
%         fprintf('  mov(set) %d, D=%d, reading %d frames\n',iMovSet,D,nFrmRead);
%         
%         if tfWB
%           hWB.Name = 'Reading movies';
%           wbStr = sprintf('Reading movie %s',movID);
%           waitbar(0,hWB,wbStr);
%         end
%         for iFrm = 1:nFrmRead
%           if tfWB
%             waitbar(iFrm/nFrmRead,hWB);
%           end
%           
%           f = frms2Read(iFrm);
% 
%           if noImg
%             % none; ITmp(iFrm,:) will have [] els
%           else
%             for iVw=1:nView
%               im = mr(iVw).readframe(f);
%               if size(im,3)==3 && isequal(im(:,:,1),im(:,:,2),im(:,:,3))
%                 im = rgb2gray(im);
%               end
%               ITmp{iFrm,iVw} = im;
%             end
%           end
%           
%           lblsFrmXY = lpos(:,:,f);
%           tags = lpostag(:,f);
%           
%           if tblMovArray
%             assert(false,'Unsupported codepath');
%             %s(end+1,1).mov = movieNamesID(iMovSet,:); %#ok<AGROW>
%           else
%             s(end+1,1).mov = iMovSet; %#ok<AGROW>
%           end
%           %s(end).movS = movS1;
%           s(end).frm = f;
%           s(end).p = Shape.xy2vec(lblsFrmXY);
%           s(end).tfocc = strcmp('occ',tags(:)');
%           if tfLposTS
%             lts = lposTS{iMovSet};
%             s(end).pTS = lts(:,f)';
%           end
%         end
%         
%         I = [I;ITmp]; %#ok<AGROW>
%       end
%       tbl = struct2table(s,'AsArray',true);      
%     end
        
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
      if ~all(strcmpi(obj.viewNames(:),crObj.viewNames(:)))
        warningNoTrace('Labeler:viewCal',...
          'Project viewnames do not match viewnames in calibration object.');
      end
    end
    
    function [tfAllSame,movWidths,movHeights] = viewCalCheckMovSizes(obj)
      % Check for consistency of movie sizes in current proj. Throw
      % warndlgs for each view where sizes differ.
      % 
      % tfAllSame: [1 nView] logical. If true, all movies in that view
      % have the same size. This includes both .movieInfoAll AND 
      % .movieInfoAllGT.
      % movWidths, movHeights: [nMovSetxnView] arrays
            
      ifo = cat(1,obj.movieInfoAll,obj.movieInfoAllGT);
      movWidths = cellfun(@(x)x.info.Width,ifo);
      movHeights = cellfun(@(x)x.info.Height,ifo);
      nrow = obj.nmovies + obj.nmoviesGT;
      nView = obj.nview;
      szassert(movWidths,[nrow nView]);
      szassert(movHeights,[nrow nView]);
      
      tfAllSame = true(1,nView);
      if nrow>0
        for iVw=1:nView
          tfAllSame(iVw) = ...
            all(movWidths(:,iVw)==movWidths(1,iVw)) && ...
            all(movHeights(:,iVw)==movHeights(1,iVw));          
        end
        if ~all(tfAllSame)
          warnstr = 'The movies in this project have varying view/image sizes. This probably doesn''t work well with calibrations. Proceed at your own risk.';
          warndlg(warnstr,'Image sizes vary','modal');
        end
      end
    end
    
    function viewCalSetCheckViewSizes(obj,iMov,crObj,tfSet)
      % Check/set movie image size for movie iMov on calrig object 
      %
      % iMov: movie index, applied in GT-aware fashion (eg .currMovie)
      % crObj: scalar calrig object
      % tfSet: if true, set the movie size on the calrig (with diagnostic
      % printf); if false, throw warndlg if the sizes don't match
            
      assert(iMov>0);
      movInfo = obj.movieInfoAllGTaware(iMov,:);
      movWidths = cellfun(@(x)x.info.Width,movInfo);
      movHeights = cellfun(@(x)x.info.Height,movInfo);
      vwSizes = [movWidths(:) movHeights(:)];
      if tfSet
        % If movie sizes differ in this project, setting of viewsizes may
        % be hazardous. Assume warning has been thrown if necessary
        crObj.viewSizes = vwSizes;
        for iVw=1:obj.nview
          fprintf(1,'Calibration obj: set [width height] = [%d %d] for view %d (%s).\n',...
            vwSizes(iVw,1),vwSizes(iVw,2),iVw,crObj.viewNames{iVw});
        end
      else
        % Check view sizes
        if ~isequal(crObj.viewSizes,vwSizes)
          warnstr = sprintf('View sizes in calibration object (%s) do not match movie %d (%s).',...
            mat2str(crObj.viewSizes),iMov,mat2str(vwSizes));
          warndlg(warnstr,'View size mismatch','non-modal');
        end
      end
    end
    
  end
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
    
    function viewCalSetProjWide(obj,crObj,varargin)
      % Set project-wide calibration object.
      %
      % .viewCalibrationData or .viewCalibrationDataGT set depending on
      % .gtIsGTMode.
      
      tfSetViewSizes = myparse(varargin,...
        'tfSetViewSizes',false); % If true, set viewSizes on crObj per current movieInfo
      
      if obj.nmovies==0 || obj.currMovie==0
        error('Labeler:calib',...
          'Add/select a movie first before setting the calibration object.');
      end
      
      obj.viewCalCheckCalRigObj(crObj);
      
      vcdPW = obj.viewCalProjWide;
      if ~isempty(vcdPW) && ~vcdPW
        warningNoTrace('Labeler:viewCal',...
          'Discarding movie-specific calibration data. Calibration data will apply to all movies.');
        obj.viewCalProjWide = true;
        obj.viewCalibrationData = [];
        obj.viewCalibrationDataGT = [];
      end
      
      obj.viewCalCheckMovSizes();
      obj.viewCalSetCheckViewSizes(obj.currMovie,crObj,tfSetViewSizes);
      
      obj.viewCalProjWide = true;
      obj.viewCalibrationData = crObj;
      obj.viewCalibrationDataGT = [];

      lc = obj.lblCore;
      if lc.supportsCalibration
        lc.projectionSetCalRig(crObj);
      else
        warning('Labeler:viewCal','Current labeling mode does not utilize view calibration.');
      end
    end
    
    function viewCalSetCurrMovie(obj,crObj,varargin)
      % Set calibration object for current movie

      tfSetViewSizes = myparse(varargin,...
        'tfSetViewSizes',false); % If true, set viewSizes on crObj per current movieInfo
      
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
      
      obj.viewCalSetCheckViewSizes(obj.currMovie,crObj,tfSetViewSizes);
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
  
  methods (Static)
    function nptsLbled = labelPosNPtsLbled(lpos)
      % poor man's export of LabelPosLabeledFramesStats
      %
      % lpos: [nPt x d x nFrm x nTgt]
      % 
      % nptsLbled: [nFrm]. Number of labeled (non-nan) points for each frame
      
      [~,d,nfrm,ntgt] = size(lpos);
      assert(d==2);
      assert(ntgt==1,'One target only.');
      lposnnan = ~isnan(lpos);
      
      nptsLbled = nan(nfrm,1);
      for f = 1:nfrm
        tmp = all(lposnnan(:,:,f),2); % both x/y must be labeled for pt to be labeled
        nptsLbled(f) = sum(tmp);
      end
    end
  end
  
  methods (Access=private)
    
    function labelsUpdateNewFrame(obj,force)
      if obj.isinit
        return;
      end
      if exist('force','var')==0
        force = false;
      end
      if ~isempty(obj.lblCore) && (obj.prevFrame~=obj.currFrame || force)
        obj.lblCore.newFrame(obj.prevFrame,obj.currFrame,obj.currTarget);
      end
      obj.prevAxesLabelsUpdate();
      if ~obj.gtIsGTMode
      obj.labels2VizUpdate();
    end
    end
    
    function labelsUpdateNewTarget(obj,prevTarget)
      if ~isempty(obj.lblCore)
        obj.lblCore.newTarget(prevTarget,obj.currTarget,obj.currFrame);
      end
      obj.prevAxesLabelsUpdate();
      if ~obj.gtIsGTMode
        obj.labels2VizUpdate();
      end
    end
    
    function labelsUpdateNewFrameAndTarget(obj,prevFrm,prevTgt)
      if ~isempty(obj.lblCore)
        obj.lblCore.newFrameAndTarget(...
          prevFrm,obj.currFrame,...
          prevTgt,obj.currTarget);
      end
      obj.prevAxesLabelsUpdate();
      if ~obj.gtIsGTMode
        obj.labels2VizUpdate();
      end
    end
        
  end
   
  %% GT mode
  methods
    function gtSetGTMode(obj,tf)
      validateattributes(tf,{'numeric' 'logical'},{'scalar'});
      tf = logical(tf);
      if tf==obj.gtIsGTMode
        % none, value unchanged
      else
        obj.gtIsGTMode = tf;
        nMov = obj.nmoviesGTaware;
        if nMov==0
          obj.movieSetNoMovie();
        else
          IMOV = 1; % FUTURE: remember last/previous iMov in "other" gt mode
          obj.movieSet(IMOV);
        end
        obj.updateFrameTableComplete();
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
    function gtInitSuggestions(obj,gtSuggType,nSamp)
      tblMFT = obj.gtGenerateSuggestions(gtSuggType,nSamp);
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
      
      lposCell = obj.labeledposGT;
      fcn = @(zm,zf,zt) (nnz(isnan(lposCell{-zm}(:,:,zf,zt)))==0);
      % a mft row is labeled if all pts are either labeled, or estocc, or
      % fullocc (lpos will be inf which is not nan)
      tfAllTgtsLbled = rowfun(fcn,tbl,...
        'InputVariables',{'mov' 'frm' 'iTgt'},...
        'OutputFormat','uni');
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
        lposXY = obj.labeledposGT{iMov}(:,:,frm,iTgt);
        obj.gtSuggMFTableLbled(tfInTbl) = nnz(isnan(lposXY))==0;
        obj.notify('gtSuggMFTableLbledUpdated');
      end
    end
    function tblMFT = gtGenerateSuggestions(obj,gtSuggType,nSamp)
      assert(isa(gtSuggType,'GTSuggestionType'));
      
      % Start with full table (every frame), then sample
      mfts = MFTSetEnum.AllMovAllTgtAllFrm;
      tblMFT = mfts.getMFTable(obj);
      tblMFT = gtSuggType.sampleMFTTable(tblMFT,nSamp);
    end
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
    function tblGTres = gtComputeGTPerformance(obj)
      tblMFTSugg = obj.gtSuggMFTable;
      mfts = MFTSet(MovieIndexSetVariable.AllGTMov,...
        FrameSetVariable.LabeledFrm,FrameDecimationFixed(1),...
        TargetSetVariable.AllTgts);    
      tblMFTLbld = mfts.getMFTable(obj);
      
      [tf,loc] = ismember(tblMFTSugg,tblMFTLbld);
      assert(isequal(tf,obj.gtSuggMFTableLbled));
      nSuggLbled = nnz(tf);
      nSuggUnlbled = nnz(~tf);
      if nSuggUnlbled>0
        warningNoTrace('Labeler:gt',...
          '%d suggested GT frames have not been labeled.',nSuggUnlbled);
      end
      
      nTotGTLbled = height(tblMFTLbld);
      if nTotGTLbled>nSuggLbled
        warningNoTrace('Labeler:gt',...
          '%d labeled GT frames were not in list of suggestions. These labels will NOT be used in assessing GT performance.',...
          nTotGTLbled-nSuggLbled);
      end 
      
      % Labeled GT table, in order of tblMFTSugg
      tblMFT_SuggAndLbled = tblMFTLbld(loc(tf),:);
        
      tObj = obj.tracker;
      tObj.track(tblMFT_SuggAndLbled);
            
      % get results and compute GT perf
      tblTrkRes = tObj.getAllTrackResTable();
      [tf,loc] = tblismember(tblMFT_SuggAndLbled,tblTrkRes,MFTable.FLDSID);
      assert(all(tf));
      tblTrkRes = tblTrkRes(loc,:);
      tblMFT_SuggAndLbled = obj.labelAddLabelsMFTable(tblMFT_SuggAndLbled);
      pTrk = tblTrkRes.pTrk;
      pLbl = tblMFT_SuggAndLbled.p;
      nrow = size(pTrk,1);
      npts = obj.nLabelPoints;
      szassert(pTrk,[nrow 2*npts]);
      szassert(pLbl,[nrow 2*npts]);
      
      % L2 err matrix: [nrow x npt]
      pTrk = reshape(pTrk,[nrow npts 2]);
      pLbl = reshape(pLbl,[nrow npts 2]);
      err = sqrt(sum((pTrk-pLbl).^2,3));
      muerr = mean(err,2);
      
      tblTmp = tblMFT_SuggAndLbled(:,{'p' 'pTS' 'tfocc' 'pTrx'});
      tblTmp.Properties.VariableNames = {'pLbl' 'pLblTS' 'tfoccLbl' 'pTrx'};
      tblGTres = [tblTrkRes tblTmp table(err,muerr,'VariableNames',{'L2err' 'meanL2err'})];
      
      obj.gtTblRes = tblGTres;
      obj.notify('gtResUpdated');
    end
    function h = gtReport(obj)
      t = obj.gtTblRes;
      t.meanOverPtsL2err = mean(t.L2err,2);
      clrs =  obj.labelPointsPlotInfo.Colors;
      nclrs = size(clrs,1);
      npts = size(t.L2err,2);
      assert(npts==obj.nLabelPoints);
      if nclrs~=npts
        warningNoTrace('Labeler:gt',...
          'Number of colors do not match number of points.');
      end
      
      % Err by landmark
      h = figure('Name','GT err by landmark');
      ax = axes;
      boxplot(t.L2err,'colors',clrs,'boxstyle','filled');
      args = {'fontweight' 'bold' 'interpreter' 'none'};
      xlabel(ax,'Landmark/point',args{:});
      ylabel(ax,'L2 err (px)',args{:});
      title(ax,'GT err by landmark',args{:});
      ax.YGrid = 'on';
      
      % AvErrAcrossPts by movie
      h(end+1,1) = figurecascaded(h(end),'Name','Mean GT err by movie');
      ax = axes;
      [iMovAbs,gt] = t.mov.get;
      assert(all(gt));
      grp = categorical(iMovAbs);
      grplbls = arrayfun(@(z1,z2)sprintf('mov%s (n=%d)',z1{1},z2),...
        categories(grp),countcats(grp),'uni',0);
      boxplot(t.meanOverPtsL2err,grp,'colors',clrs,'boxstyle','filled',...
        'labels',grplbls);
      args = {'fontweight' 'bold' 'interpreter' 'none'};
      xlabel(ax,'Movie',args{:});
      ylabel(ax,'L2 err (px)',args{:});
      title(ax,'Mean (over landmarks) GT err by movie',args{:});
      ax.YGrid = 'on';
      
      % Mean err by movie, pt
      h(end+1,1) = figurecascaded(h(end),'Name','Mean GT err by movie, landmark');
      ax = axes;
      tblStats = grpstats(t(:,{'mov' 'L2err'}),{'mov'});
      tblStats.mov = tblStats.mov.get;
      tblStats = sortrows(tblStats,{'mov'});
      movUnCnt = tblStats.GroupCount; % [nmovx1]
      meanL2Err = tblStats.mean_L2err; % [nmovxnpt]
      nmovUn = size(movUnCnt,1);
      szassert(meanL2Err,[nmovUn npts]);
      meanL2Err(:,end+1) = nan; % pad for pcolor
      meanL2Err(end+1,:) = nan;       
      hPC = pcolor(meanL2Err);
      hPC.LineStyle = 'none';
      colorbar;
      xlabel(ax,'Landmark/point',args{:});
      ylabel(ax,'Movie',args{:});
      xticklbl = arrayfun(@num2str,1:npts,'uni',0);
      yticklbl = arrayfun(@(x)sprintf('mov%d (n=%d)',x,movUnCnt(x)),1:nmovUn,'uni',0);
      set(ax,'YTick',0.5+(1:nmovUn),'YTickLabel',yticklbl);
      set(ax,'XTick',0.5+(1:npts),'XTickLabel',xticklbl);
      axis(ax,'ij');
      title(ax,'Mean GT err (px) by movie, landmark',args{:});
      
      % Montage
      obj.trackLabelMontage(t,'meanOverPtsL2err','hPlot',h);
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
      obj.suspScore = cell(size(obj.labeledpos));
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
    
    function suspComputeUI(obj)
      tfsucc = obj.suspCompute();
      if ~tfsucc
        return;
      end
      figtitle = sprintf('Suspicious frames: %s',obj.suspDiag);
      hF = figure('Name',figtitle);
      tbl = obj.suspSelectedMFT;
      tblFlds = tbl.Properties.VariableNames;
      nt = NavigationTable(hF,[0 0 1 1],@(i,rowdata)obj.suspCbkTblNaved(i),...
        'ColumnName',tblFlds);
      nt.setData(tbl);
%       nt.navOnSingleClick = true;
      hF.UserData = nt;
%       kph = SuspKeyPressHandler(nt);
%       setappdata(hF,'keyPressHandler',kph);

      % See LabelerGUI/addDepHandle
      handles = obj.gdata;
      handles.depHandles(end+1,1) = hF;
      guidata(obj.hFig,handles);
    end
    
    function suspCbkTblNaved(obj,i)
      % i: row index into .suspSelectedMFT;
      tbl = obj.suspSelectedMFT;
      nrow = height(tbl);
      if i<1 || i>nrow
        error('Labeler:susp','Row ''%d'' out of bounds.',i);
      end
      mftrow = tbl(i,:);
      if obj.currMovie~=mftrow.mov
        obj.movieSet(mftrow.mov);
      end
      obj.setFrameAndTarget(mftrow.frm,mftrow.iTgt);
    end
  end
  
  methods (Hidden)
    
    function suspVerifyScore(obj,suspscore)
      nmov = obj.nmovies;
      if ~(iscell(suspscore) && numel(suspscore)==nmov)
        error('Labeler:susp',...
          'Invalid ''suspscore'' output from suspicisouness computation.');
      end
      lpos = obj.labeledpos;
      for imov=1:nmov
        [~,~,nfrm,ntgt] = size(lpos{imov});
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
      obj.preProcParams = [];
      obj.preProcH0 = [];
      obj.preProcInitData();
    end
    
    function preProcInitData(obj)
      % Initialize .preProcData*
      
      I = cell(0,1);
      tblP = MFTable.emptyTable(MFTable.FLDSCORE);
      obj.preProcData = CPRData(I,tblP);
      obj.preProcDataTS = now;
    end
    
    function tfPPprmsChanged = preProcSetParams(obj,ppPrms)
      assert(isstruct(ppPrms));      
      ppPrms0 = obj.preProcParams;
      tfPPprmsChanged = ~isequaln(ppPrms0,ppPrms);
      if tfPPprmsChanged
        warningNoTrace('Preprocessing parameters altered; data cache cleared.');
        obj.preProcInitData();
      end
      obj.preProcParams = ppPrms;
    end
    
    function tblP = preProcCropLabelsToRoiIfNec(obj,tblP)
      % Add .roi column to table if appropriate/nec
      %
      % If hasTrx, modify tblP as follows:
      % - add .roi
      % - set .p to be .pRoi, ie relative to ROI
      % - set .pAbs to be absolute .p
      % If not hasTrx,
      % - .p will be pAbs
      % - no .roi
      
      if obj.hasTrx
        tf = tblfldscontains(tblP,{'roi' 'pRoi' 'pAbs'});
        assert(all(tf) || ~any(tf));
        if ~any(tf)
          roiRadius = obj.preProcParams.TargetCrop.Radius;
          tblP = obj.labelMFTableAddROI(tblP,roiRadius);
          tblP.pAbs = tblP.p;
          tblP.p = tblP.pRoi;
        end
      else
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
      % - .roi is guaranteed when .hasTrx.

      wbObj = myparse(varargin,...
        'wbObj',[] ... % optional WaitBarWithCancel. If cancel:
                   ... % 1. obj const 
                   ... % 2. tblP indeterminate
        ); 
      tfWB = ~isempty(wbObj);
      
      tblP = obj.labelGetMFTableLabeled('wbObj',wbObj);
      if tfWB && wbObj.isCancel
        % tblP indeterminate, return it anyway
        return;
      end
      
      tblP = obj.preProcCropLabelsToRoiIfNec(tblP);
      tfnan = any(isnan(tblP.p),2);
      nnan = nnz(tfnan);
      if nnan>0
        warningNoTrace('Labeler:nanData',...
          'Not including %d partially-labeled rows.',nnan);
      end
      tblP = tblP(~tfnan,:);
    end
    
    % Hist Eq Notes
    %
    % The Labeler has the ability/responsibility to compute a "typical" 
    % image histogram H0 representative of all movies in the current 
    % project. At the moment it does this by sampling frames at intervals 
    % (weighting all frames equally regardless of movie), but this is an 
    % impl detail.
    %
    % .preProcH0 is the image histogram used to generate the current
    % .preProcData. Conceptually, .preProcH0 is like .preProcParams in that
    % it is a parameter (vector) governing how data is preprocessed.
    % Instead of being user-set like other parameters, .preProcH0 is
    % updated/learned from the project movies.
    %
    % .preProcH0 is set at retrain- (fulltraining-) time. It is not updated
    % during tracking, or during incremental trains. If users are adding
    % movies/labels, they should periodically retrain fully, to refresh 
    % .preProcH0 relative to the movies.
    %
    % Note that .preProcData has its own .H0 for historical reasons. This
    % should always coincide with .preProcH0, except possibly in edge cases
    % where one or the other is empty.
    % 
    % SUMMARY MAIN OPERATIONS
    % retrain: this updates .preProcH0 and also forces .preProcData.H0
    % to match; .preProcData is initially cleared if necessary.
    % inctrain: .preProcH0 is untouched. .preProcData.H0 continues to match 
    % .preProcH0.
    % track: same as inctrain.
    
    function preProcUpdateH0IfNec(obj)
      ppPrms = obj.preProcParams;
      if ppPrms.histeq
        assert(obj.nview==1,...
          'Histogram Equalization currently unsupported for multiview projects.');
        assert(~obj.hasTrx,...
          'Histogram Equalization currently unsupported for multitarget projects.');
        
        nFrmSampH0 = ppPrms.histeqH0NumFrames;
        H0 = obj.movieEstimateImHist(nFrmSampH0);
        
        data = obj.preProcData;
        if ~isequal(data.H0,obj.preProcH0)
          assert(data.N==0 || ... % empty .data
                 isempty(obj.preProcH0),... 
                 '.preProcData.H0 differs from .preProcH0');
        end
        if ~isequal(H0,data.H0)
          obj.preProcInitData();
        end
        obj.preProcH0 = H0;
      else
        assert(isempty(obj.preProcData.H0));
        assert(isempty(obj.preProcH0));
      end      
    end
    
    function [data,dataIdx] = preProcDataFetch(obj,tblP,varargin)
      % dataUpdate, then retrieve
      %
      % Input args: See PreProcDataUpdate
      %
      % data: CPRData handle, equal to obj.preProcData
      % dataIdx. data.I(dataIdx,:) gives the rows corresponding to tblP
      %   (order preserved)
      
      wbObj = myparse(varargin,...
        'wbObj',[]); % WaitBarWithCancel. If cancel: obj unchanged, data and dataIdx are [].
      tfWB = ~isempty(wbObj);
      
      obj.preProcDataUpdate(tblP,'wbObj',wbObj);
      if tfWB && wbObj.isCancel
        data = [];
        dataIdx = [];
        return;
      end
      
      data = obj.preProcData;
      [tf,dataIdx] = tblismember(tblP,data.MD,MFTable.FLDSID);
      assert(all(tf));
    end
    
    function preProcDataUpdate(obj,tblP,varargin)
      % Update .preProcData to include tblP
      %
      % tblP:
      %   - MFTable.FLDSCORE: required.
      %   - .roi: optional, USED WHEN PRESENT. (prob needs to be either
      %   consistently there or not-there for a given obj or initData()
      %   "session"
      %   - .pTS: optional (if present, deleted)
      
      wbObj = myparse(varargin,...
        'wbObj',[]); % WaitBarWithCancel. If cancel, obj unchanged.
      
      if any(strcmp('pTS',tblP.Properties.VariableNames))
        % AL20170530: Not sure why we do this
        tblP(:,'pTS') = [];
      end
      [tblPnew,tblPupdate] = obj.preProcData.tblPDiff(tblP);
      obj.preProcDataUpdateRaw(tblPnew,tblPupdate,'wbObj',wbObj);
    end
    
    function preProcDataUpdateRaw(obj,tblPnew,tblPupdate,varargin)
      % Incremental data update
      %
      % * Rows appended and pGT/tfocc updated; but other information
      % untouched
      % * histeq (if enabled) uses .preProcH0. See "Hist Eq Notes" below.
      % .preProcH0 is NOT updated here.
      %
      % QUESTION: why is pTS not updated?
      %
      % tblPNew: new rows. MFTable.FLDSCORE are required fields. .roi may 
      %   be present and if so WILL BE USED to grab images and included in 
      %   data/MD. Other fields are ignored.
      % tblPupdate: updated rows (rows with updated pGT/tfocc).
      %   MFTable.FLDSCORE fields are required. Only .pGT and .tfocc are 
      %   otherwise used. Other fields ignored.
      %
      % Updates .preProcData, .preProcDataTS
      
      wbObj = myparse(varargin,...
        'wbObj',[]); % Optional WaitBarWithCancel obj. If cancel, obj unchanged.
      tfWB = ~isempty(wbObj);
      
      FLDSREQUIRED = MFTable.FLDSCORE;
      FLDSALLOWED = [MFTable.FLDSCORE {'roi' 'nNborMask'}];
      tblfldscontainsassert(tblPnew,FLDSREQUIRED);
      tblfldscontainsassert(tblPupdate,FLDSREQUIRED);
      
      prmpp = obj.preProcParams;
      if isempty(prmpp)
        error('Please specify tracking parameters.');
      end
      
      dataCurr = obj.preProcData;
      
      if prmpp.histeq
        assert(dataCurr.N==0 || isequal(dataCurr.H0,obj.preProcH0));
        assert(obj.nview==1,...
          'Histogram Equalization currently unsupported for multiview tracking.');
        assert(~obj.hasTrx,...
          'Histogram Equalization currently unsupported for multitarget tracking.');
      end
      if ~isempty(prmpp.channelsFcn)
        assert(obj.nview==1,...
          'Channels preprocessing currently unsupported for multiview tracking.');
      end
      
      %%% NEW ROWS read images + PP. Append to dataCurr. %%%
      FLDSID = MFTable.FLDSID;
      assert(~any(tblismember(tblPnew,dataCurr.MD,FLDSID)));
      
      tblPNewConcrete = obj.mftTableConcretizeMov(tblPnew);
      nNew = height(tblPnew);
      if nNew>0
        fprintf(1,'Adding %d new rows to data...\n',nNew);
        
        [I,nNborMask] = CPRData.getFrames(tblPNewConcrete,...
          'wbObj',wbObj,...
          'trxCache',obj.trxCache,...
          'maskNeighbors',prmpp.NeighborMask.Use,...
          'maskNeighborsMeth',prmpp.NeighborMask.SegmentMethod,...
          'maskNeighborsEmpPDF',obj.fgEmpiricalPDF,...
          'bgType',prmpp.NeighborMask.BGType,...
          'bgReadFcn',prmpp.NeighborMask.BGReadFcn,...
          'fgThresh',prmpp.NeighborMask.FGThresh);
        if tfWB && wbObj.isCancel
          % obj unchanged
          return;
        end
        % Include only FLDSALLOWED in metadata to keep CPRData md
        % consistent (so can be appended)
        
        tfColsAllowed = ismember(tblPnew.Properties.VariableNames,...
          FLDSALLOWED);
        tblPnewMD = tblPnew(:,tfColsAllowed);
        tblPnewMD = [tblPnewMD table(nNborMask)];
        dataNew = CPRData(I,tblPnewMD);
        if prmpp.histeq
          H0 = obj.preProcH0;
          assert(~isempty(H0),'H0 unavailable for histeq/preprocessing.');
          gHE = categorical(dataNew.MD.mov);
          dataNew.histEq('g',gHE,'H0',H0,'wbObj',wbObj);
          if tfWB && wbObj.isCancel
            % obj unchanged
            return;
          end
          assert(isequal(dataNew.H0,H0));
          if dataCurr.N==0
            dataCurr.H0 = H0;
            % dataCurr.H0, dataNew.H0 need to match for append()
          end
        end
        if ~isempty(prmpp.channelsFcn)
          feval(prmpp.channelsFcn,dataNew);
          assert(~isempty(dataNew.IppInfo),...
            'Preprocessing channelsFcn did not set .IppInfo.');
          if isempty(dataCurr.IppInfo)
            assert(dataCurr.N==0,'Ippinfo can be empty only for empty/new data.');
            dataCurr.IppInfo = dataNew.IppInfo;
          end
        end
        
        dataCurr.append(dataNew);
      end
      
      %%% EXISTING ROWS -- just update pGT and tfocc. Existing images are
      %%% OK and already histeq'ed correctly
      nUpdate = size(tblPupdate,1);
      if nUpdate>0 % AL 20160413 Shouldn't need to special-case, MATLAB 
                   % table indexing API may not be polished
        fprintf(1,'Updating labels for %d rows...\n',nUpdate);
        [tf,loc] = tblismember(tblPupdate,dataCurr.MD,FLDSID);
        assert(all(tf));
        dataCurr.MD{loc,'tfocc'} = tblPupdate.tfocc; % AL 20160413 throws if nUpdate==0
        dataCurr.pGT(loc,:) = tblPupdate.p;
      end
      
      if nUpdate>0 || nNew>0
        assert(obj.preProcData==dataCurr); % handles
        obj.preProcDataTS = now;
      else
        warningNoTrace('Nothing to update in data.');
      end
    end
   
  end
  

  %% Tracker
  methods    

    function trackSetParams(obj,sPrm)
      % sPrm: scalar struct containing both preproc + tracking params

      tObj = obj.tracker;
      if isempty(tObj)
        error('No tracker is set.');
      end
       
      ppPrms = sPrm.PreProc;
      sPrm = rmfield(sPrm,'PreProc');
      
      tfPPprmsChanged = obj.preProcSetParams(ppPrms);
      tObj.setParamContentsSmart(sPrm,tfPPprmsChanged);
    end
    
    function sPrm = trackGetParams(obj)
      % sPrm: scalar struct containing both preproc + tracking params
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('No tracker is set.');
      end
      
      sPrm = tObj.sPrm;
      if isempty(sPrm)
        % new tracker
        sPrm = struct();
      end
      
      assert(~isfield(sPrm,'PreProc'));
      sPrm.PreProc = obj.preProcParams;
    end
    
    function trackTrain(obj)
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      if ~obj.hasMovie
        error('Labeler:track','No movie.');
      end
      tObj.train();
    end
    
    function trackRetrain(obj,varargin)
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      if ~obj.hasMovie
        error('Labeler:track','No movie.');
      end
      obj.preProcUpdateH0IfNec();
      tObj.retrain(varargin{:});
    end
    
    function track(obj,mftset,varargin)
      % mftset: an MFTSet
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      assert(isa(mftset,'MFTSet'));
      tblMFT = mftset.getMFTable(obj);
      tObj.track(tblMFT,varargin{:});
      
      % For template mode to see new tracking results
      obj.labelsUpdateNewFrame(true);
      
      fprintf('Tracking complete at %s.\n',datestr(now));
    end
    
    function trackTbl(obj,tblMFT,varargin)
      tObj = obj.tracker;
      tObj.track(tblMFT,varargin{:});
      % For template mode to see new tracking results
      obj.labelsUpdateNewFrame(true);
      
      fprintf('Tracking complete at %s.\n',datestr(now));
    end
    
    function trackAndExport(obj,mftset,varargin)
      % Track one movie at a time, exporting results to .trk files and 
      % clearing data in between
      %
      % mftset: scalar MFTSet
            
      [trackArgs,rawtrkname] = myparse(varargin,...
        'trackArgs',{},...
        'rawtrkname',[]...
        );
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      tblMFT = mftset.getMFTable(obj);
      
      iMovsUn = unique(tblMFT.mov);      
      [tfok,trkfiles] = obj.resolveTrkfilesVsTrkRawname(iMovsUn,[],...
        rawtrkname,{});
      if ~tfok
        return;
      end
      
      nMov = numel(iMovsUn);
      nVw = obj.nview;
      szassert(trkfiles,[nMov nVw]);
      if obj.isMultiView
        moviestr = 'movieset';
      else
        moviestr = 'movie';
      end
      for i=1:nMov
        iMov = iMovsUn(i);
        fprintf('Tracking %s %d (%d/%d)\n',moviestr,double(iMov),i,nMov);
        
        tfMov = tblMFT.mov==iMov;
        tblMFTmov = tblMFT(tfMov,:);
        tObj.track(tblMFTmov,trackArgs{:});
        trkFile = tObj.getTrackingResults(iMov);
        szassert(trkFile,[1 nVw]);
        for iVw=1:nVw
          trkFile(iVw).save(trkfiles{i,iVw});
          fprintf('...saved: %s\n',trkfiles{i,iVw});
        end
        tObj.clearTrackingResults();
        obj.preProcInitData();
      end
    end
    
    function trackExportResults(obj,iMovs,varargin)
      % Export tracking results to trk files.
      %
      % iMovs: [nMov] vector of movie(set)s whose tracking should be
      % exported.
      %
      % If a movie has no current tracking results, a warning is thrown and
      % no trkfile is created.
      
      [trkfiles,rawtrkname] = myparse(varargin,...
        'trkfiles',[],... % [nMov nView] cellstr, fullpaths to trkfilenames to export to
        'rawtrkname',[]... % string, basename to apply over iMovs to generate trkfiles
        );
      
      assert(~obj.gtIsGTMode);
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end 
      
      [tfok,trkfiles] = obj.resolveTrkfilesVsTrkRawname(iMovs,trkfiles,...
        rawtrkname,{});
      if ~tfok
        return;
      end
      
      movfiles = obj.movieFilesAllFull(iMovs,:);
      mIdx = MovieIndex(iMovs);
      [trkFileObjs,tfHasRes] = tObj.getTrackingResults(mIdx);
      nMov = numel(iMovs);
      nVw = obj.nview;
      szassert(trkFileObjs,[nMov nVw]);
      szassert(trkfiles,[nMov nVw]);
      for iMv=1:nMov
        if tfHasRes(iMv)
          for iVw=1:nVw
            trkFileObjs(iMv,iVw).save(trkfiles{iMv,iVw});
            fprintf('Saved %s.\n',trkfiles{iMv,iVw});
          end
        else
          if obj.isMultiView
            moviestr = 'movieset';
          else
            moviestr = 'movie';
          end
          warningNoTrace('Labeler:noRes','No current tracking results for %s %s.',...
            moviestr,MFTable.formMultiMovieID(movfiles(iMv,:)));
        end
      end
    end
        
    function trackCrossValidate(obj,varargin)
      % Run k-fold crossvalidation. Results stored in .xvResults
      
      [kFold,initData,wbObj,tblMFgt] = myparse(varargin,...
        'kfold',7,... % number of folds
        'initData',true,... % if true, call .initData() between folds to minimize mem usage
        'wbObj',[],... % WaitBarWithCancel
        'tblMFgt',[]... % optional, labeled/gt data to use
        );
      
      tfWB = ~isempty(wbObj);      
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:tracker','No tracker is available for this project.');
      end
      
      % Get labeled/gt data
      if isempty(tblMFgt)
        tblMFgt = obj.preProcGetMFTableLbled();
      end
      
      % Partition MFT table
      movC = categorical(tblMFgt.mov);
      tgtC = categorical(tblMFgt.iTgt);
      grpC = movC.*tgtC;
      cvPart = cvpartition(grpC,'kfold',kFold);

      obj.preProcUpdateH0IfNec();
      
      % Basically an initHook() here
      if initData
        obj.preProcInitData();
      end
      tObj.trnDataInit(); % not strictly necessary as .retrain() should do it 
      tObj.trnResInit(); % not strictly necessary as .retrain() should do it 
      tObj.trackResInit();
      tObj.vizInit();
      tObj.asyncReset();
      
      npts = obj.nLabelPoints;
      pTrkCell = cell(kFold,1);
      dGTTrkCell = cell(kFold,1);
      if tfWB
        wbObj.startPeriod('Fold','shownumden',true,'denominator',kFold);
      end
      for iFold=1:kFold
        if tfWB
          wbObj.updateFracWithNumDen(iFold);
        end
        tblMFgtTrain = tblMFgt(cvPart.training(iFold),:);
        tblMFgtTrack = tblMFgt(cvPart.test(iFold),:);
        if tfWB
          wbObj.startPeriod('Training','nobar',true);
        end
        tObj.retrain('tblPTrn',tblMFgtTrain);
        if tfWB
          wbObj.endPeriod();
        end
        tObj.track(tblMFgtTrack,'wbObj',wbObj);        
        [tblTrkRes,pTrkiPt] = tObj.getAllTrackResTable(); % if wbObj.isCancel, partial tracking results
        if initData
          obj.preProcInitData();
        end
        tObj.trnDataInit();
        tObj.trnResInit();
        tObj.trackResInit();
        if tfWB && wbObj.isCancel
          return;
        end
        
        assert(isequal(pTrkiPt(:)',1:npts));
        assert(isequal(tblTrkRes(:,MFTable.FLDSID),...
                       tblMFgtTrack(:,MFTable.FLDSID)));
        if obj.hasTrx
          pGT = tblMFgtTrack.pAbs;
        else
          tblfldsdonotcontainassert(tblMFgtTrack,'pAbs');
          pGT = tblMFgtTrack.p;
        end
        d = tblTrkRes.pTrk - pGT;
        [ntst,Dtrk] = size(d);
        assert(Dtrk==npts*2); % npts=nPhysPts*nview
        d = reshape(d,ntst,npts,2);
        d = sqrt(sum(d.^2,3)); % [ntst x npts]
        
        pTrkCell{iFold} = tblTrkRes;
        dGTTrkCell{iFold} = d;
      end

      % create output table
      for iFold=1:kFold
        tblFold = table(repmat(iFold,height(pTrkCell{iFold}),1),...
          'VariableNames',{'fold'});
        pTrkCell{iFold} = [tblFold pTrkCell{iFold}];
      end
      pTrkAll = cat(1,pTrkCell{:});
      dGTTrkAll = cat(1,dGTTrkCell{:});
      assert(isequal(height(pTrkAll),height(tblMFgt),size(dGTTrkAll,1)));
      [tf,loc] = tblismember(tblMFgt,pTrkAll,MFTable.FLDSID);
      assert(all(tf));
      pTrkAll = pTrkAll(loc,:);
      dGTTrkAll = dGTTrkAll(loc,:);
      
      if tblfldscontains(tblMFgt,'roi')
        flds = MFTable.FLDSCOREROI;
      else
        flds = MFTable.FLDSCORE;
      end
      tblXVres = tblMFgt(:,flds);
      if tblfldscontains(tblMFgt,'pAbs')
        tblXVres.p = tblMFgt.pAbs;
      end
      tblXVres.pTrk = pTrkAll.pTrk;
      tblXVres.dGTTrk = dGTTrkAll;
      tblXVres = [pTrkAll(:,'fold') tblXVres];
      
      obj.xvResults = tblXVres;
      obj.xvResultsTS = now;
    end
    
    function trackCrossValidateVizPrctiles(obj,tblXVres,varargin)
      ptiles = myparse(varargin,...
        'prctiles',[50 75 90 95]);
      
      assert(~obj.isMultiView,'Not supported for multiview.');
      assert(~obj.gtIsGTMode,'Not supported in GT mode.');
      
      err = tblXVres.dGTTrk;
      npts = obj.nLabelPoints;
      assert(size(err,2)==npts);
      nptiles = numel(ptiles);
      ptl = prctile(err,ptiles,1)';
      szassert(ptl,[npts nptiles]);
      
      % for now always viz with first row
      iVizRow = 1;
      vizrow = tblXVres(iVizRow,:);
      mr = MovieReader;
      mr.open(obj.movieFilesAllFull{vizrow.mov,1});
      mr.forceGrayscale = obj.movieForceGrayscale;
      im = mr.readframe(vizrow.frm);
      pLbl = vizrow.p;
      if tblfldscontains(vizrow,'roi')
        xlo = vizrow.roi(1);
        xhi = vizrow.roi(2);
        ylo = vizrow.roi(3);
        yhi = vizrow.roi(4);
        szassert(pLbl,[1 2*npts]);
        pLbl(1:npts) = pLbl(1:npts)-xlo+1;
        pLbl(npts+1:end) = pLbl(npts+1:end)-ylo+1;
        im = im(ylo:yhi,xlo:xhi);
      end
      xyLbl = Shape.vec2xy(pLbl);
      szassert(xyLbl,[npts 2]);
      
      figure('Name','Tracker performance');
      imagesc(im);
      colormap gray;
      hold on;
      clrs = hsv(nptiles)*.75;
      for ipt=1:npts
        for iptile=1:nptiles        
          r = ptl(ipt,iptile);
          pos = [xyLbl(ipt,:)-r 2*r 2*r];
           rectangle('position',pos,'curvature',1,'edgecolor',clrs(iptile,:));
        end
      end
      axis square xy
      grid on;
      tstr = sprintf('Cross-validation ptiles: %s. n=%d, mean err=%.3fpx.',...
        mat2str(ptiles),height(tblXVres),mean(err(:)));
      title(tstr,'fontweight','bold','interpreter','none');
      hLeg = arrayfun(@(x)plot(nan,nan,'-','Color',clrs(x,:)),1:nptiles,...
        'uni',0);      
      legend([hLeg{:}],arrayfun(@num2str,ptiles,'uni',0));
    end
    
    function [tf,lposTrk] = trackIsCurrMovFrmTracked(obj,iTgt)
      % tf: scalar logical, true if tracker has results/predictions for 
      %   currentMov/frm/iTgt 
      % lposTrk: [nptsx2] if tf is true, xy coords from tracker; otherwise
      %   indeterminate
      
      tObj = obj.tracker;
      if isempty(tObj)
        tf = false;
        lposTrk = [];
      else
        xy = tObj.getPredictionCurrentFrame(); % [nPtsx2xnTgt]
        szassert(xy,[obj.nLabelPoints 2 obj.nTargets]);
        lposTrk = xy(:,:,iTgt);
        tfnan = isnan(xy);
        tf = any(~tfnan(:));
      end
    end
    
    function trackLabelMontage(obj,tbl,errfld,varargin)
      [nr,nc,h,npts,nphyspts] = myparse(varargin,...
        'nr',3,...
        'nc',4,...
        'hPlot',[],...
        'npts',nan,... % hack
        'nphyspts',nan); % hack
      tbl = sortrows(tbl,{errfld},{'descend'});
      % Could look first in .tracker.data, and/or use .tracker.updateData()
      tConcrete = obj.mftTableConcretizeMov(tbl);
      I = CPRData.getFrames(tConcrete);
      pTrk = tbl.pTrk;
      pLbl = tbl.pLbl;
      if isnan(npts)
        npts = obj.nLabelPoints;
      end
      if tblfldscontains(tbl,{'roi'})
        roi = tbl.roi;
        assert(size(pTrk,2)==2*npts);
        assert(size(pLbl,2)==2*npts);
        xlo = roi(:,1);
        ylo = roi(:,3);
        pTrk(:,1:npts) = bsxfun(@minus,pTrk(:,1:npts),xlo)+1;
        pLbl(:,1:npts) = bsxfun(@minus,pLbl(:,1:npts),xlo)+1;
        pTrk(:,npts+1:end) = bsxfun(@minus,pTrk(:,npts+1:end),ylo)+1;
        pLbl(:,npts+1:end) = bsxfun(@minus,pLbl(:,npts+1:end),ylo)+1; 
      end
      
      frmLblsAll = arrayfun(@(x1,x2)sprintf('frm=%d,err=%.2f',x1,x2),...
        tbl.frm,tbl.(errfld),'uni',0);
      if isnan(nphyspts)
        nphyspts = obj.nPhysPoints;
      end
      
      nrows = height(tbl);
      startIdxs = 1:nr*nc:nrows;
      for i=1:numel(startIdxs)
        plotIdxs = startIdxs(i):min(startIdxs(i)+nr*nc-1,nrows);
        frmLblsThis = frmLblsAll(plotIdxs);
        for iView=1:obj.nview
          h(end+1,1) = figure('Name','Tracking Error Montage','windowstyle','docked'); %#ok<AGROW>
          pColIdx = (1:nphyspts)+(iView-1)*nphyspts;
          pColIdx = [pColIdx pColIdx+npts]; %#ok<AGROW>
          Shape.montage(I(:,iView),pLbl(:,pColIdx),'fig',h(end),...
            'nr',nr,'nc',nc,'idxs',plotIdxs,...
            'framelbls',frmLblsThis,'framelblscolor',[1 1 .75],'p2',pTrk(:,pColIdx),...
            'p2marker','+','titlestr','Tracking Montage, descending err (''+'' is tracked)');
        end
      end
    end
  end
  methods (Static)
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
  end
  methods    
    function tblBig = trackGetBigLabeledTrackedTable(obj,varargin)
      % tbl: MFT table indcating isLbled, isTrked, trkErr, etc.
      
      wbObj = myparse(varargin,...
        'wbObj',[]); % optional WaitBarWithContext. If .isCancel:
                     % 1. tblBig is indeterminate
                     % 2. obj should be logically const
      tfWB = ~isempty(wbObj);
      
      tblLbled = obj.labelGetMFTableLabeled('wbObj',wbObj);
      if tfWB && wbObj.isCancel
        tblBig = [];
        return;
      end      
      tblLbled = Labeler.hlpTblLbled(tblLbled);
      
      tblLbled2 = obj.labelGetMFTableLabeled('wbObj',wbObj,'useLabels2',true);
      if tfWB && wbObj.isCancel
        tblBig = [];
        return;
      end

      tblLbled2 = Labeler.hlpTblLbled(tblLbled2);
      tblfldsassert(tblLbled2,[MFTable.FLDSID {'p' 'isLbled'}]);
      tblLbled2.Properties.VariableNames(end-1:end) = {'pImport' 'isImported'};
      
      npts = obj.nLabelPoints;
      
      tObj = obj.tracker;
      if ~isempty(tObj)
        tblTrked = tObj.trkPMD(:,MFTable.FLDSID);
        tblTrked.mov = int32(tblTrked.mov); % from MovieIndex
        pTrk = tObj.trkP;
        if isempty(pTrk)
          % edge case, tracker not initting properly
          pTrk = nan(0,npts*2);
        end
        isTrked = true(height(tblTrked),1);
        tblTrked = [tblTrked table(pTrk,isTrked)];
      else
        tblTrked = table(nan(0,1),nan(0,1),nan(0,1),nan(0,npts*2),false(0,1),...
          'VariableNames',{'mov' 'frm' 'iTgt' 'pTrk' 'isTrked'});
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
        tblXVerr = rowfun(@nanmean,xvres,'InputVariables',{'dGTTrk'},...
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
    end
    
    function tblSumm = trackGetSummaryTable(obj,tblBig)
      % tblSumm: Big summary table, one row per (mov,tgt)
      
      assert(~obj.gtIsGTMode,'Currently unsupported in GT mode.');
      
      % generate tblSummBase
      if obj.hasTrx
        assert(obj.nview==1,'Currently unsupported for multiview projects.');
        tfaf = obj.trxFilesAllFull;
        dataacc = nan(0,4); % mov, tgt, trajlen, frm1
        for iMov=1:obj.nmovies
          tfile = tfaf{iMov,1};
          tifo = obj.trxCache(tfile);
          frm2trxI = tifo.frm2trx;

          nTgt = size(frm2trxI,2);
          for iTgt=1:nTgt
            tflive = frm2trxI(:,iTgt);
            sp = get_interval_ends(tflive);
            if isempty(sp)
              trajlen = 0;
              frm1 = nan;
            else
              if numel(sp)>1
                warningNoTrace('Movie %d, target %d is live over non-consecutive frames.',...
                  iMov,iTgt);
              end
              trajlen = nnz(tflive); % when numel(sp)>1, track is 
              % non-consecutive and this won't strictly be trajlen
              frm1 = sp(1);
            end
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
  end
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
        lblTrkMeanErr = nanmean(trkErr(tfLbledTrked));
        
        tfLbledImported = isLbled & isImported;
        nFrmLblImported = nnz(tfLbledImported);
        lblImportedMeanErr = nanmean(importedErr(tfLbledImported));        
        
        nFrmXV = nnz(hasXV);
        xvMeanErr = mean(xvErr(hasXV));
      end
      
%     function trackSaveResults(obj,fname)
%       tObj = obj.tracker;
%       if isempty(tObj)
%         error('Labeler:track','No tracker set.');
%       end
%       s = tObj.getSaveToken(); %#ok<NASGU>
%       
%       save(fname,'-mat','-struct','s');
%       obj.projFSInfo = ProjectFSInfo('tracking results saved',fname);
%       RC.saveprop('lastTrackingResultsFile',fname);
%     end
%     
%     function trackLoadResults(obj,fname)
%       tObj = obj.tracker;
%       if isempty(tObj)
%         error('Labeler:track','No tracker set.');
%       end
%       s = load(fname);
%       tObj.loadSaveToken(s);
%       
%       obj.projFSInfo = ProjectFSInfo('tracking results loaded',fname);
%       RC.saveprop('lastTrackingResultsFile',fname);
%     end
          
%     function [success,fname] = trackSaveResultsAs(obj)
%       [success,fname] = obj.trackSaveLoadAsHelper('lastTrackingResultsFile',...
%         'uiputfile','Save tracking results','trackSaveResults');
%     end
%     
%     function [success,fname] = trackLoadResultsAs(obj)
%       [success,fname] = obj.trackSaveLoadAsHelper('lastTrackingResultsFile',...
%         'uigetfile','Load tracking results','trackLoadResults');
%     end
%     
%     function [success,fname] = trackSaveLoadAsHelper(obj,rcprop,uifcn,...
%         promptstr,rawMeth)
%       % rcprop: Name of RC property for guessing path
%       % uifcn: either 'uiputfile' or 'uigetfile'
%       % promptstr: used in uiputfile
%       % rawMeth: track*Raw method to call when a file is specified
%       
%       % Guess a path/location for save/load
%       lastFile = RC.getprop(rcprop);
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
   
  %% Video
  methods
    
    function videoCenterOnCurrTarget(obj)
      % Shift axis center/target and CameraUpVector without touching zoom.
      % 
      % Potential TODO: CamViewAngle treatment looks a little bizzare but
      % seems to work ok. Theoretically better (?), at movieSet time, cache
      % a default CameraViewAngle, and at movieRotateTargetUp set time, set
      % the CamViewAngle to either the default or the default/2 etc.
    
      [x0,y0] = obj.videoCurrentCenter;
      [x,y,th] = obj.currentTargetLoc();
      
      dx = x-x0;
      dy = y-y0;
      ax = obj.gdata.axes_curr;
      axisshift(ax,dx,dy);
      ax.CameraPositionMode = 'auto'; % issue #86, behavior differs between 16b and 15b. Use of manual zoom toggles .CPM into manual mode
      ax.CameraTargetMode = 'auto'; % issue #86, etc Use of manual zoom toggles .CTM into manual mode
      %ax.CameraViewAngleMode = 'auto';
      if obj.movieRotateTargetUp
        ax.CameraUpVector = [cos(th) sin(th) 0];
        if verLessThan('matlab','R2016a')
          % See iss#86. In R2016a, the zoom/pan behavior of axes in 3D mode
          % (currently, any axis with CameraViewAngleMode manually set)
          % changed. Prior to R2016a, zoom on such an axis altered camera
          % position via .CameraViewAngle, etc, with the axis limits
          % unchanged. Starting in R2016a, zoom on 3D axes changes the axis
          % limits while the camera position is unchanged.
          %
          % Currently we prefer the modern treatment and the
          % center-on-target, rotate-target, zoom slider, etc treatments
          % are built around that treatment. For prior MATLABs, we work
          % around -- it is a little awkward as the fundamental strategy
          % behind zoom is different. For prior MATLABs users should prefer
          % the Zoom slider in the Targets panel as opposed to using the
          % zoom tools in the toolbar.
          hF = obj.gdata.figure;
          tf = getappdata(hF,'manualZoomOccured');
          if tf
            ax.CameraViewAngleMode = 'auto';
            setappdata(hF,'manualZoomOccured',false);
          end
        end
        if strcmp(ax.CameraViewAngleMode,'auto')
          cva = ax.CameraViewAngle;
          ax.CameraViewAngle = cva/2;
        end
      else
        ax.CameraUpVectorMode = 'auto';
      end
    end
    
    function videoZoom(obj,zoomRadius)
      % Zoom to square window over current frame center with given radius.
      
      [x0,y0] = obj.videoCurrentCenter();
      lims = [x0-zoomRadius,x0+zoomRadius,y0-zoomRadius,y0+zoomRadius];
      axis(obj.gdata.axes_curr,lims);
    end    
    function [xsz,ysz] = videoCurrentSize(obj)
      v = axis(obj.gdata.axes_curr);
      xsz = v(2)-v(1);
      ysz = v(4)-v(3);
    end
    function [x0,y0] = videoCurrentCenter(obj)
      v = axis(obj.gdata.axes_curr);
      x0 = mean(v(1:2));
      y0 = mean(v(3:4));
    end
    
    function xy = videoClipToVideo(obj,xy)
      % Clip coords to video size.
      %
      % xy (in): [nx2] xy-coords
      %
      % xy (out): [nx2] xy-coords, clipped so that x in [1,nc] and y in [1,nr]
      
      xy(:,1) = min(max(xy(:,1),1),obj.movienc);
      xy(:,2) = min(max(xy(:,2),1),obj.movienr);      
    end
    function dxdy = videoCurrentUpVec(obj)
      % The main axis can be rotated, flipped, etc; Get the current unit 
      % "up" vector in (x,y) coords
      %
      % dxdy: [2] unit vector [dx dy] 
      
      ax = obj.gdata.axes_curr;
      if obj.hasTrx && obj.movieRotateTargetUp
        v = ax.CameraUpVector; % should be norm 1
        dxdy = v(1:2);
      else
        dxdy = [0 1];
      end
    end
    function dxdy = videoCurrentRightVec(obj)
      % The main axis can be rotated, flipped, etc; Get the current unit 
      % "right" vector in (x,y) coords
      %
      % dxdy: [2] unit vector [dx dy] 

      ax = obj.gdata.axes_curr;
      if obj.hasTrx && obj.movieRotateTargetUp
        v = ax.CameraUpVector; % should be norm 1
        parity = mod(strcmp(ax.XDir,'normal') + strcmp(ax.YDir,'normal'),2);
        if parity
          dxdy = [-v(2) v(1)]; % etc
        else
          dxdy = [v(2) -v(1)]; % rotate v by -pi/2.
        end
      else
        dxdy = [1 0];
      end      
    end
    
    function videoPlay(obj)
      obj.videoPlaySegmentCore(obj.currFrame,obj.nframes,...
        'setFrameArgs',{'updateTables',false});
    end
    
    function videoPlaySegment(obj)
      % Play segment centererd at .currFrame
      
      f = obj.currFrame;
      df = obj.moviePlaySegRadius;
      fstart = max(1,f-df);
      fend = min(obj.nframes,f+df);
      obj.videoPlaySegmentCore(fstart,fend,'freset',f,...
        'setFrameArgs',{'updateTables',false,'updateLabels',false});
    end
    
    function videoPlaySegmentCore(obj,fstart,fend,varargin)
      
      [setFrameArgs,freset] = myparse(varargin,...
        'setFrameArgs',{},...
        'freset',nan);
      tfreset = ~isnan(freset);
            
      ticker = tic;
      while true
        % Ways to exit loop:
        % 1. user cancels playback through GUI mutation of gdata.isPlaying
        % 2. fend reached
        % 3. ctrl-c
        
        guidata = obj.gdata;
        if ~guidata.isPlaying
          break;
        end
                  
        dtsec = toc(ticker);
        df = dtsec*obj.moviePlayFPS;
        f = ceil(df)+fstart;
        if f > fend
          break;
        end

        obj.setFrame(f,setFrameArgs{:});
        drawnow;

%         dtsec = toc(ticker);
%         pause_time = (f-fstart)/obj.moviePlayFPS - dtsec;
%         if pause_time <= 0,
%           if handles.guidata.mat_lt_8p4
%             drawnow;
%             % MK Aug 2015: There is a drawnow in status update so no need to draw again here
%             % for 2014b onwards.
%             %     else
%             %       drawnow('limitrate');
%           end
%         else
%           pause(pause_time);
%         end
      end
      
      if tfreset
        % AL20170619 passing setFrameArgs a bit fragile; needed for current
        % callers (don't update labels in videoPlaySegment)
        obj.setFrame(freset,setFrameArgs{:}); 
      end
      
      % - icon managed by caller      
    end
    
  end
 
  
  %% Navigation
  methods
    
    function navPrefsUI(obj)
      NavPrefs(obj);
    end
    
    function setMFT(obj,iMov,frm,iTgt)
      if obj.currMovie~=iMov
        obj.movieSet(iMov);
      end
      obj.setFrameAndTarget(frm,iTgt);
    end
  
    function tfSetOccurred = setFrameProtected(obj,frm,varargin)
      % Protected set against frm being out-of-bounds for current target.
      
      if obj.hasTrx 
        iTgt = obj.currTarget;
        if ~obj.frm2trx(frm,iTgt)
          tfSetOccurred = false;
          return;
        end
      end
      
      tfSetOccurred = true;
      obj.setFrame(frm,varargin{:});      
    end
    
    function setFrame(obj,frm,varargin)
      % Set movie frame, maintaining current movie/target.
      %
      % CTRL-C note: This is fairly ctrl-c safe; a ctrl-c break may leave
      % obj state a little askew but it should be cosmetic and another
      % (full/completed) setFrame() call should fix things up. We could
      % prob make it even more Ctrl-C safe with onCleanup-plus-a-flag.
      
      [tfforcereadmovie,tfforcelabelupdate,updateLabels,updateTables,...
        updateTrajs] = myparse(varargin,...
        'tfforcereadmovie',false,...
        'tfforcelabelupdate',false,...
        'updateLabels',true,...
        'updateTables',true,...
        'updateTrajs',true);
            
      if obj.hasTrx
        assert(~obj.isMultiView,'MultiView labeling not supported with trx.');
        if ~obj.frm2trx(frm,obj.currTarget)
          error('Labeler:target','Target idx %d not live in frame %d.',...
            obj.currTarget,frm);
        end
      end
      
      % Remainder nearly identical to setFrameAndTarget()
      obj.hlpSetCurrPrevFrame(frm,tfforcereadmovie);
      
      if obj.hasTrx && obj.movieCenterOnTarget
        assert(~obj.isMultiView);
        obj.videoCenterOnCurrTarget();
      end
      
      if updateLabels
        obj.labelsUpdateNewFrame(tfforcelabelupdate);
      end
      if updateTables
        obj.updateTrxTable();
%         obj.updateCurrSusp();
      end
      if updateTrajs
        obj.updateShowTrx();
      end
    end
    
%     function setTargetID(obj,tgtID)
%       % Set target ID, maintaining current movie/frame.
%       
%       iTgt = obj.trxIdPlusPlus2Idx(tgtID+1);
%       assert(~isnan(iTgt),'Invalid target ID: %d.');
%       obj.setTarget(iTgt);
%     end
    
    function setTarget(obj,iTgt)
      % Set target index, maintaining current movie/frameframe.
      % iTgt: INDEX into obj.trx
      
      validateattributes(iTgt,{'numeric'},...
        {'positive' 'integer' '<=' obj.nTargets});
      
      frm = obj.currFrame;
      if ~obj.frm2trx(frm,iTgt)
        error('Labeler:target',...
          'Target idx %d is not live at current frame (%d).',iTgt,frm);
      end
      
      prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      if obj.hasTrx 
        obj.labelsUpdateNewTarget(prevTarget);
        if obj.movieCenterOnTarget        
          obj.videoCenterOnCurrTarget();
        end
      end
%       obj.updateCurrSusp();
      obj.updateShowTrx();
    end
        
    function setFrameAndTarget(obj,frm,iTgt)
      % Set to new frame and target for current movie.
      % Prefer setFrame() or setTarget() if possible to
      % provide better continuity wrt labeling etc.
     
      validateattributes(iTgt,{'numeric'},...
        {'positive' 'integer' '<=' obj.nTargets});

      if ~obj.isinit && obj.hasTrx && ~obj.frm2trx(frm,iTgt)
        error('Labeler:target',...
          'Target idx %d is not live at current frame (%d).',iTgt,frm);
      end
        
      % 2nd arg true to match legacy
      obj.hlpSetCurrPrevFrame(frm,true);
      
      prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      if obj.hasTrx && obj.movieCenterOnTarget
        obj.videoCenterOnCurrTarget();
      end
      if ~obj.isinit
        obj.labelsUpdateNewFrameAndTarget(obj.prevFrame,prevTarget);
        obj.updateTrxTable();
%         obj.updateCurrSusp();
        obj.updateShowTrx();
      end
    end    
    
    function tfSetOccurred = frameUpDF(obj,df)
      f = min(obj.currFrame+df,obj.nframes);
      tfSetOccurred = obj.setFrameProtected(f); 
    end
    
    function tfSetOccurred = frameDownDF(obj,df)
      f = max(obj.currFrame-df,1);
      tfSetOccurred = obj.setFrameProtected(f);
    end
    
    function tfSetOccurred = frameUp(obj,tfBigstep)
      if tfBigstep
        df = obj.movieFrameStepBig;
      else
        df = 1;
      end
      tfSetOccurred = obj.frameUpDF(df);
    end
  end
  methods (Static) % seek utils
    function [tffound,f] = seekBigLpos(lpos,f0,df,iTgt)
      % lpos: [npts x d x nfrm x ntgt]
      % f0: starting frame
      % df: frame increment
      % iTgt: target of interest
      % 
      % tffound: logical
      % f: first frame encountered with (non-nan) label, applicable if
      %   tffound==true
      
      [npts,d,nfrm,ntgt] = size(lpos); %#ok<ASGLU>
      assert(d==2);
      
      f = f0+df;
      while 0<f && f<=nfrm
        for ipt = 1:npts
          %for j = 1:2
          if ~isnan(lpos(ipt,1,f,iTgt))
            tffound = true;
            return;
          end
          %end
        end
        f = f+df;
      end
      tffound = false;
      f = nan;
    end
    function [tffound,f] = seekSmallLpos(lpos,f0,df)
      % lpos: [npts x nfrm]
      % f0: starting frame
      % df: frame increment
      % 
      % tffound: logical
      % f: first frame encountered with (non-nan) label, applicable if
      %   tffound==true
      
      [npts,nfrm] = size(lpos);
      
      f = f0+df;
      while 0<f && f<=nfrm
        for ipt=1:npts
          if ~isnan(lpos(ipt,f))
            tffound = true;
            return;
          end
        end
        f = f+df;
      end
      tffound = false;
      f = nan;
    end
  end
  methods
    function tfSetOccurred = frameDown(obj,tfBigstep)
      if tfBigstep
        df = obj.movieFrameStepBig;
      else
        df = 1;
      end
      tfSetOccurred = obj.frameDownDF(df);
    end
    
    function frameUpNextLbled(obj,tfback,varargin)
      % call obj.setFrame() on next labeled frame. 
      % 
      % tfback: optional. if true, seek backwards.
            
      if ~obj.hasMovie || obj.currMovie==0
        return;
      end
      
      lpos = myparse(varargin,...
        'lpos','__UNSET__'); % optional, provide "big" lpos array to use instead of .labeledposCurrMovie
      
      if tfback
        df = -1;
      else
        df = 1;
      end
      
      if strcmp(lpos,'__UNSET__')
        lpos = obj.labeledposCurrMovie;
      elseif isempty(lpos)
        % edge case
        return;
      else
        szassert(lpos,[obj.nLabelPoints 2 obj.nframes obj.nTargets]);
      end
        
      [tffound,f] = Labeler.seekBigLpos(lpos,obj.currFrame,df,...
        obj.currTarget);
      if tffound
        obj.setFrameProtected(f);
      end
    end
    
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
          x = round(obj.movienc/2);
          y = round(obj.movienr/2);
          th = 0;
        else
          i = cfrm - ctrx.firstframe + 1;
          x = ctrx.x(i);
          y = ctrx.y(i);
          th = ctrx.theta(i);
        end
      else
        x = round(obj.movienc/2);
        y = round(obj.movienr/2);
        th = 0;
      end
    end
    
    function setSelectedFrames(obj,frms)
      if isempty(frms)
        obj.selectedFrames = frms;        
      elseif ~obj.hasMovie
        error('Labeler:noMovie',...
          'Cannot set selected frames when no movie is loaded.');
      else
        validateattributes(frms,{'numeric'},{'integer' 'vector' '>=' 1 '<=' obj.nframes});
        obj.selectedFrames = frms;  
      end
    end
                
    function updateTrxTable(obj)
      % based on .frm2trxm, .currFrame, .labeledpos
      
      tbl = obj.gdata.tblTrx;
      if ~obj.hasTrx || ~obj.hasMovie || obj.currMovie==0 % Can occur during movieSet(), when invariants momentarily broken
        set(tbl,'Data',cell(0,2));
        return;
      end
      
      f = obj.currFrame;
      tfLive = obj.frm2trx(f,:);
      idxLive = find(tfLive);
      idxLive = idxLive(:);
      lpos = obj.labeledposCurrMovie;
      tfLbled = arrayfun(@(x)any(lpos(:,1,f,x)),idxLive); % nans counted as 0
      tbldat = [num2cell(idxLive) num2cell(tfLbled)];      
      set(tbl,'Data',tbldat);
    end
    
    % TODO: Move this into UI
    function updateFrameTableIncremental(obj)
      % assumes .labelpos and tblFrames differ at .currFrame at most
      %
      % might be unnecessary/premature optim
      
      tbl = obj.gdata.tblFrames;
      dat = get(tbl,'Data');
      tblFrms = cell2mat(dat(:,1));      
      cfrm = obj.currFrame;
      tfRow = (tblFrms==cfrm);
      
      [nTgtsCurFrm,nPtsCurFrm] = obj.labelPosLabeledFramesStats(cfrm);
      if nTgtsCurFrm>0
        if any(tfRow)
          assert(nnz(tfRow)==1);
          iRow = find(tfRow);
          dat(iRow,2:3) = {nTgtsCurFrm nPtsCurFrm};
        else          
          dat(end+1,:) = {cfrm nTgtsCurFrm nPtsCurFrm};
          n = size(dat,1);
          tblFrms(end+1,1) = cfrm;
          [~,idx] = sort(tblFrms);
          dat = dat(idx,:);
          iRow = find(idx==n);
        end
        set(tbl,'Data',dat);
      else
        iRow = [];
        if any(tfRow)
          assert(nnz(tfRow)==1);
          dat(tfRow,:) = [];
          set(tbl,'Data',dat);
        end
      end
      
      tbl.SelectedRows = iRow;
            
      % dat should equal get(tbl,'Data')
      if obj.hasMovie
        PROPS = obj.gtGetSharedProps();
        %obj.gdata.labelTLInfo.setLabelsFrame();
        obj.(PROPS.MFAHL)(obj.currMovie) = size(dat,1)>0;
      end
      
      tx = obj.gdata.txTotalFramesLabeled;
      nTgtsTot = sum(cell2mat(dat(:,2)));
      tx.String = num2str(nTgtsTot);
    end    
    function updateFrameTableComplete(obj)
      [nTgts,nPts] = obj.labelPosLabeledFramesStats();
      assert(isequal(nTgts>0,nPts>0));
      tfFrm = nTgts>0;
      iFrm = find(tfFrm);

      nTgtsLbledFrms = nTgts(tfFrm);
      dat = [num2cell(iFrm) num2cell(nTgtsLbledFrms) num2cell(nPts(tfFrm)) ];
      tbl = obj.gdata.tblFrames;
      set(tbl,'Data',dat);

      if obj.hasMovie
        PROPS = obj.gtGetSharedProps();
        %obj.gdata.labelTLInfo.setLabelsFrame(1:obj.nframes);
        obj.(PROPS.MFAHL)(obj.currMovie) = size(dat,1)>0;
      end
      
      tx = obj.gdata.txTotalFramesLabeled;
      nTgtsTot = sum(nTgtsLbledFrms);
      tx.String = num2str(nTgtsTot);
    end
  end
  
  methods (Hidden)

    function hlpSetCurrPrevFrame(obj,frm,tfforce)
      % helper for setFrame, setFrameAndTarget
      
      currFrmOrig = obj.currFrame;
      currIm1Orig = obj.currIm{1};
      
      gd = obj.gdata;
      if obj.currFrame~=frm || tfforce
        imsall = gd.images_all;
        for iView=1:obj.nview
          obj.currIm{iView} = obj.movieReader(iView).readframe(frm);
          set(imsall(iView),'CData',obj.currIm{iView});
        end
        obj.currFrame = frm;
      end
     
      obj.prevFrame = currFrmOrig;
      if ~isequal(size(currIm1Orig),size(obj.currIm{1}))
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
        obj.prevIm = zeros(size(obj.currIm{1}));
      else
        obj.prevIm = currIm1Orig;
      end
      obj.prevAxesImFrmUpdate();
    end
    
  end
  
  %% 
  methods % PrevAxes
    
    function setPrevAxesMode(obj,pamode,pamodeinfo)
      % Set .prevAxesMode, .prevAxesModeInfo
      %
      % pamode: PrevAxesMode
      % pamodeinfo: (optional) userdata for pamode.
      
      if exist('pamodeinfo','var')==0
        pamodeinfo = [];
      end
      
      switch pamode
        case PrevAxesMode.LASTSEEN
          obj.prevAxesMode = pamode;
          obj.prevAxesModeInfo = pamodeinfo;
          obj.prevAxesImFrmUpdate();
          obj.prevAxesLabelsUpdate();
          gd = obj.gdata;
          axp = gd.axes_prev;
          set(axp,...
            'CameraUpVectorMode','auto',...
            'CameraViewAngleMode','auto');
          gd.hLinkPrevCurr.Enabled = 'on'; % links X/Ylim, X/YDir
        case PrevAxesMode.FROZEN
          obj.prevAxesFreeze(pamodeinfo);
        otherwise
          assert(false);
      end
    end
    
    function prevAxesFreeze(obj,freezeInfo)
      % Freeze the current frame/labels in the previous axis. Sets
      % .prevAxesMode, .prevAxesModeInfo.
      %
      % freezeInfo: Optional freezeInfo to apply. If not supplied,
      % image/labels taken from current movie/frame/etc.
      
      gd = obj.gdata;
      if isequal(freezeInfo,[])
        axc = gd.axes_curr;
        freezeInfo = struct(...
          'iMov',obj.currMovie,...
          'frm',obj.currFrame,...
          'iTgt',obj.currTarget,...
          'im',obj.currIm{1},...
          'axes_curr',struct('XLim',axc.XLim,'YLim',axc.YLim,...
                             'XDir',axc.XDir,'YDir',axc.YDir,...
                             'CameraUpVector',axc.CameraUpVector));
        if strcmp(axc.CameraViewAngleMode,'auto')
          freezeInfo.axes_curr.CameraViewAngleMode = 'auto';
        else
          freezeInfo.axes_curr.CameraViewAngle = axc.CameraViewAngle;
        end
      end
      
      gd.image_prev.CData = freezeInfo.im;
      gd.txPrevIm.String = num2str(freezeInfo.frm);
      obj.prevAxesSetLabels(freezeInfo.iMov,freezeInfo.frm,freezeInfo.iTgt);
      
      gd.hLinkPrevCurr.Enabled = 'off';
      axp = gd.axes_prev;
      axcProps = freezeInfo.axes_curr;
      for prop=fieldnames(axcProps)',prop=prop{1}; %#ok<FXSET>
        axp.(prop) = axcProps.(prop);
      end
      % Setting XLim/XDir etc unnec coming from PrevAxesMode.LASTSEEN, but 
      % sometimes nec eg for a "refreeze"
      
      obj.prevAxesMode = PrevAxesMode.FROZEN;
      obj.prevAxesModeInfo = freezeInfo;
    end
    
    function prevAxesImFrmUpdate(obj)
      % update prevaxes image and txframe based on .prevIm, .prevFrame
      switch obj.prevAxesMode
        case PrevAxesMode.LASTSEEN
          gd = obj.gdata;
          gd.image_prev.CData = obj.prevIm;
          gd.txPrevIm.String = num2str(obj.prevFrame);
      end
    end
    
    function prevAxesLabelsUpdate(obj)
      % Update (if required) .lblPrev_ptsH, .lblPrev_ptsTxtH based on 
      % .prevFrame etc 
      
      if obj.isinit || ~obj.hasMovie || obj.prevAxesMode==PrevAxesMode.FROZEN
        return;
      end
      if ~isnan(obj.prevFrame) && ~isempty(obj.lblPrev_ptsH)
        obj.prevAxesSetLabels(obj.currMovie,obj.prevFrame,obj.currTarget);
      else
        LabelCore.setPtsOffaxis(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
      end
    end
    
  end
  methods (Access=private)
    function prevAxesSetLabels(obj,iMov,frm,iTgt)
      persistent tfWarningThrownAlready
      
      lpos = obj.labeledposGTaware;
      lpostag = obj.labeledpostagGTaware;
      lpos = lpos{iMov}(:,:,frm,iTgt);
      lpostag = lpostag{iMov}(:,frm,iTgt);
      ipts = 1:obj.nPhysPoints;
      txtOffset = obj.labelPointsPlotInfo.LblOffset;
      LabelCore.assignLabelCoordsStc(lpos(ipts,:),...
        obj.lblPrev_ptsH(ipts),obj.lblPrev_ptsTxtH(ipts),txtOffset);
      if any(lpostag(ipts))
        if isempty(tfWarningThrownAlready)
          warningNoTrace('Labeler:labelsPrev',...
            'Label tags in previous frame not visualized.');
          tfWarningThrownAlready = true;
        end
      end
    end
  end
  
  %% Labels2/OtherTarget labels
  methods
    
    function labels2BulkSet(obj,lpos)
      assert(numel(lpos)==numel(obj.labeledpos2));
      for i=1:numel(lpos)
        assert(isequal(size(lpos{i}),size(obj.labeledpos2{i})))
        obj.labeledpos2{i} = lpos{i};
      end
      if ~obj.gtIsGTMode
      obj.labels2VizUpdate();
    end
    end
    
    function labels2SetCurrMovie(obj,lpos)
      assert(~obj.gtIsGTMode);
      iMov = obj.currMovie;
      assert(isequal(size(lpos),size(obj.labeledpos{iMov})));
      obj.labeledpos2{iMov} = lpos;
    end
    
    function labels2Clear(obj)
      for i=1:numel(obj.labeledpos2)
        obj.labeledpos2{i}(:) = nan;
      end
      if ~obj.gtIsGTMode
        obj.labels2VizUpdate();
      end
    end
    
    function labels2ImportTrkPrompt(obj,iMovs)
      % See labelImportTrkPrompt() 
      
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmovies;
      end
      
      assert(~obj.gtIsGTMode);
      obj.labelImportTrkPromptGeneric(iMovs,'labels2ImportTrk');
    end
   
    function labels2ImportTrk(obj,iMovs,trkfiles)
      obj.labelImportTrkGeneric(iMovs,trkfiles,'labeledpos2',[],[]);
      if ~obj.gtIsGTMode
      obj.labels2VizUpdate();
      end
      RC.saveprop('lastTrkFileImported',trkfiles{end});
    end
    
    function labels2ImportTrkCurrMov(obj)
      % Try to import default trk file for current movie into labels2. If
      % the file is not there, error.
      
      assert(~obj.gtIsGTMode,'Unsupported in GT mode.');
      if ~obj.hasMovie
        error('Labeler:nomov','No movie is loaded.');
      end
      obj.labels2ImportTrkPrompt(obj.currMovie);
    end
    
    function labels2ExportTrk(obj,iMovs,varargin)
      % Export label data to trk files.
      %
      % iMov: optional, indices into .movieFilesAll to export. Defaults to 1:obj.nmovies.
      
      % TODO: prob behind labelExportTrk, trackExportResults
      
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmovies;
      end

      assert(~obj.gtIsGTMode);
      
      movfiles = obj.movieFilesAllFull(iMovs,1);
      rawname = obj.defaultExportTrkRawname();
      [tfok,trkfiles] = obj.getTrkFileNamesForExportUI(movfiles,rawname);
      if tfok
        nMov = numel(iMovs);
        assert(numel(trkfiles)==nMov);
        for i=1:nMov
          iMv = iMovs(i);
          trkfile = TrkFile(obj.labeledpos2{iMv});
          trkfile.save(trkfiles{i});
        end
      end      
    end
    
    function labelsMiscInit(obj)
      % Initialize view stuff for labels2, lblOtherTgts
      
      trkPrefs = obj.projPrefs.Track;
      if ~isempty(trkPrefs)
        ptsPlotInfo = trkPrefs.PredictPointsPlot;
        ptsPlotInfo.Colors = obj.labelPointsPlotInfo.Colors;
      else
        ptsPlotInfo = obj.labelPointsPlotInfo;
      end
      ptsPlotInfo.PickableParts = 'none';
      
      obj.genericInitLabelPointViz('labeledpos2_ptsH','labeledpos2_ptsTxtH',...
        obj.gdata.axes_curr,ptsPlotInfo);
      obj.genericInitLabelPointViz('lblOtherTgts_ptsH',[],...
        obj.gdata.axes_curr,ptsPlotInfo);
    end
    
    function labels2VizUpdate(obj)
      assert(~obj.gtIsGTMode);
      iMov = obj.currMovie;
      frm = obj.currFrame;
      iTgt = obj.currTarget;
      lpos = obj.labeledpos2{iMov}(:,:,frm,iTgt);
      txtOffset = obj.labelPointsPlotInfo.LblOffset;
      LabelCore.setPtsCoordsStc(lpos,obj.labeledpos2_ptsH,...
        obj.labeledpos2_ptsTxtH,txtOffset);
    end
    
    function labels2VizShow(obj)
      [obj.labeledpos2_ptsH.Visible] = deal('on');
      [obj.labeledpos2_ptsTxtH.Visible] = deal('on');
      obj.labels2Hide = false;
    end
    
    function labels2VizHide(obj)
      [obj.labeledpos2_ptsH.Visible] = deal('off');
      [obj.labeledpos2_ptsTxtH.Visible] = deal('off');
      obj.labels2Hide = true;
    end
    
    function labels2VizToggle(obj)
      if obj.labels2Hide
        obj.labels2VizShow();
      else
        obj.labels2VizHide();
      end
    end
     
  end
  
  methods % OtherTarget
    
    function labelsOtherTargetShowIdxs(obj,iTgts)
      frm = obj.currFrame;
      lpos = obj.labeledposCurrMovie;
      lpos = squeeze(lpos(:,:,frm,iTgts)); % [npts x 2 x numel(iTgts)]

      npts = obj.nLabelPoints;     
      hPts = obj.lblOtherTgts_ptsH;
      for ipt=1:npts
        xnew = squeeze(lpos(ipt,1,:));
        ynew = squeeze(lpos(ipt,2,:));
        set(hPts(ipt),'XData',[hPts(ipt).XData xnew'],...
                      'YData',[hPts(ipt).YData ynew']);
      end
    end
    
    function labelsOtherTargetHideAll(obj)
      npts = obj.nLabelPoints;
      hPts = obj.lblOtherTgts_ptsH;
      for ipt=1:npts
        set(hPts(ipt),'XData',[],'YData',[]);
      end      
    end
    
  end
  
  %% Emp PDF
  methods
    function updateFGEmpiricalPDF(obj,varargin)
      
      if ~obj.hasTrx
        error('Method only supported for projects with trx.');
      end
      if obj.gtIsGTMode
        error('Method is not supported in GT mode.');
      end
      if obj.nview>1
        error('Method is not supported for multiple views.');
      end
      tObj = obj.tracker;
      if isempty(tObj)
        error('Method only supported for projects with trackers.');
      end
      
      prmNborMask = tObj.sPrm.PreProc.NeighborMask;
      if ~prmNborMask.Use
        warningNoTrace('Neighbor masking is currently not turned on in your tracking parameters.');
      end
      if isempty(prmNborMask.BGReadFcn)
        error('Method requires background read function to be defined in Neighbor Masking tracking parameters.');
      end
      
      % Start with all labeled rows. Prefer these b/c user apparently cares
      % more about these frames
      wbObj = WaitBarWithCancel('Empirical foreground PDF');
      oc = onCleanup(@()delete(wbObj));
      tblMFTlbled = obj.labelGetMFTableLabeled('wbObj',wbObj);
      if wbObj.isCancel
        return;
      end
      
      roiRadius = tObj.sPrm.PreProc.TargetCrop.Radius;
      tblMFTlbled = obj.labelMFTableAddROI(tblMFTlbled,roiRadius);

      amu = mean(tblMFTlbled.aTrx);
      bmu = mean(tblMFTlbled.bTrx);
      %fprintf('amu/bmu: %.4f/%.4f\n',amu,bmu);
      
      % get stuff we will need for movies: movieReaders, bgimages, etc
      movieStuff = cell(obj.nmovies,1);
      iMovsLbled = unique(tblMFTlbled.mov);
      for iMov=iMovsLbled(:)'
        
        s = struct();
        movfile = obj.movieFilesAllFull{iMov,1};
        mr = MovieReader();
        mr.open(movfile);
        s.movRdr = mr;
        
        trxfname = obj.trxFilesAllFull{iMov,1};
        movIfo = obj.movieInfoAll{iMov};        
        [s.trx,s.frm2trx] = obj.getTrx(trxfname,movIfo.nframes);
        
        [s.bg,s.bgdev] = feval(prmNborMask.BGReadFcn,movfile,movIfo.info);
        
        movieStuff{iMov} = s;
      end    
      
      hFigViz = figure;
      ax = axes;
    
      xroictr = -roiRadius:roiRadius;
      yroictr = -roiRadius:roiRadius;
      [xgrid,ygrid] = meshgrid(xroictr,yroictr); % xgrid, ygrid give coords for each pixel where (0,0) is the central pixel (at target)
      pdfRoiAcc = zeros(2*roiRadius+1,2*roiRadius+1);
      nAcc = 0;
      n = height(tblMFTlbled);
      for i=1:n
        trow = tblMFTlbled(i,:);
        iMov = trow.mov;
        frm = trow.frm;
        iTgt = trow.iTgt;
        
        sMovStuff = movieStuff{iMov};
        
        [tflive,trxxs,trxys,trxths,trxas,trxbs] = ...
          PxAssign.getTrxStuffAtFrm(sMovStuff.trx,frm);
        assert(isequal(tflive(:),sMovStuff.frm2trx(frm,:)'));

        % Skip roi if it contains more than 1 trxcenter.
        roi = trow.roi; % [xlo xhi ylo yhi]
        roixlo = roi(1);
        roixhi = roi(2);
        roiylo = roi(3);
        roiyhi = roi(4);
        tfCtrInRoi = roixlo<=trxxs & trxxs<=roixhi & roiylo<=trxys & trxys<=roiyhi;
        if nnz(tfCtrInRoi)>1
          continue;
        end
      
        % In addition run CC pxAssign and keep only the central CC to get
        % rid of any objects at the periphery
        im = sMovStuff.movRdr.readframe(frm);
        im = double(im);
        imbwl = PxAssign.asgnCC(im,sMovStuff.bg,sMovStuff.bgdev,...
          sMovStuff.trx,frm,...
          'bgtype',prmNborMask.BGType,...
          'fgthresh',prmNborMask.FGThresh);
        xTgtCtrRound = round(trxxs(iTgt));
        yTgtCtrRound = round(trxys(iTgt));
        ccKeep = imbwl(yTgtCtrRound,xTgtCtrRound);
        if ccKeep==0
          warningNoTrace('Unexpected non-foreground pixel for (mov,frm,tgt)=(%d,%d,%d) at (r,c)=(%d,%d).',...
            iMov,frm,iTgt,yTgtCtrRound,xTgtCtrRound);
        else
          imfgUse = zeros(size(imbwl));
          imfgUse(imbwl==ccKeep) = 1;                    
          imfgUseRoi = padgrab(imfgUse,0,roiylo,roiyhi,roixlo,roixhi);
          
          th = trxths(iTgt);
          a = trxas(iTgt);
          b = trxbs(iTgt);
          imforebwcanon = readpdf(imfgUseRoi,xgrid,ygrid,xgrid,ygrid,0,0,-th);
          xfac = a/amu;
          yfac = b/bmu;
          imforebwcanonscale = interp2(xgrid,ygrid,imforebwcanon,...
            xgrid*xfac,ygrid*yfac,'linear',0);
          
          imshow(imforebwcanonscale,'Parent',ax);
          tstr = sprintf('row %d/%d',i,n);
          title(tstr,'fontweight','bold','interpreter','none');
          drawnow;
          
          pdfRoi = imforebwcanonscale/sum(imforebwcanonscale(:)); % each row equally weighted
          pdfRoiAcc = pdfRoiAcc + pdfRoi;
          nAcc = nAcc + 1;
        end
      end
      
      fgpdf = pdfRoiAcc/nAcc;
      
      imshow(fgpdf,[],'xdata',xroictr,'ydata',yroictr);
      colorbar;
      tstr = sprintf('N=%d, amu=%.3f, bmu=%.3f. FGThresh=%.2f',...
        nAcc,amu,bmu,prmNborMask.FGThresh);
      title(tstr,'fontweight','bold','interpreter','none');
      
      obj.fgEmpiricalPDF = struct(...
        'amu',amu,'bmu',bmu,...
        'xpdfctr',xroictr,'ypdfctr',yroictr,...        
        'fgpdf',fgpdf,...
        'n',nAcc,...
        'roiRadius',roiRadius,...
        'prmNborMask',prmNborMask);
    end
  end
  
  
  %% Util
  methods
    
    function tblConcrete = mftTableConcretizeMov(obj,tbl)
      % tbl: MFTable where .mov is MovieIndex array
      %
      % tblConcrete: Same table where .mov is [NxNview] cellstr
      
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
        tblConcrete = [tblConcrete table(trxFile)];
      end
    end
    
    function genericInitLabelPointViz(obj,hProp,hTxtProp,ax,plotIfo)
      deleteValidHandles(obj.(hProp));
      obj.(hProp) = gobjects(obj.nLabelPoints,1);
      if ~isempty(hTxtProp)
        deleteValidHandles(obj.(hTxtProp));
        obj.(hTxtProp) = gobjects(obj.nLabelPoints,1);
      end
      
      % any extra plotting parameters
      allowedPlotParams = {'HitTest' 'PickableParts'};
      ism = ismember(cellfun(@lower,allowedPlotParams,'Uni',0),...
                     cellfun(@lower,fieldnames(plotIfo),'Uni',0));
      extraParams = {};
      for i = find(ism)
        extraParams = [extraParams,{allowedPlotParams{i},plotIfo.(allowedPlotParams{i})}]; %#ok<AGROW>
      end

      for i = 1:obj.nLabelPoints
        obj.(hProp)(i) = plot(ax,nan,nan,plotIfo.Marker,...
          'MarkerSize',plotIfo.MarkerSize,...
          'LineWidth',plotIfo.LineWidth,...
          'Color',plotIfo.Colors(i,:),...
          'UserData',i,...
          extraParams{:});
        if ~isempty(hTxtProp)
        obj.(hTxtProp)(i) = text(nan,nan,num2str(i),'Parent',ax,...
          'Color',plotIfo.Colors(i,:),'PickableParts','none');
        end
      end      
    end
    
  end

end
