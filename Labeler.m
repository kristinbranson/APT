classdef Labeler < handle
% Bransonlab Animal Video Labeler/Tracker

  properties (Constant,Hidden)
    VERSION = '1.2';
    DEFAULT_LBLFILENAME = '%s.lbl';
    DEFAULT_CFG_FILENAME = 'config.default.yaml';
    DEFAULT_TRKFILE = '$movfile_$projname';
    DEFAULT_TRKFILE_NOPROJ = '$movfile';
    
    % non-config props
    SAVEPROPS = { ...
      'VERSION' 'projname' ...
      'movieFilesAll' 'movieInfoAll' 'trxFilesAll' 'projMacros'...
      'viewCalibrationData' 'viewCalProjWide' ...
      'labeledpos' 'labeledpostag' 'labeledposTS' 'labeledposMarked' 'labeledpos2' ...
      'currMovie' 'currFrame' 'currTarget' ...
      'labelTemplate' ...
      'suspScore'};
    LOADPROPS = { ...
      'projname' ...
      'movieFilesAll' 'movieInfoAll' 'trxFilesAll' 'projMacros' ...
      'viewCalibrationData' 'viewCalProjWide' ...
      'labeledpos' 'labeledpostag' 'labeledposTS' 'labeledposMarked' 'labeledpos2' ...
      'labelTemplate' ...
      'suspScore'};
    
    TBLTRX_STATIC_COLSTBL = {'id' 'labeled'};
    TBLTRX_STATIC_COLSTRX = {'id' 'labeled'};
    
    TBLFRAMES_COLS = {'frame' 'tgts' 'pts'};
    
    NEIGHBORING_FRAME_MAXRADIUS = 10;
    NEIGHBORING_FRAME_OFFSETS = neighborIndices(Labeler.NEIGHBORING_FRAME_MAXRADIUS);    
  end
  
  events
    newProject
    newMovie
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
    % .viewCalProjWide=true, .vCD=<scalar Calrig obj>. Scalar Calrig obj apples to all movies.
    % .viewCalProjWide=false, .vCD=[nMovSet] cell array of calRigs. .vCD
    % applies element-wise to movies. .vCD{i} can be empty indicating unset
    % calibration object for that movie.
    viewCalProjWide % [], true, or false. init: PN
    viewCalibrationData % Opaque calibration 'useradata' for multiview. init: PN
    
    movieReader = []; % [1xnview] MovieReader objects. init: C
    movieInfoAll = {}; % cell-of-structs, same size as movieFilesAll
    movieDontAskRmMovieWithLabels = false; % If true, won't warn about removing-movies-with-labels    
  end
  properties (Dependent)
    viewCalibrationDataCurrent % view calibration data applicable to current movie
  end
  properties (SetObservable)
    movieFilesAll = {}; % [nmovset x nview] column cellstr, full paths to movies; can include macros 
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
  end
  properties (SetObservable)
    moviename; % short 'pretty' name, cosmetic purposes only. For multiview, primary movie name.
    movieCenterOnTarget = false; % scalar logical.
    movieRotateTargetUp = false;
    movieForceGrayscale = false; % scalar logical. In future could make [1xnview].
    movieFrameStepBig; % scalar positive int
    moviePlaySegRadius; % scalar int
    moviePlayFPS; 
    movieInvert; % [1xnview] logical. If true, movie should be inverted when read. This is to compensate for codec issues where movies can be read inverted on platform A wrt platform B
    
    movieIsPlaying = false;
  end
  properties (Dependent)
    isMultiView;
    movieFilesAllFull; % like movieFilesAll, but macro-replaced and platformized
    movieIDsAll; % like movieFilesAll, but standardized
    movieSetIDsAll; % [nmovsetx1] ids, single char for each movieset
    hasMovie;
    moviefile;
    nframes;
    movienr; % [nview]
    movienc; % [nview]
    nmovies;
    moviesSelected; % [nSel] vector of movie indices currently selected in MovieManager
  end
  
  %% Trx
  properties (SetObservable)
    trxFilesAll = {};  % column cellstr, full paths to trxs. Same size as movieFilesAll.
  end
  properties
    trxfile = '';             % full path current trxfile
    trx = [];                 % trx object
    %zoomRadiusDefault = 100;  % default zoom box size in pixels
    %zoomRadiusTight = 10;     % zoom size on maximum zoom (smallest pixel val)
    frm2trx = [];             % nFrm x nTrx logical. frm2trx(iFrm,iTrx) is true if trx iTrx is live on frame iFrm
    trxIdPlusPlus2Idx = [];   % (max(trx ids)+1) x 1 vector of indices into obj.trx. 
                              % Since IDs start at 0, THIS VECTOR IS INDEXED BY ID+1.
                              % ie: .trx(trxIdPlusPlus2Idx(ID+1)).id = ID. Nonexistent IDs map to NaN.
  end
  properties (Dependent,SetObservable)
    targetZoomRadiusDefault;
  end
  properties (Dependent)
    hasTrx
    currTrx
    currTrxID
    nTrx
    nTargets % nTrx, or 1 if no Trx
  end
  
  %% ShowTrx
  properties (SetObservable)
    showTrx;                  % true to show trajectories
    showTrxCurrTargetOnly;    % if true, plot only current target
  end
  properties
    hTraj;                    % nTrx x 1 vector of line handles
    hTrx;                     % nTrx x 1 vector of line handles    
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
    labeledpostag;        % column cell vec with .nmovies elements. labeledpostag{iMov} is npts x nFrm(iMov) x nTrx(iMov) cell array. init: PN
    
    labeledpos2;          % identical size/shape with labeledpos. aux labels (eg predicted, 2nd set, etc). init: PN
    labels2Hide;      % scalar logical
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
  
  %% Suspiciousness
  properties (SetObservable)
    suspScore; % column cell vec same size as labeledpos. suspScore{iMov} is nFrm(iMov) x nTrx(iMov)
    suspNotes; % column cell vec same size as labeledpos. suspNotes{iMov} is a nFrm x nTrx column cellstr
    currSusp; % suspScore for current mov/frm/tgt. Can be [] indicating 'N/A'
  end
  
  %% Tracking
  properties (SetObservable)
    tracker % LabelTracker object. init: PLPN
    trackNFramesSmall % small/fine frame increment for tracking. init: C
    trackNFramesLarge % big/coarse ". init: C
    trackNFramesNear % neighborhood radius. init: C
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
  properties (AbortSet)
    currMovie; % idx into .movieFilesAll (row index, when obj.multiView is true)
  end
  properties (SetObservable)
    currFrame = 1; % current frame
  end
  properties
    currIm = [];            % [nview] cell vec of image data. init: C
    isinit = false;         % scalar logical; true during initialization, when some invariants not respected
    selectedFrames = [];    % vector of frames currently selected frames; typically t0:t1
    hFig; % handle to main LabelerGUI figure
  end
  properties (Dependent)
    gdata; % handles structure for LabelerGUI
  end

  
  %% Prop access
  methods % dependent prop getters
    function v = get.viewCalibrationDataCurrent(obj)
      vcdPW = obj.viewCalProjWide;
      vcd = obj.viewCalibrationData;
      if isempty(vcdPW)
        v = [];
      elseif vcdPW
        assert(isequal(vcd,[]) || isscalar(vcd));
        v = vcd;
      else % ~vcdPW
        assert(iscell(vcd) && numel(vcd)==obj.nmovies);
        if obj.nmovies==0 || obj.currMovie==0
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
      sMacro = obj.projMacros;
      if ~isfield(sMacro,'projroot')
        % This conditional allows user to explictly specify project root
        sMacro.projroot = obj.projectroot;
      end
      v = FSPath.fullyLocalizeStandardize(obj.movieFilesAll,obj.projMacros);
      FSPath.warnUnreplacedMacros(v);
    end
    function v = get.movieIDsAll(obj)
      v = FSPath.standardPath(obj.movieFilesAll);
    end
    function v = get.movieSetIDsAll(obj)
      v = MFTable.formMultiMovieIDArray(obj.movieIDsAll);
    end
    function v = get.hasMovie(obj)
      v = obj.movieReader(1).isOpen;
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
        ifo = obj.movieInfoAll{obj.currMovie,1};
        v = ifo.nframes;
      end
    end
    function v = get.moviesSelected(obj) %#GUIREQ
      % Find MovieManager in LabelerGUI
      handles = obj.gdata;
      if isfield(handles,'movieMgr')
        hMM = handles.movieMgr;
      else
        hMM = [];
      end
      if ~isempty(hMM) && isvalid(hMM)
        mmgd = guidata(hMM);
        v = mmgd.cbkGetSelectedMovies();
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
    function v = get.currTrxID(obj)
      if obj.hasTrx
        v = obj.trx(obj.currTarget).id;
      else
        v = nan;
      end
    end
    function v = get.nTrx(obj)
      v = numel(obj.trx);
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
    function v = get.labeledposCurrMovie(obj)
      if obj.currMovie==0
        v = [];
      else
        v = obj.labeledpos{obj.currMovie};
      end
    end
    function v = get.labeledpostagCurrMovie(obj)
      if obj.currMovie==0
        v = [];
      else
        v = obj.labeledpostag{obj.currMovie};
      end
    end
    function v = get.nPhysPoints(obj)
      v = size(obj.labeledposIPtSetMap,1);
    end
    function v = get.gdata(obj)
      v = guidata(obj.hFig);
    end    
  end
  
  methods % prop access
    % TODO get rid of setter, use listeners
    function set.labeledpos(obj,v)
      obj.labeledpos = v;
      if ~obj.isinit %#ok<MCSUP> 
        obj.updateTrxTable();
        obj.updateFrameTableIncremental(); 
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
       fprintf('Configuring your path ...');
       APT.setpath;
      end
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
         
%       if isempty(cfg.LabelPointMap) && obj.nview==1
%         % create default map
%         tmpFields = arrayfun(@(x)sprintf('pt%d',x),(1:npts)','uni',0);
%         tmpVals = num2cell((1:npts)');
%         lblPtMap = cell2struct(tmpVals,tmpFields,1);
%       else
%         lblPtMap = cfg.LabelPointMap;
%       end
       
      % pts, sets, views
      setnames = cfg.LabelPointNames;%fieldnames(lblPtMap);
      nSet = size(setnames,1);
      ipt2view = nan(npts,1);
      ipt2set = nan(npts,1);
      setmap = nan(nSet,obj.nview);
      for iSet = 1:nSet
        set = setnames{iSet};
        %iPts = lblPtMap.(set);
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
 
      % For unclear reasons, creation of new tracker occurs downstream in
      % projLoad() or projNew()
      if ~isempty(obj.tracker)
        delete(obj.tracker);
        obj.tracker = [];
      end
      
      obj.showTrx = cfg.Trx.ShowTrx;
      obj.showTrxCurrTargetOnly = cfg.Trx.ShowTrxCurrentTargetOnly;
      
      obj.labels2Hide = false;

      % New projs must start with LASTSEEN as there is nothing to freeze
      % yet. projLoad() will further set any loaded info
      obj.setPrevAxesMode(PrevAxesMode.LASTSEEN,[]);

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
        'PlaySegmentRadius',obj.moviePlaySegRadius,...
        'PlayFPS',obj.moviePlayFPS);

      cfg.LabelPointsPlot = obj.labelPointsPlotInfo;
      cfg.Trx.ShowTrx = obj.showTrx;
      cfg.Trx.ShowTrxCurrentTargetOnly = obj.showTrxCurrTargetOnly;
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
      
      obj.projname = name;
      obj.projFSInfo = [];
      obj.movieFilesAll = cell(0,obj.nview);
      obj.movieFilesAllHaveLbls = false(0,1);
      obj.movieInfoAll = cell(0,obj.nview);
      obj.trxFilesAll = cell(0,obj.nview);
      obj.projMacros = struct();
      obj.viewCalProjWide = [];
      obj.viewCalibrationData = [];
      obj.isinit = true;
      obj.movieSetNoMovie(); % order important here
      obj.labeledpos = cell(0,1);
      obj.labeledposTS = cell(0,1);
      obj.labeledposMarked = cell(0,1);
      obj.labeledpostag = cell(0,1);
      obj.labeledpos2 = cell(0,1);
      obj.labelTemplate = [];
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

        obj.gdata.labelTLInfo.setTracker(obj.tracker);
      end

      props = obj.gdata.propsNeedInit;
      for p = props(:)', p=p{1}; %#ok<FXSET>
        obj.(p) = obj.(p);
      end
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
      
      projfile = sprintf(obj.DEFAULT_LBLFILENAME,obj.projname);
      filterspec = fullfile(savepath,projfile);
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
    
    function s = projGetSaveStruct(obj)
      s = struct();
      s.cfg = obj.getCurrentConfig();
      
      for f = obj.SAVEPROPS, f=f{1}; %#ok<FXSET>
        s.(f) = obj.(f);
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
    
    function projLoad(obj,fname,varargin)
      % Load a lbl file, along with moviefile and trxfile referenced therein
            
      nomovie = myparse(varargin,...
        'nomovie',false ... % If true, call movieSetNoMovie() instead of movieSet(currMovie)
        );
      
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
      
      for f = obj.LOADPROPS,f=f{1}; %#ok<FXSET>
        if isfield(s,f)
          obj.(f) = s.(f);          
        else
          warningNoTrace('Labeler:load','Missing load field ''%s''.',f);
          %obj.(f) = [];
        end
      end
     
      obj.movieFilesAllHaveLbls = cellfun(@(x)any(~isnan(x(:))),obj.labeledpos);
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
      
      if obj.nmovies==0 || s.currMovie==0 || nomovie
        obj.movieSetNoMovie();
      else
        obj.movieSet(s.currMovie);
        obj.setFrameAndTarget(s.currFrame,s.currTarget);
      end
      
      %assert(isa(s.labelMode,'LabelMode'));      
      obj.labeledposNeedsSave = false;
      obj.suspScore = obj.suspScore;
            
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI

      if ~isempty(obj.tracker)
        fprintf(1,'Loading tracker info: %s.\n',tCls);
        obj.tracker.loadSaveToken(s.trackerData);
      end
      
      % This needs to occur after .labeledpos etc has been set
      pamode = PrevAxesMode.(s.cfg.PrevAxes.Mode);
      obj.setPrevAxesMode(pamode,s.cfg.PrevAxes.ModeInfo);
      
      props = obj.gdata.propsNeedInit;
      for p = props(:)', p=p{1}; %#ok<FXSET>
        obj.(p) = obj.(p);
      end
    end
    
    function projImport(obj,fname)
      % 'Import' the project fname, MERGING movies/labels into the current project.
          
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
      
      if isfield(s,'projMacros') && ~isfield(s.projMacros,'projroot')
        s.projMacros.projroot = fileparts(fname);
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
        if ~isempty(obj.suspNotes)
          obj.suspNotes{end+1,1} = [];
        end
      end

      obj.labeledposNeedsSave = true;
      obj.projFSInfo = ProjectFSInfo('imported',fname);
      
      if ~isempty(obj.tracker)
        warning('Labeler:projImport','Re-initting tracker.');
        obj.tracker.init();
      end
    end
    
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

      resp = inputdlg(macrosdisp,'Project macros',1,vals);
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
    
    function tf = projMacroIsMacro(obj,macro)
      tf = isfield(obj.projMacros,macro);
    end
   
    function s = projMacroStrs(obj)
      m = obj.projMacros;
      flds = fieldnames(m);
      vals = struct2cell(m);
      n = numel(flds);
      s = cellfun(@(x,y)sprintf('%s -> %s',x,y),flds,vals,'uni',0);
    end
    
    function p = projLocalizePath(obj,p)
      p = FSPath.platformizePath(FSPath.macroReplace(p,obj.projMacros));
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
      
      [xyGT,xyTstT,xyTstTRed,tstITst] = myparse(varargin,...
        'xyGT',[],...
        'xyTstT',[],...
        'xyTstTRed',[],...
        'tstITst',[]);

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
    end
    
    % Legacy meth. labelGetMFTableLabeledStc is new method but assumes
    % .hasTrx
    function [I,tbl] = lblCompileContents(movieNames,labeledposes,...
        labeledpostags,type,varargin)
      % convenience signature 
      %
      % type: either 'all' or 'lbl'

      nMov = size(movieNames,1); 
      switch type
        case 'all'
          frms = repmat({'all'},nMov,1);
        case 'lbl'
          frms = repmat({'lbl'},nMov,1);
        otherwise
          assert(false);
      end
      [I,tbl] = Labeler.lblCompileContentsRaw(movieNames,labeledposes,...
        labeledpostags,1:nMov,frms,varargin{:});
    end
    
    % Legacy meth. labelGetMFTableLabeledStc is new method but assumes
    % .hasTrx
    %#3DOK
    function [I,tbl] = lblCompileContentsRaw(...
        movieNames,lposes,lpostags,iMovs,frms,varargin)
      % Read moviefiles with landmark labels
      %
      % movieNames: [NxnView] cellstr of movienames
      % lposes: [N] cell array of labeledpos arrays [npts x 2 x nfrms x ntgts]. 
      %   For multiview, npts=nView*NumLabelPoints.
      % lpostags: [N] cell array of labeledpostags [npts x nfrms x ntgts]
      % iMovs. [M] (row) indices into movieNames to read.
      % frms. [M] cell array. frms{i} is a vector of frames to read for
      % movie iMovs(i). frms{i} may also be:
      %     * 'all' indicating "all frames" 
      %     * 'lbl' indicating "all labeled frames" (currently includes partially-labeled)
      %
      % I: [NtrlxnView] cell vec of images
      % tbl: [NTrl rows] labels/metadata MFTable.
      %   MULTIVIEW NOTE: tbl.p is the 2d/projected label positions, ie
      %   each shape has nLabelPoints*nView*2 coords, raster order is 1. pt
      %   index, 2. view index, 3. coord index (x vs y)
      %
      % Optional PVs:
      % - hWaitBar. Waitbar object
      % - noImg. logical scalar default false. If true, all elements of I
      % will be empty.
      % - lposTS. [N] cell array of labeledposTS arrays [nptsxnfrms]
      % - movieNamesID. [NxnView] Like movieNames (input arg). Use these
      % names in tbl instead of movieNames. The point is that movieNames
      % may be macro-replaced, platformized, etc; otoh in the MD table we
      % might want macros unreplaced, a standard format etc.
      % - tblMovArray. Scalar logical, defaults to false. Only relevant for
      % multiview data. If true, use array of movies in tbl.mov. Otherwise, 
      % use single compactified string ID.
      
      [hWB,noImg,lposTS,movieNamesID,tblMovArray] = myparse(varargin,...
        'hWaitBar',[],...
        'noImg',false,...
        'lposTS',[],...
        'movieNamesID',[],...
        'tblMovArray',false);
      assert(numel(iMovs)==numel(frms));
      for i = 1:numel(frms)
        val = frms{i};
        assert(isnumeric(val) && isvector(val) || ismember(val,{'all' 'lbl'}));
      end
      
      tfWB = ~isempty(hWB);
      
      assert(iscellstr(movieNames));
      [N,nView] = size(movieNames);
      assert(iscell(lposes) && iscell(lpostags));
      assert(isequal(N,numel(lposes),numel(lpostags)));
      tfLposTS = ~isempty(lposTS);
      if tfLposTS
        assert(numel(lposTS)==N);
      end
      for i=1:N
        assert(size(lposes{i},1)==size(lpostags{i},1) && ...
               size(lposes{i},3)==size(lpostags{i},2));
        if tfLposTS
          assert(isequal(size(lposTS{i}),size(lpostags{i})));
        end
      end
      
      if ~isempty(movieNamesID)
        assert(iscellstr(movieNamesID));
        szassert(movieNamesID,size(movieNames)); 
      else
        movieNamesID = movieNames;
      end
      
      for iVw=nView:-1:1
        mr(iVw) = MovieReader();
      end

      I = [];
      % Here, for multiview, mov are for the first movie in each set
      s = struct('mov',cell(0,1),'frm',[],'p',[],'tfocc',[]);
      
      nMov = numel(iMovs);
      fprintf('Reading %d movies.\n',nMov);
      if nView>1
        fprintf('nView=%d.\n',nView);
      end
      for i = 1:nMov
        iMovSet = iMovs(i);
        lpos = lposes{iMovSet}; % npts x 2 x nframes
        lpostag = lpostags{iMovSet};

        [npts,d,nFrmAll] = size(lpos);
        assert(d==2);
        if isempty(lpos)
          assert(isempty(lpostag));
          lpostag = cell(npts,nFrmAll); % edge case: when lpos/lpostag are [], uninitted/degenerate case
        end
        szassert(lpostag,[npts nFrmAll]);
        D = d*npts;
        % Ordering of d is: {x1,x2,x3,...xN,y1,..yN} which for multiview is
        % {xp1v1,xp2v1,...xpnv1,xp1v2,...xpnvk,yp1v1,...}. In other words,
        % in decreasing raster order we have 1. pt index, 2. view index, 3.
        % coord index (x vs y)
        
        for iVw=1:nView
          movfull = movieNames{iMovSet,iVw};
          mr(iVw).open(movfull);
        end
        
        movID = MFTable.formMultiMovieID(movieNamesID(iMovSet,:));
        
        % find labeled/tagged frames (considering ALL frames for this
        % movie)
        tfLbled = arrayfun(@(x)nnz(~isnan(lpos(:,:,x)))>0,(1:nFrmAll)');
        frmsLbled = find(tfLbled);
        tftagged = ~cellfun(@isempty,lpostag); % [nptxnfrm]
        ntagged = sum(tftagged,1);
        frmsTagged = find(ntagged);
        assert(all(ismember(frmsTagged,frmsLbled)));

        frms2Read = frms{i};
        if strcmp(frms2Read,'all')
          frms2Read = 1:nFrmAll;
        elseif strcmp(frms2Read,'lbl')
          frms2Read = frmsLbled;
        end
        nFrmRead = numel(frms2Read);
        
        ITmp = cell(nFrmRead,nView);
        fprintf('  mov(set) %d, D=%d, reading %d frames\n',iMovSet,D,nFrmRead);
        
        if tfWB
          hWB.Name = 'Reading movies';
          wbStr = sprintf('Reading movie %s',movID);
          waitbar(0,hWB,wbStr);
        end
        for iFrm = 1:nFrmRead
          if tfWB
            waitbar(iFrm/nFrmRead,hWB);
          end
          
          f = frms2Read(iFrm);

          if noImg
            % none; ITmp(iFrm,:) will have [] els
          else
            for iVw=1:nView
              im = mr(iVw).readframe(f);
              if size(im,3)==3 && isequal(im(:,:,1),im(:,:,2),im(:,:,3))
                im = rgb2gray(im);
              end
              ITmp{iFrm,iVw} = im;
            end
          end
          
          lblsFrmXY = lpos(:,:,f);
          tags = lpostag(:,f);
          
          if tblMovArray
            s(end+1,1).mov = movieNamesID(iMovSet,:); %#ok<AGROW>
          else
            s(end+1,1).mov = movID; %#ok<AGROW>
          end
          %s(end).movS = movS1;
          s(end).frm = f;
          s(end).p = Shape.xy2vec(lblsFrmXY);
          s(end).tfocc = strcmp('occ',tags(:)');
          if tfLposTS
            lts = lposTS{iMovSet};
            s(end).pTS = lts(:,f)';
          end
        end
        
        I = [I;ITmp]; %#ok<AGROW>
      end
      tbl = struct2table(s,'AsArray',true);      
    end    
        
  end 
  
  %% Movie
  methods
    
    function movieAdd(obj,moviefile,trxfile)
      % Add movie/trx to end of movie/trx list.
      %
      % moviefile: string or cellstr (can have macros)
      % trxfile: (optional) string or cellstr 
      
      assert(~obj.isMultiView,'Unsupported for multiview labeling.');
      
      if exist('trxfile','var')==0 || isequal(trxfile,[])
        if ischar(moviefile)
          trxfile = '';
        elseif iscellstr(moviefile)
          trxfile = repmat({''},size(moviefile));
        else
          error('Labeler:movieAdd','''Moviefile'' must be a char or cellstr.');
        end
      end
      moviefile = cellstr(moviefile);
      trxfile = cellstr(trxfile);
      if numel(moviefile)~=numel(trxfile)
        error('Labeler:movieAdd',...
          '''Moviefile'' and ''trxfile'' arguments must have same size.');
      end
      nMov = numel(moviefile);
      
      mr = MovieReader();
      for iMov = 1:nMov
        movFile = moviefile{iMov};
        tFile = trxfile{iMov};
      
        movfilefull = obj.projLocalizePath(movFile);
        assert(exist(movfilefull,'file')>0,'Cannot find file ''%s''.',movfilefull);
        if any(strcmp(movFile,obj.movieFilesAll))
          if nMov==1
            error('Labeler:dupmov',...
              'Movie ''%s'' is already in project.',movFile);
          else
            warningNoTrace('Labeler:dupmov',...
              'Movie ''%s'' is already in project and will not be added to project.',movFile);
            continue;
          end
        end
        if any(strcmp(movfilefull,obj.movieFilesAllFull))
          warningNoTrace('Labeler:dupmov',...
            'Movie ''%s'', macro-expanded to ''%s'', is already in project.',...
            movFile,movfilefull);
        end
        assert(isempty(tFile) || exist(tFile,'file')>0,'Cannot find file ''%s''.',tFile);

        mr.open(movfilefull);
        ifo = struct();
        ifo.nframes = mr.nframes;
        ifo.info = mr.info;
        mr.close();
        
        if ~isempty(tFile)
          tmp = load(tFile);
          nTgt = numel(tmp.trx);
        else
          nTgt = 1;
        end
        
        obj.movieFilesAll{end+1,1} = movFile;
        obj.movieFilesAllHaveLbls(end+1,1) = false;
        obj.movieInfoAll{end+1,1} = ifo;
        obj.trxFilesAll{end+1,1} = tFile;
        obj.labeledpos{end+1,1} = nan(obj.nLabelPoints,2,ifo.nframes,nTgt);
        obj.labeledposTS{end+1,1} = -inf(obj.nLabelPoints,ifo.nframes,nTgt);
        obj.labeledposMarked{end+1,1} = false(obj.nLabelPoints,ifo.nframes,nTgt);
        obj.labeledpostag{end+1,1} = cell(obj.nLabelPoints,ifo.nframes,nTgt);
        obj.labeledpos2{end+1,1} = nan(obj.nLabelPoints,2,ifo.nframes,nTgt);

        if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
          obj.viewCalibrationData{end+1,1} = [];
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
        obj.movieAdd(movs);
      else
        fprintf('Importing %d movie sets from file ''%s''.\n',nMovSetImport,bfile);
        for i=1:nMovSetImport
          try
            obj.movieSetAdd(movs(i,:));
          catch ME
            warningNoTrace('Labeler:mov',...
              'Error trying to add movieset %d: %s Movieset not added to project.',...
              i,ME.message);
          end
        end
      end
    end

    function movieSetAdd(obj,moviefiles)
      % Add a set of movies (Multiview mode) to end of movie list.
      %
      % moviefiles: cellstr (can have macros)

      if obj.nTargets~=1
        error('Labeler:movieSetAdd','Unsupported for nTargets>1.');
      end
      
      moviefiles = cellstr(moviefiles);
      if numel(moviefiles)~=obj.nview
        error('Labeler:movieAdd',...
          'Number of moviefiles supplied (%d) must match number of views (%d).',...
          numel(moviefiles),obj.nview);
      end
      moviefilesfull = cellfun(@(x)obj.projLocalizePath(x),moviefiles,'uni',0);
      cellfun(@(x)assert(exist(x,'file')>0,'Cannot find file ''%s''.',x),moviefilesfull);
      tfMFeq = arrayfun(@(x)strcmp(moviefiles{x},obj.movieFilesAll(:,x)),1:obj.nview,'uni',0);
      tfMFFeq = arrayfun(@(x)strcmp(moviefilesfull{x},obj.movieFilesAllFull(:,x)),1:obj.nview,'uni',0);
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
      
      obj.movieFilesAll(end+1,:) = moviefiles(:)';
      obj.movieFilesAllHaveLbls(end+1,1) = false;
      obj.movieInfoAll(end+1,:) = ifos;
      obj.trxFilesAll(end+1,:) = repmat({''},1,obj.nview);
      obj.labeledpos{end+1,1} = nan(obj.nLabelPoints,2,nFrms,nTgt);
      obj.labeledposTS{end+1,1} = -inf(obj.nLabelPoints,nFrms,nTgt); 
      obj.labeledposMarked{end+1,1} = false(obj.nLabelPoints,nFrms,nTgt);
      obj.labeledpostag{end+1,1} = cell(obj.nLabelPoints,nFrms,nTgt);      
      obj.labeledpos2{end+1,1} = nan(obj.nLabelPoints,2,nFrms,nTgt);
      
      if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
        obj.viewCalibrationData{end+1,1} = [];
      end
      
      % This clause does not occur in movieAdd(), b/c movieAdd is called
      % from UI functions which do this for the user. Currently movieSetAdd
      % does not have any UI so do it here.
      if ~obj.hasMovie && obj.nmovies>0
        obj.movieSet(1,'isFirstMovie',true);
      end
    end

    function tfSucc = movieRmName(obj,movName)
      % movName: compared to .movieFilesAll (macros UNreplaced)
      iMov = find(strcmp(movName,obj.movieFilesAll));
      if isscalar(iMov)
        tfSucc = obj.movieRm(iMov);
      end
    end
    function tfSucc = movieRm(obj,iMov)
      % tfSucc: true if movie removed, false otherwise
      
      assert(isscalar(iMov));
      assert(any(iMov==1:obj.nmovies),'Invalid movie index ''%d''.');
      if iMov==obj.currMovie
        error('Labeler:movieRm','Cannot remove current movie.');
      end
      
      tfProceedRm = true;
      if obj.labelposMovieHasLabels(iMov) && ~obj.movieDontAskRmMovieWithLabels
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
        nMovOrig = obj.nmovies;
        
        obj.movieFilesAll(iMov,:) = [];
        obj.movieFilesAllHaveLbls(iMov,:) = [];
        obj.movieInfoAll(iMov,:) = [];
        obj.trxFilesAll(iMov,:) = [];
        
        tfOrig = obj.isinit;
        obj.isinit = true; % AL20160808. we do not want set.labeledpos side effects, listeners etc.
        obj.labeledpos(iMov,:) = []; % should never throw with .isinit==true        
        obj.labeledposTS(iMov,:) = [];
        obj.labeledposMarked(iMov,:) = [];
        obj.labeledpostag(iMov,:) = [];
        obj.labeledpos2(iMov,:) = [];
        obj.isinit = tfOrig;
        
        if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
          szassert(obj.viewCalibrationData,[nMovOrig 1]);
          obj.viewCalibrationData(iMov,:) = [];
        end
        
        if obj.currMovie>iMov
          obj.movieSet(obj.currMovie-1);
        end
      end
      
      tfSucc = tfProceedRm;
    end
    
    function movieFilesMacroize(obj,str,macro)
      % Replace a string with a macro throughout .movieFilesAll. A project
      % macro is also added for macro->string.
      %
      %
      % str: a string 
      % macro: macro which will replace all matches of string (macro should
      % NOT include leading $)
      
      strpat = regexprep(str,'\\','\\\\');
      obj.movieFilesAll = regexprep(obj.movieFilesAll,strpat,['$' macro]);
      
      if isfield(obj.projMacros,macro) 
        currVal = obj.projMacros.(macro);
        if strcmp(currVal,str)
          % good; current macro val is equal to str          
        else
          warningNoTrace('Labeler:macro',...
            'Project macro ''%s'' is currently defined as ''%s''.',...
            macro,currVal);
        end
      else
        obj.projMacroAdd(macro,str);
      end
    end
    
    function tfsuccess = movieCheckFilesExist(obj,iMov) % NOT obj const
      % Helper function for movieSet(), check that movie/trxfiles exist
      %
      % tfsuccess: false indicates user canceled or similar. True indicates
      % that i) obj.movieFilesAllFull(iMov,:) all exist; ii) if obj.hasTrx,
      % obj.trxFilesAll(iMov,:) all exist.
      %
      % This function can also harderror.
      %
      % This function is NOT obj const -- macro-related state can be
      % mutated eg in multiview projects when the user does something for
      % movie 1 but then movie 2 errors out. This is not that bad so we
      % accept it for now.
      
      tfsuccess = false;
      
      if ~all(cellfun(@isempty,obj.trxFilesAll(iMov,:)))
        assert(~obj.isMultiView,'Multiview labeling with targets unsupported.');
      end
                
      for iView = 1:obj.nview
        movfile = obj.movieFilesAll{iMov,iView};
        movfileFull = obj.movieFilesAllFull{iMov,iView};
        FSPath.errUnreplacedMacros(movfileFull);
        
        if exist(movfileFull,'file')==0
          if isdeployed
            error('Labeler:mov',...
              'Cannot find movie ''%s'', macro-expanded to ''%s''.',...
              movfile,movfileFull);
          end
          
          tfBrowse = false;
          if FSPath.hasMacro(movfile)
            qstr = sprintf('Cannot find movie ''%s'', macro-expanded to ''%s''.',...
              movfile,movfileFull);
            resp = questdlg(qstr,'Movie not found','Redefine macros','Browse to movie','Cancel','Cancel');
            if isempty(resp)
              resp = 'Cancel';
            end
            switch resp
              case 'Redefine macros'
                obj.projMacroSetUI();
                movfileFull = obj.movieFilesAllFull{iMov,iView};
                FSPath.errUnreplacedMacros(movfileFull);
                if exist(movfileFull,'file')==0
                  error('Labeler:mov','Cannot find movie ''%s'', macro-expanded to ''%s''',...
                    movfile,movfileFull);
                end
              case 'Browse to movie'
                tfBrowse = true;
              case 'Cancel'
                return;
            end
          else
            qstr = sprintf('Cannot find movie ''%s''.',movfile);
            
            mfaAll = obj.movieFilesAll;
            tfMfaAllHasMacro = cellfun(@FSPath.hasMacro,mfaAll);
            mfaNoMacro = mfaAll(~tfMfaAllHasMacro);
            mfaNoMacroBase = FSPath.commonbase(mfaNoMacro);
            while ~isempty(mfaNoMacroBase) && ...
                (mfaNoMacroBase(end)=='/' || mfaNoMacroBase(end)=='\')
              mfaNoMacroBase = mfaNoMacroBase(1:end-1);
            end
            if ~isempty(mfaNoMacroBase) && numel(mfaNoMacro)>=3
              resp = questdlg(qstr,'Movie not found','Browse to movie','Create/set path macro','Cancel','Cancel');
            else
              resp = questdlg(qstr,'Movie not found','Browse to movie','Cancel','Cancel');
            end
            if isempty(resp)
              resp = 'Cancel';
            end
            switch resp
              case 'Browse to movie'
                tfBrowse = true;
              case 'Create/set path macro'
                macrostrs = obj.projMacroStrs;
                if isempty(macrostrs)
                  macrostrs = '<none>';
                else
                  macrostrs = sprintf('\n%s',macrostrs{:});
                end
                macrostr = sprintf('%d movies share common base path: %s. Existing macros: %s\n',...
                  numel(mfaNoMacro),mfaNoMacroBase,macrostrs);
                respMacro = questdlg(macrostr,'Create/set macro','Use existing macro','Create new macro','Cancel','Cancel');
                if isempty(respMacro)
                  respMacro = 'Cancel';
                end
                switch respMacro
                  case 'Create new macro'
                    answ = inputdlg('Enter new macro name','Create macro',1);
                    if isempty(answ)
                      return;
                    end
                    macroName = answ{1};
                    if obj.projMacroIsMacro(macroName)
                      error('Labeler:macro','A macro named ''%s'' already exists.',macroName);
                    end
                    answ = inputdlg('Enter path represented by macro:','Set macro',1);
                    if isempty(answ)
                      return;
                    end
                    answ = answ{1};
                    obj.movieFilesMacroize(mfaNoMacroBase,macroName);
                    obj.projMacroSet(macroName,answ);
                    movfileFull = obj.movieFilesAllFull{iMov,iView};
                    if exist(movfileFull,'file')==0
                      error('Labeler:mov','Cannot find movie ''%s'', macro-expanded to ''%s''',...
                        obj.movieFilesAll{iMov,iView},movfileFull);
                    end
                  case 'Use existing macro'
                    assert(false,'Currently unsupported.');
                    %                     macros = fieldnames(obj.projMacros);
                    %                     [sel,ok] = listdlg(...
                    %                       'ListString',macros,...
                    %                       'SelectionMode','single',...
                    %                       'Name','Select macro',...
                    %                       'PromptString',sprintf('Select macro to replace base path %s',mfaNoMacroBase));
                    %                     if ~ok
                    %                       return;
                    %                     end
                  case 'Cancel'
                    return;
                end
              case 'Cancel'
                return;
            end
          end
          
          if tfBrowse
            lastmov = RC.getprop('lbl_lastmovie');
            if isempty(lastmov)
              lastmov = pwd;
            end
            [newmovfile,newmovpath] = uigetfile('*.*','Select movie',lastmov);
            if isequal(newmovfile,0)
              error('Labeler:mov','Cannot find movie ''%s''.',movfileFull);
            end
            movfileFull = fullfile(newmovpath,newmovfile);
            if exist(movfileFull,'file')==0
              error('Labeler:mov','Cannot find movie ''%s''.',movfileFull);
            end
            obj.movieFilesAll{iMov,iView} = movfileFull;
          end
          
          % At this point, either we have i) harderrored, ii)
          % early-returned with tfsuccess=false, or iii) movfileFull is set
          assert(exist(movfileFull,'file')>0);
          assert(strcmp(movfileFull,obj.movieFilesAllFull{iMov,iView}));
        end        
        
        trxFile = obj.trxFilesAll{iMov,iView};
        tfTrx = ~isempty(trxFile);
        if tfTrx
          if exist(trxFile,'file')==0
            qstr = sprintf('Cannot find trxfile ''%s''.',trxFile);
            resp = questdlg(qstr,'Trx file not found','Browse to trxfile','Cancel','Cancel');
            if isempty(resp)
              resp = 'Cancel';
            end
            switch resp
              case 'Browse to trxfile'
                % none
              case 'Cancel'
                return;
            end
            
            lasttrxfile = RC.getprop('lbl_lasttrxfile');
            if isempty(lasttrxfile)
              lasttrxfile = RC.getprop('lbl_lastmovie');
            end
            if isempty(lasttrxfile)
              lasttrxfile = pwd;
            end
            [newtrxfile,newtrxfilepath] = uigetfile('*.*','Select trxfile',lasttrxfile);
            if isequal(newtrxfile,0)
              return;
            end
            trxFile = fullfile(newtrxfilepath,newtrxfile);
            if exist(trxFile,'file')==0
              error('Labeler:trx','Cannot find trxfile ''%s''.',trxFile);
            end
            
            assert(exist(trxFile,'file')>0);
            obj.trxFilesAll{iMov,iView} = trxFile;
          end
          RC.saveprop('lbl_lasttrxfile',trxFile);
        end
      end
      
      tfsuccess = true;
    end
    
    function tfsuccess = movieSet(obj,iMov,varargin)
      % iMov: If multivew, movieSet index (row index into .movieFilesAll)      
      
      %# MVOK
      
      assert(any(iMov==1:obj.nmovies),'Invalid movie index ''%d''.',iMov);
      
      isFirstMovie = myparse(varargin,...
        'isFirstMovie',false); % passing true for the first time a movie is added to a proj helps the UI
      
      tfsuccess = obj.movieCheckFilesExist(iMov); % throws
      if ~tfsuccess
        return;
      end
      
      for iView=1:obj.nview
        movfileFull = obj.movieFilesAllFull{iMov,iView};
        obj.movieReader(iView).open(movfileFull);
        RC.saveprop('lbl_lastmovie',movfileFull);
        if iView==1
          [path0,movname] = myfileparts(obj.moviefile);
          [~,parent] = fileparts(path0);
          obj.moviename = fullfile(parent,movname);
        end
      end
      
      obj.isinit = true; % Initialization hell, invariants momentarily broken
      obj.currMovie = iMov;
      
      % for fun debugging
      %       obj.gdata.axes_all.addlistener('XLimMode','PreSet',@(s,e)lclTrace('preset'));
      %       obj.gdata.axes_all.addlistener('XLimMode','PostSet',@(s,e)lclTrace('postset'));
      obj.setFrameAndTarget(1,1);
      
      trxFile = obj.trxFilesAll{iMov,1};
      tfTrx = ~isempty(trxFile);
      if tfTrx
        assert(~obj.isMultiView,...
          'Multiview labeling with targets is currently unsupported.');
        RC.saveprop('lbl_lasttrxfile',trxFile);
        tmp = load(trxFile,'trx');
        if isfield(tmp,'trx')
          trxvar = tmp.trx;
        else
          warningNoTrace('Labeler:trx','No ''trx'' variable found in trxfile %s.',trxFile);
          trxvar = [];
        end
      else
        trxvar = [];
      end
      obj.trxSet(trxvar);
      obj.trxfile = trxFile; % this must come after .trxSet() call
        
      obj.isinit = false; % end Initialization hell      

      % AL20160615: omg this is the plague.
      % AL20160605: These three calls semi-obsolete. new projects will not
      % have empty .labeledpos, .labeledpostag, or .labeledpos2 elements;
      % these are set at movieAdd() time.
      %
      % However, some older projects will have these empty els; and
      % maybe it's worth keeping the ability to have empty els for space
      % reasons (as opposed to eg filling in all els in lblModernize()).
      % Wait and see.
      if isempty(obj.labeledpos{iMov})
        obj.labeledpos{iMov} = nan(obj.nLabelPoints,2,obj.nframes,obj.nTargets);
      end
      if isempty(obj.labeledposTS{iMov})
        obj.labeledposTS{iMov} = -inf(obj.nLabelPoints,obj.nframes,obj.nTargets); 
      end
      if isempty(obj.labeledposMarked{iMov})
        obj.labeledposMarked{iMov} = false(obj.nLabelPoints,obj.nframes,obj.nTargets); 
      end
      if isempty(obj.labeledpostag{iMov})
        obj.labeledpostag{iMov} = cell(obj.nLabelPoints,obj.nframes,obj.nTargets);
      end
      if isempty(obj.labeledpos2{iMov})
        obj.labeledpos2{iMov} = nan(obj.nLabelPoints,2,obj.nframes,obj.nTargets);
      end      
      
      % KB 20161213: moved this up here so that we could redo in initHook
      obj.labelsMiscInit();
      obj.labelingInit();
      
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
    
    function movieSetNoMovie(obj)
      % Set .currMov to 0
           
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
      obj.trxfile = '';
      obj.isinit = true;
      obj.currMovie = 0;
      obj.trxSet([]);
      obj.currFrame = 1;
      obj.currTarget = 0;
      obj.isinit = false;
      
      obj.labelsMiscInit();
      obj.labelingInit();
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
      
      obj.currSusp = [];
    end
    
    function H0 = movieEstimateImHist(obj,nFrmSamp)
      % Estimate the typical image histogram H0 of movies in the project.
      %
      % nFrmSamp: number of frames to sample. Currently the typical image
      % histogram is computed with all frames weighted equally, even if
      % some movies have far more frames than others. An alternative would
      % be eg equal-weighting by movie. nFrmSamp is only an
      % estimate/target, the actual number of frames sampled may differ.
      
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
    
  end
  
  %% Trx
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
      id2t = nan(maxID+1,1);
      for i = 1:obj.nTrx
        id2t(obj.trx(i).id+1) = i;
      end
      obj.trxIdPlusPlus2Idx = id2t;
      if isnan(obj.nframes)
        obj.frm2trx = [];
      else
        obj.frm2trx = Labeler.trxHlpComputeF2t(obj.nframes,trx);
      end
      
      obj.currImHud.updateReadoutFields('hasTgt',obj.hasTrx);
      obj.initShowTrx();
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
  end  
  methods    
    function trxSave(obj)
      % Save the current .trx to .trxfile. 
    
      tfile = obj.trxfile;
      tExist = load(tfile);
      tNew = struct('trx',obj.trx);
      if isequaln(tExist,tNew)
        msgbox(sprintf('Current trx matches that in ''%s''.',tfile));
      else
        SAVEBTN = 'OK, save and overwrite';
        CANCBTN = 'Cancel';
        str = sprintf('This will OVERWRITE the file ''%s''. Please back up your original trx!',tfile);
        ret = questdlg(str,'Save Trx',SAVEBTN,CANCBTN,CANCBTN);
        if isempty(ret)
          ret = CANCBTN;
        end
        switch ret
          case SAVEBTN
            save(tfile,'-struct','tNew');
          case CANCBTN
            % none
          otherwise
            assert(false);
        end
      end
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
    
  end
   
  %% Labeling
  methods
    
    function labelingInit(obj,varargin)
      % Create LabelCore and call labelCore.init() based on current 
      % .labelMode, .nLabelPoints, .labelPointsPlotInfo, .labelTemplate      
      
      lblmode = myparse(varargin,...
        'labelMode',[]);
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
      x = obj.labeledpos{iMov}(:,:,iFrm,iTgt);
      if all(isnan(x(:)))
        % none; short-circuit set to avoid triggering .labeledposNeedsSave
      else        
        obj.labeledpos{iMov}(:,:,iFrm,iTgt) = nan;
        obj.labeledposNeedsSave = true;        
      end
      
      obj.labeledposTS{iMov}(:,iFrm,iTgt) = now();
      obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
      obj.labeledpostag{iMov}(:,iFrm,iTgt) = {[]};
    end
    
    function labelPosClearI(obj,iPt)
      % Clear labels and tags for current movie/frame/target, point iPt
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;      
      xy = obj.labeledpos{iMov}(iPt,:,iFrm,iTgt);
      if all(isnan(xy))
        % none; short-circuit set to avoid triggering .labeledposNeedsSave
      else
        obj.labeledpos{iMov}(iPt,:,iFrm,iTgt) = nan;
        obj.labeledposNeedsSave = true;
      end
      
      obj.labeledposTS{iMov}(iPt,iFrm,iTgt) = now();
      obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
      obj.labeledpostag{iMov}{iPt,iFrm,iTgt} = [];
    end
    
    function [tf,lpos,lpostag,lposmarked] = labelPosIsLabeled(obj,iFrm,iTrx)
      % For current movie. Labeled includes fullyOccluded
      %
      % tf: scalar logical
      % lpos: [nptsx2] xy coords for iFrm/iTrx
      % lpostag: [npts] cell array of tags 
      % lposmarked: [npts] logical array
      
      iMov = obj.currMovie;
      lpos = obj.labeledpos{iMov}(:,:,iFrm,iTrx);
      tfnan = isnan(lpos);
      %assert(all(tfnan(:)) || ~any(tfnan(:)));
      tf = any(~tfnan(:));
      
      if nargout>=3
        lpostag = obj.labeledpostag{iMov}(:,iFrm,iTrx);
        lposmarked = obj.labeledposMarked{iMov}(:,iFrm,iTrx);
      end
    end 
    
    function tf = labelPosIsLabeledMov(obj,iMov)
      % iMov: movie index (row index into .movieFilesAll)
      %
      % tf: [nframes-for-iMov], true if any point labeled in that mov/frame

      %#MVOK
      
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
      lpos = obj.labeledpos{iMov}(:,:,iFrm,iTrx);
      tf = isinf(lpos(:,1));
    end
    
    function labelPosSet(obj,xy)
      % Set labelpos from labelPtsH for current movie/frame/target
            
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledpos{iMov}(:,:,iFrm,iTgt) = xy;
      obj.labeledposTS{iMov}(:,iFrm,iTgt) = now();
      obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;

      obj.labeledposNeedsSave = true;
    end
        
    function labelPosSetI(obj,xy,iPt)
      % Set labelpos for current movie/frame/target, point iPt
      
      assert(~any(isnan(xy(:))));
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledpos{iMov}(iPt,:,iFrm,iTgt) = xy;
      obj.labeledposTS{iMov}(iPt,iFrm,iTgt) = now();
      obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;

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
      
      obj.labeledpos{iMov}(iPt,1,frms,iTgt) = xy(1);
      obj.labeledpos{iMov}(iPt,2,frms,iTgt) = xy(2);
      obj.updateFrameTableComplete(); % above sets mutate .labeledpos{obj.currMovie} in more than just .currFrame
      
      obj.labeledposTS{iMov}(iPt,frms,iTgt) = now();
      obj.labeledposMarked{iMov}(iPt,frms,iTgt) = true;

      obj.labeledposNeedsSave = true;
    end
    
    function labelPosSetFromLabeledPos2(obj)
      % copy .labeledpos2 to .labeledpos for current movie/frame/target
      
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
      
      iMov = obj.currMovie;
      lposOld = obj.labeledpos{iMov};
      szassert(xy,size(lposOld));
      obj.labeledpos{iMov} = xy;
      obj.labeledposTS{iMov}(:) = now();
      obj.labeledposMarked{iMov}(:) = true; % not sure of right treatment
      
      obj.updateFrameTableComplete();
      obj.labeledposNeedsSave = true;
    end
    
    function labelPosSetUnmarkedFramesMovieFramesUnmarked(obj,xy,iMov,frms)
      % Set all unmarked labels for given movie, frames. Newly-labeled 
      % points are NOT marked in .labeledposmark
      %
      % xy: [nptsx2xnumel(frms)xntgts]
      % iMov: scalar movie index
      % frms: frames for iMov; labels 3rd dim of xy
      
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
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledposMarked{iMov}(:,iFrm,iTgt) = false;
    end
    
    function labelPosSetAllMarked(obj,val)
      % Clear .labeledposMarked for current movie, all frames/targets
      obj.labeledposMarked{iMov}(:) = val;
    end
        
    function labelPosSetOccludedI(obj,iPt)
      % Occluded is "pure occluded" here
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledpos{iMov}(iPt,:,iFrm,iTgt) = inf;
      obj.labeledposTS{iMov}(iPt,iFrm,iTgt) = now();
      obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
      
      obj.labeledposNeedsSave = true;
    end
        
    function labelPosTagSetI(obj,tag,iPt)
      % Set a single tag onto points
      %
      % tag: char. 
      % iPt: can be vector
      %
      % The same tag value will be set to all elements of iPt.      
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledposTS{iMov}(iPt,iFrm,iTgt) = now();
      [obj.labeledpostag{iMov}{iPt,iFrm,iTgt}] = deal(tag); 
    end
    
    function labelPosTagClearI(obj,iPt)
      % iPt: can be vector
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledposTS{iMov}(iPt,iFrm,iTgt) = now();
      [obj.labeledpostag{iMov}{iPt,iFrm,iTgt}] = deal([]);
    end
    
    function labelPosTagSetFramesI(obj,tag,iPt,frms)
      % Set tags for current movie/target, given pt/frames

      obj.trxCheckFramesLiveErr(frms);
      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      obj.labeledpostag{iMov}(iPt,frms,iTgt) = {tag};
    end
    
    function labelPosTagClearFramesI(obj,iPt,frms)
      % Clear tags for current movie/target, given pt/frames
      
      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      obj.labeledpostag{iMov}(iPt,frms,iTgt) = {[]};    
    end
    
    function [tfneighbor,iFrm0,lpos0] = labelPosLabeledNeighbor(obj,iFrm,iTrx)
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
      lposTrx = obj.labeledpos{iMov}(:,:,:,iTrx);
      for dFrm = 0:obj.NEIGHBORING_FRAME_OFFSETS % xxx AL apparent bug
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
    
    function [nTgts,nPts] = labelPosLabeledFramesStats(obj,frms)
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
      %npts = obj.nLabelPoints;
      ntgts = obj.nTargets;
      lpos = obj.labeledpos{obj.currMovie};
      tflpos = ~isnan(lpos); % true->labeled (either regular or occluded)      
      
      nTgts = zeros(nf,1);
      nPts = zeros(nf,1);
      if tfWaitBar
        hWB = waitbar(0,'Updating frame table');
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
%           tfTgtLabeled = false;
%           for iPt = 1:npts
          z = sum(tflpos(:,1,f,iTgt));
          tmpNPts = tmpNPts+z;
          tfTgtLabeled = (z>0);
%             if tflpos(iPt,1,f,iTgt)
%               tmpNPts = tmpNPts+1;
%               tfTgtLabeled = true;
%             end
%           end
          if tfTgtLabeled
            tmpNTgts = tmpNTgts+1;
          end
        end
        nTgts(i) = tmpNTgts;
        nPts(i) = tmpNPts;        
      end
    end
    
    function tf = labelposMovieHasLabels(obj,iMov)
      lpos = obj.labeledpos{iMov};
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
    function [tfok,trkfiles] = checkTrkFileNamesExport(trkfiles)
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
      sMacro = struct();
      sMacro.projname = obj.projname;
      [sMacro.projdir,sMacro.projfile] = fileparts(obj.projectfile);
    end
    
    function trkfile = defaultTrkFileName(obj,movfile)
      trkfile = Labeler.genTrkFileName(obj.defaultTrkRawname(),...
        obj.baseTrkFileMacros(),movfile);
    end
    
    function rawname = defaultTrkRawname(obj)
      prjname = obj.projname;
      if isempty(prjname)
        basename = Labeler.DEFAULT_TRKFILE_NOPROJ;
      else
        basename = Labeler.DEFAULT_TRKFILE;        
      end
      rawname = fullfile('$movdir',basename);
    end
        
    function [tfok,trkfiles] = getTrkFileNamesForExport(obj,movfiles,rawname)
      sMacro = obj.baseTrkFileMacros();
      trkfiles = cellfun(@(x)Labeler.genTrkFileName(rawname,sMacro,x),movfiles,'uni',0);
      [tfok,trkfiles] = Labeler.checkTrkFileNamesExport(trkfiles);
    end
    
    function [tfok,trkfiles] = resolveTrkfilesVsRawname(obj,iMovs,...
        trkfiles,rawname)
      % Input arg helper -- use basename if trkfiles not supplied; check
      % sizes. 
      %
      % tfok: scalar, if true then trkfiles is usable; if false then user
      %   canceled
      % trkfiles: [iMovs] cellstr, trkfiles (full paths) to export to
      % 
      % This call can also throw.
      
      movfiles = obj.movieFilesAllFull(iMovs,:);
      if isempty(trkfiles)
        if isempty(rawname)
          rawname = obj.defaultTrkRawname;
        end
        [tfok,trkfiles] = obj.getTrkFileNamesForExport(movfiles,rawname);
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
	
    function [trkfilesCommon,kwCommon,trkfilesAll] = getTrkFileNamesForImport(obj,movfiles)
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
      % iMov: optional, indices into (rows of) .movieFilesAll to export. 
      %   Defaults to 1:obj.nmovies.
      
      [trkfiles,rawtrkname] = myparse(varargin,...
        'trkfiles',[],... % [nMov nView] cellstr, fullpaths to trkfilenames to export to
        'rawtrkname',[]... % string, rawname to apply over iMovs to generate trkfiles
        );
      
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmovies;
      end
      
      [tfok,trkfiles] = obj.resolveTrkfilesVsRawname(iMovs,trkfiles,rawtrkname);
      if ~tfok
        return;
      end
        
      nMov = numel(iMovs);
      nView = obj.nview;
      nPhysPts = obj.nPhysPoints;
      for i=1:nMov
        iMvSet = iMovs(i);
        lposFull = obj.labeledpos{iMvSet};
        lposTSFull = obj.labeledposTS{iMvSet};
        lposTagFull = obj.labeledpostag{iMvSet};
        
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
      msgbox(sprintf('Results for %d moviesets exported.',nMov),'Export complete.');
    end
    
    function labelImportTrkGeneric(obj,iMovSets,trkfiles,lposFld,lposTSFld,lposTagFld)
      % iMovStes: [N] vector of movie set indices
      % trkfiles: [Nxnview] cellstr of trk filenames
      % lpos*Fld: property names for labeledpos, labeledposTS,
      % labeledposTag. Can be empty to not set that prop.
      
      nMovSets = numel(iMovSets);
      szassert(trkfiles,[nMovSets obj.nview]);
      nPhysPts = obj.nPhysPoints;   
      tfMV = obj.isMultiView;
      nView = obj.nview;
      
      for i=1:nMovSets
        iMov = iMovSets(i);
        lpos = nan(size(obj.labeledpos{iMov}));
        lposTS = -inf(size(obj.labeledposTS{iMov}));
        lpostag = cell(size(obj.labeledpostag{iMov}));
        assert(size(lpos,1)==nPhysPts*nView);
        
        if tfMV
          fprintf('MovieSet %d...\n',iMov);
        end
        for iVw = 1:nView
          tfile = trkfiles{i,iVw};
          s = load(tfile,'-mat');
          
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
      
      obj.labelImportTrkGeneric(iMovs,trkfiles,'labeledpos',...
          'labeledposTS','labeledpostag');
      
      obj.movieFilesAllHaveLbls(iMovs) = ...
        cellfun(@(x)any(~isnan(x(:))),obj.labeledpos(iMovs));
      
      obj.updateFrameTableComplete();
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
    
    function tblMF = labelGetMFTableLabeled(obj)
      % Compile mov/frm/tgt MFTable; include all labeled frames/tgts
      %
      % tblMF: See MFTable.FLDSFULLTRX.
      
      movIDs = FSPath.standardPath(obj.movieFilesAll);
      if obj.hasTrx
        tblMF = Labeler.labelGetMFTableLabeledStc(movIDs,obj.labeledpos,...
          obj.labeledpostag,obj.labeledposTS,obj.trxFilesAll);
      else
        [~,tblMF] = Labeler.lblCompileContents(obj.movieFilesAllFull,...
          obj.labeledpos,obj.labeledpostag,'lbl',...
          'noImg',true,'lposTS',obj.labeledposTS,'movieNamesID',movIDs);
        tblMF.iTgt = ones(height(tblMF),1);
        tblMF.pTrx = nan(height(tblMF),2);
      end
      
      tblfldsassert(tblMF,MFTable.FLDSFULLTRX);
    end
    
    function tblMF = labelGetMFTableAll(obj,iMov,frmCell)
      % Compile mov/frm/tgt MFTable for given movies/frames.
      %
      % iMov: [n] vector of movie(set) indices
      % frmsCell: [n] cell vector. frms{i} is a vector of frames to read 
      %   for movie iMov(i), or the string 'all' for all frames in the
      %   movie.
      % roiRadius: scalar, roi crop radius. Ignored if ~.hasTrx.
      %
      % tblMF: See MFTable.FLDSFULLTRX.
      
      movIDs = FSPath.standardPath(obj.movieFilesAll);
      if obj.hasTrx        
        tblMF = Labeler.labelGetMFTableLabeledStc(movIDs,obj.labeledpos,...
          obj.labeledpostag,obj.labeledposTS,obj.trxFilesAll,...
          'iMovRead',iMov,'frmReadCell',frmCell,'tgtsRead','live');
      else
        [~,tblMF] = Labeler.lblCompileContentsRaw(obj.movieFilesAllFull,...
          obj.labeledpos,obj.labeledpostag,iMovs,frmCell,...
          'noImg',true,'lposTS',obj.labeledposTS,'movieNamesID',movIDs);
        tblMF.iTgt = ones(height(tblMF),1);
        tblMF.pTrx = nan(height(tblMF,2));
      end
      
      tblfldsassert(tblMF,MFTable.FLDSFULLTRX);
    end
    
    function tblMF = labelGetMFTableCurrMovFrmTgt(obj)
      % Get MFTable for current movie/frame/target (single-row table)
      %
      % tblMF: See MFTable.FLDSFULLTRX.
                  
      iMov = obj.currMovie;
      frm = obj.currFrame;
      iTgt = obj.currTarget;
      lposFrmTgt = obj.labeledpos{iMov}(:,:,frm,iTgt);
      lpostagFrmTgt = obj.labeledpostag{iMov}(:,frm,iTgt);
      lposTSFrmTgt = obj.labeledposTS{iMov}(:,frm,iTgt);      
      movID = FSPath.standardPath(obj.movieFilesAll(iMov,:));

      mov = movID;
      p = Shape.xy2vec(lposFrmTgt); % absolute position
      pTS = lposTSFrmTgt';
      tfocc = strcmp(lpostagFrmTgt','occ');
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
    
    function tblMF = labelMFTableAddROI(obj,tblMF,roiRadius)
      % Add .pRoi and .roi to tblMF
      %
      % tblMF.pRoi: Just like tblMF.p, but relative to tblMF.roi (p==1 => 
      %   first row/col of ROI)
      % tblMF.roi: [nrow x 2*2*nview]. Raster order {lo,hi},{x,y},view
      
      tblfldsassert(tblMF,MFTable.FLDSFULLTRX);
      
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
            'Movie(set) ''%s'', frame %d, target %d: shape out of bounds of target ROI. Not including row.',...
            tblMF.mov{i},tblMF.frm(i),tblMF.iTgt(i));
          tfRmRow(i) = true;
        else
          pRoi(i,:) = Shape.xy2vec(xyROIcurr);
          roi(i,:) = roiCurr;
        end
      end
      
      tblMF = [tblMF table(pRoi,roi)];
      tblMF(tfRmRow,:) = [];
    end
    
  end
  
  methods (Static)
    
    function tblMF = labelGetMFTableLabeledStc(movID,lpos,lpostag,lposTS,...
        trxFilesAll,varargin)
      % Compile MFtable, by default for all labeled mov/frm/tgts
      %
      % movID: [NxnView] cellstr of movie IDs (use non-macro-replaced etc)
      % lpos: [N] cell array of labeledpos arrays [npts x 2 x nfrms x ntgts]. 
      %   For multiview, npts=nView*NumLabelPoints.
      % lpostag: [N] cell array of labeledpostags [npts x nfrms x ntgts]
      % lposTS: [N] cell array of labeledposTS [npts x nfrms x ntgts]
      % trxFilesAll: [NxnView] cellstr of trxfiles corresponding to movID
      %
      % tblMF: [NTrl rows] MFTable, one row per labeled movie/frame/target.
      %   MULTIVIEW NOTE: tbl.p* is the 2d/projected label positions, ie
      %   each shape has nLabelPoints*nView*2 coords, raster order is 1. pt
      %   index, 2. view index, 3. coord index (x vs y)
      %   
      %   Fields: {'mov' 'frm' 'iTgt' 'p' 'pTS' 'tfocc' 'pTrx'}
      %   Here 'p' is 'pAbs' or absolute position
      
       [iMovRead,frmReadCell,tgtsRead] = myparse(varargin,...
        'iMovRead',[],... % row indices into movID for movies to read/include
        'frmReadCell',[], ... % [N], or [numel(iMovRead)] if iMovRead supplied. Cell array of frames to read for each movie. If not supplied, all labeled frame/trx are included
        'tgtsRead','lbl' ... % char. Either 'lbl' to include all labeled targets for each mov/frame read; or 'live' to include all live targets for each mov/frame read
        );
      
      s = structconstruct(MFTable.FLDSFULLTRX,[0 1]);
      
      [nMov,nView] = size(movID);
      szassert(lpos,[nMov 1]);
      szassert(lpostag,[nMov 1]);
      szassert(lposTS,[nMov 1]);
      szassert(trxFilesAll,[nMov nView]);
      
      if isequal(iMovRead,[])
        iMovRead = 1:nMov;
      end
      nMovRead = numel(iMovRead);
      if isequal(frmReadCell,[])
        frmReadCell = repmat({'all'},nMovRead,1);
      end
      assert(iscell(frmReadCell) && numel(frmReadCell)==nMovRead);
      
      for iRead = 1:nMovRead
        iMov = iMovRead(iRead);
%         if tfWB
%           hWB.Name = 'Scanning movies';
%           wbStr = sprintf('Reading movie %s',movID);
%           waitbar(0,hWB,wbStr);
%         end        
        
        movIDI = movID(iMov,:);
        lposI = lpos{iMov};
        lpostagI = lpostag{iMov};
        lposTSI = lposTS{iMov};
        [npts,d,nfrms,ntgts] = size(lposI);
        assert(d==2);
        szassert(lpostagI,[npts nfrms ntgts]);
        szassert(lposTSI,[npts nfrms ntgts]);
        
        % load trx for all views
        trxI = cell(1,nView);
        frm2trx = cell(1,nView);
        for iView=1:nView
          tfile = trxFilesAll{iMov,iView};
          if exist(tfile,'file')==0
            error('Labeler:file','Cannot find trxfile ''%s''.',tfile);
          end
          tmp = load(tfile,'-mat','trx');
          trx = tmp.trx;
          assert(numel(trx)==ntgts);
          trxI{iView} = trx;
          frm2trx{iView} = Labeler.trxHlpComputeF2t(nfrms,trx);
        end
        % In multiview multitarget projs, the each view's trx must contain
        % the same number of els and these elements must correspond across
        % views. 
        if nView>1 && isfield(trxI{1},'id')
          trxids = cellfun(@(x)[x.id],trxI,'uni',0);
          assert(isequal(trxids{:}),'Trx ids differ.');
        end
        frm2trxOverall = frm2trx{1};
        for iView=2:nView
          frm2trxOverall = or(frm2trxOverall,frm2trx{iView});
        end
        % frm2trxOverall: [nfrm x ntgts] logical array, true at (i,j) iff
        % target j is live in any view at frame i 
        
%         if isempty(lpos)
%           assert(isempty(lpostag));
%           lpostag = cell(npts,nFrmAll); % edge case: when lpos/lpostag are [], uninitted/degenerate case
%         end
        frmsRead = frmReadCell{iRead};
        if ischar(frmsRead) && strcmp(frmsRead,'all')
          frmsRead = 1:nfrms; %all frames in this movie
        end
        nFrmsRead = numel(frmsRead);
      
        for iF=1:nFrmsRead
          f = frmsRead(iF);
          for iTgt=1:ntgts
            lposIFrmTgt = lposI(:,:,f,iTgt);
            switch tgtsRead
              case 'lbl'
                % read if any point (in any view) is labeled for this 
                % (frame,target)
                tfReadTgt = any(~isnan(lposIFrmTgt(:)));
                if tfReadTgt
                  assert(frm2trxOverall(f,iTgt),'Labeled target is not live.');
                end
              case 'live'
                tfReadTgt = frm2trxOverall(f,iTgt);                
              otherwise
                assert(false);
            end
            if tfReadTgt
              lpostagIFrmTgt = lpostagI(:,f,iTgt);
              lposTSIFrmTgt = lposTSI(:,f,iTgt);
              xtrxs = cellfun(@(xx)xx(iTgt).x(f+xx(iTgt).off),trxI);
              ytrxs = cellfun(@(xx)xx(iTgt).y(f+xx(iTgt).off),trxI);
              
              s(end+1,1).mov = movIDI; %#ok<AGROW> % xxx apparent bug, need FSPath.formMultiMovieID; but only for hasTrx + multiview which is currently unsupported
              s(end).frm = f;
              s(end).iTgt = iTgt;
              s(end).p = Shape.xy2vec(lposIFrmTgt);
              s(end).pTS = lposTSIFrmTgt';
              s(end).tfocc = strcmp(lpostagIFrmTgt','occ');
              s(end).pTrx = [xtrxs(:)' ytrxs(:)'];
            end
          end
        end
      end
      tblMF = struct2table(s,'AsArray',true);      
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
      % have the same size.
      % movWidths, movHeights: [nMovSetxnView] arrays
            
      nView = obj.nview;
      ifo = obj.movieInfoAll;
      movWidths = cellfun(@(x)x.info.Width,ifo);
      movHeights = cellfun(@(x)x.info.Height,ifo);
      szassert(movWidths,[obj.nmovies nView]);
      szassert(movHeights,[obj.nmovies nView]);
      
      tfAllSame = true(1,nView);
      if obj.nmovies>0
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
  end  
  methods
    
    function viewCalClear(obj)
      obj.viewCalProjWide = [];
      obj.viewCalibrationData = [];
      % Currently lblCore is not cleared, change will be reflected in
      % labelCore at next movie change etc
      
%       lc = obj.lblCore;      
%       if lc.supportsCalibration
%         warning('Labeler:viewCal','');
%       end
    end
    
    function viewCalSetProjWide(obj,crObj,varargin)
      % Set project-wide calibration object.
      
      tfSetViewSizes = myparse(varargin,...
        'tfSetViewSizes',false); % If true, set viewSizes on crObj per current movieInfo
      
      if obj.nmovies==0
        error('Labeler:calib','Add a movie first before setting the calibration object.');
      end
      
      obj.viewCalCheckCalRigObj(crObj);
      
      vcdPW = obj.viewCalProjWide;
      if ~isempty(vcdPW) && ~vcdPW
        warningNoTrace('Labeler:viewCal',...
          'Discarding movie-specific calibration data. Calibration data will apply to all movies.');
        obj.viewCalProjWide = true;
        obj.viewCalibrationData = [];    
      end
      [tfAllSame,movWidths,movHeights] = obj.viewCalCheckMovSizes();
      
      if tfSetViewSizes
        if all(tfAllSame)
          iMovUse = 1;
        else
          iMovUse = obj.currMovie;
          if iMovUse==0
            iMovUse = 1; % dangerous for user, but we already warned them
          end
        end
        vwSizes = [movWidths(iMovUse,:)' movHeights(iMovUse,:)'];
        crObj.viewSizes = vwSizes;
        for iVw=1:obj.nview
          fprintf(1,'Calibration obj: set [width height] = [%d %d] for view %d (%s).\n',...
            vwSizes(iVw,1),vwSizes(iVw,2),iVw,crObj.viewNames{iVw});
        end
      else
        % Check view sizes        
        iMovCheck = obj.currMovie;
        if iMovCheck==0
          if obj.nmovies==0
            iMovCheck = nan;
          else
            iMovCheck = 1;
          end
        end
        if ~isnan(iMovCheck)
          vwSizesExpect = [movWidths(iMovCheck,:)' movHeights(iMovCheck,:)'];
          if ~isequal(crObj.viewSizes,vwSizesExpect)
            warnstr = sprintf('View sizes in calibration object (%s) do not match movie (%s).',...
              mat2str(crObj.viewSizes),mat2str(vwSizesExpect));
            warndlg(warnstr,'View size mismatch','non-modal');
          end
        end
      end      
      
      obj.viewCalProjWide = true;
      obj.viewCalibrationData = crObj;

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
      
      if obj.nmovies==0 || obj.currMovie==0
        error('Labeler:calib','Add/select a movie first before setting the calibration object.');
      end

      obj.viewCalCheckCalRigObj(crObj);      

      vcdPW = obj.viewCalProjWide;
      if isempty(vcdPW)
        obj.viewCalProjWide = false;
        obj.viewCalibrationData = cell(obj.nmovies,1);
      elseif vcdPW
        warningNoTrace('Labeler:viewCal',...
          'Discarding project-wide calibration data. Calibration data will need to be set on other movies.');
        obj.viewCalProjWide = false;
        obj.viewCalibrationData = cell(obj.nmovies,1);
      else
        assert(iscell(obj.viewCalibrationData));
        szassert(obj.viewCalibrationData,[obj.nmovies 1]);
      end        
      
      ifo = obj.movieInfoAll(obj.currMovie,:);
      movWidths = cellfun(@(x)x.info.Width,ifo);
      movHeights = cellfun(@(x)x.info.Height,ifo);
      vwSizes = [movWidths' movHeights'];
      if tfSetViewSizes
        crObj.viewSizes = vwSizes;
        arrayfun(@(x)fprintf(1,'Calibration obj: set [width height] = [%d %d] for view %d (%s).\n',...
            vwSizes(x,1),vwSizes(x,2),x,crObj.viewNames{x}),1:obj.nview);
      else
        if ~isequal(crObj.viewSizes,vwSizes)
          warnstr = sprintf('View sizes in calibration object (%s) do not match movie (%s).',...
            mat2str(crObj.viewSizes),mat2str(vwSizes));
          warndlg(warnstr,'View size mismatch','non-modal');
        end
      end
      
      obj.viewCalibrationData{obj.currMovie} = crObj;
      
      lc = obj.lblCore;
      if lc.supportsCalibration
        lc.projectionSetCalRig(crObj);
      else
        warning('Labeler:viewCal','Current labeling mode does not utilize view calibration.');
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
      obj.labels2VizUpdate();
    end
    
    function labelsUpdateNewTarget(obj,prevTarget)
      if ~isempty(obj.lblCore)
        obj.lblCore.newTarget(prevTarget,obj.currTarget,obj.currFrame);
      end
      obj.prevAxesLabelsUpdate();
      obj.labels2VizUpdate();
    end
    
    function labelsUpdateNewFrameAndTarget(obj,prevFrm,prevTgt)
      if ~isempty(obj.lblCore)
        obj.lblCore.newFrameAndTarget(...
          prevFrm,obj.currFrame,...
          prevTgt,obj.currTarget);
      end
      obj.prevAxesLabelsUpdate();
      obj.labels2VizUpdate();
    end
        
  end
   
  %% Susp 
  methods
    
    function setSuspScore(obj,ss)
      assert(~obj.isMultiView);
      
      if isequal(ss,[])
        % none; this is ok
      else
        nMov = obj.nmovies;
        nTgt = obj.nTargets;
        assert(iscell(ss) && isvector(ss) && numel(ss)==nMov);
        for iMov = 1:nMov
          ifo = obj.movieInfoAll{iMov,1}; 
          assert(isequal(size(ss{iMov}),[ifo.nframes nTgt]),...
            'Size mismatch for score for movie %d.',iMov);
        end
      end
      
      obj.suspScore = ss;
    end
    
    function updateCurrSusp(obj)
      % Update .currSusp from .suspScore, currMovie, .currFrm, .currTarget
      
      tfDoSusp = ~isempty(obj.suspScore) && ...
                  obj.hasMovie && ...
                  obj.currMovie>0 && ...
                  obj.currFrame>0;
      if tfDoSusp
        ss = obj.suspScore{obj.currMovie};
        obj.currSusp = ss(obj.currFrame,obj.currTarget);       
      else
        obj.currSusp = [];
      end
      if ~isequal(obj.currSusp,[])
        obj.currImHud.updateSusp(obj.currSusp);
      end
    end
      
  end
  
  %% Tracker
  methods
    
    function setTrackParamFile(obj,prmFile)
      trker = obj.tracker;
      assert(~isempty(trker),'No tracker object currently set.');
      trker.setParamFile(prmFile);      
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
    
    function trackRetrain(obj)
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      if ~obj.hasMovie
        error('Labeler:track','No movie.');
      end
      tObj.retrain();
    end
    
    function track(obj,tm,varargin)
      % tm: a TrackMode
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end      
      [iMovs,frms] = tm.getMovsFramesToTrack(obj);
      tObj.track(iMovs,frms,varargin{:});
    end
    
    function trackAndExport(obj,tm,varargin)
      % Track one movie at a time, exporting results to .trk files and 
      % clearing data in between
      %
      % tm: scalar TrackMode
            
      [trackArgs,rawtrkname] = myparse(varargin,...
        'trackArgs',{},...
        'rawtrkname',[]...
        );
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      [iMovs,frms] = tm.getMovsFramesToTrack(obj);
      
      [tfok,trkfiles] = obj.resolveTrkfilesVsRawname(iMovs,[],rawtrkname);
      if ~tfok
        return;
      end
      
      nMov = numel(iMovs);
      nVw = obj.nview;
      szassert(trkfiles,[nMov nVw]);
      if obj.isMultiView
        moviestr = 'movieset';
      else
        moviestr = 'movie';
      end
      for i=1:nMov
        fprintf('Tracking %s %d (%d/%d)\n',moviestr,iMovs(i),i,nMov);
        tObj.track(iMovs(i),frms(i),trackArgs{:});
        trkFile = tObj.getTrackingResults(iMovs(i));
        szassert(trkFile,[1 nVw]);
        for iVw=1:nVw
          trkFile(iVw).pTrkFull = single(trkFile(iVw).pTrkFull);
          trkFile(iVw).save(trkfiles{i,iVw});
          fprintf('...saved: %s\n',trkfiles{i,iVw});
        end
        tObj.clearTrackingResults();
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
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end 
      
      [tfok,trkfiles] = obj.resolveTrkfilesVsRawname(iMovs,trkfiles,rawtrkname);
      if ~tfok
        return;
      end
      
      movfiles = obj.movieFilesAllFull(iMovs,:);
      [trkFileObjs,tfHasRes] = tObj.getTrackingResults(iMovs);
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
        
    function [dGTTrkCell,pTrkCell,tblMFgt,cvPart] = ...
                                  trackCrossValidate(obj,varargin)
      % Run k-fold crossvalidation
      %
      % dGTTrkCell: [nfoldx1] cell. dGTTrkCell{iFold} is [nTrkIxnpts] L2 
      %   dist between trk and GT for ith fold
      % pTrkCell: [nfoldx1] cell. pTrkCell{iFold} is [nTrkIxncol] 
      %   tracking results table for ith fold (nTrkI can be diff across 
      %   folds)
      % tblMFgt: [nGTxncol] FLDSCORE or FLDSCOREROI table for all lbled/gt
      %   data in proj
      % cvPart: cvpartition used in crossvalidation
      
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
        tblMFgt = obj.labelGetMFTableLabeled();
        tblMFgt = tObj.hlpAddRoiIfNec(tblMFgt);
      end
      
      % Partition MFT table
      movC = categorical(tblMFgt.mov);
      tgtC = categorical(tblMFgt.iTgt);
      grpC = movC.*tgtC;
      cvPart = cvpartition(grpC,'kfold',kFold);

      % Basically an initHook() here
      if initData
        tObj.initData();
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
        tObj.track([],[],'tblP',tblMFgtTrack,'wbObj',wbObj);        
        [tblTrkRes,pTrkiPt] = tObj.getAllTrackResTable(); % if wbObj.isCancel, partial tracking results
        if initData
          tObj.initData();
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
    end
    
  end
  methods (Static)
    
    function [nGT,nFold,muErr,muErrPt,tblErrMov,tbl] = ...
        trackCrossValidateStats(dGTTrkCell,pTrkCell)

      assert(iscell(dGTTrkCell) && iscell(pTrkCell) && ...
            numel(dGTTrkCell)==numel(pTrkCell));
          
      tbl = cat(1,pTrkCell{:});
      tbl.err = cat(1,dGTTrkCell{:});
      nGT = height(tbl);
      nFold = numel(dGTTrkCell);
      muErrPt = nanmean(tbl.err,1); % [1xnpt]
      muErr = nanmean(muErrPt); % each pt equal wt
      movUn = unique(tbl.mov);
      muErrMov = cellfun(@(x) nanmean(nanmean(tbl.err(strcmp(tbl.mov,x),:),1)), ...
        movUn); % each pt equal wt
      movUnCnt = cellfun(@(x)nnz(strcmp(tbl.mov,x)),movUn);
      tblErrMov = table(movUn,movUnCnt,muErrMov,...
        'VariableNames',{'mov' 'count' 'err'});
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
%       % XXX unfinished, rationalize track save/load/export
%     end
%     
%     function [success,fname] = trackLoadResultsAs(obj)
%       [success,fname] = obj.trackSaveLoadAsHelper('lastTrackingResultsFile',...
%         'uigetfile','Load tracking results','trackLoadResults');
%       % XXX unfinished, rationalize track save/load/export
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
  
  %% showTrx
  methods
    
    function initShowTrx(obj)
      deleteValidHandles(obj.hTraj);
      deleteValidHandles(obj.hTrx);
      obj.hTraj = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrx = matlab.graphics.primitive.Line.empty(0,1);
      
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
          'Color',pref.TrajColor',...
          'MarkerSize',pref.TrxMarkerSize,...
          'LineWidth',pref.TrxLineWidth);
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
            xTrx = trxCurr.x(t+trxCurr.off);
            yTrx = trxCurr.y(t+trxCurr.off);
          else
            xTrx = nan;
            yTrx = nan;
          end
          set(obj.hTrx(iTrx),'XData',xTrx,'YData',yTrx,'Color',color);          
        end
      end
      set(obj.hTraj(tfShow),'Visible','on');
      set(obj.hTraj(~tfShow),'Visible','off');
      set(obj.hTrx(tfShow),'Visible','on');
      set(obj.hTrx(~tfShow),'Visible','off');
    end
    
  end
  
  %% Navigation
  methods
  
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
        obj.updateCurrSusp();
      end
      if updateTrajs
        obj.updateShowTrx();
      end
    end
    
    function setTargetID(obj,tgtID)
      % Set target ID, maintaining current movie/frame.
      
      iTgt = obj.trxIdPlusPlus2Idx(tgtID+1);
      assert(~isnan(iTgt),'Invalid target ID: %d.');
      obj.setTarget(iTgt);
    end
    
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
      obj.updateCurrSusp();
      obj.updateShowTrx();
    end
        
    function setFrameAndTarget(obj,frm,iTgt)
      % Set to new frame and target for current movie.
      % Prefer setFrame() or setTarget() if possible to
      % provide better continuity wrt labeling etc.
     
      validateattributes(iTgt,{'numeric'},{'positive' 'integer' '<=' obj.nTargets});

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
        obj.updateCurrSusp();
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
    
    function tfSetOccurred = frameDown(obj,tfBigstep)
      if tfBigstep
        df = obj.movieFrameStepBig;
      else
        df = 1;
      end
      tfSetOccurred = obj.frameDownDF(df);
    end
    
    function frameUpNextLbled(obj,tfback)
      % call obj.setFrame() on next labeled frame. 
      % 
      % tfback: optional. if true, go backwards.
      
      if exist('tfback','var')==0
        tfback = false;
      end
      if tfback
        df = -1;
      else
        df = 1;
      end
      
      lpos = obj.labeledpos{obj.currMovie};
      f = obj.currFrame;
      nf = obj.nframes;
      npt = obj.nLabelPoints;
      
      f = f+df;
      while 0<f && f<=nf
        for iPt = 1:npt
        for j = 1:2
          if ~isnan(lpos(iPt,j,f))
            obj.setFrameProtected(f);
            return;
          end
        end
        end        
        f = f+df;
      end
    end
    
    function [x,y,th] = currentTargetLoc(obj)
      % Return current target loc, or movie center if no target
      
      assert(~obj.isMultiView,'Not supported for MultiView.');
      
      if obj.hasTrx
        cfrm = obj.currFrame;
        ctrx = obj.currTrx;

        if cfrm < ctrx.firstframe || cfrm > ctrx.endframe
          warningNoTrace('Labeler:target','No track for current target at frame %d.',cfrm);
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
                
    % TODO prob use listener/event for this; maintain relevant
    % datastructure in Labeler
    function updateTrxTable(obj)
      % based on .frm2trxm .currFrame, .labeledpos
      
      tbl = obj.gdata.tblTrx;
      if ~obj.hasTrx || ~obj.hasMovie || obj.currMovie==0 % Can occur during movieSet(), when invariants momentarily broken
        set(tbl,'Data',[]);
        return;
      end
      
      colnames = obj.TBLTRX_STATIC_COLSTRX(1:end-1);

      tfLive = obj.frm2trx(obj.currFrame,:);
      trxLive = obj.trx(tfLive);
      trxLive = trxLive(:);
      trxLive = rmfield(trxLive,setdiff(fieldnames(trxLive),colnames));
      trxLive = orderfields(trxLive,colnames);
      tbldat = struct2cell(trxLive)';
      
      iTrxLive = find(tfLive);
      tfLbled = false(size(iTrxLive(:)));
      lpos = obj.labeledpos{obj.currMovie};
      cfrm = obj.currFrame;
      for i = 1:numel(iTrxLive)
        tfLbled(i) = any(lpos(:,1,cfrm,iTrxLive(i)));
      end
      tbldat(:,end+1) = num2cell(tfLbled);
      
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
          dat(tfRow,2:3) = {nTgtsCurFrm nPtsCurFrm};
        else
          dat(end+1,:) = {cfrm nTgtsCurFrm nPtsCurFrm};
          [~,idx] = sort(cell2mat(dat(:,1)));
          dat = dat(idx,:);
        end
        set(tbl,'Data',dat);
      else
        if any(tfRow)
          assert(nnz(tfRow)==1);
          dat(tfRow,:) = [];
          set(tbl,'Data',dat);
        end
      end
            
      % dat should equal get(tbl,'Data')     
      if obj.hasMovie
        obj.gdata.labelTLInfo.setLabelsFrame();
        obj.movieFilesAllHaveLbls(obj.currMovie) = size(dat,1)>0;
      end
      
      tx = obj.gdata.txTotalFramesLabeled;
      nrow = size(dat,1);
      tx.String = num2str(nrow);
    end    
    function updateFrameTableComplete(obj)
      [nTgts,nPts] = obj.labelPosLabeledFramesStats();
      assert(isequal(nTgts>0,nPts>0));
      tfFrm = nTgts>0;
      iFrm = find(tfFrm);
      
      dat = [num2cell(iFrm) num2cell(nTgts(tfFrm)) num2cell(nPts(tfFrm)) ];
      tbl = obj.gdata.tblFrames;
      set(tbl,'Data',dat);

      if obj.hasMovie
        obj.gdata.labelTLInfo.setLabelsFrame(1:obj.nframes);
        obj.movieFilesAllHaveLbls(obj.currMovie) = size(dat,1)>0;
      end
      
      tx = obj.gdata.txTotalFramesLabeled;
      nrow = size(dat,1);
      tx.String = num2str(nrow);
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
      
      lpos = obj.labeledpos{iMov}(:,:,frm,iTgt);
      lpostag = obj.labeledpostag{iMov}(:,frm,iTgt);
      ipts = 1:obj.nPhysPoints;
      LabelCore.assignLabelCoordsStc(lpos(ipts,:),...
        obj.lblPrev_ptsH(ipts),obj.lblPrev_ptsTxtH(ipts));
      if ~all(cellfun(@isempty,lpostag(ipts)))
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
      obj.labels2VizUpdate();
    end
    
    function labels2SetCurrMovie(obj,lpos)
      iMov = obj.currMovie;
      assert(isequal(size(lpos),size(obj.labeledpos{iMov})));
      obj.labeledpos2{iMov} = lpos;
    end
    
    function labels2Clear(obj)
      for i=1:numel(obj.labeledpos2)
        obj.labeledpos2{i}(:) = nan;
      end
      obj.labels2VizUpdate();
    end
    
    function labels2ImportTrkPrompt(obj,iMovs)
      % See labelImportTrkPrompt() 
      
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmovies;
      end
      obj.labelImportTrkPromptGeneric(iMovs,'labels2ImportTrk');
    end
   
    function labels2ImportTrk(obj,iMovs,trkfiles)
      obj.labelImportTrkGeneric(iMovs,trkfiles,'labeledpos2',[],[]);
      obj.labels2VizUpdate();
      RC.saveprop('lastTrkFileImported',trkfiles{end});
    end
    
    function labels2ImportTrkCurrMov(obj)
      % Try to import default trk file for current movie into labels2. If
      % the file is not there, error.
      
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
      
      movfiles = obj.movieFilesAllFull(iMovs,1);
      rawname = obj.defaultTrkRawname();
      [tfok,trkfiles] = obj.getTrkFileNamesForExport(movfiles,rawname);
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
      ptsPlotInfo.HitTest = 'off';
      
      obj.genericInitLabelPointViz('labeledpos2_ptsH','labeledpos2_ptsTxtH',...
        obj.gdata.axes_curr,ptsPlotInfo);
      obj.genericInitLabelPointViz('lblOtherTgts_ptsH',[],...
        obj.gdata.axes_curr,ptsPlotInfo);
    end
    
    function labels2VizUpdate(obj)
      iMov = obj.currMovie;
      frm = obj.currFrame;
      iTgt = obj.currTarget;
      lpos = obj.labeledpos2{iMov}(:,:,frm,iTgt);
      LabelCore.setPtsCoords(lpos,obj.labeledpos2_ptsH,obj.labeledpos2_ptsTxtH);
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
    
    function labelsOtherTargetShowIDs(obj,tgtIDs)
      iTgts = obj.trxIdPlusPlus2Idx(tgtIDs+1);
      frm = obj.currFrame;
      iMov = obj.currMovie;
      lpos = squeeze(obj.labeledpos{iMov}(:,:,frm,iTgts)); % [npts x 2 x numel(iTgts)]

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
  
  %% Util
  methods
    
    function genericInitLabelPointViz(obj,hProp,hTxtProp,ax,plotIfo)
      deleteValidHandles(obj.(hProp));
      obj.(hProp) = gobjects(obj.nLabelPoints,1);
      if ~isempty(hTxtProp)
      deleteValidHandles(obj.(hTxtProp));
      obj.(hTxtProp) = gobjects(obj.nLabelPoints,1);
      end
      
      % any extra plotting parameters
      allowedPlotParams = {'HitTest'};
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
          'Color',plotIfo.Colors(i,:),'Hittest','off');
        end
      end      
    end
    
  end

end
