classdef Labeler < handle
% Labeler  Bransonlab Animal Video Labeler

  properties (Constant,Hidden)
    VERSION = '0.5';
    DEFAULT_LBLFILENAME = '%s.lbl';
    PREF_DEFAULT_FILENAME = 'pref.default.yaml';
    PREF_LOCAL_FILENAME = 'pref.yaml';
    
    SAVEPROPS = { ...
      'VERSION' ...
      'projname' 'projMacros' ...
      'movieFilesAll' 'movieInfoAll' 'trxFilesAll' ...
      'labeledpos' 'labeledpostag' 'labeledposTS' 'labeledpos2' ...      
      'currMovie' 'currFrame' 'currTarget' ...
      'labelMode' 'nLabelPoints' 'labelPointsPlotInfo' 'labelTemplate' ...
      'minv' 'maxv' 'movieForceGrayscale'...
      'suspScore'};
    LOADPROPS = {...
      'projname' 'projMacros' ...
      'movieFilesAll' 'movieInfoAll' 'trxFilesAll' ...
      'labeledpos' 'labeledpostag' 'labeledposTS' 'labeledpos2' ...
      'labelMode' 'nLabelPoints' 'labelTemplate' ...
      'minv' 'maxv' 'movieForceGrayscale' ...
      'suspScore'};
    
    TBLTRX_STATIC_COLSTBL = {'id' 'labeled'};
    TBLTRX_STATIC_COLSTRX = {'id' 'labeled'};
    
    TBLFRAMES_COLS = {'frame' 'tgts' 'pts'};
    
    NEIGHBORING_FRAME_MAXRADIUS = 10;
    NEIGHBORING_FRAME_OFFSETS = neighborIndices(Labeler.NEIGHBORING_FRAME_MAXRADIUS);
  end
  
  events
    newMovie
  end
  
  %% Project
  properties (SetObservable)
    projname              % 
    projFSInfo;           % ProjectFSInfo
  end
  properties (SetAccess=private)
    projMacros = struct(); % scalar struct containing user-defined macros
  end
  properties
    projPrefs;
  end
  properties (Dependent)
    projectfile;          % Full path to current project 
    projectroot;          % Parent dir of projectfile, if it exists
  end

  %% Movie/Video
  % Originally "Movie" referred to high-level data units/elements, eg
  % things added, removed, managed by the MovieManager etc; while "Video"
  % referred to visual details, display, playback, etc. But I sort of
  % forgot and mixed them up so that Movie sometimes applies to the latter.
  properties
    movieReader = []; % MovieReader object
    minv = 0;
    maxv = inf;
    movieFrameStepBig = 10;
    movieInfoAll = cell(0,1); % column cell-of-structs, same size as movieFilesAll
    movieDontAskRmMovieWithLabels = false; % If true, won't warn about removing-movies-with-labels    
  end
  properties (SetObservable)
    movieFilesAll = cell(0,1); % column cellstr, full paths to movies; can include macros 
    movieFilesAllHaveLbls = false(0,1); % [numel(movieFilesAll)x1] logical. 
        % How MFAHL is maintained
        % - At project load, it is updated fully.
        % - Trivial update on movieRm/movieAdd.
        % - Otherwise, all labeling operations can only affect the current
        % movie; meanwhile the FrameTable contains all necessary info to
        % update movieFilesAllHaveLbls. So we piggyback onto
        % updateFrameTable*(). 
    targetZoomFac;
    moviename; % short 'pretty' name, cosmetic purposes only
    movieCenterOnTarget = false; % scalar logical.
    movieForceGrayscale = false; % scalar logical
  end
  properties (Dependent)
    movieFilesAllFull; % like movieFilesAll, but macro-replaced and platformized
    hasMovie;
    moviefile;
    nframes;
    movienr;
    movienc;
    nmovies;
  end
  
  %% Trx
  properties (SetObservable)
    trxFilesAll = cell(0,1);  % column cellstr, full paths to trxs. Same size as movieFilesAll.
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
  properties (Dependent)
    hasTrx
    currTrx
    currTrxID
    nTrx
    nTargets % nTrx, or 1 if no Trx
  end  
  
  %% ShowTrx
  properties (SetObservable)
    showTrxMode;              % scalar ShowTrxMode
  end
  properties
    hTraj;                    % nTrx x 1 vector of line handles
    hTrx;                     % nTrx x 1 vector of line handles    
    showTrxPreNFrm = 15;      % number of preceding frames to show in traj
    showTrxPostNFrm = 5;      % number of following frames to show in traj
  end
  
  %% Labeling
  properties (SetObservable)
    labelMode;            % scalar LabelMode
  end
  properties (SetAccess=private)
    nLabelPoints;         % scalar integer
  end
  properties % make public setaccess
    labelPointsPlotInfo;  % struct containing cosmetic info for labelPoints        
  end
  properties (SetAccess=private)
    labelTemplate;    
    labeledpos;           % column cell vec with .nmovies elements. labeledpos{iMov} is npts x 2 x nFrm(iMov) x nTrx(iMov) double array; labeledpos{1}(:,1,:,:) is X-coord, labeledpos{1}(:,2,:,:) is Y-coord
    labeledposTS;         % labeledposTS{iMov} is nptsxnFrm(iMov)xnTrx(iMov). It is the last time .labeledpos or .labeledpostag was touched.
    labeledposMarked;     % labeledposMarked{iMov} is a nptsxnFrm(iMov)xnTrx(iMov) logical array. Elements are set to true when the corresponding pts have their labels set; users can set elements to false at random.
    labeledpostag;        % column cell vec with .nmovies elements. labeledpostag{iMov} is npts x nFrm(iMov) x nTrx(iMov) cell array
    
    labeledpos2;          % identical size/shape with labeledpos. aux labels (eg predicted, 2nd set, etc)
  end
  properties (SetObservable)
    labeledposNeedsSave;  % scalar logical, .labeledpos has been touched since last save. Currently does NOT account for labeledpostag
  end
  properties (Dependent)
    labeledposCurrMovie;
    labeledpostagCurrMovie;
  end
  properties
    lblCore;
    
    lblPrev_ptsH;         % TODO: encapsulate labelsPrev (eg in a LabelCore)
    lblPrev_ptsTxtH;
    
    labeledpos2_ptsH;
    labeledpos2_ptsTxtH;
  end 
  
  %% Suspiciousness
  properties (SetObservable)
    suspScore; % column cell vec same size as labeledpos. suspScore{iMov} is nFrm(iMov) x nTrx(iMov)
    suspNotes; % column cell vec same size as labeledpos. suspNotes{iMov} is a nFrm x nTrx column cellstr
    currSusp; % suspScore for current mov/frm/tgt. Can be [] indicating 'N/A'
  end
  
  %% Tracking
  properties (SetObservable)
    tracker % LabelTracker object
    trackNFramesSmall % small/fine frame increment for tracking
    trackNFramesLarge % big/coarse "
    trackNFramesNear % neighborhood radius
  end
  properties
    trackPrefs % Track preferences substruct
  end
  
  %% Misc
  properties (SetObservable, AbortSet)
    currMovie;            % idx into .movieFilesAll
    prevFrame = nan;      % last previously VISITED frame
    currTarget = nan;
    
    currImHud; % scalar AxisHUD object
  end
  properties (SetObservable)
    currFrame = 1; % current frame
  end
  properties
    currIm = [];
    prevIm = [];
    gdata = [];             % handles structure for figure
    depHandles = cell(0,1); % vector of handles that should be deleted when labeler is deleted
    
    isinit = false;         % scalar logical; true during initialization, when some invariants not respected
    
    selectedFrames = [];    % vector of frames currently selected frames; typically t0:t1
  end
  
  %% Prop access
  methods % dependent prop getters
    function v = get.movieFilesAllFull(obj)
      sMacro = obj.projMacros;
      if ~isfield(sMacro,'projroot')
        % This conditional allows user to explictly specify project root
        sMacro.projroot = obj.projectroot;
      end
      v = obj.movieFilesAll;
      v = cellfun(@(x)obj.projLocalizePath(x),v,'uni',0);
      Labeler.warnUnreplacedMacros(v);
    end
    function v = get.hasMovie(obj)
      v = obj.movieReader.isOpen;
    end    
    function v = get.moviefile(obj)
      mr = obj.movieReader;
      if isempty(mr)
        v = [];
      else
        v = mr.filename;
      end
    end
    function v = get.movienr(obj)
      mr = obj.movieReader;
      if mr.isOpen
        v = mr.nr;        
      else
        v = [];
      end
    end
    function v = get.movienc(obj)
      mr = obj.movieReader;
      if mr.isOpen
        v = mr.nc;        
      else
        v = [];
      end
    end    
    function v = get.nframes(obj)
      mr = obj.movieReader;
      if isempty(mr)
        v = nan;
      else
        v = obj.movieReader.nframes;
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
      v = numel(obj.movieFilesAll);
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
      obj.movieReader.forceGrayscale = v; %#ok<MCSUP>
      obj.movieForceGrayscale = v;
    end
  end
  
  %% Ctor/Dtor
  methods 
  
    function obj = Labeler(varargin)
      % lObj = Labeler();
     
      prefdeffile = fullfile(APT.Root,Labeler.PREF_DEFAULT_FILENAME);
      assert(exist(prefdeffile,'file')>0,...
        'Cannot find default preferences ''%s''.',prefdeffile);
      prefdef = ReadYaml(prefdeffile,[],1);
      fprintf(1,'Default preferences: %s\n',prefdeffile);

      preflocfile = fullfile(APT.Root,Labeler.PREF_LOCAL_FILENAME);
      if exist(preflocfile,'file')>0
        preflocal = ReadYaml(preflocfile,[],1);
        fprintf(1,'Found local preferences: %s\n',preflocfile);
        pref = structoverlay(prefdef,preflocal);
      else
        pref = prefdef;
      end      
      obj.initFromPrefs(pref);

      hFig = LabelerGUI(obj);
      obj.gdata = guidata(hFig);      
      obj.movieReader = MovieReader;  
      obj.currImHud = AxisHUD(obj.gdata.axes_curr);
      obj.movieSetNoMovie();
      
      for prop = obj.gdata.propsNeedInit(:)', prop=prop{1}; %#ok<FXSET>
        obj.(prop) = obj.(prop);
      end

      if obj.trackPrefs.Enable
        obj.tracker = feval(obj.trackPrefs.Type,obj);
        obj.tracker.init();
      end        
    end
    
    function initFromPrefs(obj,pref)
      obj.labelMode = LabelMode.(pref.LabelMode);
      obj.nLabelPoints = pref.NumLabelPoints;
      %obj.zoomRadiusDefault = pref.Trx.ZoomRadius;
      %obj.zoomRadiusTight = pref.Trx.ZoomRadiusTight;
      obj.targetZoomFac = pref.Trx.ZoomFactorDefault;
      obj.movieFrameStepBig = pref.Movie.FrameStepBig;
      
      % AL20160614 TODO: ideally get rid of labelPointsPlotInfo and just 
      % use .pref.LabelPointsPlot
      lpp = pref.LabelPointsPlot;
      if isfield(lpp,'ColorMapName') && ~isfield(lpp,'ColorMap')
        lpp.Colors = feval(lpp.ColorMapName,pref.NumLabelPoints);
      end
      obj.labelPointsPlotInfo = lpp;
            
      prfTrk = pref.Track; 
      obj.trackNFramesSmall = prfTrk.PredictFrameStep;
      obj.trackNFramesLarge = prfTrk.PredictFrameStepBig;
      obj.trackNFramesNear = prfTrk.PredictNeighborhood;
      obj.trackPrefs = prfTrk;      
      
      obj.projPrefs = pref; % redundant with some other props
    end
    
    function addDepHandle(obj,h)
      % GC dead handles
      tfValid = cellfun(@isvalid,obj.depHandles);
      obj.depHandles = obj.depHandles(tfValid);

      obj.depHandles{end+1,1} = h;
    end
    
    function delete(obj)
      tfValid = cellfun(@isvalid,obj.depHandles);
      hValid = obj.depHandles(tfValid);
      cellfun(@delete,hValid);
      obj.depHandles = cell(0,1);
    end
    
  end
  
  %% Project/Lbl files
  methods
    
    function projQuickOpen(obj,movfile,trxfile)
      % Create a new project; add the mov/trx; open the movie
      
      assert(exist(movfile,'file')>0);
      assert(isempty(trxfile) || exist(trxfile,'file')>0);
      
      [~,projName,~] = fileparts(movfile);
      obj.projNew(projName);      
      obj.movieAdd(movfile,trxfile);
      obj.movieSet(1);
    end
    
    function projNew(obj,name)
      if exist('name','var')==0
        resp = inputdlg('Project name:','New Project');
        if isempty(resp)
          return;
        end
        name = resp{1};
      end
      
      obj.projname = name;
      obj.projFSInfo = [];
      obj.movieFilesAll = cell(0,1);
      obj.movieFilesAllHaveLbls = false(0,1);
      obj.movieInfoAll = cell(0,1);
      obj.trxFilesAll = cell(0,1);
      obj.movieSetNoMovie(); % order important here
      obj.labeledpos = cell(0,1);
      obj.labeledposTS = cell(0,1);
      obj.labeledposMarked = cell(0,1);
      obj.labeledpostag = cell(0,1);
      obj.labeledpos2 = cell(0,1);
      obj.updateFrameTableComplete();  
      obj.labeledposNeedsSave = false;
      
      if ~isempty(obj.tracker)
        obj.tracker.init();
      end
    end
      
    function projSaveRaw(obj,fname)
      s = obj.projGetSaveStruct(); %#ok<NASGU>
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
      for f = obj.SAVEPROPS, f=f{1}; %#ok<FXSET>
        s.(f) = obj.(f);
      end
      
      switch obj.labelMode
        case LabelMode.TEMPLATE
          s.labelTemplate = obj.lblCore.getTemplate();
      end
      
      tObj = obj.tracker;
      if ~isempty(tObj)
        sTrk = tObj.getSaveToken();
        tObjCls = class(tObj);
        assert(~isfield(s,tObjCls));
        s.(tObjCls) = sTrk;
      end        
    end
    
    function projLoad(obj,fname)
      % Load a lbl file, along with moviefile and trxfile referenced therein
            
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
      for f = obj.LOADPROPS,f=f{1}; %#ok<FXSET>
        if isfield(s,f)
          obj.(f) = s.(f);          
        else
          warningNoTrace('Labeler:load','Missing load field ''%s''.',f);
          %obj.(f) = [];
        end
      end
      % labeledposMarked: special treatment. This is not serialized. All
      % labels are initted as unmarked
      obj.labeledposMarked = cellfun(@(x)false(size(x)),obj.labeledposTS,'uni',0);
      % labelPointsPlotInfo: special treatment. For old projects,
      % obj.labelPointsPlotInfo can have new/removed fields relative to
      % s.labelPointsPlotInfo. I guess by overlaying we are not removing
      % obsolete fields...
      obj.labelPointsPlotInfo = structoverlay(obj.labelPointsPlotInfo,...
        s.labelPointsPlotInfo);
      obj.movieFilesAllHaveLbls = cellfun(@(x)any(~isnan(x(:))),obj.labeledpos);
      obj.isinit = false;
      
      % need this before setting movie so that .projectroot exists
      obj.projFSInfo = ProjectFSInfo('loaded',fname);

      tObj = obj.tracker;
      % AL20160517 initialization hell unresolved
%       if ~isempty(tObj)
%         tObj.init(); % AL 20160508: needs to occur before .movieSet() below
%       end
      
      if obj.nmovies==0 || s.currMovie==0
        obj.movieSetNoMovie();
      else
        obj.movieSet(s.currMovie);
      end
      
      % AL20160517 initialization hell unresolved
      if ~isempty(tObj)
        tObj.init(); % AL 20160508: needs to occur before .movieSet() below
      end

      assert(isa(s.labelMode,'LabelMode'));      
      obj.labeledposNeedsSave = false;

      obj.setFrameAndTarget(s.currFrame,s.currTarget);
      obj.suspScore = obj.suspScore;
            
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI

      if ~isempty(tObj)        
        tObjCls = class(tObj);
        if isfield(s,tObjCls)
          fprintf(1,'Loading tracker info: %s.\n',tObjCls);
          trkTok = s.(tObjCls);
          tObj.loadSaveToken(trkTok);
        end
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
      
      if isfield(s,'projMacros') && ~isfield(s.projMacros,'projroot')
        s.projMacros.projroot = fileparts(fname);
      else
        s.projMacros = struct();
      end
      
      nMov = numel(s.movieFilesAll);
      for iMov = 1:nMov
        movfile = s.movieFilesAll{iMov};
        movfileFull = Labeler.platformize(Labeler.macroReplace(movfile,s.projMacros));
        movifo = s.movieInfoAll{iMov};
        trxfl = s.trxFilesAll{iMov};
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
        obj.movieFilesAllHaveLbls(end+1,1) = nnz(~isnan(lpos))>0;
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
      vals = struct2cell(s);
      
      resp = inputdlg(macros,'Project macros',1,vals);
      if ~isempty(resp)
        assert(isequal(numel(macros),numel(vals),numel(resp)));
        for i=1:numel(macros)
          try
            obj.projMacroSet(macros{i},resp{i});
          catch ME
            warningNoTrace('Labeler:macro','Cannot set macro ''%s'': %s',...
              macros{i},ME.message);
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
      tf = isfield(obj.projMacro,macro);
    end
    
    function p = projLocalizePath(obj,p)
      p = Labeler.platformizePath(Labeler.macroReplace(p,obj.projMacros));
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
    
    function str = macroReplace(str,sMacro)
      % sMacro: macro struct
      
      macros = fieldnames(sMacro);
      for i=1:numel(macros)
        mpat = ['\$' macros{i}];
        val = sMacro.(macros{i});
        val = regexprep(val,'\\','\\\\');
        str = regexprep(str,mpat,val);
      end
    end
    
    function str = platformizePath(str)
      % Depending on platform, replace / with \ or vice versa
      
      if ispc
        str = regexprep(str,'/','\');
      else
        str = regexprep(str,'\\','/');
      end
    end
    
    function tf = hasMacro(str)
      tf = ~isempty(regexp(str,'\$','once'));
    end
    
    function warnUnreplacedMacros(strs)
      toks = cellfun(@(x)regexp(x,'\$([a-zA-Z]+)','tokens'),strs,'uni',0);
      toks = [toks{:}];
      toks = [toks{:}];
      if ~isempty(toks)
        toks = unique(toks);
        cellfun(@(x)warningNoTrace('Labeler:macro','Unreplaced macro: $%s',x),toks);
      end
    end
    
    function errUnreplacedMacros(strs)
      strs = cellstr(strs);
      toks = cellfun(@(x)regexp(x,'\$([a-zA-Z]+)','tokens'),strs,'uni',0);
      toks = [toks{:}];
      toks = [toks{:}];
      if ~isempty(toks)
        tokstr = String.cellstr2CommaSepList(toks);
        error('Labeler:macro','Unreplaced macros: $%s',tokstr);
      end
    end
    
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
    end  
    
    function [I,p,md] = lblRead(lblFiles,varargin)
      % lblFiles: [N] cellstr
      % Optional PVs:
      %  - tfAllFrames. scalar logical, defaults to false. If true, read in
      %  unlabeled as well as labeled frames.
      % 
      % I: [Nx1] cell array of images (frames)
      % p: [NxD] positions
      % md: [Nxm] metadata table
      
      assert(false,'TODO: deal with movieFilesAll, macros etc.');

      assert(iscellstr(lblFiles));
      nLbls = numel(lblFiles);

      tfAllFrames = myparse(varargin,'tfAllFrames',false);
      if tfAllFrames
        readMovsLblsType = 'all';
      else
        readMovsLblsType = 'lbl';
      end
      I = cell(0,1);
      p = [];
      md = [];
      for iLbl = 1:nLbls
        lblName = lblFiles{iLbl};
        lbl = load(lblName,'-mat');
        fprintf('Lblfile: %s\n',lblName);
        
        movFiles = lbl.movieFilesAllFull; % TODO
        
        [ILbl,tMDLbl] = Labeler.lblCompileContents(movFiles,...
          lbl.labeledpos,lbl.labeledpostag,readMovsLblsType);
        pLbl = tMDLbl.p;
        tMDLbl(:,'p') = [];
        
        nrows = numel(ILbl);
        tMDLbl.lblFile = repmat({lblName},nrows,1);
        [~,lblNameS] = myfileparts(lblName);
        tMDLbl.lblFileS = repmat({lblNameS},nrows,1);
        
        I = [I;ILbl]; %#ok<AGROW>
        p = [p;pLbl]; %#ok<AGROW>
        md = [md;tMDLbl]; %#ok<AGROW>
      end
      
      assert(isequal(size(md,1),numel(I),size(p,1),size(bb,1)));
    end
    
    function [I,tbl] = lblCompileContents(movieNames,labeledposes,...
        labeledpostags,type,varargin)
      % convenience signature 
      %
      % type: either 'all' or 'lbl'

      nMov = numel(movieNames);      
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
    
    function [I,tbl] = lblCompileContentsRaw(...
        movieNames,lposes,lpostags,iMovs,frms,varargin)
      % Read moviefiles with landmark labels
      %
      % movieNames: [N] cellstr of movienames
      % lposes: [N] cell array of labeledpos arrays [nptsx2xnfrms]
      % lpostags: [N] cell array of labeledpostags [nptsxnfrms]      
      % iMovs. [M] indices into movieNames to read.
      % frms. [M] cell array. frms{i} is a vector of frames to read for
      % movie iMovs(i). frms{i} may also be:
      %     * 'all' indicating "all frames" 
      %     * 'lbl' indicating "all labeled frames" (currently includes partially-labeled)   
      %
      % I: [Ntrl] cell vec of images
      % tbl: [NTrl rows] labels/metadata table.
      %
      % Optional PVs:
      % - hWaitBar. Waitbar object
      % - noImg. logical scalar default false. If true, all elements of I
      % will be empty.
      % - lposTS. [N] cell array of labeledposTS arrays [nptsxnfrms]
      
      [hWB,noImg,lposTS] = myparse(varargin,...
        'hWaitBar',[],...
        'noImg',false,...
        'lposTS',[]);
      assert(numel(iMovs)==numel(frms));
      for i = 1:numel(frms)
        val = frms{i};
        assert(isnumeric(val) && isvector(val) || ismember(val,{'all' 'lbl'}));
      end
      
      tfWB = ~isempty(hWB);
      
      assert(iscellstr(movieNames));
      assert(iscell(lposes) && iscell(lpostags));
      assert(isequal(numel(movieNames),numel(lposes),numel(lpostags)));
      tfLposTS = ~isempty(lposTS);
      if tfLposTS
        cellfun(@(x,y)assert(size(x,1)==size(y,1) && size(x,2)==size(y,3)),...
          lposTS,lposes);
      end
      
      mr = MovieReader();

      I = [];
      s = struct('mov',cell(0,1),'movS',[],'frm',[],'p',[],'tfocc',[]);
      
      nMov = numel(iMovs);
      fprintf('Reading %d movies.\n',nMov);
      for i = 1:nMov
        iMov = iMovs(i);
        mov = movieNames{iMov};
        [~,movS] = myfileparts(mov);
        lpos = lposes{iMov}; % npts x 2 x nframes
        lpostag = lpostags{iMov};

        [npts,d,nFrmAll] = size(lpos);
        if isempty(lpos)
          assert(isempty(lpostag));
          lpostag = cell(npts,nFrmAll); % edge case: when lpos/lpostag are [], uninitted/degenerate case
        end
        assert(isequal(size(lpostag),[npts nFrmAll]));
        D = d*npts;
        
        mr.open(mov);
        
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
        
        ITmp = cell(nFrmRead,1);
        fprintf('  mov %d, D=%d, reading %d frames\n',iMov,D,nFrmRead);
        
        if tfWB
          hWB.Name = 'Reading movies';
          wbStr = sprintf('Reading movie %s',movS);
          waitbar(0,hWB,wbStr);          
        end
        for iFrm = 1:nFrmRead
          if tfWB
            waitbar(iFrm/nFrmRead,hWB);
          end
          
          f = frms2Read(iFrm);
          if noImg
            im = [];
          else
            im = mr.readframe(f);
            if size(im,3)==3 && isequal(im(:,:,1),im(:,:,2),im(:,:,3))
              im = rgb2gray(im);
            end
          end
          
          %fprintf('iMov=%d, read frame %d (%d/%d)\n',iMov,f,iFrm,nFrmRead);
          
          ITmp{iFrm} = im;
          lblsFrmXY = lpos(:,:,f);
          tags = lpostag(:,f);
          
          s(end+1,1).mov = mov; %#ok<AGROW>
          s(end).movS = movS;
          s(end).frm = f;
          s(end).p = Shape.xy2vec(lblsFrmXY);
          s(end).tfocc = strcmp('occ',tags(:)');
          if tfLposTS
            lts = lposTS{iMov};
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
      % moviefile: can have macros
      % trxfile: optional
      
      if exist('trxfile','var')==0
        trxfile = '';
      end
      
      movfilefull = obj.projLocalizePath(moviefile);
      
      assert(exist(movfilefull,'file')>0,'Cannot find file ''%s''.',movfilefull);
      assert(isempty(trxfile) || exist(trxfile,'file')>0,'Cannot find file ''%s''.',trxfile);
      
      mr = MovieReader();
      mr.open(movfilefull);
      ifo = struct();
      ifo.nframes = mr.nframes;
      ifo.info = mr.info;
      mr.close();
      
      if ~isempty(trxfile)
        tmp = load(trxfile);
        nTgt = numel(tmp.trx);
      else
        nTgt = 1;
      end
      
      obj.movieFilesAll{end+1,1} = moviefile;
      obj.movieFilesAllHaveLbls(end+1,1) = false;
      obj.movieInfoAll{end+1,1} = ifo;
      obj.trxFilesAll{end+1,1} = trxfile;
      obj.labeledpos{end+1,1} = nan(obj.nLabelPoints,2,ifo.nframes,nTgt);
      obj.labeledposTS{end+1,1} = -inf(obj.nLabelPoints,ifo.nframes,nTgt); 
      obj.labeledposMarked{end+1,1} = false(obj.nLabelPoints,ifo.nframes,nTgt); 
      obj.labeledpostag{end+1,1} = cell(obj.nLabelPoints,ifo.nframes,nTgt);      
      obj.labeledpos2{end+1,1} = nan(obj.nLabelPoints,2,ifo.nframes,nTgt);
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
        obj.movieFilesAll(iMov,:) = [];
        obj.movieFilesAllHaveLbls(iMov,:) = [];
        obj.movieInfoAll(iMov,:) = [];
        obj.trxFilesAll(iMov,:) = [];
        obj.labeledpos(iMov,:) = [];
        obj.labeledposTS(iMov,:) = [];
        obj.labeledposMarked(iMov,:) = [];
        obj.labeledpostag(iMov,:) = [];
        obj.labeledpos2(iMov,:) = [];        
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
    
    function movieSet(obj,iMov)
      assert(any(iMov==1:obj.nmovies),'Invalid movie index ''%d''.');
      
      % 1. Set the movie
      
      movfile = obj.movieFilesAll{iMov};
      movfileFull = obj.movieFilesAllFull{iMov};
      Labeler.errUnreplacedMacros(movfileFull);
      
      if exist(movfileFull,'file')==0
        if Labeler.hasMacro(movfile)
          qstr = sprintf('Cannot find movie ''%s'', macro-expanded to ''%s''.',...
            movfile,movfileFull);
          resp = questdlg(qstr,'Movie not found','Redefine macros','Browse to movie','Cancel','Cancel');
          if isempty(resp)
            resp = 'Cancel';
          end          
          switch resp
            case 'Redefine macros'
              obj.projMacroSetUI();
              movfileFull = obj.movieFilesAllFull{iMov};
              Labeler.errUnreplacedMacros(movfileFull);
              if exist(movfileFull,'file')==0
                error('Labeler:mov','Cannot find movie ''%s'', macro-expanded to ''%s''',...
                  movfile,movfileFull);
              end
            case 'Browse to movie'
              % none
            case 'Cancel'
              return;
          end
        end
                
        if exist(movfileFull,'file')==0
          % Either
          % i) no macro in moviename OR
          % ii) has macro but user selected browse to movie
          
          warningNoTrace('Labeler:mov',...
            'Cannot find movie ''%s''. Please browse to movie location.',...
            movfileFull);
          lastmov = RC.getprop('lbl_lastmovie');
          if isempty(lastmov)
            lastmov = pwd;
          end
          [newmovfile,newmovpath] = uigetfile('*.*','Select movie',lastmov);
          if isequal(newmovfile,0)
            error('Labeler:mov','Cannot find movie ''%s''.',movfileFull);
          end
          movfileFull = fullfile(newmovpath,newmovfile);
          obj.movieFilesAll{iMov} = movfileFull;
        end
      end        
      
      obj.movieReader.open(movfileFull);
      RC.saveprop('lbl_lastmovie',movfileFull);
      [path0,movname] = myfileparts(obj.moviefile);
      [~,parent] = fileparts(path0);
      obj.moviename = fullfile(parent,movname);
            
      obj.isinit = true; % Initialization hell, invariants momentarily broken
      obj.currMovie = iMov;
      obj.setFrameAndTarget(1,1);
      
      % 2. Set the trx
      
      trxFile = obj.trxFilesAll{iMov};
      tfTrx = ~isempty(trxFile);
      if tfTrx
        tmp = load(trxFile);
        obj.trxSet(tmp.trx);
        obj.videoSetTargetZoomFac(obj.targetZoomFac);        
      else
        obj.trxSet([]);
      end
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
      if isempty(obj.labeledposTS{iMov});
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
      
      obj.labelingInit();
      obj.labels2VizInit();
      
      notify(obj,'newMovie');
      
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
      % Set to iMov==0

      obj.currMovie = 0;
      
      obj.movieReader.close();
      obj.moviename = '';
      obj.trxfile = '';
      obj.trx = [];
      obj.frm2trx = [];
      obj.trxIdPlusPlus2Idx = [];

      obj.currFrame = 1;
      imcurr = obj.gdata.image_curr;
      set(imcurr,'CData',0);
      imprev = obj.gdata.image_prev;
      set(imprev,'CData',0);
      
      obj.initShowTrx();
      
      obj.currTarget = 0;
      obj.currSusp = [];
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
        assert(isequal(numel(trx),obj.nTargets));

        % TODO: check for labels that are "out of bounds" for new trx
      end
      
      obj.trx = trx;
                  
      f2t = false(obj.nframes,obj.nTrx);
      if obj.hasTrx
        if ~isfield(obj.trx,'id'),
          for i = 1:numel(obj.trx),
            obj.trx(i).id = i;
          end
        end
        maxID = max([obj.trx.id]);
      else
        maxID = -1;
      end
      id2t = nan(maxID+1,1);
      for i = 1:obj.nTrx
        frm0 = obj.trx(i).firstframe;
        frm1 = obj.trx(i).endframe;
        f2t(frm0:frm1,i) = true;
        id2t(obj.trx(i).id+1) = i;
      end
      obj.frm2trx = f2t;
      obj.trxIdPlusPlus2Idx = id2t;
      
      obj.currImHud.updateReadoutFields('hasTgt',obj.hasTrx);
      obj.initShowTrx();
    end
    
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
   
    function trxCheckFramesLive(obj,frms)
      % Check that current target is live for given frames; err if not
      
      iTgt = obj.currTarget;
      if obj.hasTrx
        tfLive = obj.frm2trx(frms,iTgt);
        if ~all(tfLive)
          error('Labeler:labelpos',...
            'Target %d is not live during all desired frames.',iTgt);
        end
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
        delete(lc);
        obj.lblCore = [];
      end
      obj.lblCore = LabelCore.create(obj,lblmode);      
      obj.lblCore.init(nPts,lblPtsPlotInfo);
      if lblmode==LabelMode.TEMPLATE && ~isempty(template)
        obj.lblCore.setTemplate(template);
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
      % iMov: movie index (into .movieFilesAll)
      %
      % tf: [nframes-for-iMov], true if any point labeled in that mov/frame

      ifo = obj.movieInfoAll{iMov};
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
    
    function labelPosSetFramesI(obj,frms,xy,iPt)
      % Set labelpos for current movie/target to a single (constant) point
      % across multiple frames
      %
      % frms: vector of frames
      
      assert(isvector(frms));
      assert(numel(xy)==2 && ~any(isnan(xy(:))));
      assert(isscalar(iPt));
      
      obj.trxCheckFramesLive(frms);

      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      
      obj.labeledpos{iMov}(iPt,1,frms,iTgt) = xy(1);
      obj.labeledpos{iMov}(iPt,2,frms,iTgt) = xy(2);
      obj.updateFrameTableComplete(); % above sets mutate .labeledpos{obj.currMovie} in more than just .currFrame
      
      obj.labeledposTS{iMov}(iPt,frms,iTgt) = now();
      obj.labeledposMarked{iMov}(iPt,frms,iTgt) = true;

      obj.labeledposNeedsSave = true;
    end
    
    function labelPosBulkImport(obj,xy)
      % Set ALL labels for current movie/target
      %
      % xy: [nptx2xnfrm]
      
      iMov = obj.currMovie;      
      lposOld = obj.labeledpos{iMov};      
      assert(isequal(size(xy),size(lposOld)));      
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
      nfrmsMov = obj.movieInfoAll{iMov}.nframes;
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
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledposTS{iMov}(iPt,iFrm,iTgt) = now();
      obj.labeledpostag{iMov}{iPt,iFrm,iTgt} = tag;      
    end
    
    function labelPosTagClearI(obj,iPt)
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledpostag{iMov}{iPt,iFrm,iTgt} = [];
    end
    
    function labelPosTagSetFramesI(obj,tag,iPt,frms)
      % Set tags for current movie/target, given pt/frames

      obj.trxCheckFramesLive(frms);
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
      for dFrm = 0:obj.NEIGHBORING_FRAME_OFFSETS 
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
    
    % trk files, label export
    
    function s = labelCreateTrkContents(obj,iMov)
      % Create .trk contents from .labeledpos, .labeledpostag etc
      %
      % iMov: scalar movie index
      %
      % s: scalar struct
      
      assert(isscalar(iMov));      
      s = struct();
      s.pTrk = obj.labeledpos{iMov};
      s.pTrkTS = obj.labeledposTS{iMov};
      s.pTrkTag = obj.labeledpostag{iMov};
      s.pTrkiPt = 1:size(s.pTrk,1);
    end
  end
  
  methods (Static)
    function trkfile = defaultTrkFileName(movfile)
      [p,movS] = fileparts(movfile);
      trkfile = fullfile(p,[movS '.trk']);
    end    
    function [tfok,trkfiles] = getTrkFileNames(movfiles)
      % Generate trkfile names for movfiles. If trkfiles exist, ask before
      % overwriting etc.
      %
      % movfiles: cellstr of movieFilesAllFull
      %
      % tfcontinue: if true, trkfiles is valid, go ahead and write to
      % those. (This may involve overwrites but user OKed it)
      % trkfiles: cellstr, same size as movfiles. .trk filenames
      % corresponding to movfiles
      
      [movpaths,movS] = cellfun(@fileparts,movfiles,'uni',0);
      trkfiles = cellfun(@(x,y)fullfile(x,[y '.trk']),movpaths,movS,'uni',0);
      tfok = true;      
      tfexist = cellfun(@(x)exist(x,'file')>0,trkfiles);
      if any(tfexist)
        iExist = find(tfexist,1);
        queststr = sprintf('One or more .trk files already exist, eg: %s.',trkfiles{iExist});
        btn = questdlg(queststr,'Files exist','Overwrite','Add datetime to filenames',...
          'Cancel','Add datetime to filenames');
        if isempty(btn)
          btn = 'Cancel';
        end
        switch btn
          case 'Overwrite'
            % none; use trkfiles as-is
          case 'Add datetime to filenames'
            nowstr = datestr(now,'yyyymmddTHHMMSS');
            trkfiles = cellfun(@(x,y)fullfile(x,[y '.' nowstr '.trk']),movpaths,movS,'uni',0);
          otherwise
            tfok = false;
            trkfiles = [];
        end
      end
    end
  end
  
  methods
    function labelExportTrk(obj,iMov)
      % Export label data to trk files.
      %
      % iMov: optional, indices into .movieFilesAll to export. Defaults to 1:obj.nmovies.
      
      if exist('iMov','var')==0
        iMov = 1:obj.nmovies;
      end
      
      movfiles = obj.movieFilesAllFull(iMov);
      [tfok,trkfiles] = Labeler.getTrkFileNames(movfiles);
      if tfok
        nMov = numel(iMov);
        for i=1:nMov
          s = obj.labelCreateTrkContents(iMov(i)); %#ok<NASGU>
          save(trkfiles{i},'-mat','-struct','s');
        end
      end
    end
    
    function labelImportTrkGeneric(obj,iMovs,trkfiles,lposFld,lposTSFld,lposTagFld)
      % iMovs: [N] vector of movie indices
      % trkfiles: [N] cellstr of trk filenames
      % lpos*Fld: property names for labeledpos, labeledposTS,
      % labeledposTag. Can be empty to not set that prop.
      
      nMov = numel(iMovs);
      assert(nMov==numel(trkfiles));
      
      for i=1:nMov
        iM = iMovs(i);
        s = load(trkfiles{i},'-mat');
        fprintf(1,'Loaded trk file: %s\n',trkfiles{i});
        
        lpos = nan(size(obj.labeledpos{iM}));
        lposTS = -inf(size(obj.labeledposTS{iM}));
        lpostag = cell(size(obj.labeledpostag{iM}));
        if isfield(s,'pTrkiPt')
          iPt = s.pTrkiPt;
        else
          iPt = 1:size(lpos,1); % all pts
        end
        lpos(iPt,:,:,:) = s.pTrk;
        lposTS(iPt,:,:) = s.pTrkTS;
        lpostag(iPt,:,:) = s.pTrkTag;
        
        obj.(lposFld){iM} = lpos;
        if ~isempty(lposTSFld)
          obj.(lposTSFld){iM} = lposTS;
        end
        if ~isempty(lposTagFld)
          obj.(lposTagFld){iM} = lpostag;
        end
      end      
    end
    
    function labelImportTrk(obj,iMovs,trkfiles)
      % Import label data from trk files.
      %
      % iMovs: [nMovie]. Optional, movie indices for which to import.
      %   Defaults to 1:obj.nmovies.
      % trkfiles: [nMovie] cellstr. Optional, full filenames to trk files
      %   corresponding to iMov. Defaults to <movpath>/<movname>.trk.
      
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmovies;
      end      
      if exist('trkfiles','var')==0
        movfiles = obj.movieFilesAllFull(iMovs);
        trkfiles = cellfun(@Labeler.defaultTrkFileName,movfiles,'uni',0);
      end
      
      obj.labelImportTrkGeneric(iMovs,trkfiles,'labeledpos',...
        'labeledposTS','labeledpostag');
      
      obj.updateFrameTableComplete();
      %obj.labeledposNeedsSave = true; AL 20160609: don't touch this for
      %now, since what we are importing is already in the .trk file.
      obj.labelsUpdateNewFrame(true);
    end
    
    function labelImportTrkCurrMov(obj)
      % Try to import default trk file for current movie into labeledpos. If
      % the file is not there, error.
      
      if ~obj.hasMovie
        error('Labeler:nomov','No movie is loaded.');
      end
      obj.labelImportTrk(obj.currMovie);
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
      obj.labelsPrevUpdate();
      obj.labels2VizUpdate();
    end
    
    function labelsUpdateNewTarget(obj,prevTarget)
      if ~isempty(obj.lblCore)
        obj.lblCore.newTarget(prevTarget,obj.currTarget,obj.currFrame);
      end
      obj.labelsPrevUpdate();
      obj.labels2VizUpdate();
    end
    
    function labelsUpdateNewFrameAndTarget(obj,prevFrm,prevTgt)
      if ~isempty(obj.lblCore)
        obj.lblCore.newFrameAndTarget(...
          prevFrm,obj.currFrame,...
          prevTgt,obj.currTarget);
      end
      obj.labelsPrevUpdate();
      obj.labels2VizUpdate();
    end
    
    % CONSIDER: encapsulating labelsPrev (eg in a LabelCore)
    function labelsPrevUpdate(obj)
      persistent tfWarningThrownAlready

      if obj.isinit
        return;
      end
      if ~isnan(obj.prevFrame) && ~isempty(obj.lblPrev_ptsH)
        iMov = obj.currMovie;
        frm = obj.prevFrame;
        iTgt = obj.currTarget;
        
        lpos = obj.labeledpos{iMov}(:,:,frm,iTgt);
        obj.lblCore.assignLabelCoords(lpos,...
          'hPts',obj.lblPrev_ptsH,...
          'hPtsTxt',obj.lblPrev_ptsTxtH);
        
        lpostag = obj.labeledpostag{iMov}(:,frm,iTgt);
        if ~all(cellfun(@isempty,lpostag))
          if isempty(tfWarningThrownAlready)
            warningNoTrace('Labeler:labelsPrev','TODO: label tags in previous frame not visualized.');
            tfWarningThrownAlready = true;
          end
        end
      else
        LabelCore.setPtsOffaxis(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
      end
    end
    
  end
   
  %% Susp 
  methods
    
    function setSuspScore(obj,ss)
      if isequal(ss,[])
        % none; this is ok
      else
        nMov = obj.nmovies;
        nTgt = obj.nTargets;
        assert(iscell(ss) && isvector(ss) && numel(ss)==nMov);
        for iMov = 1:nMov
          ifo = obj.movieInfoAll{iMov};        
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
    
    function setTracker(obj,tObj)
      obj.tracker = tObj;
    end
    
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
    
    function track(obj,tm)
      % tm: a TrackMode
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end      
      [iMovs,frms] = tm.getMovsFramesToTrack(obj);
      tObj.track(iMovs,frms);      
    end
    
    function trackSaveResults(obj,fname)
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      s = tObj.getSaveToken(); %#ok<NASGU>
      
      save(fname,'-mat','-struct','s');
      obj.projFSInfo = ProjectFSInfo('tracking results saved',fname);
      RC.saveprop('lastTrackingResultsFile',fname);
    end
    
    function trackLoadResults(obj,fname)
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      s = load(fname);
      tObj.loadSaveToken(s);
      
      obj.projFSInfo = ProjectFSInfo('tracking results loaded',fname);
      RC.saveprop('lastTrackingResultsFile',fname);
    end
          
    function [success,fname] = trackSaveResultsAs(obj)
      [success,fname] = obj.trackSaveLoadAsHelper('lastTrackingResultsFile',...
        'uiputfile','Save tracking results','trackSaveResults');
    end
    
    function [success,fname] = trackLoadResultsAs(obj)
      [success,fname] = obj.trackSaveLoadAsHelper('lastTrackingResultsFile',...
        'uigetfile','Load tracking results','trackLoadResults');
    end
    
    function [success,fname] = trackSaveLoadAsHelper(obj,rcprop,uifcn,...
        promptstr,rawMeth)
      % rcprop: Name of RC property for guessing path
      % uifcn: either 'uiputfile' or 'uigetfile'
      % promptstr: used in uiputfile
      % rawMeth: track*Raw method to call when a file is specified
      
      % Guess a path/location for save/load
      lastFile = RC.getprop(rcprop);
      if isempty(lastFile)
        projFile = obj.projectfile;
        if ~isempty(projFile)
          savepath = fileparts(projFile);
        else
          savepath = pwd;
        end
      else
        savepath = fileparts(lastFile);
      end
      
      filterspec = fullfile(savepath,'*.mat');
      [fname,pth] = feval(uifcn,filterspec,promptstr);
      if isequal(fname,0)
        fname = [];
        success = false;
      else
        fname = fullfile(pth,fname);
        success = true;
        obj.(rawMeth)(fname);
      end
    end
    
  end
   
  %% Video
  methods
    
    function videoResetView(obj)
      axis(obj.gdata.axes_curr,'auto','image');
      %axis(obj.gdata.axes_prev,'auto','image');
    end
    
    function videoCenterOnCurrTarget(obj)
      [x0,y0] = obj.videoCurrentCenter;
      [x,y] = obj.currentTargetLoc();
      
      dx = x-x0;
      dy = y-y0;
      axisshift(obj.gdata.axes_curr,dx,dy);
      %axisshift(obj.gdata.axes_prev,dx,dy);
    end
    
    function videoZoom(obj,zoomRadius)
      % Zoom to square window over current frame center with given radius.
      
      [x0,y0] = obj.videoCurrentCenter();      
      lims = [x0-zoomRadius,x0+zoomRadius,y0-zoomRadius,y0+zoomRadius];
      axis(obj.gdata.axes_curr,lims);
      axis(obj.gdata.axes_prev,lims);      
    end
    function videoSetTargetZoomFac(obj,zoomFac)
      % zoomFac: 0 for no-zoom; 1 for max zoom
      
      if zoomFac < 0
        zoomFac = 0;
        warning('Labeler:zoomFac','Zoom factor must be in [0,1].');
      end
      if zoomFac > 1
        zoomFac = 1;
        warning('Labeler:zoomFac','Zoom factor must be in [0,1].');
      end
      
      obj.targetZoomFac = zoomFac;
        
      zr0 = max(obj.movienr,obj.movienc)/2; % no-zoom: large radius
      zr1 = obj.projPref.Trx.ZoomRadiusTight; % tight zoom: small radius
      
      if zr1>zr0
        zr = zr0;
      else
        zr = zr0 + zoomFac*(zr1-zr0);
      end
      obj.videoZoom(zr);
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
    
    function videoSetContrastFromAxesCurr(obj)
      % Get video contrast from axes_curr and record/set
      clim = get(obj.gdata.axes_curr,'CLim');
      if isempty(clim) 
        % none; can occur when Labeler is closed
      else
        set(obj.gdata.axes_prev,'CLim',clim);
        obj.minv = clim(1);
        obj.maxv = clim(2);
      end
    end
    
    function videoApplyGammaGrayscale(obj,gamma)
      % Applies gamma-corrected grayscale colormap
      
      validateattributes(gamma,{'numeric'},{'scalar' 'real' 'positive'});
      
      im = obj.gdata.image_curr;
      if size(im.CData,3)~=1
        error('Labeler:gamma','Gamma correction currently only supported for grayscale/intensity images.');
      end
      
      m0 = gray(256);
      m1 = imadjust(m0,[],[],gamma);
      colormap(obj.gdata.axes_curr,m1);
      colormap(obj.gdata.axes_prev,m1);
    end
    
    function videoFlipUD(obj)
      % flip entire axes; movies + labels
      
      gd = obj.gdata;
      gd.axes_curr.YDir = toggleAxisDir(gd.axes_curr.YDir);
      gd.axes_prev.YDir = toggleAxisDir(gd.axes_prev.YDir);
    end
    function videoFlipLR(obj)
      % flip entire axes; movies + labels
      
      gd = obj.gdata;
      gd.axes_curr.XDir = toggleAxisDir(gd.axes_curr.XDir);
      gd.axes_prev.XDir = toggleAxisDir(gd.axes_prev.XDir);
    end
    
    function videoFlipUDVidOnly(obj)
      obj.movieReader.flipVert = ~obj.movieReader.flipVert;
      if obj.hasMovie
        obj.setFrame(obj.currFrame,true);
      end
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
      
      if isempty(obj.showTrxMode)
        obj.showTrxMode = ShowTrxMode.ALL;
      end
      onoff = onIff(obj.hasTrx);
      obj.gdata.menu_setup_trajectories.Enable = onoff;
    end
    
    function setShowTrxMode(obj,mode)
      assert(isa(mode,'ShowTrxMode'));      
      obj.showTrxMode = mode;
      obj.updateShowTrx();
    end
    
    function updateShowTrx(obj)
      % Update .hTrx, .hTraj based on .trx, .tfShowTrx, .currFrame
      
      if ~obj.hasTrx
        return;
      end
      
      t = obj.currFrame;
      trxAll = obj.trx;
      nPre = obj.showTrxPreNFrm;
      nPst = obj.showTrxPostNFrm;
      pref = obj.projPrefs.Trx;
      
      switch obj.showTrxMode
        case ShowTrxMode.NONE
          tfShow = false(obj.nTrx,1);
        case ShowTrxMode.CURRENT
          tfShow = false(obj.nTrx,1);
          tfShow(obj.currTarget) = true;
        case ShowTrxMode.ALL
          tfShow = true(obj.nTrx,1);
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
  
    function setFrame(obj,frm,tfforcereadmovie)
      % Set movie frame, maintaining current movie/target.
      
      if nargin<3
        tfforcereadmovie = false;
      end
            
      if obj.hasTrx
        tfTargetLive = obj.frm2trx(frm,:);      
        if ~tfTargetLive(obj.currTarget)
          iTgt = find(tfTargetLive,1);
          if isempty(iTgt)
            error('Labeler:noTarget','No live targets in frame %d.',frm);
          end

          warningNoTrace('Labeler:targetNotLive',...
            'Current target idx=%d is not live in frame %d. Switching to target idx=%d.',...
            obj.currTarget,frm,iTgt);
          obj.setFrameAndTarget(frm,iTgt);
          return;
        end
      end
      
      % Remainder nearly identical to setFrameAndTarget()
      
      obj.prevFrame = obj.currFrame;
      obj.prevIm = obj.currIm;
      set(obj.gdata.image_prev,'CData',obj.prevIm);
      
      if obj.currFrame~=frm || tfforcereadmovie
        obj.currIm = obj.movieReader.readframe(frm);
        obj.currFrame = frm;
      end            
      set(obj.gdata.image_curr,'CData',obj.currIm);
      
      if obj.hasTrx && obj.movieCenterOnTarget
        obj.videoCenterOnCurrTarget();
      end
      obj.labelsUpdateNewFrame();
      obj.updateTrxTable();
      obj.updateCurrSusp();
      obj.updateShowTrx();
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
      
      validateattributes(iTgt,{'numeric'},{'positive' 'integer' '<=' obj.nTargets});
      
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

      obj.prevFrame = obj.currFrame;
      obj.prevIm = obj.currIm;
      set(obj.gdata.image_prev,'CData',obj.prevIm);
     
      obj.currIm = obj.movieReader.readframe(frm);
      obj.currFrame = frm;
      set(obj.gdata.image_curr,'CData',obj.currIm);
      
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
    
    function frameUpDF(obj,df)
      f = min(obj.currFrame+df,obj.nframes);
      obj.setFrame(f); 
    end
    
    function frameDownDF(obj,df)
      f = max(obj.currFrame-df,1);
      obj.setFrame(f);
    end
    
    function frameUp(obj,tfBigstep)
      if tfBigstep
        df = obj.movieFrameStepBig;
      else
        df = 1;
      end
      obj.frameUpDF(df);
    end
    
    function frameDown(obj,tfBigstep)
      if tfBigstep
        df = obj.movieFrameStepBig;
      else
        df = 1;
      end
      obj.frameDownDF(df);
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
            obj.setFrame(f);
            return;
          end
        end
        end        
        f = f+df;
      end
    end
    
    function [x,y,th] = currentTargetLoc(obj)
      % Return current target loc, or movie center if no target
      
      if obj.hasTrx
        cfrm = obj.currFrame;
        ctrx = obj.currTrx;

        if cfrm < ctrx.firstframe || cfrm > ctrx.endframe
          warning('Labeler:target','No track for current target at frame %d.',cfrm);
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
      if ~obj.hasMovie
        error('Labeler:noMovie',...
          'Cannot set selected frames when no movie is loaded.');
      end
      validateattributes(frms,{'numeric'},{'integer' 'vector' '>=' 1 '<=' obj.nframes});
      obj.selectedFrames = frms;
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
        obj.gdata.labelTLManual.setLabelsFrame();
        obj.movieFilesAllHaveLbls(obj.currMovie) = size(dat,1)>0;
      end
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
        obj.movieFilesAllHaveLbls(obj.currMovie) = size(dat,1)>0;
      end
    end
   
  end
  
  %% Labels2
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
    
    function s = labels2CreateTrkContents(obj,iMov)
      % Create .trk contents from .labeledpos2
      %
      % iMov: scalar movie index
      %
      % s: scalar struct
      
      assert(isscalar(iMov));      
      s = struct();
      s.pTrk = obj.labeledpos2{iMov};
      [npts,~,nfrm,ntgt] = size(s.pTrk);
      s.pTrkTS = -inf(npts,nfrm,ntgt);
      s.pTrkTag = cell(npts,nfrm,ntgt);
      s.pTrkiPt = 1:size(s.pTrk,1);
    end
    
    function labels2ImportTrk(obj,iMovs,trkfiles)
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmovies;
      end      
      if exist('trkfiles','var')==0
        movfiles = obj.movieFilesAllFull(iMovs);
        trkfiles = cellfun(@Labeler.defaultTrkFileName,movfiles,'uni',0);
      end

      obj.labelImportTrkGeneric(iMovs,trkfiles,'labeledpos2',[],[]);      
      obj.labels2VizUpdate();
    end
    
    function labels2ImportTrkCurrMov(obj)
      % Try to import default trk file for current movie into labels2. If
      % the file is not there, error.
      
      if ~obj.hasMovie
        error('Labeler:nomov','No movie is loaded.');
      end
      obj.labels2ImportTrk(obj.currMovie);
    end
    
    function labels2ExportTrk(obj,iMov)
      % Export label data to trk files.
      %
      % iMov: optional, indices into .movieFilesAll to export. Defaults to 1:obj.nmovies.
      
      if exist('iMov','var')==0
        iMov = 1:obj.nmovies;
      end
      
      movfiles = obj.movieFilesAllFull(iMov);
      [tfok,trkfiles] = Labeler.getTrkFileNames(movfiles);
      if tfok
        nMov = numel(iMov);
        for i=1:nMov
          s = obj.labels2CreateTrkContents(iMov(i)); %#ok<NASGU>
          save(trkfiles{i},'-mat','-struct','s');
        end
      end
      
    end
    
    function labels2VizInit(obj)
      % Initialize view stuff for labels2  
      
      if ~isempty(obj.trackPrefs)
        ptsPlotInfo = obj.trackPrefs.PredictPointsPlot;
        ptsPlotInfo.Colors = obj.labelPointsPlotInfo.Colors;
      else
        ptsPlotInfo = obj.labelPointsPlotInfo;
      end
      
      obj.genericInitLabelPointViz('labeledpos2_ptsH','labeledpos2_ptsTxtH',...
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
    end
    
    function labels2VizHide(obj)
      [obj.labeledpos2_ptsH.Visible] = deal('off');
      [obj.labeledpos2_ptsTxtH.Visible] = deal('off');
    end
     
  end
  
  %% Util
  methods
    
    function genericInitLabelPointViz(obj,hProp,hTxtProp,ax,plotIfo)
      deleteValidHandles(obj.(hProp));
      deleteValidHandles(obj.(hTxtProp));
      obj.(hProp) = gobjects(obj.nLabelPoints,1);
      obj.(hTxtProp) = gobjects(obj.nLabelPoints,1);
      for i = 1:obj.nLabelPoints
        obj.(hProp)(i) = plot(ax,nan,nan,plotIfo.Marker,...
          'MarkerSize',plotIfo.MarkerSize,...
          'LineWidth',plotIfo.LineWidth,...
          'Color',plotIfo.Colors(i,:),...
          'UserData',i);
        obj.(hTxtProp)(i) = text(nan,nan,num2str(i),'Parent',ax,...
          'Color',plotIfo.Colors(i,:),'Hittest','off');
      end      
    end
    
  end

end
