classdef Labeler < handle
% Labeler  Bransonlab Animal Video Labeler

  properties (Constant,Hidden)
    VERSION = '0.2';
    DEFAULT_LBLFILENAME = '%s.lbl';
    PREF_DEFAULT_FILENAME = 'pref.default.yaml';
    PREF_LOCAL_FILENAME = 'pref.yaml';

    SAVEPROPS = { ...
      'VERSION' ...
      'projname' ...
      'movieFilesAll' 'movieInfoAll' 'trxFilesAll' 'labeledpos' 'labeledpostag' ...      
      'currMovie' 'currFrame' 'currTarget' ...
      'labelMode' 'nLabelPoints' 'labelPointsPlotInfo' 'labelTemplate' ...
      'minv' 'maxv' 'movieForceGrayscale'...
      'suspScore'};
    LOADPROPS = {...
      'projname' ...
      'movieFilesAll' 'movieInfoAll' 'trxFilesAll' 'labeledpos' 'labeledpostag' ...
      'labelMode' 'nLabelPoints' 'labelTemplate' ...
      'minv' 'maxv' 'movieForceGrayscale' ...
      'suspScore'};
    
    TBLTRX_STATIC_COLSTBL = {'id' 'labeled'};
    TBLTRX_STATIC_COLSTRX = {'id' 'labeled'};
    
    TBLFRAMES_COLS = {'frame' 'tgts' 'pts'};
    
    NEIGHBORING_FRAME_MAXRADIUS = 10;
    NEIGHBORING_FRAME_OFFSETS = neighborIndices(Labeler.NEIGHBORING_FRAME_MAXRADIUS);
  end
  
  %% Project
  properties (SetObservable)
    projname              % 
    projFSInfo;           % ProjectFSInfo
  end
  properties (Dependent)
    projectfile;          % Full path to current project 
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
    movieFilesAll = cell(0,1); % column cellstr, full paths to movies
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
    zoomRadiusDefault = 100;  % default zoom box size in pixels
    zoomRadiusTight = 10;     % zoom size on maximum zoom (smallest pixel val)
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
    trxPrefs;                 % struct, 'Trx' section of prefs
  end
  
  %% Labeling
  properties (SetObservable)
    labelMode;            % scalar LabelMode
  end
  properties (SetAccess=private)
    %labels = cell(0,1);  % cell vector with nTarget els. labels{iTarget} is nModelPts x 2 x "numFramesTarget"
    nLabelPoints;         % scalar integer
    labelPointsPlotInfo;  % struct containing cosmetic info for labelPoints        
    labelTemplate;    
    labeledpos;           % column cell vec with .nmovies elements. labeledpos{iMov} is npts x 2 x nFrm(iMov) x nTrx(iMov) double array; labeledpos{1}(:,1,:,:) is X-coord, labeledpos{1}(:,2,:,:) is Y-coord
    labeledpostag;        % column cell vec with .nmovies elements. labeledpostag{iMov} is npts x nFrm(iMov) x nTrx(iMov) cell array
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
  end 
  
  %% Suspiciousness
  properties (SetObservable)
    suspScore; % column cell vec same size as labeledpos. suspScore{iMov} is nFrm(iMov) x nTrx(iMov)
    suspNotes; % column cell vec same size as labeledpos. suspNotes{iMov} is a nFrm x nTrx column cellstr
    currSusp; % suspScore for current mov/frm/tgt. Can be [] indicating 'N/A'
  end
  
  %% Tracking
  properties (SetObservable)
    tracker
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
    
    isinit;                 % scalar logical; true during initialization, when some invariants not respected
  end
  
  %% Prop access
  methods % dependent prop getters
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
    function set.labeledpos(obj,v)
      obj.labeledpos = v;
      if ~obj.isinit %#ok<MCSUP>
        obj.updateTrxTable();
        obj.updateFrameTableIncremental(); % TODO use listener/event for this
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
    end
    
    function initFromPrefs(obj,pref)
      obj.labelMode = LabelMode.(pref.LabelMode);
      obj.nLabelPoints = pref.NumLabelPoints;
      obj.zoomRadiusDefault = pref.Trx.ZoomRadius;
      obj.zoomRadiusTight = pref.Trx.ZoomRadiusTight;
      obj.targetZoomFac = pref.Trx.ZoomFactorDefault;
      obj.movieFrameStepBig = pref.Movie.FrameStepBig;
      lpp = pref.LabelPointsPlot;
      if isfield(lpp,'ColorMapName') && ~isfield(lpp,'ColorMap')
        lpp.Colors = feval(lpp.ColorMapName,pref.NumLabelPoints);
      end
      obj.labelPointsPlotInfo = lpp;
      obj.trxPrefs = pref.Trx;
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
  
  %% Project
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
      obj.labeledpostag = cell(0,1);
      obj.updateFrameTableComplete();  
      obj.labeledposNeedsSave = false;
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
      
      assert(exist(fname,'file')>0,'File ''%s'' not found.',fname);
      
      s = load(fname,'-mat');
      if ~all(isfield(s,{'VERSION' 'labeledpos'}))
        error('Labeler:load','Unexpected contents in Label file.');
      end
      RC.saveprop('lastLblFile',fname);
      
      obj.isinit = true;
      for f = obj.LOADPROPS,f=f{1}; %#ok<FXSET>
        if isfield(s,f)
          obj.(f) = s.(f);          
        else
          warningNoTrace('Labeler:load','Missing load field ''%s''.',f);
          %obj.(f) = [];
        end
      end
      % labelPointsPlotInfo: special treatment. For old projects,
      % obj.labelPointsPlotInfo can have new/removed fields relative to
      % s.labelPointsPlotInfo. I guess by overlaying we are not removing
      % obsolete fields...
      obj.labelPointsPlotInfo = structoverlay(obj.labelPointsPlotInfo,...
        s.labelPointsPlotInfo);
      obj.movieFilesAllHaveLbls = cellfun(@(x)any(~isnan(x(:))),obj.labeledpos);
      obj.isinit = false;
      
      if obj.nmovies==0 || s.currMovie==0
        obj.movieSetNoMovie();
      else
        obj.movieSet(s.currMovie);
      end
      
      assert(isa(s.labelMode,'LabelMode'));      
      obj.labeledposNeedsSave = false;
      obj.projFSInfo = ProjectFSInfo('loaded',fname);

      obj.setFrameAndTarget(s.currFrame,s.currTarget);
      obj.suspScore = obj.suspScore;
            
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI
    end
    
  end
  
  %% Movie
  methods
    
    function movieAdd(obj,moviefile,trxfile)
      % Add movie/trx to end of movie/trx list.
      % trxfile: optional
      
      if exist('trxfile','var')==0
        trxfile = '';
      end
      
      assert(exist(moviefile,'file')>0,'Cannot find file ''%s''.',moviefile);
      assert(isempty(trxfile) || exist(trxfile,'file')>0,'Cannot find file ''%s''.',trxfile);
      
      mr = MovieReader();
      mr.open(moviefile);
      ifo = struct();
      ifo.nframes = mr.nframes;
      ifo.info = mr.info;
      mr.close();
      
      obj.movieFilesAll{end+1,1} = moviefile;
      obj.movieFilesAllHaveLbls(end+1,1) = false;
      obj.movieInfoAll{end+1,1} = ifo;
      obj.trxFilesAll{end+1,1} = trxfile;
      obj.labeledpos{end+1,1} = [];
      obj.labeledpostag{end+1,1} = [];
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
        obj.labeledpostag(iMov,:) = [];      
        if obj.currMovie>iMov
          obj.movieSet(obj.currMovie-1);
        end
      end
      
      tfSucc = tfProceedRm;
    end
    
    function movieSet(obj,iMov)
      assert(any(iMov==1:obj.nmovies),'Invalid movie index ''%d''.');
      
      % 1. Set the movie
      
      movfile = obj.movieFilesAll{iMov};
      obj.movieReader.open(movfile);
      RC.saveprop('lbl_lastmovie',movfile);
      [path0,movname] = myfileparts(obj.moviefile);
      [~,parent] = fileparts(path0);
      obj.moviename = fullfile(parent,movname);
      
      obj.movieSetHelperUI();
      
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

      if isempty(obj.labeledpos{iMov})
        obj.labelPosInitCurrMovie();
      end
      if isempty(obj.labeledpostag{iMov})
        obj.labelPosTagInitCurrMovie();
      end
      obj.labelingInit();
      
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
  
  methods (Hidden)
    
    function movieSetHelperUI(obj)
      movRdr = obj.movieReader;
      nframes = movRdr.nframes;
                 
      im1 = movRdr.readframe(1);
      if isfield(movRdr.info,'bitdepth')
        obj.maxv = min(obj.maxv,2^movRdr.info.bitdepth-1);
      elseif isa(im1,'uint16')
        obj.maxv = min(2^16 - 1,obj.maxv);
      elseif isa(im1,'uint8')
        obj.maxv = min(obj.maxv,2^8 - 1);
      else
        obj.maxv = min(obj.maxv,2^(ceil(log2(max(im1(:)))/8)*8));
      end
      
      %#UI      
      axcurr = obj.gdata.axes_curr;
      axprev = obj.gdata.axes_prev;
      imcurr = obj.gdata.image_curr;
      set(imcurr,'CData',im1);
      set(axcurr,'CLim',[obj.minv,obj.maxv],...
                 'XLim',[.5,size(im1,2)+.5],...
                 'YLim',[.5,size(im1,1)+.5]);
      set(axprev,'CLim',[obj.minv,obj.maxv],...
                 'XLim',[.5,size(im1,2)+.5],...
                 'YLim',[.5,size(im1,1)+.5]);
      zoom(axcurr,'reset');
      zoom(axprev,'reset');
      
      %#UI
      sliderstep = [1/(nframes-1),min(1,100/(nframes-1))];
      set(obj.gdata.slider_frame,'Value',0,'SliderStep',sliderstep);      
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
      obj.lblCore = LabelCore.create(obj,obj.labelMode);      
      obj.lblCore.init(nPts,lblPtsPlotInfo);
      if obj.labelMode==LabelMode.TEMPLATE && ~isempty(template)
        obj.lblCore.setTemplate(template);
      end

      % TODO: encapsulate labelsPrev (eg in a LabelCore)
      deleteValidHandles(obj.lblPrev_ptsH);
      deleteValidHandles(obj.lblPrev_ptsTxtH);
      obj.lblPrev_ptsH = nan(obj.nLabelPoints,1);
      obj.lblPrev_ptsTxtH = nan(obj.nLabelPoints,1);
      axprev = obj.gdata.axes_prev;
      for i = 1:obj.nLabelPoints
        obj.lblPrev_ptsH(i) = plot(axprev,nan,nan,lblPtsPlotInfo.Marker,...
          'MarkerSize',lblPtsPlotInfo.MarkerSize,...
          'LineWidth',lblPtsPlotInfo.LineWidth,...
          'Color',lblPtsPlotInfo.Colors(i,:),...
          'UserData',i);
        obj.lblPrev_ptsTxtH(i) = text(nan,nan,num2str(i),'Parent',axprev,...
          'Color',lblPtsPlotInfo.Colors(i,:),'Hittest','off');
      end
    end
    
    %%% labelpos
      
    function labelPosInitCurrMovie(obj)
      obj.labeledpos{obj.currMovie} = nan(obj.nLabelPoints,2,obj.nframes,obj.nTargets); 
    end
    function labelPosTagInitCurrMovie(obj)
      obj.labeledpostag{obj.currMovie} = cell(obj.nLabelPoints,obj.nframes,obj.nTargets); 
    end
        
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
      
      obj.labeledpostag{iMov}{iPt,iFrm,iTgt} = [];
    end
    
    function [tf,lpos,lpostag] = labelPosIsLabeled(obj,iFrm,iTrx)
      % For current movie. Labeled includes fullyOccluded
      %
      % tf: scalar logical
      % lpos: [nptsx2] xy coords for iFrm/iTrx
      % lpostag: [npts] cell array of tags 
      
      iMov = obj.currMovie;
      lpos = obj.labeledpos{iMov}(:,:,iFrm,iTrx);
      tfnan = isnan(lpos);
      %assert(all(tfnan(:)) || ~any(tfnan(:)));
      tf = any(~tfnan(:));
      
      if nargout>=3
        lpostag = obj.labeledpostag{iMov}(:,iFrm,iTrx);
      end
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
      
      assert(~any(isnan(xy(:))));
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledpos{iMov}(:,:,iFrm,iTgt) = xy;

      obj.labeledposNeedsSave = true;
    end
    
    function labelPosSetI(obj,xy,iPt)
      % Set labelpos for current movie/frame/target, point iPt
      
      assert(~any(isnan(xy(:))));
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledpos{iMov}(iPt,:,iFrm,iTgt) = xy;
      
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
      
      obj.updateFrameTableComplete();
      obj.labeledposNeedsSave = true;      
    end
    
    function labelPosSetOccludedI(obj,iPt)
      % Occluded is "pure occluded" here
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledpos{iMov}(iPt,:,iFrm,iTgt) = inf;
      
      obj.labeledposNeedsSave = true;
    end
        
    function labelPosTagSetI(obj,tag,iPt)
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
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
        hWB = waitbar(0,'Writing video');
        for i = 1:nFrmsLbled
          f = frmsLbled(i);
          obj.setFrame(f);
          tmpFrame = getframe(ax);
          vr.writeVideo(tmpFrame);
          waitbar(i/nFrmsLbled,hWB,sprintf('Wrote frame %d\n',f));
        end
      catch ME
        vr.close();
        ME.rethrow();
      end
      vr.close();
      delete(hWB);
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
    end
    
    function labelsUpdateNewTarget(obj,prevTarget)
      if ~isempty(obj.lblCore)
        obj.lblCore.newTarget(prevTarget,obj.currTarget,obj.currFrame);
      end
      obj.labelsPrevUpdate();
    end
    
    function labelsUpdateNewFrameAndTarget(obj,prevFrm,prevTgt)
      if ~isempty(obj.lblCore)
        obj.lblCore.newFrameAndTarget(...
          prevFrm,obj.currFrame,...
          prevTgt,obj.currTarget);
      end
      obj.labelsPrevUpdate();
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
    
    function track(obj)
      trker = obj.tracker;
      if isempty(trker)
        error('Labeler:track','No tracker set.');
      end
      if ~obj.hasMovie
        error('Labeler:track','No movie.');
      end
      lpos = obj.labeledpos{obj.currMovie};
      trxNew = trker.track(obj.trx,lpos,[],[],[]);
      obj.trxSet(trxNew);
      obj.setFrameAndTarget(obj.currFrame,obj.currTarget);
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
      zr1 = obj.zoomRadiusTight; % tight zoom: small radius
      
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
      gd = obj.gdata;
      gd.axes_curr.YDir = toggleAxisDir(gd.axes_curr.YDir);
      gd.axes_prev.YDir = toggleAxisDir(gd.axes_prev.YDir);
    end
    function videoFlipLR(obj)
      gd = obj.gdata;
      gd.axes_curr.XDir = toggleAxisDir(gd.axes_curr.XDir);
      gd.axes_prev.XDir = toggleAxisDir(gd.axes_prev.XDir);
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
      pref = obj.trxPrefs;
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
      pref = obj.trxPrefs;
      
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

end
