classdef Labeler < handle
% Labeler  Bransonlab Animal Video Labeler

  properties (Constant,Hidden)
    VERSION = '0.0';
    DEFAULT_LBLFILENAME = '%s_lbl.mat';    
    PREF_FILENAME = 'pref.yaml';

    SAVEPROPS = { ...
      'VERSION' 'projname' ...
      'movieFilesAll' 'trxFilesAll' 'labeledpos' ...      
      'currMovie' 'currFrame' 'currTarget' 'minv' 'maxv' ...
      'labelMode' 'nLabelPoints' 'labelPointsPlotInfo'};
    LOADPROPS = {...
      'movieFilesAll' 'trxFilesAll' 'labeledpos' ...
      'minv' 'maxv'};
    
    TBLTRX_STATIC_COLSTBL = {'id' 'labeled'};
    TBLTRX_STATIC_COLSTRX = {'id' 'labeled'};
    
    TBLFRAMES_COLS = {'frame' 'tgts' 'pts'};
    
    NEIGHBORING_FRAME_MAXRADIUS = 10;
    NEIGHBORING_FRAME_OFFSETS = neighborIndices(Labeler.NEIGHBORING_FRAME_MAXRADIUS);
  end
  
  %% Project
  properties (SetObservable)
    projname
    projFSInfo;           % ProjectFSInfo
  end
  properties (Dependent)
    projectfile;          % Full path to current project 
  end

  %% Movie
  properties
    movieReader = []; % MovieReader object
    minv = 0;
    maxv = inf;
    movieFrameStepBig = 10;
  end
  properties (SetObservable)
    movieFilesAll = cell(0,1); % column cellstr, full paths to movies
    targetZoomFac;
    moviename; % short name, moviefile
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
  
  %% Labeling
  properties (SetAccess=private)
    %labels = cell(0,1);  % cell vector with nTarget els. labels{iTarget} is nModelPts x 2 x "numFramesTarget"
    nLabelPoints;         % scalar integer
    labelPointsPlotInfo;  % struct containing cosmetic info for labelPoints
    
    labelMode;            % scalar LabelMode
    labeledpos;           % column cell vec with .nmovies elements. labeledpos{iMov} is npts x 2 x nFrm(iMov) x nTrx(iMov)
  end
  properties (SetObservable)
    labeledposNeedsSave;  % scalar logical, .labeledpos has been touched since last save
  end
  properties
    lblCore;
    
    lblPrev_ptsH;         % TODO: encapsulate labelsPrev (eg in a LabelCore)
    lblPrev_ptsTxtH;                          
  end  
  
  %% Misc
  properties (SetObservable, AbortSet)
    currMovie;            % idx into .movieFilesAll
    currFrame = 1;        % current frame
    prevFrame = nan;      % last previously VISITED frame
    currTarget = nan;
    prevTarget = nan;    
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
  end
  
  methods % prop access
    function set.labeledpos(obj,v)
      obj.labeledpos = v;
      if ~obj.isinit %#ok<MCSUP>
        obj.updateTrxTable();
        obj.updateFrameTableIncremental(); % TODO use listener/event for this
      end
    end
  end
  
  %% Ctor/Dtor
  methods 
  
    function obj = Labeler(varargin)
      % lObj = Labeler();  
      
      if nargin==0
        preffile = fullfile(APT.Root,Labeler.PREF_FILENAME);
        pref = ReadYaml(preffile,[],1);
        obj.initFromPrefs(pref);        
      else
        assert(false,'Currently unsupported');
      end      
      hFig = LabelerGUI(obj);
      obj.gdata = guidata(hFig);      
      obj.movieReader = MovieReader;      
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
    end
    
    function addDepHandle(obj,h)
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
    
    function projNew(obj,name)
      if exist('name','var')==0
        resp = inputdlg('Project name:','New Project');
        if isempty(resp)
          return;
        end
        name = resp{1};
      end
      
      obj.projname = name;
      obj.movieFilesAll = cell(0,1);
      obj.trxFilesAll = cell(0,1);
      obj.movieSetNoMovie(); % order important here
      obj.labeledpos = cell(0,1);

      %obj.notify('newProject');
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
      
      defaultFname = sprintf(obj.DEFAULT_LBLFILENAME,obj.projname);
      filterspec = fullfile(savepath,defaultFname);
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
          s.template = obj.lblCore.getTemplate();
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
      
      assert(exist(fname,'file')>0,'File ''%s'' not found.');
      
      s = load(fname,'-mat');
      if ~all(isfield(s,{'VERSION' 'labeledpos'}))
        error('Labeler:load','Unexpected contents in Label file.');
      end
      
      for f = obj.LOADPROPS,f=f{1}; %#ok<FXSET>
        obj.(f) = s.(f);
      end
      
      if obj.nmovies==0
        obj.movieSetNoMovie();
      else
        obj.movieSet(s.currMovie);
      end
      
      assert(isa(s.labelMode,'LabelMode'));
      if isfield(s,'template')
        template = s.template;
      else 
        template = [];
      end
      
      obj.labelingInit('labelMode',s.labelMode,'nPts',s.nLabelPoints,...
        'labelPointsPlotInfo',s.labelPointsPlotInfo,'template',template);
%       obj.labeledpos = s.labeledpos;      
      obj.labeledposNeedsSave = false;
      
      obj.projFSInfo = ProjectFSInfo('loaded',fname);

      obj.setTarget(s.currTarget);
      obj.setFrame(s.currFrame,'forceUpdate',true);
            
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
      
      obj.movieFilesAll{end+1,1} = moviefile;
      obj.trxFilesAll{end+1,1} = trxfile;
      obj.labeledpos{end+1,1} = [];   
    end
    
    function movieRm(obj,iMov)
      assert(any(iMov==1:obj.nmovies),'Invalid movie index ''%d''.');
      if iMov==obj.currMovie
        error('Labeler:movieRm','Cannot remove current movie.');
      end
      if obj.labelposMovieHasLabels(iMov)
        warning('Labeler:movieRm','Movie index ''%d'' has labels. Removing...',iMov);
      end
      
      obj.movieFilesAll(iMov,:) = [];
      obj.trxFilesAll(iMov,:) = [];
      obj.labeledpos(iMov,:) = [];
      if obj.currMovie>iMov
        obj.movieSet(obj.currMovie-1);
      end
    end
    
    function movieSet(obj,iMov)
      assert(any(iMov==1:obj.nmovies),'Invalid movie index ''%d''.');
      movfile = obj.movieFilesAll{iMov};
      trxFile = obj.trxFilesAll{iMov};      
           
      obj.movieReader.open(movfile);
      RC.saveprop('lbl_lastmovie',movfile);
      [~,obj.moviename] = myfileparts(obj.moviefile);
      
      obj.trxfile = trxFile;
      obj.newMovieAndTrx();
            
      obj.isinit = true; % Initialization hell, invariants momentarily broken
      obj.currMovie = iMov; 
      if isempty(obj.labeledpos{iMov})
        obj.labelPosInitCurrMovie();
      end
      obj.currFrame = 2; % probably dumb, to force update in setFrame
      obj.setTarget(1);
      if obj.hasTrx
        obj.setFrame(obj.trx(1).firstframe);
      else
        obj.setFrame(1);
      end      
      obj.isinit = false; % end Initialization hell

      if obj.hasTrx
        obj.videoSetTargetZoomFac(obj.targetZoomFac);
      end
      
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI 
      
      obj.labelingInit();
    end
    
    function movieSetNoMovie(obj)
      % Set to iMov==0

      obj.currMovie = 0;
      
      %obj.movieReader = [];
      obj.trx = [];
      obj.frm2trx = [];
      obj.trxIdPlusPlus2Idx = [];

      obj.currFrame = 1;
      set(obj.gdata.txCurrImTarget,'Visible','off');
      imcurr = obj.gdata.image_curr;
      set(imcurr,'CData',0);
      
      obj.currTarget = 0; 
    end
    
  end
    
  %% Labeling
  methods
    
    function labelingInit(obj,varargin)
      % Initialize labeling state
      % 
      % Optional PVs:
      % - labelMode. Defaults to .labelMode
      % - nPts. Defaults to current nPts
      % - labelPointsPlotInfo. Defaults to current
      % - template.
      
      [lblmode,nPts,lblPtsPlotInfo,template] = myparse(varargin,...
        'labelMode',obj.labelMode,...
        'nPts',obj.nLabelPoints,...
        'labelPointsPlotInfo',obj.labelPointsPlotInfo,...
        'template',[]);
      assert(isa(lblmode,'LabelMode'));
      validateattributes(nPts,{'numeric'},{'scalar' 'positive' 'integer'});
      % assert(iscellstr(ptNames) && numel(ptNames)==nPts);
      % assert(isequal(size(labelPointsPlotInfo),[nPts 3]));
      
      obj.labelMode = lblmode;
      obj.nLabelPoints = nPts;
      obj.labelPointsPlotInfo = lblPtsPlotInfo;
      
      gd = obj.gdata;
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
      switch lblmode
        case LabelMode.SEQUENTIAL
          obj.lblCore = LabelCoreSeq(obj);
          gd.menu_setup_sequential_mode.Enable = 'on';
          gd.menu_setup_sequential_mode.Checked = 'on';
          gd.menu_setup_template_mode.Enable = 'off';
          gd.menu_setup_template_mode.Checked = 'off';
          gd.menu_setup_highthroughput_mode.Enable = 'off';
          gd.menu_setup_highthroughput_mode.Checked = 'off';
          gd.menu_setup_createtemplate.Enable = 'off';
  
          obj.lblCore.init(nPts,lblPtsPlotInfo);
          
        case LabelMode.TEMPLATE
          obj.lblCore = LabelCoreTemplate(obj);
          gd.menu_setup_sequential_mode.Enable = 'off';
          gd.menu_setup_sequential_mode.Checked = 'off';
          gd.menu_setup_template_mode.Enable = 'on';
          gd.menu_setup_template_mode.Checked = 'on';
          gd.menu_setup_highthroughput_mode.Enable = 'off';
          gd.menu_setup_highthroughput_mode.Checked = 'off';
          gd.menu_setup_createtemplate.Enable = 'off';

          obj.lblCore.init(nPts,lblPtsPlotInfo);
          if ~isempty(template)
            obj.lblCore.setTemplate(template);
          end
        case LabelMode.HIGHTHROUGHPUT
          obj.lblCore = LabelCoreHT(obj);
          gd.menu_setup_sequential_mode.Enable = 'off';
          gd.menu_setup_sequential_mode.Checked = 'off';
          gd.menu_setup_template_mode.Enable = 'off';
          gd.menu_setup_template_mode.Checked = 'off';
          gd.menu_setup_highthroughput_mode.Enable = 'on';
          gd.menu_setup_highthroughput_mode.Checked = 'on';
          gd.menu_setup_createtemplate.Enable = 'off';
  
          obj.lblCore.init(nPts,lblPtsPlotInfo);
      end
      
      %fprintf(2,'Remove labelPosInitWithLocked');
      %obj.labelPosInitWithLocked();
      %obj.lblCore.clearLabels(); 
      
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
    
%     function labelPosInitWithLocked(obj)
%       obj.labeledpos = nan(obj.nLabelPoints,2,obj.nframes,obj.nTargets); 
%       obj.labeledposNeedsSave = false;
%       fprintf(2,'deleteme\n');
%     end
    
    function labelPosClear(obj)
      % Clear all labels for current movie/frame/target
      
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
    end
    
    function labelPosClearI(obj,iPt)
      % Clear labels for current movie/frame/target, point iPt
      
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
    end
    
    function [tf,lpos] = labelPosIsLabeled(obj,iFrm,iTrx)
      % for current movie. Labeled includes occluded
      
      lpos = obj.labeledpos{obj.currMovie}(:,:,iFrm,iTrx);
      tfnan = isnan(lpos);
      %assert(all(tfnan(:)) || ~any(tfnan(:)));
      tf = any(~tfnan(:));
    end 
    
    function tf = labelPosIsOccluded(obj,iFrm,iTrx)
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
      % Set labelpos from labelPtsH for current movie/frame/target, point 
      % iPt
      
      assert(~any(isnan(xy(:))));
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledpos{iMov}(iPt,:,iFrm,iTgt) = xy;
      
      obj.labeledposNeedsSave = true;
    end
    
    function labelPosSetOccludedI(obj,iPt)
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledpos{iMov}(iPt,:,iFrm,iTgt) = inf;
      
      obj.labeledposNeedsSave = true;
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
        frms = 1:obj.nframes;
        tfWaitBar = true;
      else
        tfWaitBar = false;
      end
      
      if ~obj.hasMovie || obj.currMovie==0 % invariants temporarily broken
        nTgts = nan;
        nPts = nan;
        return;
      end
      
      nf = numel(frms);
      npts = obj.nLabelPoints;
      ntgts = obj.nTargets;
      lpos = obj.labeledpos{obj.currMovie};
      tflpos = ~isnan(lpos); % true->labeled (either regular or occluded)      
      
      nTgts = zeros(nf,1);
      nPts = zeros(nf,1);
      if tfWaitBar
        hWB = waitbar(0,'Scanning existing labels');
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
      lpos = obj.labelpos{iMov};
      tf = any(~isnan(lpos(:)));
    end
           
  end
  
  methods (Access=private)
    
    function labelsUpdateNewFrame(obj,force)
      if exist('force','var')==0
        force = false;
      end
      if ~isempty(obj.lblCore) && (obj.prevFrame~=obj.currFrame || force)
        obj.lblCore.newFrame(obj.prevFrame,obj.currFrame,obj.currTarget);
      end
      obj.labelsPrevUpdate();
    end
    
    function labelsUpdateNewTarget(obj)
      if ~isempty(obj.lblCore)
        obj.lblCore.newTarget(obj.prevTarget,obj.currTarget,obj.currFrame);
      end
      obj.labelsPrevUpdate();
    end
    
    % TODO: encapsulate labelsPrev (eg in a LabelCore)
    function labelsPrevUpdate(obj)
      if ~isnan(obj.prevFrame) && ~isempty(obj.lblPrev_ptsH)
        iMov = obj.currMovie;
        lpos = obj.labeledpos{iMov}(:,:,obj.prevFrame,obj.currTarget);
        obj.lblCore.assignLabelCoords(lpos,'hPts',obj.lblPrev_ptsH,...
          'hPtsTxt',obj.lblPrev_ptsTxtH);
      else
        LabelCore.setPtsOffaxis(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
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
      set(obj.gdata.axes_prev,'CLim',clim);
      obj.minv = clim(1);
      obj.maxv = clim(2);
    end
    
  end
  
  %%
  methods (Hidden)
    
    function newMovieAndTrx(obj)
      % .movieReader and .trxfilename set

      movRdr = obj.movieReader;
      nframes = movRdr.nframes;

      tfTrx = ~isempty(obj.trxfile);
      if tfTrx
        tmp = load(obj.trxfile);
        obj.trx = tmp.trx;
        set(obj.gdata.txCurrImTarget,'Visible','on'); %UI      
      else
        obj.trx = [];
        set(obj.gdata.txCurrImTarget,'Visible','off'); %UI
      end
                  
      f2t = false(obj.nframes,obj.nTrx);
      if tfTrx
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
    
    function setFrame(obj,frm,varargin)
      forceUpdate = myparse(varargin,'forceUpdate',false);
      
      obj.prevFrame = obj.currFrame;
      obj.prevIm = obj.currIm;
      set(obj.gdata.image_prev,'CData',obj.prevIm);
      
      if obj.currFrame~=frm || forceUpdate
        obj.currIm = obj.movieReader.readframe(frm);
        obj.currFrame = frm;
      end            
      set(obj.gdata.image_curr,'CData',obj.currIm);
      
      if obj.hasTrx
        tfTargetLive = obj.frm2trx(frm,:);
        if ~tfTargetLive(obj.currTarget)
          % In the new frame, the current target does not exist.
          % Automatically select a new target.
          
          iTgt1stLive = find(tfTargetLive,1);
          assert(~isempty(iTgt1stLive),'TODO: No targets present in current frame.');          
          warningNoTrace('Labeler:newTarget',...
            'Current target (ID %d) is not present in frame=%d. Switching to target ID %d.',...
            obj.currTrxID,frm,obj.trx(iTgt1stLive).id);
          
          %%% This section follows what Labeler.setTarget() does
          obj.prevTarget = iTgt1stLive;
          obj.currTarget = iTgt1stLive;
          
          % Mildly dangerous thing to do. labelUpdateNewTarget will 
          % 1. Tell LabelCore to transition from .prevTarget->.currTarget 
          % on obj.currFrame, which is not really what is occurring;
          % 2. Update the previous-frame-labels for this .currTarget (I
          % guess this is innocuous).
          
          % In general, a flaw in the current framework is that transitions
          % are expected to be either:
          % 1. Fixed frame, target1 -> target2
          % 2. Fixed target, frame1 -> frame2
          % 
          % In general, this may not be possible.
          % ALTODO: This may be worth cleaning up, eg have setTarget(),
          % setFrame(), resetFrameTarget().
          
          obj.labelsUpdateNewTarget();
          
          %obj.setTargetTxt('txCurrImTarget',iTgt1stLive);
          
          %%% End follow Labeler.setTarget() 
        end
          
        obj.videoCenterOnCurrTarget();
      end

      obj.labelsUpdateNewFrame(forceUpdate);
      
      obj.updateTrxTable();  
    end
    
    function setTargetID(obj,tgtID)
      iTgt = obj.trxIdPlusPlus2Idx(tgtID+1);
      assert(~isnan(iTgt),'Invalid target ID: %d.');
      obj.setTarget(iTgt);
    end
    
    function setTarget(obj,iTgt)
      % Change the target to iTgt keeping the current frame fixed.
      % iTgt: INDEX into obj.trx
      
      validateattributes(iTgt,{'numeric'},{'positive' 'integer' '<=' obj.nTargets});
      
      obj.prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      if obj.hasTrx
        obj.videoCenterOnCurrTarget();
        obj.labelsUpdateNewTarget();
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
    
    % TODO Prob use listener/event for this
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
    end    
    function updateFrameTableComplete(obj)     
      [nTgts,nPts] = obj.labelPosLabeledFramesStats();
      assert(isequal(nTgts>0,nPts>0));
      tfFrm = nTgts>0;
      iFrm = find(tfFrm);
      
      dat = [num2cell(iFrm) num2cell(nTgts(tfFrm)) num2cell(nPts(tfFrm)) ];
      tbl = obj.gdata.tblFrames;
      set(tbl,'Data',dat);
    end
   
  end

end
