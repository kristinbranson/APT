classdef Labeler < handle
% Labeler  Bransonlab Animal Video Labeler
%
% Takes a movie + trx (optional) and creates/edits animal labels.

  properties (Constant,Hidden)
    VERSION = '0.0';
    DEFAULT_LBLFILENAME = '%s.lbl.mat';    
    PREF_FILENAME = 'pref.yaml';

    SAVEPROPS = { ...
      'VERSION' 'moviefile' 'nframes' 'trxFilename' 'nTrx' ...
      'labelMode' 'nLabelPoints' 'labelPointsPlotInfo' ...
      'labeledpos' 'currFrame' 'currTarget' 'minv' 'maxv'};
    LOADPROPS = {'minv' 'maxv'};
    
    TBLTRX_STATIC_COLSTBL = {'id' 'labeled'};
    TBLTRX_STATIC_COLSTRX = {'id' 'labeled'};
    
    TBLFRAMES_COLS = {'frame' 'labeled targets'};
    
    NEIGHBORING_FRAME_MAXRADIUS = 10;
    NEIGHBORING_FRAME_OFFSETS = neighborIndices(Labeler.NEIGHBORING_FRAME_MAXRADIUS);
  end
  
  %% Movie
  properties
    movieReader = []; % MovieReader object
    minv = 0;
    maxv = inf;
    movieFrameStepBig = 10;
  end
  properties (Dependent)
    hasMovie;
    moviefile;
    nframes;
    movienr;
    movienc;
  end
  
  %% Trx
  properties
    trxFilename = '';         % full filename
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
  properties
    %labels = cell(0,1);  % cell vector with nTarget els. labels{iTarget} is nModelPts x 2 x "numFramesTarget"
    nLabelPoints;         % scalar integer
    labelPointsPlotInfo;      % struct containing cosmetic info for labelPoints
    
    labelMode;            % scalar LabelMode
    labeledpos;           % labels, npts x 2 x nFrm x nTrx
    labelsLocked;         % nFrm x nTrx
%     labelNames;           % nLabelPoints-by-1 cellstr
    
    lblCore;
    
    lblPrev_ptsH;         % Maybe encapsulate this and next with axes_prev, image_prev
    lblPrev_ptsTxtH;                          
  end  
  
  properties
    gdata = [];           % handles structure for figure

    currFrame = 1;        % current frame
    currIm = [];
    prevFrame = nan;      % last previously VISITED frame
    prevIm = [];
  end
  properties (SetObservable,AbortSet)
    currTarget = nan;
    prevTarget = nan;
  end
  
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
      v = obj.movieReader.nframes;
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
  end
  
  methods % prop access
    function set.labeledpos(obj,v)
      obj.labeledpos = v;
      obj.updateTrxTable();
      obj.updateFrameTableIncremental(); % TODO use listener/event for this
    end
  end
  
  %% Save/Load
  methods
    
    function saveLblFile(obj,fname)
      % Saves a .lbl file. Currently defaults to same dir as moviefile.
      
      if exist('fname','var')==0 && obj.hasMovie
        if ~obj.hasMovie
          % extremely unlikely
          error('Labeler:save','No movie loaded.');          
        end
      
        movieFile = obj.movieReader.filename;
        [moviePath,movieFile] = myfileparts(movieFile);
        defaultFname = sprintf(obj.DEFAULT_LBLFILENAME,movieFile);
        filterspec = fullfile(moviePath,defaultFname);
        
        [fname,pth] = uiputfile(filterspec,'Save label file');
        if isequal(fname,0)
          return;
        end
        fname = fullfile(pth,fname);
      elseif exist(fname,'file')>0
        warning('Labeler:save','Overwriting file ''%s''.',fname);
      end
      
      s = obj.getSaveStruct(); %#ok<NASGU>
      save(fname,'-mat','-struct','s');
      
      RC.saveprop('lastLblFile',fname);
    end
    
    function s = getSaveStruct(obj)
      s = struct();
      s.moviefile = obj.movieReader.filename;
      for f = obj.SAVEPROPS, f=f{1}; %#ok<FXSET>
        s.(f) = obj.(f);
      end
      
      switch obj.labelMode
        case LabelMode.TEMPLATE
          s.template = obj.lblCore.getTemplate();
      end
    end
    
    function loadLblFile(obj,fname)
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
      
      if isempty(s.trxFilename)
        obj.loadMovie(s.moviefile);
      else
        obj.loadMovie(s.moviefile,s.trxFilename);
      end
                  
      assert(isa(s.labelMode,'LabelMode'));
      if isfield(s,'template')
        template = s.template;
      else 
        template = [];
      end
      obj.labelingInit('labelMode',s.labelMode,'nPts',s.nLabelPoints,...
        'labelPointsPlotInfo',s.labelPointsPlotInfo,'template',template);
      obj.labeledpos = s.labeledpos;
      
      obj.setTarget(s.currTarget);
      obj.setFrame(s.currFrame);
            
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI      
    end
    
  end
  
  %% Movie, Trx
  methods
    
    function obj = Labeler(varargin)
      % lObj = Labeler();  
      % lObj = Labeler(labelMode,npts)
      
      if nargin==0
        preffile = fullfile(APT.Root,Labeler.PREF_FILENAME);
        pref = ReadYaml(preffile,[],1);
        obj.initFromPrefs(pref);        
      elseif nargin==2
        assert(false,'Unsupported');
%         obj.labelMode = varargin{1};
%         obj.nLabelPoints = varargin{2};
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
      obj.movieFrameStepBig = pref.Movie.FrameStepBig;
      lpp = pref.LabelPointsPlot;
      if isfield(lpp,'ColorMapName') && ~isfield(lpp,'ColorMap')
        lpp.Colors = feval(lpp.ColorMapName,pref.NumLabelPoints);
      end
      obj.labelPointsPlotInfo = lpp;   
    end
    
    function loadMovie(obj,movfile,trxfile)
      % movfile: optional, movie name. If not specified, user will be
      % prompted.
      % trxname: optional, trx filename. If not specified, no trx file will
      % be used. To be prompted to specify a trxfile, specify trxname as [].
      
      if exist('movfile','var')==0 || isempty(movfile)
        lastmov = RC.getprop('lbl_lastmovie');            
        [movfile,movpath] = uigetfile('*.*','Select video to label',lastmov);
        if ~ischar(movfile)
          return;
        end
        movfile = fullfile(movpath,movfile);
      end
      assert(exist(movfile,'file')>0,'File ''%s'' not found.',movfile);

      tfTrx = exist('trxfile','var') > 0;
      if tfTrx
        if isempty(trxfile)
          [trxfile,trxpath] = uigetfile('*.mat','Select trx file',movpath);
          if ~ischar(trxfile)
            % user canceled; interpret this as "there is no trx file"
            tfTrx = false;
          else
            trxfile = fullfile(trxpath,trxfile);
          end
        end
        if tfTrx
          assert(exist(trxfile,'file')>0,'Trx file ''%s'' not found.');
        end
      else
        trxfile = [];
      end
      
      obj.movieReader.open(movfile);
      RC.saveprop('lbl_lastmovie',movfile);
      
      obj.trxFilename = trxfile;

      obj.newMovieAndTrx();
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
          gd.menu_setup_template_mode.Checked = 'on';
          gd.menu_setup_createtemplate.Enable = 'off';
  
          obj.lblCore.init(nPts,lblPtsPlotInfo);
          
        case LabelMode.TEMPLATE
          obj.lblCore = LabelCoreTemplate(obj);
          gd.menu_setup_sequential_mode.Enable = 'off';
          gd.menu_setup_sequential_mode.Checked = 'off';
          gd.menu_setup_template_mode.Enable = 'on';
          gd.menu_setup_template_mode.Checked = 'on';
          gd.menu_setup_createtemplate.Enable = 'off';

          obj.lblCore.init(nPts,lblPtsPlotInfo);
          if ~isempty(template)
            obj.lblCore.setTemplate(template);
          end
      end
      
      obj.labelPosInitWithLocked();
      
      deleteHandles(obj.lblPrev_ptsH);
      deleteHandles(obj.lblPrev_ptsTxtH);
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
      
    function labelPosInitWithLocked(obj)
      obj.labeledpos = nan(obj.nLabelPoints,2,obj.nframes,obj.nTargets); 
      obj.labelsLocked = false(obj.nframes,obj.nTargets);
    end
    
    function labelPosClear(obj)
      % Clear all labels for current frame, current target
      obj.labeledpos(:,:,obj.currFrame,obj.currTarget) = nan;
    end
    
    function [tf,lpos] = labelPosIsLabeled(obj,iFrm,iTrx)
      lpos = obj.labeledpos(:,:,iFrm,iTrx);
      tfnan = isnan(lpos);
      assert(all(tfnan(:)) || ~any(tfnan(:)));
      tf = ~any(tfnan(:));
    end 
    
    function labelPosSet(obj,xy)
      % Set labelpos from labelPtsH for current frame, current target
      
      assert(~any(isnan(xy(:))));
      
      cfrm = obj.currFrame;
      ctrx = obj.currTarget;
      obj.labeledpos(:,:,cfrm,ctrx) = xy;
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
      
      lposTrx = obj.labeledpos(:,:,:,iTrx);
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
       
  end
  
  methods (Access=private)
    
    function labelsUpdateNewFrame(obj)
      if ~isempty(obj.lblCore) && obj.prevFrame~=obj.currFrame
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
    
    function labelsPrevUpdate(obj)
      if ~isnan(obj.prevFrame) && ~isempty(obj.lblPrev_ptsH)
        lpos = obj.labeledpos(:,:,obj.prevFrame,obj.currTarget);
        LabelCore.assignCoords2Pts(lpos,obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
      else
        LabelCore.removePts(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
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
    function videoZoomFac(obj,zoomFac)
      % zoomFac: 0 for no-zoom; 1 for max zoom
      
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

      tfTrx = ~isempty(obj.trxFilename);
      if tfTrx
        tmp = load(obj.trxFilename);
        obj.trx = tmp.trx;
      else
        obj.trx = [];
      end
            
      f2t = false(obj.nframes,obj.nTrx);
      maxID = max([obj.trx.id]);
      id2t = nan(maxID+1,1);
      for i = 1:obj.nTrx
        frm0 = obj.trx(i).firstframe;
        frm1 = obj.trx(i).endframe;        
        f2t(frm0:frm1,i) = true;
        id2t(obj.trx(i).id+1) = i;
      end
      obj.frm2trx = f2t;
      obj.trxIdPlusPlus2Idx = id2t;
            
      if obj.hasTrx
        obj.currFrame = min([obj.trx.firstframe]);
        set(obj.gdata.txCurrImTarget,'Visible','on');       
      else
        obj.currFrame = 1;
        set(obj.gdata.txCurrImTarget,'Visible','off');
      end
                 
      im1 = movRdr.readframe(obj.currFrame);
      %obj.minv = max(obj.minv,0);
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
%       axis(axcurr,'image');
%       axis(axprev,'image');
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
      
      obj.labelPosInitWithLocked();

      obj.currFrame = 2; % to force update in setFrame
      obj.setTarget(1);
      obj.setFrame(1);
      
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI
   end
    
    function setFrame(obj,frm)
      obj.prevFrame = obj.currFrame;
      obj.prevIm = obj.currIm;
      set(obj.gdata.image_prev,'CData',obj.prevIm);
      obj.setFrameTxt('txPrevIm',obj.prevFrame);
      
      if obj.currFrame~=frm
        obj.currIm = obj.movieReader.readframe(frm);
        obj.currFrame = frm;
      end            
      set(obj.gdata.image_curr,'CData',obj.currIm);
      
      %#UI
      set(obj.gdata.slider_frame,'Value',(frm-1)/(obj.nframes-1));
      set(obj.gdata.edit_frame,'String',num2str(frm));

      if obj.hasTrx
        tfTargetLive = obj.frm2trx(frm,:);
        if ~tfTargetLive(obj.currTarget)
          % In the new frame, the current target does not exist.
          % Automatically select a new target.
          
          iTgt1stLive = find(tfTargetLive,1);
          assert(~isempty(iTgt1stLive),'TODO: No targets present in current frame.');          
          warning('Labeler:newTarget',...
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
      obj.setFrameTxt('txCurrImFrame',frm);

      obj.labelsUpdateNewFrame();
      
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
      assert(obj.frm2trx(obj.currFrame,iTgt),...
        'Desired target not preset in current frame.');
      
      obj.prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      if obj.hasTrx
        obj.videoCenterOnCurrTarget();
        obj.labelsUpdateNewTarget();
      end
      %obj.setTargetTxt('txCurrImTarget',iTgt);
    end
    
    function frameUp(obj,tfBigstep)
      if tfBigstep
        df = obj.movieFrameStepBig;
      else
        df = 1;
      end
      f = min(obj.currFrame+df,obj.nframes);
      obj.setFrame(f);
    end
    
    function frameDown(obj,tfBigstep)
      if tfBigstep
        df = obj.movieFrameStepBig;
      else
        df = 1;
      end
      f = max(obj.currFrame-df,1);
      obj.setFrame(f);
    end
    
    function setFrameTxt(obj,hTxtTag,frm)
      set(obj.gdata.(hTxtTag),'String',sprintf('frm: %d',frm));
    end
%     function setTargetTxt(obj,hTxtTag,tgt)
%       set(obj.gdata.(hTxtTag),'String',sprintf('tgtID: %d',tgt));
%     end
    
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
      if ~obj.hasTrx
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
      lpos = obj.labeledpos;
      cfrm = obj.currFrame;
      for i = 1:numel(iTrxLive)
        tfLbled(i) = any(lpos(:,1,cfrm,iTrxLive(i)));
      end
      tbldat(:,end+1) = num2cell(tfLbled);
      
      set(tbl,'Data',tbldat);
    end
    
    % TODO Prob use listener/event for this
    function updateFrameTableIncremental(obj)
      % assumes .labelops and tblFrames differ at .currFrame at most
      %
      % might be unnecessary/premature optim
      
      tbl = obj.gdata.tblFrames;
      dat = get(tbl,'Data');
      frames = cell2mat(dat(:,1));
      
      cfrm = obj.currFrame;
      lpos = obj.labeledpos;
      tfLbled = ~isnan(squeeze(lpos(1,1,cfrm,:)));
      nLbled = nnz(tfLbled);
      
      i = frames==cfrm;
      if nLbled>0
        if any(i)
          assert(nnz(i)==1);
          dat{i,2} = nLbled;
        else
          dat(end+1,:) = {cfrm nLbled};
          [~,idx] = sort(cell2mat(dat(:,1)));
          dat = dat(idx,:);
        end
        set(tbl,'Data',dat);
      else
        if any(i)
          assert(nnz(i)==1);
          dat(i,:) = [];
          set(tbl,'Data',dat);
        end
      end
    end    
    function updateFrameTableComplete(obj)
      lpos = obj.labeledpos;
      tf = ~isnan(squeeze(lpos(1,1,:,:)));
      assert(isequal(size(tf),[obj.nframes obj.nTargets]));
      nLbled = sum(tf,2);
      iFrm = find(nLbled>0);
      nLbled = nLbled(iFrm);
      dat = [num2cell(iFrm) num2cell(nLbled)];

      tbl = obj.gdata.tblFrames;
      set(tbl,'Data',dat);
    end
   
  end

end
