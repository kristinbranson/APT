classdef Labeler < handle
% Labeler  Bransonlab Animal Video Labeler
%
% Takes a movie, trx (optional), template (optional), and creates/edits
% "animal model" labels.

  properties (Constant,Hidden)
    DT2P = 5;
    VERSION = '0.0';
    DEFAULT_LBLFILENAME = '%s.lbl.mat';
    
    SAVEPROPS = { ...
      'VERSION' 'moviefile' 'nframes' 'trxFilename' 'nTrx' ...
      'labelMode' 'nLabelPoints' 'labelNames' ...
      'labeledpos' 'labelPtsColors' 'currFrame' 'currTarget' 'minv' 'maxv'};
    LOADPROPS = {'minv' 'maxv'};
    
    TBLTRX_STATIC_COLSTBL = {'id' 'labeled'};
    TBLTRX_STATIC_COLSTRX = {'id' 'labeled'};
    
    TBLFRAMES_COLS = {'frame' 'labeled targets'};
    
    FRAMEUP_BIGSTEP = 10;
    NEIGHBORING_FRAME_MAXRADIUS = 10;
    NEIGHBORING_FRAME_OFFSETS = neighborIndices(Labeler.NEIGHBORING_FRAME_MAXRADIUS);
  end
  
  
  %% Movie
  properties
    movieReader = [];         % MovieReader object
    minv = 0;
    maxv = inf;
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
  end  
  properties (Dependent)
    hasTrx
    currTrx
    nTrx
    nTargets          % nTrx, or 1 if no Trx
  end
  
  
  %% Labeling
  properties
    %labels = cell(0,1);  % cell vector with nTarget els. labels{iTarget} is nModelPts x 2 x "numFramesTarget"
    labeledpos;           % labels, npts x 2 x nFrm x nTrx
    labelsLocked;         % nFrm x nTrx
    
    labelMode;            % scalar LabelMode
    nLabelPoints;         % scalar integer
    labelNames;           % nLabelPoints-by-1 cellstr
    labelPtsColors;       % nLabelPoints x 3 RGB
    
    labelState;           % scalar LabelState
    labelPtsH;            % nLabelPoints x 1 handle vec, handle to points
    labelPtsTxtH;         % nLabelPoints x 1 handle vec, handle to text
    label_iPtMove;        % scalar integer. 0..nLabelPoints, point clicked and being moved

    lblPrev_ptsH;         % Maybe encapsulate this and next with axes_prev, image_prev
    lblPrev_ptsTxtH;

    % Label mode 1
    lbl1_nPtsLabeled;     % scalar integer. 0..nLabelPoints, or inf.
                          % State description; see bdfmode1
                          
    % Label mode 2
    lbl2_template;        % LabelTemplate obj. "original" template
    lbl2_templateClr = [1 1 1]; % 1 x 3 RGB
    lbl2_ptsTfAdjusted;   % nLabelPoints x 1 logical vec. If true, pt has been adjusted from template
  end 
  
  properties
    gdata = [];           % handles structure for figure

    currFrame = 1;        % current frame
    currIm = [];
    prevFrame = nan;      % last previously VISITED frame
    prevIm = [];    
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
      if obj.hasTrx
        obj.updateTrxTable();
      end
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
      switch s.labelMode
        case LabelMode.SEQUENTIAL
          obj.labelingInit(s.labelMode,s.nLabelPoints,...
            'ptNames',s.labelNames,'ptColors',s.labelPtsColors);
          obj.labeledpos = s.labeledpos; 
          obj.labelMode1Label();
          obj.setFrame(s.currFrame);
          obj.setTarget(s.currTarget);
        otherwise
          assert(false,'TODO');
      end
            
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI      
    end
    
  end
  
  
  %% Movie, Trx
  methods
    
    function obj = Labeler(varargin)
      npts = varargin{1};
      
      hFig = LabelerGUI(obj);
      obj.gdata = guidata(hFig);
      
      obj.movieReader = MovieReader;
      
      obj.nLabelPoints = npts;
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
      assert(exist(movfile,'file')>0,'File ''%s'' not found.');

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
  methods % labelpos
      
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
    
    function labelPosSet(obj)
      % Set labelpos from labelPtsH for current frame, current target
      
      cfrm = obj.currFrame;
      ctrx = obj.currTarget;
      xy = Labeler.getCoordsFromPts(obj.labelPtsH);
      assert(~any(isnan(xy(:))));
      obj.labeledpos(:,:,cfrm,ctrx) = xy;
    end
    
  end
  
  methods
    
    function labelingInit(obj,labelMode,nPts,varargin)
      % Initialize labeling state
      % 
      % Optional PVs:
      % - ptNames
      % - ptColors

      assert(isa(labelMode,'LabelMode'));
      validateattributes(nPts,{'numeric'},{'scalar' 'positive' 'integer'});
      [ptNames,ptColors] = myparse(varargin,...
        'ptNames',repmat({''},nPts,1),...
        'ptColors',jet(nPts));      
      assert(iscellstr(ptNames) && numel(ptNames)==nPts);
      assert(isequal(size(ptColors),[nPts 3]));
      
      obj.labelMode = labelMode;
      obj.nLabelPoints = nPts;
      obj.labelNames = ptNames(:);
      obj.labelPtsColors = ptColors;
      
      obj.labelPosInitWithLocked();

      deleteHandles(obj.labelPtsH);
      deleteHandles(obj.labelPtsTxtH);
      obj.labelPtsH = nan(obj.nLabelPoints,1);
      obj.labelPtsTxtH = nan(obj.nLabelPoints,1);
      ax = obj.gdata.axes_curr;
      for i = 1:obj.nLabelPoints
        obj.labelPtsH(i) = plot(ax,nan,nan,'w+','MarkerSize',20,...
                                        'LineWidth',3,'Color',ptColors(i,:),'UserData',i);
        obj.labelPtsTxtH(i) = text(nan,nan,num2str(i),'Parent',ax,...
                                          'Color',ptColors(i,:),'Hittest','off');
      end
      
      deleteHandles(obj.lblPrev_ptsH);
      deleteHandles(obj.lblPrev_ptsTxtH);
      obj.lblPrev_ptsH = nan(obj.nLabelPoints,1);
      obj.lblPrev_ptsTxtH = nan(obj.nLabelPoints,1);
      axprev = obj.gdata.axes_prev;
      for i = 1:obj.nLabelPoints
        obj.lblPrev_ptsH(i) = plot(axprev,nan,nan,'w+','MarkerSize',20,...
                                   'LineWidth',3,'Color',ptColors(i,:),'UserData',i);
        obj.lblPrev_ptsTxtH(i) = text(nan,nan,num2str(i),'Parent',axprev,...
                                      'Color',ptColors(i,:),'Hittest','off');
      end
      
      obj.label_iPtMove = nan;
                 
      gdat = obj.gdata;
      set(gdat.axes_curr,'ButtonDownFcn',@(s,e)obj.bdfMode1Ax(s,e));
      arrayfun(@(x)set(x,'HitTest','on','ButtonDownFcn',@(s,e)obj.bdfMode1Pt(s,e)),obj.labelPtsH);
      set(gdat.figure,'WindowButtonMotionFcn',@(s,e)obj.wbmfMode1(s,e));
      set(gdat.figure,'WindowButtonUpFcn',@(s,e)obj.wbufMode1(s,e));      
      set(gdat.pbClear,'Enable','on');      
    end    
    
    function labelsUpdateNewFrame(obj)
      if ~isempty(obj.labelMode)
        switch obj.labelMode
          case LabelMode.SEQUENTIAL
            obj.labelMode1NewFrameOrTarget();
            obj.labelsPrevUpdate();
        end
      end
    end
    
    function labelsUpdateNewTarget(obj)
      if ~isempty(obj.labelMode)
        switch obj.labelMode
          case LabelMode.SEQUENTIAL
            obj.labelMode1NewFrameOrTarget();
            obj.labelsPrevUpdate();
        end
      end
    end
    
    function labelsPrevUpdate(obj)
      if ~isnan(obj.prevFrame)
        lpos = obj.labeledpos(:,:,obj.prevFrame,obj.currTarget);
        Labeler.assignCoords2Pts(lpos,obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
      else
        Labeler.removePts(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
      end
    end
    
    function bdfMode1Ax(obj,~,~)
      switch obj.labelState
        case LabelState.LABEL
          ax = obj.gdata.axes_curr;
          
          nlbled = obj.lbl1_nPtsLabeled;
          if nlbled>=obj.nLabelPoints
            assert(false); % adjustment mode only
          else % 0..nLabelPoints-1
            tmp = get(ax,'CurrentPoint');
            x = tmp(1,1);
            y = tmp(1,2);
            
            i = nlbled+1;
            set(obj.labelPtsH(i),'XData',x,'YData',y);
            set(obj.labelPtsTxtH(i),'Position',[x+obj.DT2P y+obj.DT2P]);
            obj.lbl1_nPtsLabeled = i;
            
            if i==obj.nLabelPoints
              obj.labelMode1Adjust();
            end
          end
      end
    end
    
    function bdfMode1Pt(obj,src,~)
      switch obj.labelState
        case LabelState.ADJUST
          obj.label_iPtMove = get(src,'UserData');
        case LabelState.ACCEPTED
          obj.labelMode1Adjust();
          obj.label_iPtMove = get(src,'UserData');
      end
    end
    
    function wbmfMode1(obj,~,~)
      if obj.labelState==LabelState.ADJUST
        iPt = obj.label_iPtMove;      
        if ~isnan(iPt) % should always be true
          ax = obj.gdata.axes_curr;
          tmp = get(ax,'CurrentPoint');
          pos = tmp(1,1:2);
          set(obj.labelPtsH(iPt),'XData',pos(1),'YData',pos(2));
          pos(1) = pos(1) + obj.DT2P;
          set(obj.labelPtsTxtH(iPt),'Position',pos);
        end
      end
    end
    
    function wbufMode1(obj,~,~)
      if obj.labelState==LabelState.ADJUST
        obj.label_iPtMove = nan;
      end
    end    
    
  end
  
  methods % Label mode 1 (Sequential)
    
    % LABEL MODE 1 IMPL NOTES
    % There are three labeling states: 'label', 'adjust', 'accepted'. 
    % 
    % During the labeling state, points are being clicked in order. This 
    % includes the state where there are zero points clicked (fresh image). 
    % In this stage, only axBDF is on and each click labels a new point.
    %    
    % During the adjustment state, points may be adjusted by
    % click-dragging; this is enabled by ptsBDF, figWBMF, figWBUF acting in 
    % concert.
    %
    % When any/all adjustment is complete, tbAccept is clicked and we enter
    % the accepted stage. This locks the labeled points for this frame and
    % writes to .labeledpos.
    %
    % pbClear is enabled at all times. Clicking it returns to the 'label'
    % state and clears any labeled points.
    %
    % tbAccept is disabled during 'label'. During 'adjust', its name is 
    % "Accept" and clicking it moves to the 'accepted' state. During
    % 'accepted, its name is "Adjust" and clicking it moves to the 'adjust'
    % state.
    %
    % When multiple targets are present, all actions/transitions are for
    % the current target. Acceptance writes to .labeledpos for the current
    % target. Changing targets is like changing frames; all pre-acceptance
    % actions are discarded.
    
    % View according to state
    % .labeledpos is final state for accepted labels. During labeling, the
    % various lbl1_* state is used to keep track of tenative labels. The
    % only way to write to .labeledpos is via one fo the labeledpos* 
    % methods, or by loading a .lbl file. 
    % Writing from lbl1_* state to .labelpos occurs via labelMode1Accept.
    % Writing from .labelpos to lbl1_* state occurs via labelMode1NewFrameOrTarget.
        
    function labelMode1Label(obj)
      % Enter Label state and clear all mode1 label state for current
      % frame/target
      
      set(obj.gdata.tbAccept,'BackgroundColor',[0.4 0.0 0.0],...
        'String','','Enable','off','Value',0);
      
      obj.lbl1_nPtsLabeled = 0;
      arrayfun(@(x)set(x,'Xdata',nan,'ydata',nan),obj.labelPtsH);
      arrayfun(@(x)set(x,'Position',[nan nan 1],'hittest','off'),obj.labelPtsTxtH);
      obj.label_iPtMove = nan;
      obj.labelPosClear();
      
      obj.labelState = LabelState.LABEL;      
    end
       
    function labelMode1Adjust(obj)
      % Enter adjustment state for current frame/target
      
      assert(obj.lbl1_nPtsLabeled==obj.nLabelPoints);
      %assert(all(ishandle(obj.labelPtsH)));
            
      obj.label_iPtMove = nan;
      
      %obj.labelPosClear();
      set(obj.gdata.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
        'Value',0,'Enable','on');
      obj.labelState = LabelState.ADJUST;
    end
    
    function labelMode1Accept(obj,tfSetLabelPos)
      % Enter accepted state (for current frame)
      
      if tfSetLabelPos
        obj.labelPosSet();
      end
      set(obj.gdata.tbAccept,'BackgroundColor',[0,0.4,0],'String','Accepted',...
        'Value',1,'Enable','on');
      obj.labelState = LabelState.ACCEPTED;
    end    
       
    function labelMode1NewFrameOrTarget(obj)
      % React to new frame or target. Set mode1 label state (.lbl1_*) 
      % according to labelpos. If a frame is not labeled, then start fresh 
      % in Label state. Otherwise, start in Accepted state with saved labels.
      
      iFrm = obj.currFrame;
      iTrx = obj.currTarget;
      
      [tflabeled,lpos] = obj.labelPosIsLabeled(iFrm,iTrx);
      if tflabeled
        obj.lbl1_nPtsLabeled = obj.nLabelPoints;
        Labeler.assignCoords2Pts(lpos,obj.labelPtsH,obj.labelPtsTxtH);
        obj.label_iPtMove = nan;
        obj.labelMode1Accept(false); % I guess could just call with true arg
      else
        obj.labelMode1Label();
      end
    end
    
  end
  
  methods % Label mode 2 (template)
    
    % LABEL MODE 2 IMPL NOTES
    % 2 States:
    % - Adjusting
    % -- 'untouched' template points are white
    % -- 'touched' template points are colored
    % - Accepted
    % -- all points colored. 
    % -- Can go back into adjustment mode, but all points turn white as if from template.
    % 
    % TRANSITIONS W/OUT TRX
    % Note: Once a template is created/loaded, there is a set of points (either 
    %  all white/template, or partially template, or all colored) on the image 
    %  at all times.
    % - New unlabeled frame
    % -- Label point positions are unchanged, but all points become white.
    %    If scrolling forward, the template from frame-1 will be used etc.
    % -- (Enhancement? not implemented) Accepted points from 'nearest' 
    %    neighboring labeled frame (eg frm-1, or <frmLastVisited>, become 
    %    white points in new frame
    % - New labeled frame
    % -- Previously accepted labels shown as colored points.
    % 
    % TRANSITIONS W/TRX
    % - New unlabeled frame (current target)
    % -- Existing points are auto-aligned onto new frame. The existing
    %    points could represent previously accepted labels, or just the
    %    previous template state.
    % - New labeled frame (current target) 
    % -- Previously accepted labels shown as colored points.
    % - New target (same frame), unlabeled
    % -- Last labeled frame for new target acts as template; points auto-aligned.
    % -- If no previously labeled frame for this target, current white points are aligned onto new target.
    % - New target (same frame), labeled
    % -- Previously accepted labels shown as colored points.

  
    function labelMode2Adjust(obj)
      % Enter adjustment state for current frame/tgt. All points become
      % white (template/pre-adjustment)
      
      arrayfun(@(x)set(x,'Color',obj.lbl2_templateClr),obj.labelPtsH);
      obj.lbl2_ptsTfAdjusted(:) = false;
      obj.label_iPtMove = nan;
      
      set(obj.gdata.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
        'Value',0,'Enable','on');
      obj.labelState = LabelState.ADJUST;
    end
    
    function labelMode2Accept(obj,tfSetLabelPos)
      % Enter accepted state for current frame/tgt. All points colored. If
      % tfSetLabelPos, all points written to labelpos.
      
      nPts = obj.nLabelPoints;
      ptsH = obj.labelPtsH;
      clrs = obj.labelPtsColors;
      for i = 1:nPts
        set(ptsH(i),'Color',clrs(i,:));
      end
      
      obj.lbl2_ptsTFAdjusted(:) = true;
            
      if tfSetLabelPos
        obj.labelPosSet();
      end
      set(obj.gdata.tbAccept,'BackgroundColor',[0,0.4,0],'String','Accepted',...
        'Value',1,'Enable','on');
      obj.labelState = LabelState.ACCEPTED;
    end
    
    function labelMode2NewFrameOrTarget(obj,transition)
      iFrm = obj.currFrame;
      iTrx = obj.currTarget;
      [tflabeled,lpos] = obj.labelPosIsLabeled(iFrm,iTrx);
      if tflabeled
        Labeler.assignCoords2Pts(lpos,obj.labelPtsH,obj.labelPtsTxtH);
        obj.labelMode2Accept(false);
      else
        if obj.hasTrx
          switch transition
            case NEWFRAME
              % existing points are aligned onto new frame based on trx at
              % (currTarget,prevFrame) and (currTarget,currFrame)

              iFrm0 = obj.prevFrame;
              xy0 = Labeler.getCoordsFromPts(obj.labelPtsH);
              xy = Labeler.transformPtsTrx(xy0,obj.trx(iTrx),iFrm0,obj.trx(iTrx),iFrm);
            case NEWTARGET
              [tfneighbor,iFrm0,lpos0] = obj.labelPosLabeledNeighbor(iFrm,iTrx);
              if tfneighbor
                xy = Labeler.transformPtsTrx(lpos0,obj.trx(iTrx),iFrm0,obj.trx(iTrx),iFrm);
              else
                % no neighboring previously labeled points for new target.
                % Just start with current points for previous target.
                
                xy0 = Labeler.getCoordsFromPts(obj.labelPtsH);
                iTrx0 = obj.prevTarget;
                xy = Labeler.transformPtsTrx(xy0,obj.trx(iTrx0),iFrm,obj.trx(iTrx),iFrm);
              end              
            otherwise
              assert(false);
          end
          Labeler.assignCoords2Pts(xy,obj.labelPtsH,obj.labelPtsTxtH);
        end
        obj.labelMode2Adjust();
      end
    end
    
  end
  
  methods (Static)

    function xy = getCoordsFromPts(hPts)
      x = get(hPts,'XData');
      y = get(hPts,'YData');
      x = cell2mat(x);
      y = cell2mat(y);
      xy = [x y];
    end
    
    function assignCoords2Pts(xy,hPts,hTxt)
      nPts = size(xy,1);
      assert(size(xy,2)==2);
      assert(isequal(nPts,numel(hPts),numel(hTxt)));
      
      for i = 1:nPts
        set(hPts(i),'XData',xy(i,1),'YData',xy(i,2));
        set(hTxt(i),'Position',[xy(i,1)+Labeler.DT2P xy(i,2)+Labeler.DT2P 1]);
      end
    end
    
    function removePts(hPts,hTxt)
      assert(numel(hPts)==numel(hTxt));
      for i = 1:numel(hPts)
        set(hPts(i),'XData',nan,'YData',nan);
        set(hTxt(i),'Position',[nan nan nan]);
      end      
    end
    
    function uv = transformPtsTrx(uv0,trx0,iFrm0,trx1,iFrm1)
      % uv0: npts x 2 array of points
      % trx0: scalar trx
      % iFrm0: absolute frame number for trx0
      % etc
      %
      % The points uv0 correspond to trx0 @ iFrm0. Compute uv that
      % corresponds to trx1 @ iFrm1, ie so that uv relates to trx1@iFrm1 in 
      % the same way that uv0 relates to trx0@iFrm0.
      
      assert(trx0.off==1-trx0.firstframe);
      assert(trx1.off==1-trx1.firstframe);
      
      iFrm0 = iFrm0+trx0.off;
      xy0 = [trx0.x(iFrm0) trx0.y(iFrm0)];
      th0 = trx0.theta(iFrm0);
      
      iFrm1 = iFrm1+trx1.off;
      xy1 = [trx1.x(iFrm1) trx1.y(iFrm1)];
      th1 = trx1.theta(iFrm1);
      
      uv = transformPoints(uv0,xy0,th0,xy1,th1);
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
      for i = 1:obj.nTrx
        frm0 = obj.trx(i).firstframe;
        frm1 = obj.trx(i).endframe;        
        f2t(frm0:frm1,i) = true;
      end
      obj.frm2trx = f2t;
            
      if obj.hasTrx
        obj.currFrame = min([obj.trx.firstframe]);
      else
        obj.currFrame = 1;
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
   end
    
    function setFrame(obj,frm)
      obj.prevFrame = obj.currFrame;
      obj.prevIm = obj.currIm;
      set(obj.gdata.image_prev,'CData',obj.prevIm);
      set(obj.gdata.txPrevIm,'String',num2str(obj.prevFrame));
      
      if obj.currFrame~=frm
        obj.currIm = obj.movieReader.readframe(frm);
        obj.currFrame = frm;
      end
            
      set(obj.gdata.image_curr,'CData',obj.currIm);
      if obj.hasTrx
        obj.videoCenterOnCurrTarget();
      end
        
      %#UI
      set(obj.gdata.slider_frame,'Value',(frm-1)/(obj.nframes-1));
      set(obj.gdata.edit_frame,'String',num2str(frm));

      obj.labelsUpdateNewFrame();
      
      if obj.hasTrx
        obj.updateTrxTable();
      end
      
      % obj.showPreviousLabels      
      % obj.updateLockedButton(); %#UI
      
%       if obj.currFrame > 1
%         for i = 1:obj.npoints
%           if numel(handles.posprev) < i || ~ishandle(handles.posprev(i)),
%             handles.posprev(i) = plot(handles.axes_prev,nan,nan,'+','Color',handles.templatecolors(i,:),'MarkerSize',8);
%           end
%           set(handles.posprev(i),'XData',handles.labeledpos(i,1,handles.f-1,handles.animal),...
%             'YData',handles.labeledpos(i,2,handles.f-1,handles.animal));
%         end
%       else
%         set(handles.posprev,'XData',nan,'YData',nan);
%       end
      
%       if ~isempty(handles.trx) && ~isempty(handles.template),
%         pushbutton_template_Callback(hObject,[],handles);
%       end
    end
    
    function setTarget(obj,iTgt)
      validateattributes(iTgt,{'numeric'},{'positive' 'integer' '<=' obj.nTargets});
      
      obj.prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      if obj.hasTrx
        obj.videoCenterOnCurrTarget();
      end
      obj.labelsUpdateNewTarget();
    end
    
    function frameUp(obj,tfBigstep)
      if tfBigstep
        df = obj.FRAMEUP_BIGSTEP;
      else
        df = 1;
      end
      f = min(obj.currFrame+df,obj.nframes);
      obj.setFrame(f);
    end
    
    function frameDown(obj,tfBigstep)
      if tfBigstep
        df = obj.FRAMEUP_BIGSTEP;
      else
        df = 1;
      end
      f = max(obj.currFrame-df,1);
      obj.setFrame(f);
    end
    
%     function clearTarget(obj)
%       obj.currTarget = nan;      
%       obj.videoZoomFac(0);
%     end
    
    function [x,y] = currentTargetLoc(obj)
      if obj.hasTrx
        cfrm = obj.currFrame;
        ctrx = obj.currTrx;

        if cfrm < ctrx.firstframe || cfrm > ctrx.endframe
          warning('Labeler:target','No track for current target at frame %d.',cfrm);
          x = nan;
          y = nan;
        else
          i = cfrm - ctrx.firstframe + 1;
          x = ctrx.x(i);
          y = ctrx.y(i);
        end
      else
        x = round(obj.movienc/2);
        y = round(obj.movienr/2);
      end
    end
                
    % TODO prob use listener/event for this; maintain relevant
    % datastructure in Labeler
    function updateTrxTable(obj)
      % based on .currFrame, .labeledpos
      
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
      for iTrx = iTrxLive(:)'
        tfLbled(iTrx) = any(lpos(:,1,cfrm,iTrx));
      end
      tbldat(:,end+1) = num2cell(tfLbled);
      
      tbl = obj.gdata.tblTrx;
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
   
    %#UI No really
%     function updateLockedButton(obj)
%       disp('UPDATELOCK TODO');
%       btn = obj.gdata.togglebutton_lock;
%       if obj.labelsLocked(obj.currFrame,obj.currTarget)
%         set(btn,'BackgroundColor',[.6,0,0],'String','Locked','Value',1);
%       else
%         set(btn,'BackgroundColor',[0,.6,0],'String','Unlocked','Value',0);
%       end
%       setButtonImage(btn);
%     end
    
%     function showPreviousLabels(obj)
      % TODO
%       if isempty(handles.labeledpos),
%         fprev = [];
%       else
%         fprev = find(~isnan(handles.labeledpos(1,1,1:handles.f,handles.animal)),1,'last');
%       end
%       
%       for i = 1:handles.npoints,
%         if numel(handles.hpoly) < i || ~ishandle(handles.hpoly(i)),
%           handles.hpoly(i) = plot(handles.axes_curr,nan,nan,'w+','MarkerSize',20,'LineWidth',3);
%           set(handles.hpoly(i),'Color',handles.templatecolors(i,:),...
%             'ButtonDownFcn',@(hObject,eventdata) PointButtonDownCallback(hObject,eventdata,handles.figure,i));
%           handles.htext(i) = text(nan,nan,num2str(i),'Parent',handles.axes_curr,...
%             'Color',handles.templatecolors);
%         end
%         
%         %if all(~isnan(handles.labeledpos(i,:,handles.f,handles.animal))),
%         if ~isempty(fprev),
%           set(handles.hpoly(i),'XData',handles.labeledpos(i,1,fprev,handles.animal),...
%             'YData',handles.labeledpos(i,2,fprev,handles.animal),'Visible','on');
%           
%           tpos = [handles.labeledpos(i,1,fprev,handles.animal)+handles.dt2p;...
%             handles.labeledpos(i,2,fprev,handles.animal)];
%           set(handles.htext(i),'Position',tpos,'Visible','on');          
%         end        
%       end      
%     end   
    
  end

end
