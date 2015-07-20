classdef Labeler < handle
% Labeler  Bransonlab Animal Video Labeler
%
% Takes a movie, trx (optional), template (optional), and creates/edits
% "animal model" labels.

  properties (Constant,Hidden)
    DT2P = 5;
  end
  
  properties
    movieReader = []; % MovieReader object
    minv = 0; % etc
    maxv = inf; % 'model', for the moment
  end
  properties (Dependent)
    hasMovie;
    nframes;
  end
  
  properties
    trxFilename = ''; % full filename
    trx = []; % trx object
    fsizeini = 100; % zoom box size for following targets
  end  
  properties (Dependent)
    hasTrx
    currTrx
    nTrx
    nTrxOr1IfNoTrx
  end
  
  %% Labeling
  properties
    labelMode; % scalar LabelMode
    nLabelPoints; % scalar integer
    labelNames; % nLabelPoints-by-1 cellstr
    %labels = cell(0,1); % cell vector with nTarget els. labels{iTarget} is nModelPts x 2 x "numFramesTarget"
    labeledpos; % temporary labels, npts x 2 x nFrm x nTrx
    labelsLocked; % nFrm x nTrx    
    labelPtsColors; % nLabelPoints x 3 RGB
    
    % Label mode 1
    lbl1_state;       % SequentialModeState
    lbl1_ptsH;        % nLabelPoints x 1 handle vec, handle to points
    lbl1_ptsTxtH;     % nLabelPoints x 1 handle vec, handle to text
    lbl1_nPtsLabeled; % scalar integer. 0..nLabelPoints, or inf.
                      % State description; see bdfmode1
    lbl1_iPtMove;     % scalar integer. 0..nLabelPoints, point clicked and being moved
  end 
  
  properties
    template = []; % 
    
    hFig; % handle to figure window
    gdata = []; % handles structure for figure

    currFrame = nan; % current frame
    currIm = [];
    prevFrame = [];
    prevIm = [];    
    currTarget = nan;
    currModelPoint = nan; 
  end
  
  methods % dependent prop getters
    function v = get.hasMovie(obj)
      v = obj.movieReader.isOpen;
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
    function v = get.nTrxOr1IfNoTrx(obj)
      if obj.hasTrx
        v = obj.nTrx;
      else
        v = 1;
      end
    end       
  end

  %% Movie, Trx
  methods
    
    function obj = Labeler(varargin)
      obj.hFig = LabelerGUI(obj);
      obj.gdata = guidata(obj.hFig);
      
      obj.movieReader = MovieReader;
    end
    
    function loadMovie(obj,fname)
      if exist('fname','var')==0
        lastmov = RC.getprop('lbl_lastmovie');            
        [f,p] = uigetfile('*.*','Select video to label',lastmov);
        if ~ischar(f)
          return;
        end
        fname = fullfile(p,f);
      end
      
      assert(exist(fname,'file')>0,'File ''%s'' not found.');
      
      obj.movieReader.open(fname);
      RC.saveprop('lbl_lastmovie',fname);
      
      obj.newMovie();
    end
    
    function loadTrx(obj,fname)
      % load trx and enable targeted labeling
      
      if exist('fname','var')==0
        lasttrx = RC.load('lbl_lasttrx');
        [f,p] = uigetfile('*.*','Select trx file',lasttrx);
        if ~ischar(f)
          return;
        end
        fname = fullfile(p,f);
      end
      
      assert(exist(fname,'file')>0,'File ''%s'' not found.');
      
      obj.trxFilename = fname;
      tmp = load(fname);
      obj.trx = tmp.trx;
      
      obj.fsizeini = 100;
      obj.fsize = obj.fsizeini;
     
      set(obj.gdata.text_animal,'String',sprintf('Num animals:%d',obj.nTrx)); %#UI      
      
      obj.setTarget(1);
      
      set(obj.gdata.text_animal,'Visible','on');      
      set(obj.gdata.edit_num,'Visible','on');
      set(obj.gdata.pushbutton_num,'Visible','on');
      set(obj.gdata.pushbutton_template,'Visible','on');
    end
    
    function clearTrx(obj)
      % clear trx and disable targeted labeling
      
      obj.trxFilename = '';
      obj.trx = [];

      obj.clearTarget();
      
      %#UI 
      set(obj.gdata.text_animal,'Visible','off');      
      set(obj.gdata.edit_num,'Visible','off');
      set(obj.gdata.pushbutton_num,'Visible','off');
      set(obj.gdata.pushbutton_template,'Visible','off');
    end
    
    function loadTemplate(obj,fname)
      
    end
    
  end
  
  %% Labeling
  methods
      
    function initLabelPosLocked(obj)
      obj.labeledpos = nan(obj.nLabelPoints,2,obj.nframes,obj.nTrxOr1IfNoTrx); 
      obj.labelsLocked = false(obj.nframes,obj.nTrxOr1IfNoTrx);
    end
    
    function labelPosClearCurrFrame(obj)
      % Clear all labels for current frame
      obj.labeledpos(:,:,obj.currFrame,:) = nan;
    end
    
    function labelPosSetCurrFrame(obj)
      % Set labels for all pts for current frame
      
      assert(obj.nTrxOr1IfNoTrx==1);
      assert(obj.labelMode==LabelMode.SEQUENTIAL);
      
      f = obj.currFrame;
      ITRX = 1;
      x = get(obj.lbl1_ptsH,'XData');
      y = get(obj.lbl1_ptsH,'YData');
      x = cell2mat(x);
      y = cell2mat(y);
      assert(~any(isnan(x)));
      assert(~any(isnan(y)));
      obj.labeledpos(:,:,f,ITRX) = [x y];
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
    
    function setLabelMode1(obj,npts,varargin)
      % Resets label state
      % 
      % Optional PVs:
      % - ptNames
      % - ptColors

      validateattributes(npts,{'numeric'},{'scalar' 'positive' 'integer'});
      
      [ptNames,ptColors] = myparse(varargin,...
        'ptNames',repmat({''},npts,1),...
        'ptColors',jet(npts));      
      assert(iscellstr(ptNames) && numel(ptNames)==npts);
      assert(isequal(size(ptColors),[npts 3]));
      
      obj.labelMode = LabelMode.SEQUENTIAL;
      obj.nLabelPoints = npts;
      obj.labelNames = ptNames(:);
      obj.labelPtsColors = ptColors;
      
      obj.initLabelPosLocked();

      deleteHandles(obj.lbl1_ptsH);
      deleteHandles(obj.lbl1_ptsTxtH);
      obj.lbl1_ptsH = nan(obj.nLabelPoints,1);
      obj.lbl1_ptsTxtH = nan(obj.nLabelPoints,1);
      ax = obj.gdata.axes_curr;
      obj.lbl1_ptsH = arrayfun(@(i)plot(ax,nan,nan,'w+','MarkerSize',20,...
                                       'LineWidth',3,'Color',ptColors(i,:),'UserData',i),(1:npts)');
      obj.lbl1_ptsTxtH = arrayfun(@(i)text(nan,nan,num2str(i),'Parent',ax,...
                                           'Color',ptColors(i,:),'Hittest','off'),(1:npts)');
            
      obj.labelMode1Label();
      
      gdat = obj.gdata;
      set(gdat.pbClear,'Enable','on');
    end    
    
    function labelMode1Label(obj)
      % Enter Label state and clear all mode1 label state for this frame
      
      set(obj.gdata.figure,'WindowButtonMotionFcn',[],'WindowButtonUpFcn',[]);
      set(obj.gdata.axes_curr,'ButtonDownFcn',@(s,e)obj.bdfMode1Ax(s,e));
      set(obj.gdata.tbAccept,'BackgroundColor',[0.4 0.0 0.0],...
        'String','','Enable','off','Value',0);
      
      obj.lbl1_nPtsLabeled = 0;
      arrayfun(@(x)set(x,'Xdata',nan,'ydata',nan,'hittest','off'),obj.lbl1_ptsH);
      arrayfun(@(x)set(x,'Position',[nan nan 1],'hittest','off'),obj.lbl1_ptsTxtH);
      obj.lbl1_iPtMove = nan;
      obj.labelPosClearCurrFrame();
      
      obj.lbl1_state = SequentialModeState.LABEL;      
    end
       
    function labelMode1Adjust(obj)
      % Enter adjustment state (for current frame)
      
      assert(obj.lbl1_nPtsLabeled==obj.nLabelPoints);
      %assert(all(ishandle(obj.lbl1_ptsH)));
      
      arrayfun(@(x)set(x,'HitTest','on','ButtonDownFcn',@(s,e)obj.bdfMode1Pt(s,e)),obj.lbl1_ptsH);
      gdat = obj.gdata;
      set(gdat.axes_curr,'ButtonDownFcn',[]);
      set(gdat.figure,'WindowButtonMotionFcn',@(s,e)obj.wbmfMode1(s,e));
      set(gdat.figure,'WindowButtonUpFcn',@(s,e)obj.wbufMode1(s,e));
            
      obj.lbl1_iPtMove = nan;
      
      obj.labelPosClearCurrFrame();
      set(obj.gdata.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
        'Value',0,'Enable','on');
      obj.lbl1_state = SequentialModeState.ADJUST;
    end
    
    function labelMode1Accept(obj,tfSetLabelPos)
      % Enter accepted state (for current frame)
      if tfSetLabelPos
        obj.labelPosSetCurrFrame();
      end
      set(obj.gdata.tbAccept,'BackgroundColor',[0,0.4,0],'String','Accepted',...
        'Value',1,'Enable','on');
      obj.lbl1_state = SequentialModeState.ACCEPTED;
    end    
       
    function labelMode1NewFrame(obj,iFrm)
      % React to new frame. Set mode1 label state according to labelpos. If
      % a frame is not labeled, then start fresh in Label state. Otherwise,
      % start in Accepted state with saved labels.
      
      if exist('iFrm','var')==0
        iFrm = obj.currFrame;
      end
      assert(obj.nTrxOr1IfNoTrx==1);
      ITRX = 1;
      
      [tflabeled,lpos] = obj.labelMode1FrameIsLabeled(iFrm,ITRX);
      if tflabeled
        obj.lbl1_nPtsLabeled = obj.nLabelPoints;
        Labeler.assignCoords2Pts(lpos,obj.lbl1_ptsH,obj.lbl1_ptsTxtH);
        obj.lbl1_iPtMove = nan;
        obj.labelMode1Accept(false);
      else
        obj.labelMode1Label();
      end
    end
    
    function [tf,lpos] = labelMode1FrameIsLabeled(obj,iFrm,iTrx)
      lpos = obj.labeledpos(:,:,iFrm,iTrx);
      tfnan = isnan(lpos);
      assert(all(tfnan(:)) || ~any(tfnan(:)));
      tf = ~any(tfnan(:));
    end
    
    function bdfMode1Ax(obj,~,~)
      assert(obj.lbl1_state==SequentialModeState.LABEL);

      ax = obj.gdata.axes_curr;      

      nlbled = obj.lbl1_nPtsLabeled;
      if isinf(nlbled)
        warning('Labeler:mode1','All %d points labeled for this frame.',obj.nLabelPoints);
      elseif nlbled==obj.nLabelPoints
        assert(false); % adjustment mode only
      else % 0..nLabelPoints-1
        tmp = get(ax,'CurrentPoint');
        x = tmp(1,1);
        y = tmp(1,2);
        
        i = nlbled+1;
        set(obj.lbl1_ptsH(i),'XData',x,'YData',y);
        set(obj.lbl1_ptsTxtH(i),'Position',[x+obj.DT2P y+obj.DT2P]);        
        obj.lbl1_nPtsLabeled = i;
        
        if i==obj.nLabelPoints
          obj.labelMode1Adjust();
        end
      end
    end
    
    function bdfMode1Pt(obj,src,~)
      switch obj.lbl1_state
        case SequentialModeState.ADJUST
          obj.lbl1_iPtMove = get(src,'UserData');
      end
    end
    
    function wbmfMode1(obj,~,~)
      if obj.lbl1_state==SequentialModeState.ADJUST
        iPt = obj.lbl1_iPtMove;      
        if ~isnan(iPt) % should always be true
          ax = obj.gdata.axes_curr;
          tmp = get(ax,'CurrentPoint');
          pos = tmp(1,1:2);
          set(obj.lbl1_ptsH(iPt),'XData',pos(1),'YData',pos(2));
          pos(1) = pos(1) + obj.DT2P;
          set(obj.lbl1_ptsTxtH(iPt),'Position',pos);
        end
      end
    end
    
    function wbufMode1(obj,~,~)
      if obj.lbl1_state==SequentialModeState.ADJUST
        obj.lbl1_iPtMove = nan;
      end
    end
    
  end
  
  methods (Static)
    
    function assignCoords2Pts(lpos,hPts,hTxt)
      nPts = size(lpos,1);
      assert(size(lpos,2)==2);
      assert(isequal(nPts,numel(hPts),numel(hTxt)));
      
      for i = 1:nPts
        set(hPts(i),'XData',lpos(i,1),'YData',lpos(i,2));
        set(hTxt(i),'Position',[lpos(i,1)+Labeler.DT2P lpos(i,2)+Labeler.DT2P 1]);
      end
    end
    
  end
  
  %%
  
  methods (Hidden)
    
    % just put this in loadMovie?
    function newMovie(obj)
      movRdr = obj.movieReader;
      %movRdr.open();      
      nframes = movRdr.nframes;

      obj.initLabelPosLocked();
      
      if obj.hasTrx
        obj.currFrame = min([obj.trx.firstframe]);
      else
        obj.currFrame = 1;
      end
      obj.currTarget = 1;
      
      [obj.currIm,~] = movRdr.readframe(obj.currFrame);
      obj.prevIm = [];
%       handles.f_im = handles.f;
%       handles.imprev = handles.imcurr;
%       handles.fprev_im = handles.f;
      
      %#UI
      obj.minv = 0;
      obj.maxv = inf;
      %obj.minv = max(obj.minv,0);
      if isfield(movRdr.info,'bitdepth')
        obj.maxv = min(obj.maxv,2^movRdr.info.bitdepth-1);
      elseif isa(obj.currIm,'uint16')
        obj.maxv = min(2^16 - 1,obj.maxv);
      elseif isa(obj.currIm,'uint8')
        obj.maxv = min(obj.maxv,2^8 - 1);
      else
        obj.maxv = min(obj.maxv,2^(ceil(log2(max(obj.currIm(:)))/8)*8));
      end
      
      %#UI
      set(obj.gdata.axes_curr,'CLim',[obj.minv,obj.maxv],...
        'XLim',[.5,size(obj.currIm,2)+.5],...
        'YLim',[.5,size(obj.currIm,1)+.5]);
      set(obj.gdata.axes_prev,'CLim',[obj.minv,obj.maxv],...
        'XLim',[.5,size(obj.currIm,2)+.5],...
        'YLim',[.5,size(obj.currIm,1)+.5]);
      zoom(obj.gdata.axes_curr,'reset');
      zoom(obj.gdata.axes_prev,'reset');
      
      %#UI
      if obj.hasTrx
        obj.videoZoomTarget(obj.fsizeini);
      end            

      %#UI
      sliderstep = [1/(nframes-1),min(1,100/(nframes-1))];
      set(obj.gdata.slider_frame,'Value',0,'SliderStep',sliderstep);      
    end
    
    function setFrame(obj,frm)
      if obj.currFrame~=frm
        [obj.currIm,~] = obj.movieReader.readframe(frm);
        obj.currFrame = frm;
      end
      
      if frm > 1
        if obj.prevFrame~=frm-1
          [obj.prevIm,~] = obj.movieReader.readframe(frm-1);
          obj.prevFrame = frm-1;
        end
        set(obj.gdata.image_prev,'CData',obj.prevIm);
      else
        set(obj.gdata.image_prev,'CData',0);
      end
      
      set(obj.gdata.image_curr,'CData',obj.currIm);
      if obj.hasTrx
        obj.videoZoomTarget();
      end
        
      %#UI
      set(obj.gdata.slider_frame,'Value',(frm-1)/(obj.nframes-1));
      set(obj.gdata.edit_frame,'String',num2str(frm));

      if ~isempty(obj.labelMode)
        switch obj.labelMode
          case LabelMode.SEQUENTIAL
            obj.labelMode1NewFrame();
        end
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
      validateattributes(iTgt,'numeric',{'positive' 'integer' '<=' obj.nTrx});
      obj.currTarget = iTgt;
      set(obj.gdata.edit_num,'String',iTgt);
      obj.videoZoomTarget();
      
      %XXX template?
    end
    
    function clearTarget(obj)
      obj.currTarget = nan;      
      %XXX ZOOM      
    end
    
    %#UI
    function videoZoomTarget(obj,zoomRadius)
      % videoZoomTarget(obj,zoomRadius)
      % 
      % zoomRadius: optional, scalar double. Radius of zoom box around
      % target.
      
      if exist('zoomRadius','var')==0
        xsz = get(obj.gdata.axes_curr,'XLim');
        xsz = xsz(2)-xsz(1);
        ysz = get(obj.gdata.axes_curr,'YLim');
        ysz = ysz(2)-ysz(1);
        zoomRadius = max([xsz ysz])/2;
      end
      
      assert(obj.hasTrx);
      
      currTrx = obj.currTrx;       
      if obj.currFrame < currTrx.firstframe || obj.currFrame > currTrx.endframe
        warndlg('This animal does not exist for the current frame. Frame location will not be updated');
      else
        trxndx = obj.currFrame - currTrx.firstframe + 1;
        curlocx = currTrx.x(trxndx);
        curlocy = currTrx.y(trxndx);
        xlim = [curlocx-zoomRadius,curlocx+zoomRadius];
        ylim = [curlocy-zoomRadius,curlocy+zoomRadius];
        set(obj.gdata.axes_curr,'XLim',xlim,'YLim',ylim);
        set(obj.gdata.axes_prev,'XLim',xlim,'YLim',ylim);
      end      
    end
    
    %#UI No really
    function updateLockedButton(obj)
      disp('UPDATELOCK TODO');
%       btn = obj.gdata.togglebutton_lock;
%       if obj.labelsLocked(obj.currFrame,obj.currTarget)
%         set(btn,'BackgroundColor',[.6,0,0],'String','Locked','Value',1);
%       else
%         set(btn,'BackgroundColor',[0,.6,0],'String','Unlocked','Value',0);
%       end
%       setButtonImage(btn);
    end
    
    function showPreviousLabels(obj)
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
    end   
    
  end

end
