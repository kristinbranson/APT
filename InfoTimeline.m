classdef InfoTimeline < handle
  
  properties
    prefs % Timeline pref substruct
    
    enable % Enable only in tracker is enabled and the mode is error correction.

    lObj % scalar Labeler handle
    hAx % scalar handle to timeline axis
    hCurrFrame % scalar line handle current frame
    hMarked %scalar line handle, indicates marked frames

    hZoom % zoom handle for hAx
    hPan % pan handle "

    hIm % image handle (used when imMode==true)
    hPts % [npts] line handles (used when imMode==false)
    
    npts % number of label points in current movie/timeline
    nfrm % number of frames "
    
    selectH % patch handle, mouse select
    selectFrmAtInit
    
    ylims
    colors
    
    listeners % listeners
    
    tracker
    
    jumpThreshold
    jumpCondition
    
  end
  properties (SetAccess=private)
    selectInProg % scalar logical
  end
  
  properties (SetObservable)
    selectModeOn % scalar logical, if true, mouse-click-drag will select time intervals    
    props % list of properties to show
    curprop
  end  
  
  methods 
    function set.selectModeOn(obj,v)
      obj.selectInProg = false; %#ok<MCSUP>
      obj.selectModeOn = v;
      if ~obj.selectModeOn,
        set(obj.selectH,'XData',[nan nan nan nan],'YData',[nan nan nan nan],'ZData',[1 1 1 1]);
      end
    end
    
  end
  
  methods
    
    function obj = InfoTimeline(labeler,ax)
      obj.lObj = labeler;
      obj.hAx = ax;
      
      obj.hCurrFrame = plot(ax,[nan nan],[nan nan],'-','Color',[1 1 1]);
      hold(ax,'on');
      ax.Color = [0 0 0];
      ax.ButtonDownFcn = @(src,evt)obj.cbkBDF(src,evt);  

      obj.hMarked = plot(ax,[nan nan],[nan nan],'-','Color',[1 1 0]);      
      
      obj.hIm = [];
      obj.hPts = [];
      obj.npts = nan;
      obj.nfrm = nan;
      
      obj.selectH = patch([nan nan nan nan],[nan nan nan nan],[2 2 2 2],...
        'w','parent',ax,'LineWidth',0.25,'FaceAlpha',0.40,'HitTest','off');
      obj.selectInProg = false;
      obj.selectModeOn = false;
      
      fig = ax.Parent;
      hZ = zoom(fig);
      setAxesZoomMotion(hZ,ax,'horizontal');
      obj.hZoom = hZ;
      hP = pan(fig);
      setAxesPanMotion(hP,ax,'horizontal');
      obj.hPan = hP;
            
      % callback listeners.
      listeners = cell(0,1);
      listeners{end+1} = addlistener(labeler,...
        {'labeledpos','labeledposMarked','labeledpostag'},...
        'PostSet',@obj.cbkLabelUpdated);
      listeners{end+1} = addlistener(labeler,...
        'labelMode','PostSet',@obj.cbkLabelMode);
      
      obj.listeners = listeners;
      
      obj.props = {};
      obj.props(:,1) = {'x','y','dx','dy','|dx|','|dy|','occluded'};
      obj.props(:,2) = {'Labels'};
      obj.curprop = obj.props{1,1};
      
      obj.jumpThreshold = nan;
      obj.jumpCondition = nan;
    end
    
    function delete(obj)
      deleteValidHandles(obj.hCurrFrame);
      obj.hCurrFrame = [];
      if ~isempty(obj.hZoom)
        delete(obj.hZoom);
      end
      if ~isempty(obj.hPan)
        delete(obj.hPan);
      end
      deleteValidHandles(obj.hIm);
      obj.hIm = [];
      deleteValidHandles(obj.hPts);
      obj.hPts = [];
      deleteValidHandles(obj.selectH);
      obj.selectH = [];
      deleteValidHandles(obj.hMarked);
      obj.hMarked = [];
    end
    
  end
  
  methods
    function cbkBDF(obj,src,evt)
      obj.selectFrmAtInit = obj.lObj.currFrame;
      pos = get(obj.hAx,'CurrentPoint');
      frm = round(pos(1,1));
      frm = min(max(frm,1),obj.nfrm);
      if obj.selectModeOn
        obj.selectInProg = true;
        obj.selectH.Vertices = [frm obj.ylims(1) 1;frm obj.ylims(2) 1;frm+0.3 obj.ylims(2) 1;frm+0.3 obj.ylims(1) 1];
      else
        obj.lObj.setFrame(frm);
      end
    end
    function cbkWBMF(obj,src,evt)
      if obj.selectInProg
        ax = obj.hAx;
        pos = get(ax,'CurrentPoint');
        ypos = pos(1,2);
        frm = round(pos(1));
        frm = min(max(frm,1),obj.nfrm);
        xl = ax.XLim;
        yl = ax.YLim;
        if frm>xl(1) && frm < xl(2) && ypos>yl(1) && ypos<yl(2)
          obj.selectH.Vertices(3:4,:) = [frm+0.3 obj.ylims(2) 1;frm+0.3 obj.ylims(1) 1];
          obj.lObj.setFrame(frm);
        end
      end
    end
    function cbkWBUF(obj,src,evt)
      if obj.selectInProg
        verts = obj.selectH.Vertices;
        frms = sort(round(verts(2:3,1)));
        frms = min(max(frms,1),obj.nfrm);
        frms = frms(1):frms(2);        
        fprintf('Selected %d frames: [%d,%d]\n',numel(frms),frms(1),frms(end));
        obj.lObj.setSelectedFrames(frms);
        obj.lObj.setFrame(obj.selectFrmAtInit);

        obj.selectInProg = false;
      end
    end
    
    function cbkLabelMode(obj,src,evt)
      if obj.lObj.labelMode == LabelMode.ERRORCORRECT,
        set(obj.hMarked,'Visible','on');
      else
        set(obj.hMarked,'Visible','off');
      end
    end
  end
  
  methods
    
    function initNewMovie(obj)
      obj.prefs = obj.lObj.projPrefs.InfoTimelines;

      % react to new current movie in Labeler      
      obj.npts = obj.lObj.nLabelPoints;
      obj.nfrm = obj.lObj.nframes;
      %obj.setLabelsFull(false(obj.npts,obj.nfrm));      
      obj.colors = obj.lObj.labelPointsPlotInfo.Colors;

      ax = obj.hAx;
      ax.XColor = obj.prefs.XColor;
      ax.XTick = 0:obj.prefs.dXTick:obj.nfrm;
      ax.YLim = [0 1];

      deleteValidHandles(obj.hIm);
      deleteValidHandles(obj.hPts);
      obj.hPts = gobjects(obj.npts,1);
      for i=1:obj.npts
        obj.hPts(i) = plot(ax,nan,i,'.','linestyle','-','Color',obj.colors(i,:));
      end
      
      set(obj.hCurrFrame,'XData',[nan nan],'YData',[0 1],'ZData',[1 1]);
    end
    
    function setTracker(obj,tracker)
      obj.tracker = tracker;
      newprops = tracker.propList();
      for ndx = 1:numel(newprops),
        obj.props(end+1,1) = newprops{count};
        obj.props(end,2) = 'Tracks';
      end
    end
    
    function setLabelsFull(obj)
      % labeledpos for current frame is read from Labeler.
      
      if isnan(obj.npts), return; end
      lpos = obj.getData();
      for i=1:obj.npts
        set(obj.hPts(i),'XData',1:size(lpos,2),'YData',lpos(i,:));
      end
      
      y1 = min(lpos(:));
      y2 = max(lpos(:));
      
      obj.ylims = nan(1,2);
      obj.ylims(1) = y1-(y2-y1)*0.01;
      obj.ylims(2) = y2+(y2-y1)*0.01;
      if any(isnan(obj.ylims)),
        obj.ylims = [0 1];
      end
      if obj.ylims(2)-obj.ylims(1)==0
        obj.ylims(2) = obj.ylims(1) + 0.000001;
      end
      set(obj.hAx,'YLim',obj.ylims);
      set(obj.hCurrFrame,'XData',[nan nan],'YData',obj.ylims,'ZData',[1 1]);
      
      marked = find(any(obj.getMarkedData(),1));
      xxm = repmat(marked,[3 1]);
      xxm = xxm(:)+0.05; 
      % slightly off so that both current frame and labeled frame are both
      % visible.
      yym = repmat([obj.ylims nan],[1 size(marked,2)]);
      set(obj.hMarked,'XData',xxm(:),'YData',yym(:));
      obj.setCurrFrame(obj.lObj.currFrame)
    end
    
    function setLabelsFrame(obj,frm)
      % frm: [n] frame indices. Optional. If not supplied, defaults to
      % labeler.currFrame
            
      if exist('frm','var')==0
        frm = obj.lObj.currFrame;
      end
      
      lpos = obj.getData();      
      
      for i=1:obj.npts
        h = obj.hPts(i);
        set(h,'XData',1:size(lpos,2),'YData',lpos(i,:));
      end
    end
    
    function setCurrFrame(obj,frm)
      
      if isnan(obj.npts), return; end
      lpos = obj.getData();
      for i=1:obj.npts
        h = obj.hPts(i);
        set(h,'XData',1:size(lpos,2),'YData',lpos(i,:));
      end
      
      if ~obj.selectModeOn
        r = obj.prefs.FrameRadius;
        x0 = max(frm-r,1);
        x1 = min(frm+r,obj.nfrm);
        obj.hAx.XLim = [x0 x1];
        set(obj.hCurrFrame,'XData',[frm frm]);
      else
        set(obj.hCurrFrame,'XData',[obj.selectFrmAtInit obj.selectFrmAtInit]);
      end
    end
    
    function cbkLabelUpdated(obj,src,~)
      if ~obj.lObj.isinit
        obj.setLabelsFull;
      end
    end
    
    function setJumpParams(obj)
      % GUI to get jump parameters,
      f = figure('Visible','on','Position',[360,500,200,90],...
        'MenuBar','None','ToolBar','None');
      tbottom = 70;
      uicontrol('Style','text',...
                   'String','Jump To:','Value',1,'Position',[10,tbottom,190,20]);
      
      ibottom = 50;           
      hprop    = uicontrol('Style','Text',...
                   'String','drasdgx','Value',1,'Position',[10,ibottom-2,90,20],...
                   'HorizontalAlignment','left');
      hprope = get(hprop,'Extent');
      st = hprope(3) + 15;
      hCondition    = uicontrol('Style','popupmenu',...
                   'String',{'>','<='},'Value',1,'Position',[st,ibottom,40,20]);
      hval    = uicontrol('Style','edit',...
                   'String','0','Position',[st+55,ibottom-2,30,20]);           
      bbottom = 10;


      uicontrol('Style','pushbutton',...
                   'String','Cancel','Position',[30,bbottom,60,30],'Callback',{fcancel,f});
      uicontrol('Style','pushbutton',...
                   'String','Done','Position',[110,bbottom,60,30],...
                   'Callback',{fapply,f,hCondition,hval,obj});
                 
      function fcancel(~,~,f)
        delete(f);
      end
      
      function fapply(~,~,f,hCondition,hval,obj) 
        tr = str2double(get(hval,'String'));
        if isnan(tr),
          warndlg('Enter valid numerical value');
          return;
        end
        obj.jumpThreshold = tr;
        obj.jumpCondition = 2*get(hCondition,'Value')-3;
        delete(f);
      end
    end
    
  end
  
  methods %getters setters
    function props = getProps(obj)
      props = obj.props(:,1);
    end
    function props = getCurProp(obj)
      props = obj.curprop;
    end
    function setCurProp(obj,newprop)
      obj.curprop = newprop;
      obj.setLabelsFull();
    end
    
  end
  
  
  %% Private methods
  methods(Access = private)
    function lpos = getData(obj)
      
      pndx = find(strcmp(obj.props(:,1),obj.curprop));
      switch obj.props{pndx,2},
        
        case 'Labels'
      
          if obj.lObj.currMovie>0,
            if obj.lObj.hasTrx,
              currTrxId = obj.lObj.currTrxId;
            else
              currTrxId = 1;
            end
            
            switch obj.props{pndx,1}
              case 'x',
                lpos = squeeze(obj.lObj.labeledpos{obj.lObj.currMovie}(:,1,:,currTrxId));
              case 'y',
                lpos = squeeze(obj.lObj.labeledpos{obj.lObj.currMovie}(:,2,:,currTrxId));
              case 'dx',
                lpos = squeeze(obj.lObj.labeledpos{obj.lObj.currMovie}(:,1,:,currTrxId));
                lpos = lpos(:,2:end)-lpos(:,1:end-1);
                lpos(:,end+1) = nan;
              case 'dy',
                lpos = squeeze(obj.lObj.labeledpos{obj.lObj.currMovie}(:,2,:,currTrxId));
                lpos = lpos(:,2:end)-lpos(:,1:end-1);
                lpos(:,end+1) = nan;
              case '|dx|',
                lpos = squeeze(obj.lObj.labeledpos{obj.lObj.currMovie}(:,1,:,currTrxId));
                lpos = abs(lpos(:,2:end)-lpos(:,1:end-1));
                lpos(:,end+1) = nan;
              case '|dy|',
                lpos = squeeze(obj.lObj.labeledpos{obj.lObj.currMovie}(:,2,:,currTrxId));
                lpos = abs(lpos(:,2:end)-lpos(:,1:end-1));
                lpos(:,end+1) = nan;
              case 'occluded',
                curd = obj.lObj.labeledpostag{obj.lObj.currMovie}(:,:,currTrxId);
                lpos =  double(strcmp(curd,'occ'));
              otherwise,
                warndlg('Unknown property to display');
                lpos = nan(obj.lObj.nLabelPoints,1);
            end
          else
            lpos = nan(obj.lObj.nLabelPoints,1);
          end
        
        case 'Tracks',
          lpos = obj.tracker.getPropValues(obj.props{pndx,1});
        
      end      
    end
    
    function lpos = getMarkedData(obj)
      
      if obj.lObj.currMovie>0,
        if obj.lObj.hasTrx,
          currTrxId = obj.lObj.currTrxId;
        else
          currTrxId = 1;
        end
      end
      lpos = squeeze(obj.lObj.labeledposMarked{obj.lObj.currMovie}(:,:,currTrxId));

    end
    
    function nxtFrm = findFrame(obj,dr,curFr)
      % Finds the next or previous frame which satisfy conditions.
      % dr = 0 is back, 1 is forward
      nxtFrm = nan;
      if isnan(obj.jumpThreshold),
        warndlg('Threhold value is not for navigation');
        obj.thresholdGUI();
        if isnan(obj.jumpThreshold),
          return;
        end
      end
      
      data = obj.getData();
      if obj.jumpCondition > 0,
        locs = any(data>obj.jumpThreshold,1);
      else
        locs = any(data<=obj.jumpThreshold,1);
      end
      
      if dr > 0.5,
        locs = locs(curFr:end);
        nxtlocs = find( (~locs(1:end-1))&(locs(2:end)),1);
        if isempty(nxtlocs),
          return;
        end
        nxtFrm = curFr + nxtlocs - 1;
      else
        locs = locs(1:curFr);
        nxtlocs = find( (locs(1:end-1))&(~locs(2:end)),1,'last');
        if isempty(nxtlocs),
          return;
        end
        nxtFrm = nxtlocs;
      end
      
    
    end
  end
  
end
    
    
    