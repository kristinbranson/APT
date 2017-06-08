classdef InfoTimeline < handle
  
  properties
    lObj % scalar Labeler handle
    hAx % scalar handle to timeline axis
    hCurrFrame % scalar line handle current frame
    hMarked %scalar line handle, indicates marked frames

    hZoom % zoom handle for hAx
    hPan % pan handle "

    hPts % [npts] line handles    
    npts % number of label points in current movie/timeline
    nfrm % number of frames "
    
    listeners % [nlistener] col cell array of labeler prop listeners

    prefs % Timeline pref substruct
    tracker
  end
  properties (SetObservable)
    props % [npropx3]. Col 1: pretty/display name. Col 2: Type, eg 'Labels', 'Labels2' or 'Tracks'. Col3: non-pretty name/id
    curprop % str
  end  
  properties
    jumpThreshold
    jumpCondition
    
    selectH % patch handle, mouse select
  end
  properties (SetAccess=private)
    selectFrmAtInit % scalar, frm at start of select action
    selectInProg % scalar logical
  end
  properties (SetObservable)
    selectModeOn % scalar logical, if true, mouse-click-drag will select time intervals
  end
    
  methods
    function set.selectModeOn(obj,v)
      obj.selectInProg = false; %#ok<MCSUP>
      obj.selectModeOn = v;
      if ~obj.selectModeOn
        set(obj.selectH,'XData',[nan nan nan nan],...
          'YData',[nan nan nan nan],'ZData',[1 1 1 1]); %#ok<MCSUP>
      end
    end
  end
  
  methods
    
    function obj = InfoTimeline(labeler,ax)
      obj.lObj = labeler;
      ax.Color = [0 0 0];
      ax.ButtonDownFcn = @(src,evt)obj.cbkBDF(src,evt);
      hold(ax,'on');
      obj.hAx = ax;
      obj.hCurrFrame = plot(ax,[nan nan],[nan nan],'-','Color',[1 1 1]);
      obj.hMarked = plot(ax,[nan nan],[nan nan],'-','Color',[1 1 0]);      
      
      fig = ax.Parent;
      hZ = zoom(fig);
      setAxesZoomMotion(hZ,ax,'horizontal');
      obj.hZoom = hZ;
      hP = pan(fig);
      setAxesPanMotion(hP,ax,'horizontal');
      obj.hPan = hP;
      
      obj.hPts = [];
      obj.npts = nan;
      obj.nfrm = nan;
            
      listeners = cell(0,1);
      listeners{end+1,1} = addlistener(labeler,...
        {'labeledpos','labeledposMarked','labeledpostag'},...
        'PostSet',@obj.cbkLabelUpdated);
      listeners{end+1,1} = addlistener(labeler,...
        'labelMode','PostSet',@obj.cbkLabelMode);      
      obj.listeners = listeners;
      
      obj.prefs = [];
      obj.tracker = [];
      
      obj.props = {};
      props1(:,1) = {'x','y','dx','dy','|dx|','|dy|','occluded'};
      props1(:,2) = {'Labels'};
      props1(:,3) = {'x','y','dx','dy','|dx|','|dy|','occluded'};
      props2(:,1) = {'x (pred)','y (pred)','dx (pred)','dy (pred)','|dx| (pred)','|dy| (pred)'};
      props2(:,2) = {'Labels2'};
      props2(:,3) = {'x','y','dx','dy','|dx|','|dy|'};
      obj.props = [props1;props2];
      obj.curprop = obj.props{1,1};
      
      obj.jumpThreshold = nan;
      obj.jumpCondition = nan;
      
      obj.selectH = patch([nan nan nan nan],[nan nan nan nan],[2 2 2 2],...
        'w','parent',ax,'LineWidth',0.25,'FaceAlpha',0.40,'HitTest','off');
      obj.selectModeOn = false;
      obj.selectInProg = false;
      obj.selectFrmAtInit = nan;
    end
    
    function delete(obj)
      deleteValidHandles(obj.hCurrFrame);
      obj.hCurrFrame = [];
      deleteValidHandles(obj.hMarked);
      obj.hMarked = [];
      if ~isempty(obj.hZoom)
        delete(obj.hZoom);
      end
      if ~isempty(obj.hPan)
        delete(obj.hPan);
      end
      deleteValidHandles(obj.hPts);
      obj.hPts = [];
      cellfun(@delete,obj.listeners);
      obj.listeners = [];
      deleteValidHandles(obj.selectH);
      obj.selectH = [];
    end
    
  end
  
  methods
    function cbkBDF(obj,src,evt) %#ok<INUSD>
      pos = get(obj.hAx,'CurrentPoint');
      frm = round(pos(1,1));
      frm = min(max(frm,1),obj.nfrm);
      if obj.selectModeOn
        obj.selectFrmAtInit = obj.lObj.currFrame;
        obj.selectInProg = true;
        ylims = obj.hAx.YLim;
        obj.selectH.Vertices = [frm ylims(1) 1;frm ylims(2) 1;frm+0.3 ylims(2) 1;frm+0.3 ylims(1) 1];
      else
        obj.lObj.setFrame(frm);
      end
    end
    function cbkWBMF(obj,src,evt) %#ok<INUSD>
      if obj.selectInProg
        ax = obj.hAx;
        pos = get(ax,'CurrentPoint');
        ypos = pos(1,2);
        frm = round(pos(1));
        frm = min(max(frm,1),obj.nfrm);
        xl = ax.XLim;
        yl = ax.YLim;
        if frm>xl(1) && frm < xl(2) && ypos>yl(1) && ypos<yl(2)
          obj.selectH.Vertices(3:4,:) = [frm+0.3 yl(2) 1;frm+0.3 yl(1) 1];
          obj.lObj.setFrame(frm);
        end
      end
    end
    function cbkWBUF(obj,src,evt) %#ok<INUSD>
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
    
    function cbkLabelMode(obj,src,evt) %#ok<INUSD>
      if obj.lObj.labelMode==LabelMode.ERRORCORRECT
        set(obj.hMarked,'Visible','on');
      else
        set(obj.hMarked,'Visible','off');
      end
    end
  end
  
  methods
    
    function initNewMovie(obj)
      % react to new current movie in Labeler

      obj.prefs = obj.lObj.projPrefs.InfoTimelines;
      obj.npts = obj.lObj.nLabelPoints;
      obj.nfrm = obj.lObj.nframes;

      deleteValidHandles(obj.hPts);
      obj.hPts = gobjects(obj.npts,1);
      colors = obj.lObj.labelPointsPlotInfo.Colors;
      ax = obj.hAx;
      for i=1:obj.npts
        obj.hPts(i) = plot(ax,nan,i,'.','linestyle','-','Color',colors(i,:));
      end
      ax.XColor = obj.prefs.XColor;
      ax.XTick = 0:obj.prefs.dXTick:obj.nfrm;
      ax.YLim = [0 1];
      
      set(obj.hCurrFrame,'XData',[nan nan],'YData',[0 1],'ZData',[1 1]);
    end
    
    function setTracker(obj,tracker)
      obj.tracker = tracker;
      newprops = tracker.propList();
      for ndx = 1:numel(newprops)
        obj.props(end+1,1) = newprops{count};
        obj.props(end,2) = 'Tracks';
        obj.props(end,3) = newprops{count};
      end
    end
    
    function setLabelsFull(obj)
      % Set .hPts, .hMarked; adjust .hAx.YLim to fit data; update
      % .hCurrFrame
      
      if isnan(obj.npts), return; end
      
      lpos = obj.getDataCurrMovTgt();
      for i=1:obj.npts
        set(obj.hPts(i),'XData',1:size(lpos,2),'YData',lpos(i,:));
      end
      
      y1 = min(lpos(:));
      y2 = max(lpos(:));

      ydel = (y2-y1)*0.01;
      ylims = [y1-ydel y2+ydel];
      if any(isnan(ylims))
        ylims = [0 1];
      end
      if ylims(2)-ylims(1)==0
        ylims(2) = ylims(1) + 0.000001;
      end
      set(obj.hAx,'YLim',ylims);
      set(obj.hCurrFrame,'XData',[nan nan],'YData',ylims,'ZData',[1 1]);
      
      markedFrms = find(any(obj.getMarkedDataCurrMovTgt(),1));
      xxm = repmat(markedFrms,[3 1]);
      xxm = xxm(:)+0.05; 
      % slightly off so that both current frame and labeled frame are both
      % visible.
      yym = repmat([ylims nan],[1 size(markedFrms,2)]);
      set(obj.hMarked,'XData',xxm(:),'YData',yym(:));
      obj.newFrame(obj.lObj.currFrame);
    end
    
    function setLabelsFrame(obj,frm)
      % frm: [n] frame indices. Optional. If not supplied, defaults to
      % labeler.currFrame
            
%       if exist('frm','var')==0
%         frm = obj.lObj.currFrame;
%       end
      
      % AL20170607: This call looks unnec now (can be no-op) given
      % cbkLabelUpdated
      
      lpos = obj.getDataCurrMovTgt();
      for i=1:obj.npts
        h = obj.hPts(i);
        set(h,'XData',1:size(lpos,2),'YData',lpos(i,:));
      end
    end
    
    function newFrame(obj,frm)
      if isnan(obj.npts), return; end
      
%       AL20170607: why update .hPts?
%       lpos = obj.getDataCurrMovTgt();
%       for i=1:obj.npts
%         h = obj.hPts(i);
%         set(h,'XData',1:size(lpos,2),'YData',lpos(i,:));
%       end
      
      if ~obj.selectModeOn
        r = obj.prefs.FrameRadius;
        x0 = frm-r; %max(frm-r,1);
        x1 = frm+r; %min(frm+r,obj.nfrm);
        obj.hAx.XLim = [x0 x1];
        set(obj.hCurrFrame,'XData',[frm frm]);
      else
        % AL20170607: don't understand this, equivalent to no-op?
        set(obj.hCurrFrame,'XData',[obj.selectFrmAtInit obj.selectFrmAtInit]);
      end
    end
    
    function newTarget(obj)
      obj.setLabelsFull();
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
      hprop = uicontrol('Style','Text',...
                   'String','drasdgx','Value',1,'Position',[10,ibottom-2,90,20],...
                   'HorizontalAlignment','left');
      hprope = get(hprop,'Extent');
      st = hprope(3) + 15;
      hCondition = uicontrol('Style','popupmenu',...
                   'String',{'>','<='},'Value',1,'Position',[st,ibottom,40,20]);
      hval = uicontrol('Style','edit',...
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
        if isnan(tr)
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
    function props = getPropsDisp(obj)
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
  methods (Access=private)
    function lpos = getDataCurrMovTgt(obj)
      % lpos: [nptsxnfrm]
      
      pndx = find(strcmp(obj.props(:,1),obj.curprop));
      ptype = obj.props{pndx,2};
      pcode = obj.props{pndx,3};
      
      switch ptype
        case {'Labels' 'Labels2'}
          iMov = obj.lObj.currMovie;
          iTgt = obj.lObj.currTarget;
          if iMov>0
            if strcmp(ptype,'Labels')
              lObjlpos = obj.lObj.labeledpos{iMov};
            else
              lObjlpos = obj.lObj.labeledpos2{iMov};
            end
            switch pcode
              case 'x'
                lpos = squeeze(lObjlpos(:,1,:,iTgt));
              case 'y'
                lpos = squeeze(lObjlpos(:,2,:,iTgt));
              case 'dx'
                lpos = squeeze(lObjlpos(:,1,:,iTgt));
                lpos = diff(lpos,1,2);
                lpos(:,end+1) = nan;
              case 'dy'
                lpos = squeeze(lObjlpos(:,2,:,iTgt));
                lpos = diff(lpos,1,2);
                lpos(:,end+1) = nan;
              case '|dx|'
                lpos = squeeze(lObjlpos(:,1,:,iTgt));
                lpos = abs(diff(lpos,1,2));
                lpos(:,end+1) = nan;
              case '|dy|'
                lpos = squeeze(lObjlpos(:,2,:,iTgt));
                lpos = abs(diff(lpos,1,2));
                lpos(:,end+1) = nan;
              case 'occluded'
                curd = obj.lObj.labeledpostag{iMov}(:,:,iTgt);
                lpos =  double(strcmp(curd,'occ'));
              otherwise
                warndlg('Unknown property to display');
                lpos = nan(obj.npts,1);
            end
          else
            lpos = nan(obj.npts,1);
          end   
        case 'Tracks'
          lpos = obj.tracker.getPropValues(pcode);
          szassert(lpos,[obj.npts 1]);
      end
    end
    
    function lpos = getMarkedDataCurrMovTgt(obj)
      % lpos: [nptsxnfrm]
      
      iMov = obj.lObj.currMovie;
      if iMov>0
        iTgt = obj.lObj.currTarget;
        lpos = squeeze(obj.lObj.labeledposMarked{iMov}(:,:,iTgt));
      else
        lpos = false(obj.lObj.nLabelPoints,1);
      end
    end
    
    function nxtFrm = findFrame(obj,dr,curFr)
      % Finds the next or previous frame which satisfy conditions.
      % dr = 0 is back, 1 is forward
      nxtFrm = nan;
      if isnan(obj.jumpThreshold)
        warndlg('Threhold value is not for navigation');
        obj.thresholdGUI();
        if isnan(obj.jumpThreshold)
          return;
        end
      end
      
      data = obj.getDataCurrMovTgt();
      if obj.jumpCondition > 0
        locs = any(data>obj.jumpThreshold,1);
      else
        locs = any(data<=obj.jumpThreshold,1);
      end
      
      if dr > 0.5
        locs = locs(curFr:end);
        nxtlocs = find( (~locs(1:end-1))&(locs(2:end)),1);
        if isempty(nxtlocs)
          return;
        end
        nxtFrm = curFr + nxtlocs - 1;
      else
        locs = locs(1:curFr);
        nxtlocs = find( (locs(1:end-1))&(~locs(2:end)),1,'last');
        if isempty(nxtlocs)
          return;
        end
        nxtFrm = nxtlocs;
      end
    end
  end
  
end
