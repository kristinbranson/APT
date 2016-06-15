classdef LabelTimeline < handle
  
  properties
    prefs % Timeline pref substruct

    lObj % scalar Labeler handle
    hAx % scalar handle to timeline axis
    hCurrFrame % scalar line handle current frame

    hZoom % zoom handle for hAx
    hPan % pan handle "

    immode % scalar logical
    hIm % image handle (used when imMode==true)
    hPts % [npts] line handles (used when imMode==false)
    
    npts % number of label points in current movie/timeline
    nfrm % number of frames "
    
    selectH % patch handle, mouse select
  end
  properties (SetAccess=private)
    selectInProg % scalar logical
  end
  
  properties (SetObservable)
    selectModeOn % scalar logical, if true, mouse-click-drag will select time intervals    
  end  
  
  methods 
    function set.selectModeOn(obj,v)
      obj.selectInProg = false; %#ok<MCSUP>
      obj.selectModeOn = v;
    end
  end
  
  methods
    
    function obj = LabelTimeline(labeler,ax,tfimmode)
      obj.prefs = labeler.projPrefs.Timelines;

      obj.lObj = labeler;
      obj.hAx = ax;
      fig = ax.Parent;
      hZ = zoom(fig);
      hZ.Motion = 'horizontal';
      setAllowAxesZoom(hZ,ax,1);
      obj.hZoom = hZ;
      hP = pan(fig);
      hP.Motion = 'horizontal';
      setAllowAxesPan(hP,ax,1);
      obj.hPan = hP;
      
      obj.hCurrFrame = plot(ax,[nan nan],[nan nan],'-','Color',[1 1 1]);
      hold(ax,'on');
      ax.Color = [0 0 0];
      ax.XColor = obj.prefs.XColor;
      ax.ButtonDownFcn = @(src,evt)obj.cbkBDF(src,evt);  
      
      obj.immode = tfimmode;
      obj.hIm = [];
      obj.hPts = [];
      obj.npts = nan;
      obj.nfrm = nan;
      
      obj.selectH = patch([nan nan nan nan],[nan nan nan nan],[1 1 1 1],...
        'w','parent',ax,'LineWidth',0.25,'FaceAlpha',0.40,'HitTest','off');
      obj.selectInProg = false;
      obj.selectModeOn = false;
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
    end
    
  end
  
  methods
    function cbkBDF(obj,src,evt)
      pos = get(obj.hAx,'CurrentPoint');
      frm = round(pos(1,1));
      frm = min(max(frm,1),obj.nfrm);
      if obj.selectModeOn
        obj.selectInProg = true;
        obj.selectH.Vertices = [frm 0 1;frm obj.npts+1 1;frm obj.npts+1 1;frm 0 1];
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
          obj.selectH.Vertices(3:4,:) = [frm obj.npts+1 1;frm 0 1];
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
        
        obj.selectInProg = false;
      end
    end
  end
  
  methods
    
    function initNewMovie(obj)
      % react to new current movie in Labeler      
      obj.npts = obj.lObj.nLabelPoints;
      obj.nfrm = obj.lObj.nframes;
      %obj.setLabelsFull(false(obj.npts,obj.nfrm));      
      
      ax = obj.hAx;
      ax.XTick = 0:obj.prefs.dXTick:obj.nfrm;
      ax.YLim = [0 obj.npts+1];

      deleteValidHandles(obj.hIm);
      deleteValidHandles(obj.hPts);
      if obj.immode
        obj.hIm = imagesc(zeros(obj.npts,obj.nfrm),'Parent',ax,'HitTest','off');
        obj.hIm.CDataMapping = 'direct';
        colors = [[0 0 0]; obj.lObj.labelPointsPlotInfo.Colors];
        colormap(ax,colors);
      else      
        obj.hPts = gobjects(obj.npts,1);
        for i=1:obj.npts
          obj.hPts(i) = plot(ax,nan,i,'.','linestyle','none');
        end
      end
      
      set(obj.hCurrFrame,'XData',[nan nan],'YData',[0 obj.npts+1],'ZData',[1 1]);
    end
    
    function setLabelsFull(obj,lblTF)
      % lblTF: [npts x nfrm] logical. Optional, if not specified,
      % labeledpos for current frame is read from Labeler.
      
      if exist('lblTF','var')==0      
        lpos = obj.lObj.labeledpos{obj.lObj.currMovie}; % [npt x d x nfrm]
        lblTF = squeeze(all(~isnan(lpos),2));
      end
      assert(islogical(lblTF) && isequal(size(lblTF),[obj.npts obj.nfrm]));
      if obj.immode        
        obj.hIm.CData = bsxfun(@times,lblTF,(1:obj.npts)')+1;
      else
        for i=1:obj.npts
          idx = find(lblTF(i,:));
          set(obj.hPts(i),'XData',idx,'YData',repmat(i,1,numel(idx)));
        end
      end
    end
    
    function setLabelsFrame(obj,frm)
      % frm: [n] frame indices. Optional. If not supplied, defaults to
      % labeler.currFrame
            
      if exist('frm','var')==0
        frm = obj.lObj.currFrame;
      end
      
      lpos = obj.lObj.labeledpos{obj.lObj.currMovie}(:,:,frm,:);
      lblTF = squeeze(all(~isnan(lpos),2));      
      assert(islogical(lblTF) && isequal(size(lblTF),[obj.npts numel(frm)]));
      if obj.immode
        obj.hIm.CData(:,frm) = bsxfun(@times,lblTF,(1:obj.npts)')+1;
      else 
        for i=1:obj.npts
          h = obj.hPts(i);
          x = h.XData;
          tf = false(1,obj.nfrm);
          tf(x) = true;
          tf(frm) = lblTF(i,:);
          idx = find(tf);
          set(h,'XData',idx,'YData',repmat(i,1,numel(idx)));
        end
      end
    end
    
    function setCurrFrame(obj,frm)
      if ~obj.selectModeOn
        r = obj.prefs.FrameRadius;
        x0 = max(frm-r,1);
        x1 = min(frm+r,obj.nfrm);
        obj.hAx.XLim = [x0 x1];

        set(obj.hCurrFrame,'XData',[frm frm]);
      end
    end
    
  end
  
end
    
    
    