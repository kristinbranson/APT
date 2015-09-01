classdef LabelCore < handle
  % LabelCore 
  % Handles the details of labeling: the labeling state machine, 
  % managing in-progress labels, etc. 
  %
  % LabelCore intercepts all axes_curr (and children) BDFs, figure 
  % Window*Fcns, figure keypresses, and tbAccept/pbClear signals to
  % implement the labeling state machine; ie labelng is enabled by ptsBDF, 
  % figWBMF, figWBUF acting in concert. When labels are accepted, they are 
  % written back to the Labeler.
  %
  % Labeler provides LabelCore with target/frame transitions, trx info,
  % accepted labels info. LabelCore read/writes labeledpos through
  % Labeler's API, and for convenience directly manages limited uicontrols 
  % on LabelerGUI (pbClear, tbAccept).
  
  properties (Constant,Hidden)
    DT2P = 5;
    DXFAC = 500;
    DXFACBIG = 50;
  end
        
  properties
    labeler;              % scalar Labeler obj
    hFig;                 % scalar figure
    hAx;                  % scalar axis
    tbAccept;
    
    nPts;                 % scalar integer
    
    state;                % scalar state
    hPts;                 % nPts x 1 handle vec, handle to points
    hPtsTxt;              % nPts x 1 handle vec, handle to text
    ptsPlotInfo;          % struct, points plotting cosmetic info
  end
  
  methods (Sealed=true)
    
    function obj = LabelCore(labelerObj)
      obj.labeler = labelerObj;
      gd = labelerObj.gdata;
      obj.hFig = gd.figure;
      obj.hAx = gd.axes_curr;
      obj.tbAccept = gd.tbAccept;
    end
    
    function init(obj,nPts,ptsPlotInfo)
      obj.nPts = nPts;
      obj.ptsPlotInfo = ptsPlotInfo;
      
      deleteValidHandles(obj.hPts);
      deleteValidHandles(obj.hPtsTxt);
      obj.hPts = nan(obj.nPts,1);
      obj.hPtsTxt = nan(obj.nPts,1);
      ax = obj.hAx;
      for i = 1:obj.nPts
        obj.hPts(i) = plot(ax,nan,nan,ptsPlotInfo.Marker,...
          'MarkerSize',ptsPlotInfo.MarkerSize,...
          'LineWidth',ptsPlotInfo.LineWidth,...
          'Color',ptsPlotInfo.Colors(i,:),...
          'UserData',i);                 
        obj.hPtsTxt(i) = text(nan,nan,num2str(i),'Parent',ax,...        
          'Color',ptsPlotInfo.Colors(i,:),...
          'FontSize',ptsPlotInfo.FontSize,...
          'Hittest','off');
      end
            
      set(obj.hAx,'ButtonDownFcn',@(s,e)obj.axBDF(s,e));
      arrayfun(@(x)set(x,'HitTest','on','ButtonDownFcn',@(s,e)obj.ptBDF(s,e)),obj.hPts);
      set(obj.hFig,'WindowButtonMotionFcn',@(s,e)obj.wbmf(s,e));
      set(obj.hFig,'WindowButtonUpFcn',@(s,e)obj.wbuf(s,e));
      set(obj.labeler.gdata.uipanel_curr,'ButtonDownFcn',@(s,e)obj.pnlBDF);
      hTmp = findall(obj.hFig,'-property','KeyPressFcn','-not','Tag','edit_frame');
      set(hTmp,'KeyPressFcn',@(s,e)obj.kpf(s,e));
      
      set(obj.labeler.gdata.txCurrImAux,'Visible','off');
      
      obj.initHook();
    end
       
  end
  
  methods
    function delete(obj)
      deleteValidHandles(obj.hPts);
      deleteValidHandles(obj.hPtsTxt);
    end
  end
  
  methods
    
    function initHook(obj) %#ok<MANU>
      % Called from Labeler.labelingInit->LabelCore.init
    end
    
    function newFrame(obj,iFrm0,iFrm1,iTgt) %#ok<INUSD>
      % Frame has changed
      %
      % Called from Labeler.setFrame->Labeler.labelsUpdateNewFrame      
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSD>
      % Target has changed
      %
      % Called from Labeler.labelsUpdateNewTarget
    end
    
    function clearLabels(obj) %#ok<MANU>
      % Clear current labels and enter initial labeling state
      %
      % Called from pbClear and Labeler.labelingInit
    end
    
    function acceptLabels(obj) %#ok<MANU>
      % Called from tbAccept/Unaccept
    end    
    
    function unAcceptLabels(obj) %#ok<MANU>
      % Called from tbAccept/Unaccept
    end    
    
    function axBDF(obj,src,evt) %#ok<INUSD>
    end
    
    function ptBDF(obj,src,evt) %#ok<INUSD>
    end
    
    function wbmf(obj,src,evt) %#ok<INUSD>
    end
    
    function wbuf(obj,src,evt) %#ok<INUSD>
    end
    
    function pnlBDF(obj,src,evt) %#ok<INUSD>
      % This is called when uipanel_curr is clicked outside the axis.
    end
    
    function kpf(obj,src,evt) %#ok<INUSD>
    end
    
    function getKeyboardShortcutsHelp(obj) %#ok<MANU>
    end
          
  end
  
  methods (Hidden)
    
    function assignLabelCoords(obj,xy,varargin)
      % Assign specified label points xy to .hPts, .hPtsTxt
      % 
      % xy: .nPts x 2 coordinate array in Labeler format. NaNs=missing,
      % inf=occluded. NaN points get set to NaN (so will not be visible);
      % Inf points get positioned in the occluded box.
      % 
      % Optional PVs:
      % - tfClip: Default false. If true, clip xy to movie size as necessary
      % - hPts: vector of handles to assign to. Should have size(xy,1)
      % elements. Defaults to obj.hPts.
      % - hPtsTxt: vector of handles for text labels, etc. Defaults to
      % obj.hPtsTxt.

      [tfClip,hPoints,hPointsTxt] = myparse(varargin,...
        'tfClip',false,...
        'hPts',obj.hPts,...
        'hPtsTxt',obj.hPtsTxt);
      
      if tfClip
        lbler = obj.labeler;
        nr = lbler.movienr;
        nc = lbler.movienc;
        xyOrig = xy;
        
        tfRealCoord = ~isnan(xy) & ~isinf(xy);
        xy(tfRealCoord(:,1),1) = max(xy(tfRealCoord(:,1),1),1);
        xy(tfRealCoord(:,1),1) = min(xy(tfRealCoord(:,1),1),nc); 
        xy(tfRealCoord(:,2),2) = max(xy(tfRealCoord(:,2),2),1);
        xy(tfRealCoord(:,2),2) = min(xy(tfRealCoord(:,2),2),nr);
        if ~isequaln(xy,xyOrig)
          warningNoTrace('LabelCore:clipping',...
            'Clipping points that extend beyond movie size.');
        end
      end
      
      tfOcc = any(isinf(xy),2);
      obj.dispOccludedPts(tfOcc,'hPts',hPoints,'hPtsTxt',hPointsTxt);
      LabelCore.assignCoords2Pts(xy(~tfOcc,:),hPoints(~tfOcc),hPointsTxt(~tfOcc));
    end
    
    function assignLabelCoordsI(obj,xy,iPt)
      % Unlike assignLabelCoords, no clipping or occluded-handling
      LabelCore.assignCoords2Pts(xy,obj.hPts(iPt),obj.hPtsTxt(iPt));
    end
    
    function xy = getLabelCoords(obj)
      xy = LabelCore.getCoordsFromPts(obj.hPts);      
    end
    
    function xy = getLabelCoordsI(obj,iPt)
      xy = LabelCore.getCoordsFromPts(obj.hPts(iPt));
    end
    
    function dispOccludedPts(obj,tfOccluded,varargin)
      % Arrange occluded points/txt labels in occluded box.
      %
      % tfOccluded: logical vector with obj.nPts elements
      % Optional PVs:
      % - hPoints: vector of handles for points, must have obj.nPts elements.
      % Defaults to obj.nPts.
      % - hPointsTxt: etc.
      
      [hPoints,hPointsTxt] = myparse(varargin,...
        'hPts',obj.hPts,...
        'hPtsTxt',obj.hPtsTxt);
      
      assert(isvector(tfOccluded) && isvector(hPoints) && isvector(hPointsTxt));
      assert(isequal(numel(tfOccluded),numel(hPoints),numel(hPointsTxt)));
      
      if any(tfOccluded)
        iOcc = find(tfOccluded);
        nOcc = numel(iOcc);
        
        y = obj.labeler.movienr-10;
        dx = 15;
        x = 10 + dx*(1:nOcc);
        x = x(:);
        y = repmat(y,size(x));
        LabelCore.assignCoords2Pts([x y],hPoints(tfOccluded),hPointsTxt(tfOccluded));
      end        
    end
            
  end
    
  methods (Static)
    
    function xy = getCoordsFromPts(hPts)
      x = get(hPts,'XData');
      y = get(hPts,'YData');
      if iscell(x) % MATLABism. True for nonscalar hPts
        x = cell2mat(x);
        y = cell2mat(y);
      end
      xy = [x y];
    end
    
    function assignCoords2Pts(xy,hPts,hTxt)
      nPoints = size(xy,1);
      assert(size(xy,2)==2);
      assert(isequal(nPoints,numel(hPts),numel(hTxt)));
      
      for i = 1:nPoints
        set(hPts(i),'XData',xy(i,1),'YData',xy(i,2));
        set(hTxt(i),'Position',[xy(i,1)+LabelCore.DT2P xy(i,2)+LabelCore.DT2P 1]);
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
  
end
