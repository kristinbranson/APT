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
    
    LPOSTAG_OCC = 'occ';
  end
  
  properties (Abstract)
    supportsMultiView % scalar logical
	supportsCalibration % scalar logical
  end
  
  properties (SetObservable)
    hideLabels; % scalar logical
  end
        
  properties
    labeler;              % scalar Labeler obj
    hFig;                 % [nview] figure handles (first is main fig)
    hAx;                  % [nview] axis handles (first is main axis)
    hAxOcc;               % scalar handle, occluded-axis (primary axis only)
    tbAccept;             % scalar handle, togglebutton
    pbClear;              % scalar handle, clearbutton
    txLblCoreAux;         % scalar handle, auxiliary text (currently primary axis only)
    
    nPts;                 % scalar integer 

    state;                % scalar LabelState
    hPts;                 % nPts x 1 handle vec, handle to points
    hPtsTxt;              % nPts x 1 handle vec, handle to text
    hPtsOcc;
    hPtsTxtOcc;           % nPts x 1 handle vec, handle to occ points
    ptsPlotInfo;          % struct, points plotting cosmetic info    
    
    tfOcc;                % nPts x 1 logical
    tfEstOcc;             % nPts x 1 logical. Current Est-occ impl: 
                          % TemplateMode uses .tfEstOcc.
                          % SequenceMode does not have est-occ implemented.
                          % HT mode does not use .tfEstOcc, relies on
                          % markers.
  end
  
  methods (Static)
    
    function obj = create(labelerObj,labelMode)
      switch labelMode
        case LabelMode.SEQUENTIAL
          obj = LabelCoreSeq(labelerObj);
        case LabelMode.TEMPLATE
          if labelerObj.isMultiView
            obj = LabelCoreMultiViewTemplate(labelerObj);
          else
            obj = LabelCoreTemplate(labelerObj);
          end
        case LabelMode.HIGHTHROUGHPUT
          obj = LabelCoreHT(labelerObj);
        case LabelMode.ERRORCORRECT
          obj = LabelCoreErrorCorrect(labelerObj);
        case LabelMode.MULTIVIEWCALIBRATED
          obj = LabelCoreMultiViewCalibrated(labelerObj);
        case LabelMode.MULTIVIEWCALIBRATED2
          obj = LabelCoreMultiViewCalibrated2(labelerObj);
      end
    end
    
  end
  
  methods (Sealed=true)
    
    function obj = LabelCore(labelerObj)
      obj.labeler = labelerObj;
      gd = labelerObj.gdata;
      obj.hFig = gd.figs_all;
      obj.hAx = gd.axes_all;
      obj.hAxOcc = gd.axes_occ;
      obj.tbAccept = gd.tbAccept;
      obj.pbClear = gd.pbClear;
      obj.txLblCoreAux = gd.txLblCoreAux;
    end
    
    function init(obj,nPts,ptsPlotInfo)
      if obj.labeler.isMultiView && ~obj.supportsMultiView
        error('LabelCore:MV','Multiview labeling not supported by %s.',...
          class(obj));
      end
      obj.nPts = nPts;
      obj.ptsPlotInfo = ptsPlotInfo;
      
      deleteValidHandles(obj.hPts);
      deleteValidHandles(obj.hPtsOcc);
      deleteValidHandles(obj.hPtsTxt);
      deleteValidHandles(obj.hPtsTxtOcc);
      obj.hPts = gobjects(obj.nPts,1);
      obj.hPtsOcc = gobjects(obj.nPts,1);
      obj.hPtsTxt = gobjects(obj.nPts,1);
      obj.hPtsTxtOcc = gobjects(obj.nPts,1);
      
      ax = obj.hAx;
      axOcc = obj.hAxOcc;
      for i = 1:obj.nPts
        ptsArgs = {nan,nan,ptsPlotInfo.Marker,...
          'MarkerSize',ptsPlotInfo.MarkerSize,...
          'LineWidth',ptsPlotInfo.LineWidth,...
          'Color',ptsPlotInfo.Colors(i,:),...
          'UserData',i};
        obj.hPts(i) = plot(ax(1),ptsArgs{:});
        obj.hPtsOcc(i) = plot(axOcc,ptsArgs{:});
        obj.hPtsTxt(i) = text(nan,nan,num2str(i),'Parent',ax(1),...
          'Color',ptsPlotInfo.Colors(i,:),...
          'FontSize',ptsPlotInfo.FontSize,...
          'Hittest','off');
        obj.hPtsTxtOcc(i) = text(nan,nan,num2str(i),'Parent',axOcc,...
          'Color',ptsPlotInfo.Colors(i,:),...
          'FontSize',ptsPlotInfo.FontSize,...
          'Hittest','off');
      end
      axis(axOcc,[0 obj.nPts+1 0 2]);
      obj.hideLabels = false;
            
      set(obj.hAx,'ButtonDownFcn',@(s,e)obj.axBDF(s,e));      
      arrayfun(@(x)set(x,'HitTest','on','ButtonDownFcn',@(s,e)obj.ptBDF(s,e)),obj.hPts);
      set(obj.labeler.gdata.uipanel_curr,'ButtonDownFcn',@(s,e)obj.pnlBDF(s,e));
      set(obj.hAxOcc,'ButtonDownFcn',@(s,e)obj.axOccBDF(s,e));
      
      set(obj.labeler.gdata.tbAccept,'Enable','on');
      set(obj.labeler.gdata.pbClear,'Enable','on');
      obj.labeler.currImHud.updateReadoutFields('hasLblPt',false);
      
      obj.tfOcc = false(obj.nPts,1);
      obj.tfEstOcc = false(obj.nPts,1);
      
      obj.txLblCoreAux.Visible = 'off';
      
      obj.initHook();
    end
       
  end
  
  methods
    function delete(obj)
      deleteValidHandles(obj.hPts);
      deleteValidHandles(obj.hPtsTxt);
      deleteValidHandles(obj.hPtsOcc);
      deleteValidHandles(obj.hPtsTxtOcc);
    end
  end
  
  methods
    
    function initHook(obj) %#ok<MANU>
      % Called from Labeler.labelingInit->LabelCore.init
    end
    
    function newFrame(obj,iFrm0,iFrm1,iTgt) %#ok<INUSD>
      % Frame has changed, Target is the same
      %
      % Called from Labeler.setFrame->Labeler.labelsUpdateNewFrame      
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSD>
      % Target has changed, Frame is the same
      %
      % Called from Labeler.labelsUpdateNewTarget
    end
    
    function newFrameAndTarget(obj,iFrm0,iFrm1,iTgt0,iTgt1) %#ok<INUSD>
      % Frame and Target have both changed
      %
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
    
    function pnlBDF(obj,src,evt) 
      % This is called when uipanel_curr is clicked outside the axis, or
      % when points with HitTest off plotted in overlaid axes are clicked.
      pos = get(obj.hAx,'CurrentPoint');
      pos = pos(1,1:2);
      xlim = get(obj.hAx,'XLim');
      ylim = get(obj.hAx,'YLim');
      if pos(1) >= xlim(1) && pos(1) <= xlim(2) && pos(2) >= ylim(1) && pos(2) <= ylim(2),      
        obj.axBDF(src,evt);
      end
    end
    
    function axOccBDF(obj,src,evt) %#ok<INUSD>
    end
    
    function tfKPused = kpf(obj,src,evt) %#ok<INUSD>
      % KeyPressFcn
      % 
      % tfKPused: if true, keypress has been consumed by LabelCore. 
      % LabelCore has first dibs on keypresses. Unconsumed keypresses may 
      % be passed onto other clients (eg Timeline)

      % Default implementation does nothing and never consumes
      tfKPused = false;
    end
    
    function getLabelingHelp(obj) %#ok<MANU>
    end
          
  end
  
  %% Misc
  methods
    function labelsHide(obj)
      [obj.hPts.Visible] = deal('off');
      [obj.hPtsTxt.Visible] = deal('off'); 
      obj.hideLabels = true;
    end
    
    function labelsShow(obj)
      [obj.hPts.Visible] = deal('on');
      [obj.hPtsTxt.Visible] = deal('on');
      obj.hideLabels = false;      
    end
    
    function labelsHideToggle(obj)
      if obj.hideLabels
        obj.labelsShow();
      else
        obj.labelsHide();
      end
    end
  end
  
  %% Utilities to manipulate .hPts, .hPtsTxt (set position and color)
  
  methods (Hidden) 
    
    function assignLabelCoords(obj,xy,varargin)
      % Assign specified label points xy to .hPts, .hPtsTxt; set .tfOcc
      % based on xy (in case of tfMainAxis)
      % 
      % xy: .nPts x 2 coordinate array in Labeler format. NaNs=missing,
      % inf=occluded. NaN points get set to NaN (so will not be visible);
      % Inf points get positioned in the occluded box.
      % 
      % Optional PVs:
      % - tfClip: Default false. If true, clip xy to movie size as necessary
      % - hPts: vector of handles to assign to. Must have .nPts elements. 
      %   Defaults to obj.hPts.
      % - hPtsTxt: vector of handles for text labels, etc. Defaults to
      % obj.hPtsTxt.
      % - lblTags: [nptsx1] cell array of tags. Currently, only tag is 
      % LabelCore.LPOSTAG_OCC. If supplied, obj.tfEstOcc and 
      % obj.hPts.Marker are updated.

      [tfClip,hPoints,hPointsTxt,lblTags] = myparse(varargin,...
        'tfClip',false,...
        'hPts',obj.hPts,...
        'hPtsTxt',obj.hPtsTxt,...
        'lblTags',[]);
      
      assert(isequal(obj.nPts,numel(hPoints),numel(hPointsTxt),size(xy,1)));
      tfLblTags = ~isempty(lblTags);
      if tfLblTags
        validateattributes(lblTags,{'cell'},{'vector' 'numel' obj.nPts});
      end        
      
      if tfClip        
        lbler = obj.labeler;
        
        assert(~lbler.isMultiView,'Multi-view labeling unsupported.');
        
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
            
      % FullyOccluded
      tfOccld = any(isinf(xy),2);
      LabelCore.setPtsCoords(xy(~tfOccld,:),hPoints(~tfOccld),hPointsTxt(~tfOccld));
      
      tfMainAxis = isequal(hPoints,obj.hPts) && isequal(hPointsTxt,obj.hPtsTxt);
      if tfMainAxis
        obj.tfOcc = tfOccld;
        obj.refreshOccludedPts();
      else
        LabelCore.setPtsCoords(nan(nnz(tfOccld),2),hPoints(tfOccld),hPointsTxt(tfOccld));
      end
      
      % Tags
      if tfLblTags
        tfEO = strcmp(lblTags,LabelCore.LPOSTAG_OCC);
        if any(tfOccld & tfEO)
          warning('LabelCore:occ',...
            'Points labeled as both fully and estimated-occluded.');
          
        end
        obj.tfEstOcc = tfEO;
        obj.refreshEstOccPts(); % currently only implemented in TemplateMode        
      else
        % none; tfEstOcc, hPts markers unchanged
      end
    end
    
    function assignLabelCoordsIRaw(obj,xy,iPt)
      % Set coords and reset color for hPts(iPt), hPtsTxt(iPt)
      %
      % Unlike assignLabelCoords, no clipping or occluded-handling
      hPoint = obj.hPts(iPt);
      hTxt = obj.hPtsTxt(iPt);
      LabelCore.setPtsCoords(xy,hPoint,hTxt);
%       LabelCore.setPtsColor(hPoint,hTxt,obj.ptsPlotInfo.Colors(iPt,:));
    end
    
    function refreshOccludedPts(obj)
      % Based on .tfOcc: 'Hide' occluded points in main image; arrange
      % occluded points in occluded box.
      
      tf = obj.tfOcc;      
      assert(isvector(tf) && numel(tf)==obj.nPts);
      nOcc = nnz(tf);
      iOcc = find(tf);
      LabelCore.setPtsCoords(nan(nOcc,2),obj.hPts(tf),obj.hPtsTxt(tf));
      LabelCore.setPtsCoordsOcc([iOcc(:) ones(nOcc,1)],obj.hPtsOcc(tf),obj.hPtsTxtOcc(tf));
      LabelCore.setPtsCoordsOcc(nan(obj.nPts-nOcc,2),...
        obj.hPtsOcc(~tf),obj.hPtsTxtOcc(~tf));
    end
        
    function xy = getLabelCoords(obj)
      % rows matching .tfOcc are inf
      xy = LabelCore.getCoordsFromPts(obj.hPts);
      xy(obj.tfOcc,:) = inf;
    end
    
    function xy = getLabelCoordsI(obj,iPt)
      xy = LabelCore.getCoordsFromPts(obj.hPts(iPt));
    end
    
    function setLabelPosTagFromEstOcc(obj)
      iEO = find(obj.tfEstOcc);
      tag = obj.LPOSTAG_OCC;
      lObj = obj.labeler;
      for iPt = iEO(:)'
        lObj.labelPosTagSetI(tag,iPt);
      end      
    end
    
  end
    
  methods (Static) 
    
    function xy = getCoordsFromPts(hPts)
      x = get(hPts,{'XData'});
      y = get(hPts,{'YData'});
      x = cell2mat(x);
      y = cell2mat(y);
      xy = [x y];
    end
    
    function setPtsCoords(xy,hPts,hTxt)
      nPoints = size(xy,1);
      assert(size(xy,2)==2);
      assert(isequal(nPoints,numel(hPts),numel(hTxt)));
      
      for i = 1:nPoints
        set(hPts(i),'XData',xy(i,1),'YData',xy(i,2));
        set(hTxt(i),'Position',[xy(i,1)+LabelCore.DT2P xy(i,2)+LabelCore.DT2P 1]);
      end
    end
            
    function setPtsOffaxis(hPts,hTxt)
      % Set pts/txt to be "offscreen" ie positions to NaN.
      LabelCore.setPtsCoords(nan(numel(hPts),2),hPts,hTxt);
    end
    
    function setPtsColor(hPts,hTxt,colors)
      assert(numel(hPts)==numel(hTxt));
      n = numel(hPts);
      assert(isequal(size(colors),[n 3]));
      for i = 1:n
        clr = colors(i,:);
        set(hPts(i),'Color',clr);
        set(hTxt(i),'Color',clr);
      end      
    end
    
    function setPtsCoordsOcc(xy,hPts,hTxt)
      nPoints = size(xy,1);
      assert(size(xy,2)==2);
      assert(isequal(nPoints,numel(hPts),numel(hTxt)));
      
      for i = 1:nPoints
        set(hPts(i),'XData',xy(i,1),'YData',xy(i,2));
        set(hTxt(i),'Position',[xy(i,1)+0.25 xy(i,2)+0.25 1]);
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
      %
      % Note: unlabeled -> unlabeled, occluded -> occluded, ie 
      %       NaN points -> NaN points, Inf points -> Inf points.
      
      assert(trx0.off==1-trx0.firstframe);
      assert(trx1.off==1-trx1.firstframe);
      
      iFrm0 = iFrm0+trx0.off;
      xy0 = [trx0.x(iFrm0) trx0.y(iFrm0)];
      th0 = trx0.theta(iFrm0);
      
      iFrm1 = iFrm1+trx1.off;
      xy1 = [trx1.x(iFrm1) trx1.y(iFrm1)];
      th1 = trx1.theta(iFrm1);
      
      uv = transformPoints(uv0,xy0,th0,xy1,th1);
      tfinf = any(isinf(uv0),2); % [inf inf] rows in uv0 can be transformed into eg [inf nan] depending on angle
      uv(tfinf,:) = inf;
    end
    
  end
  
end
