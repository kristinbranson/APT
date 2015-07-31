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
%   properties (Hidden)
%     dx
%     dxbig
%     dy
%     dybig
%   end
        
  properties
    labeler;              % scalar Labeler obj
    hFig;                 % scalar figure
    hAx;                  % scalar axis
    %pbClear;            
    tbAccept;
    
    nPts;                 % scalar integer
    %ptNames;             % nPts-by-1 cellstr
    ptColors;             % nPts x 3 RGB
    
    state;           % scalar state
    hPts;            % nPts x 1 handle vec, handle to points
    hPtsTxt;         % nPts x 1 handle vec, handle to text
  end
  
  methods (Sealed=true)
    
    function obj = LabelCore(labelerObj)
      obj.labeler = labelerObj;
      gd = labelerObj.gdata;
      obj.hFig = gd.figure;
      obj.hAx = gd.axes_curr;
      %obj.pbClear = gd.pbClear;
      obj.tbAccept = gd.tbAccept;
    end
    
    function init(obj,nPts,ptColors)
      obj.nPts = nPts;
      %obj.ptNames = ptNames;
      obj.ptColors = ptColors;
      
      deleteHandles(obj.hPts);
      deleteHandles(obj.hPtsTxt);
      obj.hPts = nan(obj.nPts,1);
      obj.hPtsTxt = nan(obj.nPts,1);
      ax = obj.hAx;
      for i = 1:obj.nPts
        obj.hPts(i) = plot(ax,nan,nan,'w+','MarkerSize',20,...
                               'LineWidth',3,'Color',ptColors(i,:),'UserData',i);
        obj.hPtsTxt(i) = text(nan,nan,num2str(i),'Parent',ax,...
                                   'Color',ptColors(i,:),'Hittest','off');
      end
      
      set(obj.hAx,'ButtonDownFcn',@(s,e)obj.axBDF(s,e));
      arrayfun(@(x)set(x,'HitTest','on','ButtonDownFcn',@(s,e)obj.ptBDF(s,e)),obj.hPts);
      set(obj.hFig,'WindowButtonMotionFcn',@(s,e)obj.wbmf(s,e));
      set(obj.hFig,'WindowButtonUpFcn',@(s,e)obj.wbuf(s,e));
      hTmp = findall(obj.hFig,'-property','KeyPressFcn','-not','Tag','edit_frame');
      set(hTmp,'KeyPressFcn',@(s,e)obj.kpf(s,e));
      
%       lbler = obj.labeler;
%       nr = lbler.movienr;
%       nc = lbler.movienc;
%       obj.dx = nc/obj.DXFAC;
%       obj.dxbig = nc/obj.DXFACBIG;
%       obj.dy = nr/obj.DXFAC;
%       obj.dybig = nr/obj.DYFACBIG;
      
      obj.initHook();
    end
       
  end
  
  methods
    function delete(obj)
      deleteHandles(obj.hPts);
      deleteHandles(obj.hPtsTxt);
    end
  end
  
  methods
    
    function initHook(obj) %#ok<MANU>
    end
    
    function newFrame(obj,iFrm0,iFrm1,iTgt) %#ok<INUSD>
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSD>
    end
    
    function clearLabels(obj) %#ok<MANU>
    end
    
    function acceptLabels(obj) %#ok<MANU>
    end    
    
    function unAcceptLabels(obj) %#ok<MANU>
    end    
    
    function axBDF(obj,src,evt) %#ok<INUSD>
    end
    
    function ptBDF(obj,src,evt) %#ok<INUSD>
    end
    
    function wbmf(obj,src,evt) %#ok<INUSD>
    end
    
    function wbuf(obj,src,evt) %#ok<INUSD>
    end
    
    function kpf(obj,src,evt) %#ok<INUSD>
    end
    
    function getKeyboardShortcutsHelp(obj) %#ok<MANU>
    end
          
  end
  
  methods (Hidden)
    
    function assignLabelCoords(obj,xy,tfClip)
      % tfClip: if true, clip xy to movie size as necessary

      if exist('tfClip','var')==0
        tfClip = false;
      end
      
      if tfClip
        lbler = obj.labeler;
        nr = lbler.movienr;
        nc = lbler.movienc;
        xyOrig = xy;
        xy(:,1) = max(xy(:,1),1);
        xy(:,1) = min(xy(:,1),nc);
        xy(:,2) = max(xy(:,2),1);
        xy(:,2) = min(xy(:,2),nr);      
        if ~isequal(xy,xyOrig)
          warning('LabelCore:clipping',...
            'Clipping points that extend beyond movie size.');
        end      
      end
      
      LabelCore.assignCoords2Pts(xy,obj.hPts,obj.hPtsTxt);
    end
    
    function assignLabelCoordsI(obj,xy,iPt)
      LabelCore.assignCoords2Pts(xy,obj.hPts(iPt),obj.hPtsTxt(iPt));
    end
    
    function xy = getLabelCoords(obj)
      xy = LabelCore.getCoordsFromPts(obj.hPts);      
    end
    
    function xy = getLabelCoordsI(obj,iPt)
      xy = LabelCore.getCoordsFromPts(obj.hPts(iPt));
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
