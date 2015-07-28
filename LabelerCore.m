classdef LabelerCore < handle
  % LabelerCore 
  % Handles the details of labeling: the labeling state machine, 
  % managing in-progress labels, etc. 
  %
  % LabelerCore intercepts all axes_curr (and children) BDFs, figure 
  % Window*Fcns, figure keypresses, and tbAccept/pbClear signals to
  % implement the labeling state machine; ie labelng is enabled by ptsBDF, 
  % figWBMF, figWBUF acting in concert. When labels are accepted, they are 
  % written back to the Labeler.
  %
  % Labeler provides LabelerCore with target/frame transitions, trx info,
  % accepted labels info. LabelerCore read/writes labeledpos through
  % Labeler's API, and for convenience directly manages limited uicontrols 
  % on LabelerGUI (pbClear, tbAccept).
  
  properties (Constant,Hidden)
    DT2P = 5;
  end
        
  properties
    labeler;              % scalar Labeler obj
    hFig;                 % scalar figure
    hAx;                  % scalar axis
    %pbClear;            
    tbAccept;
    
    nPts;                 % scalar integer
    %ptNames;              % nPts-by-1 cellstr
    ptColors;             % nPts x 3 RGB
    
    state;           % scalar state
    hPts;            % nPts x 1 handle vec, handle to points
    hPtsTxt;         % nPts x 1 handle vec, handle to text
  end
  
  methods
    
    function obj = LabelerCore(labelerObj)
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
      set(obj.hFig,'KeyPressFcn',@(s,e)obj.kpf(s,e));
    end
    
  end
  
  methods
    
    function newFrame(obj,iFrm0,iFrm1,iTgt) %#ok<INUSD>
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSD>
    end
    
    function clearLabels(obj) %#ok<MANU>
    end
    
    function acceptLabels(obj) %#ok<MANU>
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
  
  methods (Static)
    
    function xy = getCoordsFromPts(hPts)
      x = get(hPts,'XData');
      y = get(hPts,'YData');
      x = cell2mat(x);
      y = cell2mat(y);
      xy = [x y];
    end
    
    function assignCoords2Pts(xy,hPts,hTxt)
      nPoints = size(xy,1);
      assert(size(xy,2)==2);
      assert(isequal(nPoints,numel(hPts),numel(hTxt)));
      
      for i = 1:nPoints
        set(hPts(i),'XData',xy(i,1),'YData',xy(i,2));
        set(hTxt(i),'Position',[xy(i,1)+LabelerCore.DT2P xy(i,2)+LabelerCore.DT2P 1]);
      end
    end
    
    function removePts(hPts,hTxt)
      assert(numel(hPts)==numel(hTxt));
      for i = 1:numel(hPts)
        set(hPts(i),'XData',nan,'YData',nan);
        set(hTxt(i),'Position',[nan nan nan]);
      end      
    end

  end
  
end

