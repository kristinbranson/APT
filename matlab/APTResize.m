classdef APTResize < handle
  properties
    pxTxUnsavedChangesWidth_
    pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge_
    pumTrackInitFontSize_
    pumTrackInitHeight_
    controller_
  end
  
  methods 
    function obj = APTResize(controller)
      obj.controller_ = controller ;
      obj.figure_ = figure ;
      handles = guidata(figure) ;
      % record state for txUnsavedChanges
      hTx = handles.txUnsavedChanges;
      hPnlPrev = handles.uipanel_prev;
      
      hTxUnits0 = hTx.Units;
      hPnlPrevUnits0 = hPnlPrev.Units;
      hTx.Units = 'pixels';
      hPnlPrev.Units = 'pixels';
      uiPnlPrevRightEdge = hPnlPrev.Position(1)+hPnlPrev.Position(3);
      obj.pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge_ = ...
        uiPnlPrevRightEdge-hTx.Position(1);
      obj.pxTxUnsavedChangesWidth_ = hTx.Position(3);
      hTx.Units = hTxUnits0;
      hPnlPrev.Units = hPnlPrevUnits0;
      
      pumTrack = handles.pumTrack;

      % Iss #116. Appears nec to get proper resize behavior
      pumTrack.Max = 2;
      
      obj.pumTrackInitFontSize_ = pumTrack.FontSize;
      obj.pumTrackInitHeight_ = pumTrack.Position(4);
    end
    
    function resize(obj)
      handles = guidata(obj.figure_);
      
      hTx = handles.txUnsavedChanges;
      hPnlPrev = handles.uipanel_prev;
      hTxUnits0 = hTx.Units;
      hPnlPrevUnits0 = hPnlPrev.Units;
      hTx.Units = 'pixels';
      hPnlPrev.Units = 'pixels';
      uiPnlPrevRightEdge = hPnlPrev.Position(1)+hPnlPrev.Position(3);
      hTx.Position(1) = uiPnlPrevRightEdge-obj.pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge_;
      hTx.Position(3) = obj.pxTxUnsavedChangesWidth_;
      hTx.Units = hTxUnits0;
      hPnlPrev.Units = hPnlPrevUnits0;
      % if isfield(handles,'controller'),
      %   handles.controller.updateStatus() ;
      % end
    end
  end
  
end