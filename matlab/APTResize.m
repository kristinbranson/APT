classdef APTResize < handle
  properties
    pxTxUnsavedChangesWidth
%     pxTxGTModeWidth
    pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge
%     pxPnlPrevRightEdgeMinusTxGTModeLeftEdge
    pumTrackInitFontSize;
    pumTrackInitHeight;
  end
  
  methods 
    function obj = APTResize(handles)
      % record state for txUnsavedChanges
      hTx = handles.txUnsavedChanges;
%       hTxGT = handles.txGTMode;
      hPnlPrev = handles.uipanel_prev;
      
      hTxUnits0 = hTx.Units;
%       hTxGTUnits0 = hTxGT.Units;
      hPnlPrevUnits0 = hPnlPrev.Units;
      hTx.Units = 'pixels';
%       hTxGTUnits0 = 'pixels';
      hPnlPrev.Units = 'pixels';
      uiPnlPrevRightEdge = hPnlPrev.Position(1)+hPnlPrev.Position(3);
      obj.pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge = ...
        uiPnlPrevRightEdge-hTx.Position(1);
      obj.pxTxUnsavedChangesWidth = hTx.Position(3);
      hTx.Units = hTxUnits0;
      hPnlPrev.Units = hPnlPrevUnits0;
      
%       hTxGTModeUnits0 = hTxGTMode.Units;
%       hTxGTMode.Units = 'pixels';

      pumTrack = handles.pumTrack;

      % Iss #116. Appears nec to get proper resize behavior
      pumTrack.Max = 2;
      
      obj.pumTrackInitFontSize = pumTrack.FontSize;
      obj.pumTrackInitHeight = pumTrack.Position(4);
    end
    
    function resize(obj,src,~)
      handles = guidata(src);
      
      hTx = handles.txUnsavedChanges;
      hPnlPrev = handles.uipanel_prev;
      hTxUnits0 = hTx.Units;
      hPnlPrevUnits0 = hPnlPrev.Units;
      hTx.Units = 'pixels';
      hPnlPrev.Units = 'pixels';
      uiPnlPrevRightEdge = hPnlPrev.Position(1)+hPnlPrev.Position(3);
      hTx.Position(1) = uiPnlPrevRightEdge-obj.pxPnlPrevRightEdgeMinusTxUnsavedChangesLeftEdge;
      hTx.Position(3) = obj.pxTxUnsavedChangesWidth;
%      align([hTx hPnlPrev],'Right','None');
      hTx.Units = hTxUnits0;
      hPnlPrev.Units = hPnlPrevUnits0;
      if isfield(handles,'controller'),
        handles.controller.updateStatus() ;
      end
      
%       tfDoPUMTrack = ~ispc && ~ismac;
%       if tfDoPUMTrack
%         drawnow; % Laggy drawing will result in poor pumTrack resize
% 
% %         % Resize pumTrack width otherwise text clipped on Linux
% %         newHeight = pum.Position(4);
% %         origFS0 = obj.pumTrackInitFontSize;
% %         % We resize the pum font manually to cap the maximum fontsize;
% %         % otherwise the pum gets too wide on Linux.
% %         newFS = max(newHeight/obj.pumTrackInitHeight*origFS0,origFS0);
% %         pum.FontSize = newFS;
%         pum = handles.pumTrack;
%         pb = handles.pbTrack;
%         pbPos = pb.Position;
%         rightEdge = pbPos(1)+pbPos(3);
%         width = 1.05*pum.Extent(3);
%         pum.Position(1) = rightEdge-width;
%         pum.Position(3) = width;
%       end
    end
  end
  
end