function shrinkFigureToFitContentBang(fig, outerGrid)
% Resize fig vertically to fit the natural content of outerGrid.
%
% outerGrid is a uigridlayout that fills fig and has a single column.
% Each of its direct children is a "row" whose rendered height is taken
% as that row's natural height.  After this function returns, fig is
% exactly tall enough to hold:
%
%   topPadding + sum(rowHeights) + (rowCount-1) * RowSpacing + bottomPadding
%
% Inputs:
%   fig         - matlab.ui.Figure to resize.  Must be Visible='on'
%                 before calling, since uigridlayout positions only
%                 resolve once the figure is realized.
%   outerGrid   - uigridlayout parented directly to fig
%
% uigridlayout in uifigure resolves positions asynchronously, so this
% function polls drawnow + pause until every child of outerGrid has a
% positive rendered width and height.  Polls up to 5 seconds.

maxWait = 5 ;  % secs
defaultPosition = [1 1 100 100] ;
rowCount = numel(outerGrid.Children) ;
deadline = tic() ;
while true
  drawnow() ;
  allResolved = true ;
  for i = 1:rowCount
    childPosition = getpixelposition(outerGrid.Children(i)) ;
    isResolved = ~isequal(childPosition, defaultPosition) && childPosition(4) > 0 ;
    if ~isResolved
      allResolved = false ;
      break
    end
  end
  if allResolved || toc(deadline) > maxWait
    break
  end
  pause(0.02) ;
end

rowHeights = zeros(1, rowCount) ;
for i = 1:rowCount
  childPosition = getpixelposition(outerGrid.Children(i)) ;
  rowHeights(i) = childPosition(4) ;
end

padding = outerGrid.Padding ;  % [left bottom right top]
totalContentHeight = sum(rowHeights) + (rowCount-1) * outerGrid.RowSpacing ;
neededFigHeight = padding(2) + totalContentHeight + padding(4) ;
fig.Position(4) = neededFigHeight ;
waitForFigureToSync(fig) ;

end  % function
