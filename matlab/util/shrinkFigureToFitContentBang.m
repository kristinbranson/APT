function shrinkFigureToFitContentBang(fig, outerGrid)
% Resize fig in both dimensions to fit the natural content of outerGrid.
%
% outerGrid is a uigridlayout that fills fig.  For each row and column
% in outerGrid, we take the natural size of that row/column to be the
% maximum rendered height/width of the children placed in it.  After
% this function returns, fig is exactly large enough to hold:
%
%   width  = leftPadding   + sum(colWidths)  + (cols-1)*ColumnSpacing + rightPadding
%   height = bottomPadding + sum(rowHeights) + (rows-1)*RowSpacing    + topPadding
%
% Inputs:
%   fig         - matlab.ui.Figure to resize.  Must be Visible='on'
%                 before calling, since uigridlayout positions only
%                 resolve once the figure is realized.
%   outerGrid   - uigridlayout parented directly to fig
%
% uigridlayout in uifigure resolves positions asynchronously, so this
% function polls drawnow + pause until every child of outerGrid has a
% resolved (non-default, positive-sized) pixel position.  Polls up to 5
% seconds.

maxWait = 5 ;  % secs
defaultPosition = [1 1 100 100] ;
children = outerGrid.Children ;
childCount = numel(children) ;
deadline = tic() ;
while true
  drawnow() ;
  allResolved = true ;
  for i = 1:childCount
    childPosition = getpixelposition(children(i)) ;
    isResolved = ~isequal(childPosition, defaultPosition) && childPosition(3) > 0 && childPosition(4) > 0 ;
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

% Determine the natural width of each column and natural height of each
% row by taking the max rendered size of the children placed in each
% column/row.  Children that span multiple rows/columns get their full
% span size attributed to the first row/column they occupy; this is
% conservative but fine for layouts that don't span.
rowCount = numel(outerGrid.RowHeight) ;
columnCount = numel(outerGrid.ColumnWidth) ;
rowHeights = zeros(1, rowCount) ;
columnWidths = zeros(1, columnCount) ;
for i = 1:childCount
  child = children(i) ;
  childPosition = getpixelposition(child) ;
  row = child.Layout.Row(1) ;
  column = child.Layout.Column(1) ;
  rowHeights(row) = max(rowHeights(row), childPosition(4)) ;
  columnWidths(column) = max(columnWidths(column), childPosition(3)) ;
end

padding = outerGrid.Padding ;  % [left bottom right top]
contentWidth = sum(columnWidths) + (columnCount-1) * outerGrid.ColumnSpacing ;
contentHeight = sum(rowHeights) + (rowCount-1) * outerGrid.RowSpacing ;
fig.Position(3) = padding(1) + contentWidth + padding(3) ;
fig.Position(4) = padding(2) + contentHeight + padding(4) ;
waitForFigureToSync(fig) ;

end  % function
