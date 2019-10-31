function constrainToWorkArea(h_figure)
%constrainToWorkArea Shift figure to fit inside a screen's work area.
%   constrainToWorkArea(h_figure) translates a figure horizontally and or
%   vertically as necessary, to put the figure inside a screen's work
%   area.  The work area is the region of the screen not obscured by the
%   task bar, dock or other screen decorations.
%   If those areas cannot be identified, the area is the entire display. 
%
%  Note: constrainToWorkArea assumes the figure is equal to or smaller than
%  the screen size.  It translates but does not shrink to fit.

%   Copyright 2004-2006 The MathWorks, Inc.
%   $Revision: 1.1.8.3 $  $Date: 2006/05/24 03:33:02 $

wa = getWorkArea;

old_units = get(h_figure, 'Units');
set(h_figure, 'Units', 'pixels');

% Position values in Matlab are 1-based, not 0-based.
% The -1 correction below is to adjust for that fact.
% Example: if screen resolution =  1280x1024 and there are no screen
% decorations, ScreenSize is (1,1,1280,1024). 
%
% If the figure filled the screen, the (right,top) position should be 
% (1280,1024), so the following would be: 
% 1280 = 1+1280 - 1; and 1024 = 1 + 1024 - 1;
fig_outer_position = get(h_figure, 'OuterPosition');

% The calculations that determine whether the figure must be moved 
% need the screen positions of the figure's outer edges.  
figWidth = fig_outer_position(3);
figRightScreenPos = fig_outer_position(1) + figWidth -1;

figHeight = fig_outer_position(4);
figTopScreenPos = fig_outer_position(2) + figHeight -1;

% Calculate the offset of the bottom edge of the figure from the bottom edge of
% the workarea.  
figureBottomEdgeOffset = wa.bottom - fig_outer_position(2);

% A negative result indicates that it is onscreen - no adjustment needed. 
% A positive result indicates that we need to shift it up onto the screen.
% (Screen coordinates increase as you go up)
bottomCorrection = max(0, figureBottomEdgeOffset);

% Calculate the offset of the top edge of the figure from the top edge of
% the workarea. A positive result indicates that it is onscreen
currentFigureTopEdgeOffset = wa.top - figTopScreenPos;

%Apply the bottom correction
correctedFigureTopEdgeOffset = currentFigureTopEdgeOffset + bottomCorrection;

% A negative result indicates that we need to shift it down onto the screen
topCorrection = min(0, correctedFigureTopEdgeOffset);

% The sum of the two corrections is the total  vertical translation
% amount
verticalCorrection = bottomCorrection + topCorrection;

% Calculate the offset of the left edge of the figure from the left edge of
% the workarea.  
figureLeftEdgeOffset = wa.left - fig_outer_position(1);

% A negative result indicates that it is onscreen - no adjustment needed. 
% A positive result indicates that we need to shift it onto the screen, to
% the right. (Screen coordinates increase as you go to the right)
leftCorrection = max(0, figureLeftEdgeOffset);

% Calculate the offset of the right edge of the figure from the right edge of
% the workarea. A positive result indicates that it is onscreen
 currentFigureRightEdgeOffset = wa.right - figRightScreenPos;

%Apply the left correction
 correctedFigureRightEdgeOffset = currentFigureRightEdgeOffset + leftCorrection;

% A negative result indicates that we need to shift it onto the screen, to
% the right
 rightCorrection = min(0, correctedFigureRightEdgeOffset);

% The sum of the two corrections is the total  horizontal translation amount
 horizontalCorrection = leftCorrection + rightCorrection;

% Use calculated corrections to adjust figure position in pixels
 set(h_figure, 'Position', get(h_figure, 'Position') + ...
              [horizontalCorrection verticalCorrection 0 0]);

% Restore preexisting Units
  set(h_figure, 'Units', old_units);
