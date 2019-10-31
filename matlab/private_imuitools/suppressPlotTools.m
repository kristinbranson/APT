function suppressPlotTools(h_fig)
%suppressPlotTools Prevents the plot tools from activating on figure.

%   Copyright 2008 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $  $Date: 2008/07/28 14:28:58 $

% prevent figure from entering plot edit mode
hB = hggetbehavior(h_fig,'plottools');
set(hB,'ActivatePlotEditOnOpen',false);
