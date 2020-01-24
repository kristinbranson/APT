function linkToolName(tool_fig,target_fig,tool_name)
% LINKTOOLNAME(TOOL_FIG, TARGET_FIG, TOOL_NAME) uses the function
% createFigureName to set the name of TOOL_FIG.  It then creates a listener
% that will update the TOOL_FIG name property to match changes that occur
% to the TARGET_FIG name property.

%   Copyright 2008-2010 The MathWorks, Inc.
%   $Revision: 1.1.6.3 $  $Date: 2010/06/26 04:56:45 $

% set the tool name to begin
set(tool_fig,'Name',createFigureName(tool_name,target_fig));

figure_name_listener = iptui.iptaddlistener(target_fig, ...
   'Name','PostSet',...
   getUpdateNameCallbackFun(tool_fig,target_fig,tool_name));

% store listener in tool figure appdata
setappdata(tool_fig,'figure_name_listener',figure_name_listener);

function cbFun = getUpdateNameCallbackFun(tool_fig,target_fig,tool_name)
 % We need to generate this function handle within a sub-function
 % workspace to prevent anonymous function workspace from storing a
 % copy of the iptui.iptaddlistener object. This was causing
 % lifecycle of listener to be tied to target_fig instead of
 % tool_fig. g648119
        
 cbFun = @(hobj,evt) set(tool_fig,'Name',createFigureName(tool_name,target_fig));