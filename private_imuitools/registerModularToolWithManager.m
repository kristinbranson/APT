function registerModularToolWithManager(modular_tool,target_images)
%registerModularToolWithManager registers a modular tool with the modular tool manager of a target image.
%   registerModularToolWithManager(MODULAR_TOOL,TARGET_IMAGES) registers
%   MODULAR_TOOL with the modular tool manager of each of the
%   TARGET_IMAGES.  If a modular tool manager is not already present then
%   one will be created.
%
%   See also IMOVERVIEW.

%   Copyright 2008-2010 The MathWorks, Inc.
%   $Revision: 1.1.6.2 $  $Date: 2010/11/17 11:24:08 $

for i = 1:numel(target_images)

    % create a modular tool manager if  necessary
    current_image = target_images(i);
    modular_tool_manager = getappdata(current_image,'modularToolManager');
    if isempty(modular_tool_manager)
        modular_tool_manager = iptui.modularToolManager();
    end

    % register the tool with the manager
    modular_tool_manager.registerTool(modular_tool);

    % store manager in image appdata
    setappdata(current_image,'modularToolManager',modular_tool_manager);

end
