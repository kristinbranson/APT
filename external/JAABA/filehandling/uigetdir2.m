function [pathname] = uigetdir2(start_path, dialog_title, selections)
% Pick multiple directories and/or files

import javax.swing.JFileChooser;

if nargin == 0 || isempty(start_path) || isnumeric(start_path) % Allow a null argument.
    start_path = pwd;
end

jchooser = javaObjectEDT('javax.swing.JFileChooser', start_path);

jchooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
if nargin > 1
    jchooser.setDialogTitle(dialog_title);
end

jchooser.setMultiSelectionEnabled(true);

% set selected file
if nargin > 2,
  if ~iscell(selections),
    selections = {selections};
  end
  selections = setdiff(selections,'');
  selections(cellfun(@isempty,selections)) = [];
  if ~isempty(selections),
    clear jselections;
    for i = 1:numel(selections),
      jselections(i) = javaObjectEDT('java.io.File',selections{i}); %#ok<AGROW>
    end
    jchooser.setSelectedFiles(jselections);
  end
end

status = jchooser.showOpenDialog([]);

if status == JFileChooser.APPROVE_OPTION
  jFile = jchooser.getSelectedFiles();
	pathname{size(jFile, 1)}=[];
  for i=1:size(jFile, 1)
		pathname{i} = char(jFile(i).getAbsolutePath);
	end
	
elseif status == JFileChooser.CANCEL_OPTION
    pathname = [];
else
    error('Error occured while picking file.');
end
