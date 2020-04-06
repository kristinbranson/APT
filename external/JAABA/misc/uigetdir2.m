function [pathname] = uigetdir2(start_path, dialog_title, name)
% Pick a directory with the Java widgets instead of uigetdir

import javax.swing.JFileChooser;
import java.io.File;

if nargin == 0 || isempty(start_path) || ~ischar(start_path), % Allow a null argument.
    start_path = pwd;
end

jchooser = javaObjectEDT('javax.swing.JFileChooser', start_path);

jchooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
if nargin > 1
    jchooser.setDialogTitle(dialog_title);
end
if nargin > 2,
  file = fullfile(start_path,name);
  if exist(file,'file'),
    jchooser.setSelectedFile(File(file));
  end
end

status = jchooser.showOpenDialog([]);

if status == JFileChooser.APPROVE_OPTION
    jFile = jchooser.getSelectedFile();
    pathname = char(jFile.getPath());
elseif status == JFileChooser.CANCEL_OPTION
    pathname = 0;
else
    error('Error occured while picking file.');
end