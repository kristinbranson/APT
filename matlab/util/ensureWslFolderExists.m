function ensureWslFolderExists(wslFolderPath)
  % Ensure the the given WSL folder path exists, on the local side.
  % Throws if unable to do so.
  escapedWslFolderPath = escape_string_for_bash(wslFolderPath) ;
  command = sprintf('mkdir -p %s', escapedWslFolderPath) ;
  apt.syscmd(command, 'failbehavior', 'err') ;
end
