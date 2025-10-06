function ensureWslFolderExists(rawWslFolderPath)
  % Ensure the the given WSL folder path exists, on the local side.
  % Throws if unable to do so.

  % Get the path as a char array
  if ischar(rawWslFolderPath) || isStringScalar(rawWslFolderPath)
    wslFolderPathAsChar = char(rawWslFolderPath) ;
  elseif isa(rawWslFolderPath, 'apt.Path')
    wslFolderPathAsChar = char(rawWslFolderPath) ;
  elseif isa(rawWslFolderPath, 'apt.MetaPath')
    if isequal(rawWslFolderPath.locale, apt.PathLocale.wsl)
      wslFolderPathAsChar = char(rawWslFolderPath.asWsl()) ;
    else
      error('If %s argument is an apt.MetaPath, it must have wsl or native locale', mfilename());
    end
  else
    error('%s argument must be a char array, string, apt.Path, or apt.MetaPath', mfilename());
  end

  % Run the command to ensure the folder exists
  escapedWslFolderPathAsChar = escape_string_for_bash(wslFolderPathAsChar) ;
  command = sprintf('mkdir -p %s', escapedWslFolderPathAsChar) ;
  apt.syscmd(command, 'failbehavior', 'err') ;
end
