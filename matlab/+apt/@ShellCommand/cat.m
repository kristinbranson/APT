function result = cat(varargin)
  % Concatenate any number of strings, apt.MetaPaths, and apt.ShellCommands
  % into a single apt.ShellCommand
  %
  % Args:
  %   varargin: Variable number of arguments to concatenate:
  %             - strings: literal command tokens
  %             - apt.MetaPath: path tokens
  %             - apt.ShellCommand: existing commands to merge
  %
  % Returns:
  %   apt.ShellCommand: New command containing all tokens
  %
  % Notes:
  %   - If any apt.MetaPath or apt.ShellCommand arguments have different locales,
  %     an error is thrown
  %   - The resulting command will have the locale of the first MetaPath/ShellCommand found,
  %     or 'native' if only strings are provided
  %   - Empty arguments are ignored
  %
  % Example:
  %   path1 = apt.MetaPath('/data/input.txt', 'native', 'movie');
  %   path2 = apt.MetaPath('/results/output.txt', 'native', 'cache');
  %   cmd = apt.ShellCommand.cat('python', 'script.py', '--input', path1, '--output', path2);
  
  if nargin == 0
    result = apt.ShellCommand({}, apt.PathLocale.native);
    return;
  end
  
  allTokens = cell(1,0);
  detectedLocale = [];
  
  for i = 1:nargin
    arg = varargin{i};
    
    if isempty(arg)
      % Skip empty arguments
      continue;
    elseif ischar(arg) || isstring(arg)
      % String literal token
      allTokens{1,end+1} = char(arg);  %#ok<AGROW>
    elseif isa(arg, 'apt.MetaPath')
      % MetaPath token
      if isempty(detectedLocale)
        detectedLocale = arg.locale;
      elseif detectedLocale ~= arg.locale
        error('apt:ShellCommand:LocaleMismatch', ...
              'MetaPath argument at position %d has locale %s, but previous paths have locale %s', ...
              i, apt.PathLocale.toString(arg.locale), apt.PathLocale.toString(detectedLocale));
      end
      allTokens{1,end+1} = arg; %#ok<AGROW>
    elseif isa(arg, 'apt.ShellCommand')
      % ShellCommand to merge
      if isempty(detectedLocale)
        detectedLocale = arg.locale_;
      elseif detectedLocale ~= arg.locale_
        error('apt:ShellCommand:LocaleMismatch', ...
              'ShellCommand argument at position %d has locale %s, but previous paths have locale %s', ...
              i, apt.PathLocale.toString(arg.locale_), apt.PathLocale.toString(detectedLocale));
      end
      % Add all tokens from the ShellCommand
      allTokens = horzcat(allTokens, arg.tokens_);  %#ok<AGROW>
    else
      error('apt:ShellCommand:InvalidArgument', ...
            'Argument at position %d must be a string, apt.MetaPath, or apt.ShellCommand, got %s', ...
            i, class(arg));
    end
  end
  
  % Use detected locale or default to native
  if isempty(detectedLocale)
    detectedLocale = apt.PathLocale.native;
  end
  
  result = apt.ShellCommand(allTokens, detectedLocale);
end