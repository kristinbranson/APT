classdef Path
  % apt.Path - Platform-aware path representation and manipulation
  %
  % This value class represents file system paths as lists of components with
  % platform-specific formatting and conversion capabilities. It handles the
  % differences between Windows (drive letters, backslashes) and POSIX
  % (forward slashes, root directory) path conventions.
  %
  % Key features:
  % - Platform-specific path parsing and formatting
  % - Conversion between Windows and POSIX path formats
  % - Path concatenation and manipulation operations
  % - Absolute/relative path handling
  %
  % Example usage:
  %   p = apt.Path({'C:', 'data', 'movie.avi'}, apt.Platform.windows);
  %   pathStr = p.char();  % Returns properly escaped path string
  %   posixPath = p.toPosix();  % Convert to POSIX format (/mnt/c/data/movie.avi)
  %
  % Note: To create a posix-style absolute path from a cell array of strings,
  % the first element of the cell array must be the empty string.  E.g.
  % apt.Path({'foo', 'bar'}, apt.Platform.posix) corresponds to foo/bar, whereas
  % apt.Path({'', 'foo', 'bar'}, apt.Platform.posix) corresponds to /foo/bar. By
  % the same token, if pth is a posix path representing an absolute path, then
  % pth.list{1} will be the empty string.
  
  properties
    list        % Cell array of path components
    platform    % apt.Platform enumeration.  Represents the platform the frontend is currently running on,
                % usually.  Present so that we can test Path functionality on e.g. Windows on
                % e.g. POSIX.
  end

  methods
    function obj = Path(listOrString, rawPlatform)
      % Constructor for apt.Path
      %
      % Args:
      %   listOrString (cell or char, optional): Cell array of path components or path string
      %   platform (char or apt.Platform, optional): 'posix', 'windows', or enum

      % Deal with args
      if ~exist('rawPlatform', 'var') || isempty(rawPlatform)
        rawPlatform = apt.Platform.current();
      end

      % Convert string to enum if needed
      if ischar(rawPlatform)        
        platform = apt.Platform.fromString(rawPlatform);
      else
        platform = rawPlatform;
      end

      % Handle empty path case (no arguments, '.', empty string, or empty array)
      if ~exist('listOrString', 'var') || isempty(listOrString) || (ischar(listOrString) && strcmp(listOrString, '.'))
        obj.list = cell(1,0);  % Empty row vector for empty path
        obj.platform = platform;
        return
      end

      if ischar(listOrString)
        str = listOrString ;
        % Convert string path to list
        obj.list = apt.Path.stringToList_(str, platform);
        
        % Check for Windows root path case
        if isempty(obj.list) && platform == apt.Platform.windows
          error('apt:Path:EmptyPath', 'Cannot create Windows path from path "/"');
        end
      elseif iscell(listOrString)
        lst = listOrString ;
        obj.list = lst;        
      else
        error('listOrString must be a string or a cell array of strings') ;
      end

      % Set the platform
      obj.platform = platform;
      
      % Remove '.' elements from the path list
      if ~isempty(obj.list)
        isDotFromIndex = strcmp(obj.list, '.');
        obj.list(isDotFromIndex) = [];
      end
    end  % function


    function result = tfIsAbsolute(obj)
      result = apt.Path.tfIsAbsoluteList_(obj.list, obj.platform) ;
    end

    function result = tfIsNull(obj)
      % Check if the path has no components (empty path)
      %
      % Returns:
      %   logical: true if the path list is empty, false otherwise
      result = isempty(obj.list);
    end

    function result = char(obj)
      % Get the path as a string, *escaped for bash*
      preResult = obj.charUnescaped() ;
      result = escape_string_for_bash(preResult) ;
    end

    function result = charUnescaped(obj)
      % Get the path as a string
      result = apt.Path.listToString_(obj.list, obj.platform);
    end

    function result = cat(obj, varargin)
      % Concatenate this path with multiple apt.Path objects or char arrays
      %
      % Args:
      %   varargin: Variable number of apt.Path objects or char arrays to concatenate
      %
      % Returns:
      %   apt.Path: New path object representing the concatenation of all inputs
      %
      % Notes:
      %   - All paths must be relative (not absolute)
      %   - All paths must have compatible platforms
      %   - Char arrays are converted to apt.Path objects with the same platform
      
      % Convert char arrays to apt.Path objects and validate
      pathArgs = cell(size(varargin));
      for i = 1:length(varargin)
        if ischar(varargin{i})
          % Convert char array to apt.Path with same platform
          pathArgs{i} = apt.Path(varargin{i}, obj.platform);
        elseif isa(varargin{i}, 'apt.Path')
          pathArgs{i} = varargin{i};
        else
          error('apt:Path:InvalidArgument', 'Argument %d must be an apt.Path object or char array, got %s', i, class(varargin{i}));
        end
      end
      
      % Start with this path
      newList = obj.list;
      
      % Concatenate each argument in sequence
      for i = 1:length(pathArgs)
        otherPath = pathArgs{i};
        
        % Check that the path is relative
        if otherPath.tfIsAbsolute()
          error('apt:Path:AbsolutePath', 'Cannot concatenate with an absolute path at argument %d', i);
        end
        
        % Check platform compatibility
        if obj.platform ~= otherPath.platform
          error('apt:Path:PlatformMismatch', 'Cannot concatenate paths from different platforms (argument %d)', i);
        end
        
        % Skip empty paths
        if ~isempty(otherPath.list)
          newList = horzcat(newList, otherPath.list);  %#ok<AGROW>
        end
      end
      
      % Create new path object
      result = apt.Path(newList, obj.platform);
    end

    function result = append(obj, varargin)
      % Append char array arguments to this path
      %
      % Args:
      %   varargin: Variable number of char arrays to append as path components
      %
      % Returns:
      %   apt.Path: New path object with the char arrays appended as components
      %
      % Notes:
      %   - All arguments must be char arrays and row arrays
      %   - Each char array is added as a separate component to the path
      %   - The result will have the same platform as this path
      %   - Empty arrays cause an error
      
      % Validate all arguments are char arrays and row arrays
      for i = 1:length(varargin)
        if ~ischar(varargin{i})
          error('apt:Path:InvalidArgument', 'Argument %d must be a char array, got %s', i, class(varargin{i}));
        end
        if isempty(varargin{i})
          error('apt:Path:EmptyArgument', 'Argument %d cannot be an empty array', i);
        end
        if ~isrow(varargin{i})
          error('apt:Path:InvalidArgument', 'Argument %d must be a row array', i);
        end
      end
      
      % Start with this path's components
      newList = obj.list;
      
      % Append each char array as a new component
      for i = 1:length(varargin)
        component = varargin{i};
        newList = horzcat(newList, {component});  %#ok<AGROW>
      end
      
      % Create new path object
      result = apt.Path(newList, obj.platform);
    end

    function [pathPart, filenamePart] = fileparts2(obj)
      % Works like Matlab's builtin fileparts(), but combines name and ext
      %
      % Returns:
      %   pathPart (apt.Path): Directory path portion 
      %   filenamePart (apt.Path): Filename portion (name + extension combined)
      %
      % Example:
      %   p = apt.Path('/home/user/data/movie.avi');
      %   [dir, file] = p.fileparts2();
      %   % dir will be apt.Path('/home/user/data')
      %   % file will be apt.Path('movie.avi')
      
      if isempty(obj.list)
        error('apt:Path:EmptyPath', 'Cannot get fileparts of empty path');
      end
      
      % Get the last component (filename)
      filename = obj.list{end};
      
      % Create the path part by removing the last component
      if isscalar(obj.list)
        % Only one component - create appropriate empty path
        if obj.tfIsAbsolute()
          % For absolute paths with one component, path part should be root
          if obj.platform == apt.Platform.windows
            % Windows: just the drive letter
            pathPart = apt.Path({filename}, obj.platform);
          else
            % Unix: root directory
            pathPart = apt.Path({''}, obj.platform);
          end
        else
          % For relative paths with one component, path part should be empty relative path
          pathPart = apt.Path('.', obj.platform);
        end
      else
        % Multiple components - take all but the last
        pathList = obj.list(1:end-1);
        pathPart = apt.Path(pathList, obj.platform);
      end
      
      % Create the filename part as a relative path
      filenamePart = apt.Path({filename}, obj.platform);
    end

    function result = replacePrefix(obj, rawSourcePrefixPath, rawTargetPrefixPath)
      % Replace a source prefix with a target prefix
      %
      % Args:
      %   rawSourcePrefixPath (apt.Path or char): The prefix to replace
      %   rawTargetPrefixPath (apt.Path or char): The replacement prefix
      %
      % Returns:
      %   apt.Path: New path with prefix replaced, or original path if prefix doesn't match
      %
      % Example:
      %   p = apt.Path('/old/base/file.txt');
      %   newP = p.replacePrefix('/old/base', '/new/location');
      %   % newP will be apt.Path('/new/location/file.txt')
      
      % Convert string arguments to apt.Path if needed
      if ischar(rawSourcePrefixPath)
        sourcePrefixPath = apt.Path(rawSourcePrefixPath, obj.platform) ;
      elseif isa(rawSourcePrefixPath, 'apt.Path')
        sourcePrefixPath = rawSourcePrefixPath ;
      else
        error('apt:Path:InvalidArgument', 'rawSourcePrefixPath must be an apt.Path object or string') ;
      end
      
      if ischar(rawTargetPrefixPath)
        targetPrefixPath = apt.Path(rawTargetPrefixPath, obj.platform);
      elseif isa(rawTargetPrefixPath, 'apt.Path')
        targetPrefixPath = rawTargetPrefixPath ;
      else
        error('apt:Path:InvalidArgument', 'rawTargetPrefixPath must be an apt.Path object or string');
      end
      
      % Check platform compatibility
      if obj.platform ~= sourcePrefixPath.platform
        error('apt:Path:PlatformMismatch', 'Source prefix path platform must match that of obj');
      end
      
      % Check if this path starts with the source prefix
      sourceList = sourcePrefixPath.list;
      if length(obj.list) < length(sourceList)
        % Path is shorter than source prefix, no match
        result = obj;
        return
      end
      
      % Check if the beginning of obj.list matches sourceList
      if ~isequal(obj.list(1:length(sourceList)), sourceList)
        % Prefix doesn't match
        result = obj;
        return
      end
      
      % Replace the prefix
      remainingList = obj.list(length(sourceList)+1:end);
      newList = horzcat(targetPrefixPath.list, remainingList) ;
      
      % Create new path object
      % Use the targetPrefix's platform, since that makes the most sense, and this
      % allows one to convert e.g. a posix path to windows by replacing the posix
      % prefix with a windows prefix.
      result = apt.Path(newList, targetPrefixPath.platform);
    end

    function result = toPosix(obj)
      % Convert this path to a POSIX-compatible path
      %
      % Returns:
      %   apt.Path: New path object compatible with POSIX systems
      %
      % Notes:
      %   - For Windows paths: converts to WSL equivalent paths
      %     - Windows absolute paths: converts drive letter (C: -> /mnt/c)
      %     - Windows relative paths: returns Linux path with same components
      %   - For Linux paths: returns obj unchanged (already POSIX-compatible)
      %   - For macOS paths: returns obj unchanged (already POSIX-compatible)
      %
      % Example:
      %   winPath = apt.Path('C:\Users\data', apt.Platform.windows);
      %   posixPath = winPath.toPosix();
      %   % posixPath will be apt.Path('/mnt/c/Users/data', apt.Platform.posix)
      
      if obj.platform == apt.Platform.posix
        % Linux/Mac paths are already POSIX-compatible
        result = obj;
        return
      end
      
      % Handle Windows paths
      if ~obj.tfIsAbsolute()
        % For Windows relative paths, just change platform to Linux
        result = apt.Path(obj.list, apt.Platform.posix);
        return;
      end
      
      % For Windows absolute paths, convert drive letter
      if isempty(obj.list)
        error('apt:Path:EmptyPath', 'Cannot convert empty absolute path');
      end
      
      % Extract drive letter from first component (e.g., 'C:')
      head = obj.list{1};
      if ~(length(head) == 2 && isstrprop(head(1), 'alpha') && head(2) == ':')
        error('apt:Path:InvalidWindowsPath', 'Windows absolute path must start with drive letter (e.g., ''C:'')');
      end
      
      % Convert drive letter: 'C:' -> {'', 'mnt', 'c'}
      driveLetter = lower(head(1));
      wslPrefix = {'', 'mnt', driveLetter};
      
      % Create new path list with WSL mount point
      if length(obj.list) == 1
        % Just the drive letter, no additional path components
        newPathList = wslPrefix;
      else
        % Drive letter plus additional components
        newPathList = [wslPrefix, obj.list(2:end)];
      end
      
      % Create new Linux path object
      result = apt.Path(newPathList, apt.Platform.posix);
    end

    function result = toWindows(obj)
      % Convert this path to a Windows-compatible path
      %
      % Returns:
      %   apt.Path: New path object compatible with Windows systems
      %
      % Notes:
      %   - For POSIX paths: converts WSL equivalent paths to Windows paths
      %     - POSIX absolute paths starting with /mnt/X: converts to Windows drive (X:)
      %     - Other POSIX absolute paths: returns unchanged with Windows platform
      %     - POSIX relative paths: returns Windows path with same components
      %   - For Windows paths: returns obj unchanged (already Windows-compatible)
      %
      % Example:
      %   posixPath = apt.Path('/mnt/c/Users/data', apt.Platform.posix);
      %   winPath = posixPath.toWindows();
      %   % winPath will be apt.Path('C:\Users\data', apt.Platform.windows)
      
      if obj.platform == apt.Platform.windows
        % Windows paths are already Windows-compatible
        result = obj;
        return
      end
      
      % Handle POSIX paths
      if ~obj.tfIsAbsolute()
        % For POSIX relative paths, just change platform to Windows
        result = apt.Path(obj.list, apt.Platform.windows);
        return;
      end
      
      % For POSIX absolute paths, check if it's a WSL mount point
      if length(obj.list) >= 3 && ...
         isempty(obj.list{1}) && ...
         strcmp(obj.list{2}, 'mnt') && ...
         length(obj.list{3}) == 1 && ...
         isstrprop(obj.list{3}, 'alpha')
        
        % This is a WSL mount point path like /mnt/c/...
        % Extract drive letter and convert to Windows format
        driveLetter = upper(obj.list{3});
        driveComponent = [driveLetter ':'];
        
        % Create new path list with Windows drive letter
        if length(obj.list) == 3
          % Just /mnt/c, no additional path components
          newPathList = {driveComponent};
        else
          % /mnt/c plus additional components
          newPathList = [{driveComponent}, obj.list(4:end)];
        end
        
        % Create new Windows path object
        result = apt.Path(newPathList, apt.Platform.windows);
      else
        % Not a WSL mount point, just change platform to Windows
        % This handles cases like /usr/local/bin -> C:\usr\local\bin (hypothetically)
        result = apt.Path(obj.list, apt.Platform.windows);
      end
    end  % function

    % function disp(obj)
    %   % Display the apt.Path object
    %   pathStr = obj.char();
    %   if obj.tfIsAbsolute
    %     absoluteStr = 'abs';
    %   else
    %     absoluteStr = 'rel';
    %   end
    %   fprintf('apt.Path: %s [%s:%s]\n', ...
    %           pathStr, ...
    %           char(obj.platform), ...
    %           absoluteStr);
    % end  % function

    function result = isequal(obj, other)
      % Test for equality with another apt.Path object
      %
      % Args:
      %   other: Another apt.Path object to compare with
      %
      % Returns:
      %   logical: true if the paths are equal, false otherwise
      
      result = isa(other, 'apt.Path') && ...
               obj.platform == other.platform && ...
               isequal(obj.list, other.list);
    end  % function
  end  % methods

  methods (Static)
    function result = stringToList_(pathAsString, platform)
      % Convert string path to list of components
      if platform == apt.Platform.windows
        % Windows path - split on both / and \
        preResult = strsplit(pathAsString, {'\', '/'}, 'CollapseDelimiters', false);
        % Remove empty components for Windows
        isNonemptyFromIndex = ~cellfun(@isempty, preResult);
        result = preResult(isNonemptyFromIndex);
      else
        % Unix-style path - split on /
        preResult = strsplit(pathAsString, '/', 'CollapseDelimiters', false);
        % For Unix, keep the first empty component if it exists (indicates absolute path)        
        if isempty(preResult)
          error('apt:Path:badStringInput', 'After spliting, list of raw component paths is empty') ;
        end
        first = preResult{1} ;
        if isempty(first)
          % Absolute path - keep first empty element, remove other empty elements
          preRest = preResult(2:end) ;
          isNonemptyFromIndex = ~cellfun(@isempty, preRest);
          rest = preRest(isNonemptyFromIndex);
          result = [ {first} rest ];
        else
          % Relative path - remove all empty components
          isNonemptyFromIndex = ~cellfun(@isempty, preResult);
          result = preResult(isNonemptyFromIndex);
        end
      end
    end

    function result = listToString_(pathList, platform)
      % Convert list of path components to string
      if isempty(pathList)
        result = '.';
      else
        if platform == apt.Platform.windows
          % Windows - use backslashes
          separator = '\';
          result = strjoin(pathList, separator);
        else
          % Unix-style - use forward slashes
          separator = '/';
          % For absolute paths, first element should be empty, so strjoin will create leading /
          % For relative paths, no empty first element, so no leading /
          % Special case: root path with single empty element should return '/'
          if length(pathList) == 1 && isempty(pathList{1})
            result = '/';
          else
            result = strjoin(pathList, separator);
          end
        end
      end
    end

    function result = tfIsAbsolutePath_(pathAsString, platform)
      % Determine if a path string is absolute
      if platform == apt.Platform.windows
        % Windows: absolute if starts with drive letter (e.g., "C:")
        result = length(pathAsString) >= 2 && isstrprop(pathAsString(1), 'alpha') && pathAsString(2) == ':';
      else
        % Unix-style: absolute if starts with "/"
        result = ~isempty(pathAsString) && pathAsString(1) == '/';
      end
    end

    function result = tfIsAbsoluteList_(pathList, platform)
      % Determine if a path component list represents an absolute path
      if isempty(pathList)
        result = false;
        return
      end      
      if platform == apt.Platform.windows
        % Windows: absolute if first component is a drive letter then a ':'
        firstElement = pathList{1};
        result = (length(firstElement) == 2) && isstrprop(firstElement(1), 'alpha') && firstElement(2) == ':' ;
      else
        % Unix-style: absolute if first component is empty
        result = isempty(pathList{1});
      end
    end
  end  % methods (Static)

  methods
    function result = encode_for_persistence_(obj, do_wrap_in_container)
      encoding = struct('list', {obj.list}, 'platform', {obj.platform}) ;
      if do_wrap_in_container
        result = encoding_container('apt.Path', encoding) ;
      else
        result = encoding ;
      end
    end
  end  % methods
  
  methods (Static)
    function result = decode_encoding(encoding)
      % Decode the encoded version of the object.  Used for loading from persistent
      % storage.
      result = apt.Path(encoding.list, encoding.platform) ;
    end
  end  % methods (Static)

  methods
    function result = leafName(obj)
      % Return leaf name as char array.  The leaf name is the final element of the
      % path.
      list = obj.list ;
      if isempty(list)
        error('Path is null, so no leafName available') ;
      end
      result = list{end} ;
    end

    function result = extension(obj)
      % The file extension of the the path.  Uses same conventions as fileparts().
      fileName = obj.leafName() ;
      [~,~,result] = fileparts(fileName) ;
    end

    function replaceExtension(obj, newExtension)
      originalFileName = obj.leafName() ;
      [~,baseName,~] = fileparts(originalFileName) ;
      newFileName = sprintf('%s%s', baseName, newExtension) ;
      obj.list{end} = newFileName ;
    end

    function [rest, leaf] = split(obj)
      % Return leaf name as char array, and the rest of the path as an apt.Path.  The leaf name is the final element of the
      % path.  Errors if obj is null.  If obj holds a single-element path, rest will
      % be the null path, leaf will be a the single element as a char arrray.
      list = obj.list ;
      if isempty(list)        
        error('Path is null, so can''t split') ;
      end
      leaf = list{end} ;
      restList = list(1:end-1) ;
      rest = apt.Path(restList, obj.platform) ;
    end
  end  % methods
  
end  % classdef
