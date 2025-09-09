classdef Path
  % apt.Path - A value class to represent paths, with platform awareness.
  %
  % This class encapsulates paths with knowledge of platform type.
  %
  % Example usage:
  %   p = apt.Path({'C:', 'data', 'movie.avi'});
  %   pathStr = p.char();

  properties
    list_        % Cell array of path components
    platform_    
      % apt.Platform enumeration.  Represents the platform the frontend is currently running on,
      % usually.  Present so that we can test Path functionality on e.g. Windows on
      % e.g. POSIX.
  end

  properties (Transient)
    tfIsAbsolute_ % Logical scalar indicating whether the path is absolute
  end

  properties (Dependent)
    list           % Get the path components list
    platform       % Get the platform
    tfIsAbsolute   % Get whether the path is absolute
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
        obj.list_ = cell(1,0);  % Empty row vector for empty path
        obj.platform_ = platform;
        obj.tfIsAbsolute_ = false;  % The empty path is a relative path
        return;
      end

      if ischar(listOrString)
        str = listOrString ;
        % Convert string path to list
        obj.list_ = apt.Path.stringToList_(str, platform);
        
        % Check for Windows root path case
        if isempty(obj.list_) && platform == apt.Platform.windows
          error('apt:Path:EmptyPath', 'Cannot create Windows path from path "/"');
        end
      elseif iscell(listOrString)
        lst = listOrString ;
        obj.list_ = lst;        
      else
        error('listOrString must be a string or a cell array of strings') ;
      end

      % Set the platform
      obj.platform_ = platform;
      
      % Determine if the path is absolute
      obj.tfIsAbsolute_ = apt.Path.isAbsoluteList_(obj.list_, platform);

      % Remove '.' elements from the path list
      if ~isempty(obj.list_)
        isDotFromIndex = strcmp(obj.list_, '.');
        obj.list_(isDotFromIndex) = [];
      end

      % Make sure an empty path is relative, then exit early
      if isempty(obj.list_)
        obj.tfIsAbsolute_ = false;  % The empty path is a relative path
        return
      end

      % For POSIX absolute paths, first element must be empty string
      if platform ~= apt.Platform.windows
        if obj.tfIsAbsolute_
          if isempty(obj.list_{1})
            % all is well
          else
            error('apt:Path:InvalidAbsolutePath', 'POSIX absolute paths must have empty string as first element');
          end
        end
      end
      
      % INVARIANT: For Windows absolute paths, first element must be drive letter + colon
      if platform == apt.Platform.windows
        if obj.tfIsAbsolute_
          firstElement = obj.list_{1};
          if length(firstElement) == 2 && isstrprop(firstElement(1), 'alpha') && firstElement(2) == ':'
            % all is well
          else
            error('apt:Path:InvalidWindowsAbsolutePath', 'Windows absolute paths must have drive letter format (e.g., ''C:'') as first element');
          end
        end
      end      
    end  % function

    function result = get.list(obj)
      result = obj.list_;
    end

    function result = get.platform(obj)
      result = obj.platform_;
    end

    function result = get.tfIsAbsolute(obj)
      result = obj.tfIsAbsolute_;
    end

    function result = char(obj)
      % Get the path as a string
      preResult = apt.Path.listToString_(obj.list_, obj.platform_);
      result = escape_string_for_bash(preResult) ;
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
      newList = obj.list_;
      
      % Concatenate each argument in sequence
      for i = 1:length(pathArgs)
        otherPath = pathArgs{i};
        
        % Check that the path is relative
        if otherPath.tfIsAbsolute
          error('apt:Path:AbsolutePath', 'Cannot concatenate with an absolute path at argument %d', i);
        end
        
        % Check platform compatibility
        if obj.platform ~= otherPath.platform
          error('apt:Path:PlatformMismatch', 'Cannot concatenate paths from different platforms (argument %d)', i);
        end
        
        % Skip empty paths
        if ~isempty(otherPath.list_)
          newList = horzcat(newList, otherPath.list_);
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
      newList = obj.list_;
      
      % Append each char array as a new component
      for i = 1:length(varargin)
        component = varargin{i};
        newList = horzcat(newList, {component});
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
      
      if isempty(obj.list_)
        error('apt:Path:EmptyPath', 'Cannot get fileparts of empty path');
      end
      
      % Get the last component (filename)
      filename = obj.list_{end};
      
      % Create the path part by removing the last component
      if isscalar(obj.list_)
        % Only one component - create appropriate empty path
        if obj.tfIsAbsolute_
          % For absolute paths with one component, path part should be root
          if obj.platform_ == apt.Platform.windows
            % Windows: just the drive letter
            pathPart = apt.Path({filename}, obj.platform_);
          else
            % Unix: root directory
            pathPart = apt.Path({''}, obj.platform_);
          end
        else
          % For relative paths with one component, path part should be empty relative path
          pathPart = apt.Path('.', obj.platform_);
        end
      else
        % Multiple components - take all but the last
        pathList = obj.list_(1:end-1);
        pathPart = apt.Path(pathList, obj.platform_);
      end
      
      % Create the filename part as a relative path
      filenamePart = apt.Path({filename}, obj.platform_);
    end

    function result = replacePrefix(obj, sourcePath, targetPath)
      % Replace a source prefix with a target prefix
      %
      % Args:
      %   sourcePath (apt.Path or char): The prefix to replace
      %   targetPath (apt.Path or char): The replacement prefix
      %
      % Returns:
      %   apt.Path: New path with prefix replaced, or original path if prefix doesn't match
      %
      % Example:
      %   p = apt.Path('/old/base/file.txt');
      %   newP = p.replacePrefix('/old/base', '/new/location');
      %   % newP will be apt.Path('/new/location/file.txt')
      
      % Convert string arguments to apt.Path if needed
      if ischar(sourcePath)
        sourcePath = apt.Path(sourcePath, obj.platform_);
      elseif ~isa(sourcePath, 'apt.Path')
        error('apt:Path:InvalidArgument', 'sourcePath must be an apt.Path object or string');
      end
      
      if ischar(targetPath)
        targetPath = apt.Path(targetPath, obj.platform_);
      elseif ~isa(targetPath, 'apt.Path')
        error('apt:Path:InvalidArgument', 'targetPath must be an apt.Path object or string');
      end
      
      % Check platform compatibility
      if obj.platform_ ~= sourcePath.platform_ || obj.platform_ ~= targetPath.platform_
        error('apt:Path:PlatformMismatch', 'All paths must have the same platform');
      end
      
      % Check if this path starts with the source prefix
      sourceList = sourcePath.list_;
      if length(obj.list_) < length(sourceList)
        % Path is shorter than source prefix, no match
        result = obj;
        return;
      end
      
      % Check if the beginning of obj.list_ matches sourceList
      if ~isequal(obj.list_(1:length(sourceList)), sourceList)
        % Prefix doesn't match
        result = obj;
        return;
      end
      
      % Replace the prefix
      remainingList = obj.list_(length(sourceList)+1:end);
      newList = [targetPath.list_, remainingList];
      
      % Create new path object
      result = apt.Path(newList, obj.platform_);
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
      
      if obj.platform_ == apt.Platform.posix
        % Linux/Mac paths are already POSIX-compatible
        result = obj;
        return
      end
      
      % Handle Windows paths
      if ~obj.tfIsAbsolute_
        % For Windows relative paths, just change platform to Linux
        result = apt.Path(obj.list_, apt.Platform.posix);
        return;
      end
      
      % For Windows absolute paths, convert drive letter
      if isempty(obj.list_)
        error('apt:Path:EmptyPath', 'Cannot convert empty absolute path');
      end
      
      % Extract drive letter from first component (e.g., 'C:')
      head = obj.list_{1};
      if ~(length(head) == 2 && isstrprop(head(1), 'alpha') && head(2) == ':')
        error('apt:Path:InvalidWindowsPath', 'Windows absolute path must start with drive letter (e.g., ''C:'')');
      end
      
      % Convert drive letter: 'C:' -> {'', 'mnt', 'c'}
      driveLetter = lower(head(1));
      wslPrefix = {'', 'mnt', driveLetter};
      
      % Create new path list with WSL mount point
      if length(obj.list_) == 1
        % Just the drive letter, no additional path components
        newPathList = wslPrefix;
      else
        % Drive letter plus additional components
        newPathList = [wslPrefix, obj.list_(2:end)];
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
      
      if obj.platform_ == apt.Platform.windows
        % Windows paths are already Windows-compatible
        result = obj;
        return
      end
      
      % Handle POSIX paths
      if ~obj.tfIsAbsolute_
        % For POSIX relative paths, just change platform to Windows
        result = apt.Path(obj.list_, apt.Platform.windows);
        return;
      end
      
      % For POSIX absolute paths, check if it's a WSL mount point
      if length(obj.list_) >= 3 && ...
         isempty(obj.list_{1}) && ...
         strcmp(obj.list_{2}, 'mnt') && ...
         length(obj.list_{3}) == 1 && ...
         isstrprop(obj.list_{3}, 'alpha')
        
        % This is a WSL mount point path like /mnt/c/...
        % Extract drive letter and convert to Windows format
        driveLetter = upper(obj.list_{3});
        driveComponent = [driveLetter ':'];
        
        % Create new path list with Windows drive letter
        if length(obj.list_) == 3
          % Just /mnt/c, no additional path components
          newPathList = {driveComponent};
        else
          % /mnt/c plus additional components
          newPathList = [{driveComponent}, obj.list_(4:end)];
        end
        
        % Create new Windows path object
        result = apt.Path(newPathList, apt.Platform.windows);
      else
        % Not a WSL mount point, just change platform to Windows
        % This handles cases like /usr/local/bin -> C:\usr\local\bin (hypothetically)
        result = apt.Path(obj.list_, apt.Platform.windows);
      end
    end  % function

    function disp(obj)
      % Display the apt.Path object
      pathStr = obj.char();
      if obj.tfIsAbsolute_
        absoluteStr = 'abs';
      else
        absoluteStr = 'rel';
      end
      fprintf('apt.Path: %s [%s:%s]\n', ...
              pathStr, ...
              char(obj.platform_), ...
              absoluteStr);
    end  % function

    function result = isequal(obj, other)
      % Test for equality with another apt.Path object
      %
      % Args:
      %   other: Another apt.Path object to compare with
      %
      % Returns:
      %   logical: true if the paths are equal, false otherwise
      
      result = isa(other, 'apt.Path') && ...
               obj.platform_ == other.platform_ && ...
               obj.tfIsAbsolute_ == other.tfIsAbsolute_ && ...
               isequal(obj.list_, other.list_);
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

    function result = isAbsolutePath_(pathAsString, platform)
      % Determine if a path string is absolute
      if platform == apt.Platform.windows
        % Windows: absolute if starts with drive letter (e.g., "C:")
        result = length(pathAsString) >= 2 && pathAsString(2) == ':';
      else
        % Unix-style: absolute if starts with "/"
        result = ~isempty(pathAsString) && pathAsString(1) == '/';
      end
    end

    function result = isAbsoluteList_(pathList, platform)
      % Determine if a path component list represents an absolute path
      if isempty(pathList)
        result = false;
        return;
      end
      
      if platform == apt.Platform.windows
        % Windows: absolute if first component ends with ":"
        firstComponent = pathList{1};
        result = ~isempty(firstComponent) && firstComponent(end) == ':';
      else
        % Unix-style: absolute if first component is empty (from leading "/")
        result = isempty(pathList{1});
      end
    end
  end  % methods (Static)

  methods
    function result = encode_for_persistence_(obj, do_wrap_in_container)
      encoding = struct('list_', {obj.list_}, 'platform_', {obj.platform_}, 'tfIsAbsolute_', {obj.tfIsAbsolute_}) ;
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
      result = apt.Path(encoding.list_, encoding.platform_) ;
    end
  end  % methods (Static)
end  % classdef
