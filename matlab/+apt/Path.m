classdef Path
  % apt.Path - A value class to represent paths, with platform awareness.
  %
  % This class encapsulates paths with knowledge of platform type.
  %
  % Example usage:
  %   p = apt.Path({'C:', 'data', 'movie.avi'});
  %   pathStr = p.toString();

  properties
    list_        % Cell array of path components
    platform_    
      % apt.Os enumeration.  Represents the OS the fronted is currently running on,
      % usually.  Present so that we can test Path functionality on e.g. Windows on
      % e.g. Linux.
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
      %   listOrString (cell or char): Cell array of path components or path string
      %   platform (char or apt.Os, optional): 'linux', 'windows', 'macos', or enum

      % Deal with args
      if ~exist('rawPlatform', 'var') || isempty(rawPlatform)
        rawPlatform = apt.Os.current();
      end

      % Convert string to enum if needed
      if ischar(rawPlatform)        
        platform = apt.Os.fromString(rawPlatform);
      else
        platform = rawPlatform;
      end

      if ischar(listOrString)
        str = listOrString ;
        % Check for empty string input
        if isempty(str)
          error('apt:Path:EmptyPath', 'Cannot create path from empty string');
        end
        % Convert string path to list
        obj.list_ = apt.Path.stringToList_(str, platform);
        
        % Check for Windows root path case
        if isempty(obj.list_) && platform == apt.Os.windows
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

      % .list_ cannot be empty
      if isempty(obj.list_)
        error('apt:Path:InvalidList', 'The Path list cannot be empty') ;
      end

      % For Linux absolute paths, first element must be empty string
      if platform ~= apt.Os.windows
        if obj.tfIsAbsolute_
          if isempty(obj.list_{1})
            % all is well
          else
            error('apt:Path:InvalidAbsolutePath', 'Linux absolute paths must have empty string as first element');
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

    function result = toString(obj)
      % Get the path as a string
      result = apt.Path.listToString_(obj.list_, obj.platform_);
    end

    function result = cat2(obj, other)
      % Concatenate this path with another path or string
      %
      % Args:
      %   other (apt.Path or char): The path to concatenate
      %
      % Returns:
      %   apt.Path: New path object representing the concatenation
      %
      % Notes:
      %   - If other is a string, it's converted to apt.Path with same platform
      %   - The other path must be relative (not absolute)
      %   - The result will have the same platform as this path
      %   - The result will be absolute if this path is absolute
      
      % Convert string to apt.Path if needed
      if ischar(other)
        other = apt.Path(other, obj.platform);
      elseif ~isa(other, 'apt.Path')
        error('apt:Path:InvalidArgument', 'Argument must be an apt.Path object or string');
      end
      
      if other.tfIsAbsolute
        error('apt:Path:AbsolutePath', 'Cannot concatenate with an absolute path');
      end
      
      if obj.platform ~= other.platform
        error('apt:Path:PlatformMismatch', 'Cannot concatenate paths from different platforms');
      end
      
      % Concatenate the path components
      newList = [obj.list_, other.list_];
      
      % Create new path object
      result = apt.Path(newList, obj.platform);
    end

    function result = cat(obj, varargin)
      % Concatenate this path with multiple paths or strings
      %
      % Args:
      %   varargin: Variable number of apt.Path objects or strings to concatenate
      %
      % Returns:
      %   apt.Path: New path object representing the concatenation of all inputs
      %
      % Notes:
      %   - Uses cat2() method as building block
      %   - All paths must be relative (not absolute)
      %   - If strings are provided, they're converted to apt.Path with same platform
      %   - All paths must have compatible platforms
      
      result = obj;
      
      % Concatenate each argument in sequence
      for i = 1:length(varargin)
        result = result.cat2(varargin{i});
      end
    end

    function result = eq(obj, other)
      % Check equality with another apt.Path
      if ~isa(other, 'apt.Path')
        result = false;
        return;
      end

      result = isequal(obj.list_, other.list_) && ...
               obj.platform_ == other.platform_ && ...
               obj.tfIsAbsolute_ == other.tfIsAbsolute_;
    end

    function disp(obj)
      % Display the apt.Path object
      pathStr = obj.toString();
      if obj.tfIsAbsolute_
        absoluteStr = 'abs';
      else
        absoluteStr = 'rel';
      end
      fprintf('apt.Path: %s [%s:%s]\n', ...
              pathStr, ...
              apt.Os.toString(obj.platform_), ...
              absoluteStr);
    end  % function
  end  % methods

  methods (Static)
    function result = stringToList_(pathAsString, platform)
      % Convert string path to list of components
      if platform == apt.Os.windows
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
      if platform == apt.Os.windows
        % Windows - use backslashes
        separator = '\';
        result = strjoin(pathList, separator);
      else
        % Unix-style - use forward slashes
        separator = '/';
        % For absolute paths, first element should be empty, so strjoin will create leading /
        % For relative paths, no empty first element, so no leading /
        result = strjoin(pathList, separator);
      end
    end

    function result = isAbsolutePath_(pathAsString, platform)
      % Determine if a path string is absolute
      if platform == apt.Os.windows
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
      
      if platform == apt.Os.windows
        % Windows: absolute if first component ends with ":"
        firstComponent = pathList{1};
        result = ~isempty(firstComponent) && firstComponent(end) == ':';
      else
        % Unix-style: absolute if first component is empty (from leading "/")
        result = isempty(pathList{1});
      end
    end

  end  % methods (Static)
end  % classdef