classdef MetaPath
  % apt.MetaPath - A value class to represent *absolute* paths with platform and file role
  % awareness.  All represented paths are absolute paths.
  %
  % This class encapsulates paths used in APT with knowledge of:
  % 1. Platform: native, wsl, remote
  % 2. File role: cache, or movie
  %
  % Example usage:
  %   p = apt.MetaPath({'C:', 'data', 'movie.avi'}, 'native', 'movie');
  %   wslPath = p.as('wsl');
  %   remotePath = p.as('remote');

  properties
    list_        % Cell array of path components
    locale_      % apt.PathLocale enumeration.  Describes the 'locale' (native/wsl/remote) where the absolute path represented by list_ is 
                 % appropriate.
    fileRole_    % apt.FileRole enumeration.
    platform_    
      % apt.Os enumeration.  Represents the OS the fronted is currently running on,
      % usually.  Present so that we can test Path functionality on e.g. Windows on
      % e.g. Linux.
  end

  properties (Dependent)
    locale       % Get the locale
    fileRole     % Get the file role
    list         % Get the path components list
    platform     % Get the platform
  end

  methods
    function obj = MetaPath(listOrString, locale, fileRole, platform)
      % Constructor for apt.MetaPath
      %
      % Args:
      %   pathList (cell): Cell array of path components
      %   locale (char or apt.PathLocale): 'native', 'wsl', 'remote', or enum
      %   fileRole (char or apt.FileRole): 'cache', 'movie', or enum
      %   platform (char or apt.Os, optional): 'linux', 'windows', 'macos', or enum

      if ~exist('fileRole', 'var') || isempty(fileRole)
        fileRole = apt.FileRole.cache;
      end
      if ~exist('locale', 'var') || isempty(locale)
        locale = apt.PathLocale.native;
      end
      if ~exist('platform', 'var') || isempty(platform)
        platform = apt.Os.current();
      end

      % Convert string to enum if needed
      if ischar(locale)
        locale = apt.PathLocale.fromString(locale);
      end
      if ischar(fileRole)
        fileRole = apt.FileRole.fromString(fileRole);
      end
      if ischar(platform)
        platform = apt.Os.fromString(platform);
      end

      if ischar(listOrString)
        % Check for empty string input
        if isempty(listOrString)
          error('apt:Path:EmptyPath', 'Cannot create path from empty string');
        end
        % Convert string path to list
        obj.list_ = apt.MetaPath.stringToList_(listOrString, locale, platform);
        % Check for Windows root path case
        if isempty(obj.list_) && locale == apt.PathLocale.native && platform == apt.Os.windows
          error('apt:Path:EmptyPath', 'Cannot create Windows path from root path "/"');
        end
      elseif iscell(listOrString)
        obj.list_ = listOrString;
      else
        error('listOrString must be a string or a cell array of strings') ;
      end

      obj.locale_ = locale;
      obj.fileRole_ = fileRole;
      obj.platform_ = platform;
    end

    function result = get.locale(obj)
      result = obj.locale_;
    end

    function result = get.fileRole(obj)
      result = obj.fileRole_;
    end

    function result = get.list(obj)
      result = obj.list_;
    end

    function result = get.platform(obj)
      result = obj.platform_;
    end

    function result = toString(obj)
      % Get the path as a string
      result = apt.MetaPath.listToString_(obj.list_, obj.locale_, obj.platform_);
    end

    function result = eq(obj, other)
      % Check equality with another apt.MetaPath
      if ~isa(other, 'apt.MetaPath')
        result = false;
        return;
      end

      result = isequal(obj.list_, other.list_) && ...
               obj.locale_ == other.locale_ && ...
               obj.fileRole_ == other.fileRole_ && ...
               obj.platform_ == other.platform_;
    end

    function disp(obj)
      % Display the apt.MetaPath object
      pathStr = obj.toString();
      fprintf('apt.MetaPath: %s [%s:%s:%s]\n', ...
              pathStr, ...
              apt.PathLocale.toString(obj.locale_), ...
              apt.FileRole.toString(obj.fileRole_), ...
              apt.Os.toString(obj.platform_));
    end
  end  % methods

  methods (Static)
    function result = stringToList_(pathAsString, locale, platform)
      % Convert string path to list of components
      if locale == apt.PathLocale.native && platform == apt.Os.windows
        % Windows path - split on both / and \
        preResult = strsplit(pathAsString, {'\', '/'}, 'CollapseDelimiters', false);
      else
        % Unix-style path - split on /
        preResult = strsplit(pathAsString, '/', 'CollapseDelimiters', false);
      end

      % Remove empty components
      isNonemptyFromIndex = ~cellfun(@isempty, preResult);
      result = preResult(isNonemptyFromIndex);
    end

    function result = listToString_(pathList, locale, platform)
      % Convert list of path components to string
      if locale == apt.PathLocale.native && platform == apt.Os.windows
        % Windows - use backslashes
        separator = '\';
        result = strjoin(pathList, separator);
      else
        % Unix-style - use forward slashes and prepend / since we only represent absolute paths
        separator = '/';
        result = ['/' strjoin(pathList, separator)];
      end
    end

  end  % methods (Static)
end  % classdef