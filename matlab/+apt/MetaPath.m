classdef MetaPath
  % apt.MetaPath - A value class to represent paths with locale and file role
  % awareness.
  %
  % This class encapsulates paths used in APT with knowledge of:
  % 1. Locale: native, wsl, remote
  % 2. File role: cache, or movie
  %
  % Example usage:
  %   p = apt.MetaPath(apt.Path('/data/movie.avi'), 'native', 'movie');
  %   wslPath = p.as('wsl');
  %   remotePath = p.as('remote');

  properties
    path_        % apt.Path object containing the actual path
    locale_      % apt.PathLocale enumeration.  Describes the 'locale' (native/wsl/remote) where the path is 
                 % appropriate.
    role_        % apt.FileRole enumeration.
  end

  properties (Dependent)
    locale       % Get the locale
    role         % Get the file role
    path         % Get the underlying apt.Path object
    platform     % Get the platform from the underlying path
  end

  methods
    function obj = MetaPath(path, locale, role)
      % Constructor for apt.MetaPath
      %
      % Args:
      %   pathObj (apt.Path): apt.Path object containing the actual path
      %   locale (char or apt.PathLocale): 'native', 'wsl', 'remote', or enum
      %   role (char or apt.FileRole): 'cache', 'movie', or enum

      if ~exist('role', 'var') || isempty(role)
        role = apt.FileRole.cache;
      end
      if ~exist('locale', 'var') || isempty(locale)
        locale = apt.PathLocale.native;
      end

      % Convert string to enum if needed
      if ischar(locale)
        locale = apt.PathLocale.fromString(locale);
      end
      if ischar(role)
        role = apt.FileRole.fromString(role);
      end

      % Validate that pathObj is an apt.Path object
      if ~isa(path, 'apt.Path')
        error('apt:MetaPath:InvalidPath', 'First argument must be an apt.Path object');
      end

      % Validate that the path is absolute
      if ~path.tfIsAbsolute
        error('apt:MetaPath:RelativePath', 'MetaPath requires an absolute path');
      end

      obj.path_ = path;
      obj.locale_ = locale;
      obj.role_ = role;
    end

    function result = get.locale(obj)
      result = obj.locale_;
    end

    function result = get.role(obj)
      result = obj.role_;
    end

    function result = get.path(obj)
      result = obj.path_;
    end

    function result = get.platform(obj)
      result = obj.path_.platform;
    end

    function result = toString(obj)
      % Get the path as a string
      result = obj.path_.toString();
    end

    function result = eq(obj, other)
      % Check equality with another apt.MetaPath
      if ~isa(other, 'apt.MetaPath')
        result = false;
        return;
      end

      result = obj.path_.eq(other.path_) && ...
               obj.locale_ == other.locale_ && ...
               obj.role_ == other.role_;
    end

    function disp(obj)
      % Display the apt.MetaPath object
      pathStr = obj.toString();
      fprintf('apt.MetaPath: %s [%s:%s:%s]\n', ...
              pathStr, ...
              apt.PathLocale.toString(obj.locale_), ...
              apt.FileRole.toString(obj.role_), ...
              apt.Os.toString(obj.path_.platform));
    end
  end  % methods

end  % classdef