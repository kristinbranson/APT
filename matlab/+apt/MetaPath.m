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

    function result = replacePrefix(obj, rawSourcePath, rawTargetPath)
      % Replace a source prefix with a target prefix in the underlying path
      %
      % Args:
      %   rawSourcePath (apt.Path, apt.MetaPath, or char): The prefix to replace
      %   rawTargetPath (apt.Path, apt.MetaPath, or char): The replacement prefix
      %
      % Returns:
      %   apt.MetaPath: New MetaPath with prefix replaced, preserving locale and role
      %
      % Example:
      %   mp = apt.MetaPath(apt.Path('/old/base/file.txt'), 'native', 'movie');
      %   newMp = mp.replacePrefix('/old/base', '/new/location');
      %   % newMp will have path '/new/location/file.txt' with same locale and role
      
      % Extract apt.Path objects from arguments
      if isa(rawSourcePath, 'apt.MetaPath')
        sourcePath = rawSourcePath.path_;
      elseif isa(rawSourcePath, 'apt.Path')
        sourcePath = rawSourcePath;
      elseif ischar(rawSourcePath)
        sourcePath = apt.Path(rawSourcePath, obj.path_.platform);
      else
        error('apt:MetaPath:InvalidArgument', 'rawSourcePath must be an apt.Path, apt.MetaPath, or string');
      end
      
      if isa(rawTargetPath, 'apt.MetaPath')
        targetPath = rawTargetPath.path_;
      elseif isa(rawTargetPath, 'apt.Path')
        targetPath = rawTargetPath;
      elseif ischar(rawTargetPath)
        targetPath = apt.Path(rawTargetPath, obj.path_.platform);
      else
        error('apt:MetaPath:InvalidArgument', 'rawTargetPath must be an apt.Path, apt.MetaPath, or string');
      end
      
      % Use the underlying apt.Path replacePrefix method
      newPath = obj.path_.replacePrefix(sourcePath, targetPath);
      
      % Create new MetaPath with the same locale and role
      result = apt.MetaPath(newPath, obj.locale_, obj.role_);
    end

    function result = as(obj, targetLocale)
      % Convert MetaPath to a different locale
      %
      % Args:
      %   targetLocale (char or apt.PathLocale): Target locale ('native', 'wsl', 'remote', or enum)
      %
      % Returns:
      %   apt.MetaPath: New MetaPath with converted locale, same role
      %
      % Notes:
      %   - If target locale equals source locale, returns obj unchanged
      %   - Currently supports: native -> wsl conversion
      %   - Other conversions throw unsupported error
      
      % Convert string to enum if needed
      if ischar(targetLocale)
        targetLocale = apt.PathLocale.fromString(targetLocale);
      end
      
      % If target equals source, return unchanged
      if targetLocale == obj.locale_
        result = obj;
        return;
      end
      
      % Handle supported conversions
      if obj.locale_ == apt.PathLocale.native && targetLocale == apt.PathLocale.wsl
        % Convert native path to WSL path using static method
        result = apt.MetaPath.toWslFromNative_(obj);
      else
        % Unsupported conversion
        error('apt:MetaPath:UnsupportedConversion', ...
              'Conversion from %s to %s is not yet supported', ...
              apt.PathLocale.toString(obj.locale_), ...
              apt.PathLocale.toString(targetLocale));
      end
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

  methods (Static)
    function result = toWslFromNative_(inputMetaPath)
      % Convert a native apt.MetaPath to WSL locale using path operations
      %
      % Args:
      %   inputPath (apt.MetaPath): Native path to convert
      %
      % Returns:
      %   apt.MetaPath: Converted WSL path with same role
      %
      % Notes:
      %   - Input must be native locale and absolute
      %   - On Windows: converts drive letters (C: -> /mnt/c) and backslashes
      %   - On Linux: identity operation (native paths are already Unix-style)
      %   - Uses path operations, never converts to raw strings
      
      % Validate input
      assert(isa(inputMetaPath, 'apt.MetaPath'), 'inputPath must be an apt.MetaPath instance');
      assert(inputMetaPath.locale == apt.PathLocale.native, 'inputPath must have native locale');
      assert(inputMetaPath.path.tfIsAbsolute, 'inputPath must be an absolute path');
      
      % On non-Windows platforms, native paths are already WSL-compatible
      inputPath = inputMetaPath.path ;
      if inputPath.platform ~= apt.Os.windows
        result = apt.MetaPath(inputPath, apt.PathLocale.wsl, inputMetaPath.role);
        return
      end
      
      % For a Windows path, need to convert drive letter
      newPath = inputPath.toPosix() ;
      
      % Create new metapath with converted components
      result = apt.MetaPath(newPath, apt.PathLocale.wsl, inputMetaPath.role);
    end
  end  % methods (Static)

end  % classdef
