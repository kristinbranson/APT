classdef MetaPath < apt.ShellToken
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
    path         % Get the underlying apt.Path object
    locale       % Get the locale
    role         % Get the file role
    platform     % Get the platform from the underlying path
  end

  methods
    function obj = MetaPath(rawPath, locale, role)
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

      % If rawPath is a char array, convert to apt.Path
      if ischar(rawPath) 
        if (locale == apt.PathLocale.wsl || locale == apt.PathLocale.remote)
          path = apt.Path(rawPath, apt.Platform.posix) ;  % wsl and remote are always posix
        else
          path = apt.Path(rawPath) ;  % use current platform
        end
      else
        path = rawPath ;  % assume it's already an apt.Path object
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

    function result = char(obj)
      % Get the path as a string
      result = obj.path_.char();
    end

    function result = tfDoesMatchLocale(obj, queryLocale)
      % Check if this MetaPath's locale matches the query locale
      %
      % Args:
      %   queryLocale (char or apt.PathLocale): The locale to check against
      %
      % Returns:
      %   logical: True if locales match
      
      % Convert string to enum if needed
      if ischar(queryLocale)
        queryLocale = apt.PathLocale.fromString(queryLocale);
      end
      
      result = (obj.locale_ == queryLocale);
    end

    function result = tfDoesMatchPlatform(obj, queryPlatform)
      % Check if this MetaPath's platform matches the query platform
      %
      % Args:
      %   queryPlatform (char or apt.Platform): The platform to check against
      %
      % Returns:
      %   logical: True if platforms match
      
      % Convert string to enum if needed
      if ischar(queryPlatform)
        queryPlatform = apt.Platform.fromString(queryPlatform);
      end
      
      result = (obj.path_.platform == queryPlatform);
    end

    function result = isequal(obj, other)
      % Check equality with another apt.MetaPath
      if ~isa(other, 'apt.MetaPath')
        result = false;
        return;
      end

      result = isequal(obj.path_, other.path_) && ...
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

    function result = as(obj, targetLocale, varargin)
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
      %   - Generally, how to convert to the remote locale is backend-specific, so is
      %     not handled in this class.
      
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
      elseif obj.locale_ == apt.PathLocale.wsl && targetLocale == apt.PathLocale.native
        % Convert WSL path to native path using static method
        result = apt.MetaPath.toNativeFromWsl_(obj);
      else
        % Unsupported conversion
        error('apt:MetaPath:UnsupportedConversion', ...
              'Conversion from %s to %s is not yet supported', ...
              char(obj.locale_), ...
              char(targetLocale));
      end
    end  % function

    function result = asNative(obj)
      % Convenience method
      result = obj.as(apt.PathLocale.native) ;
    end

    function result = asWsl(obj)
      % Convenience method
      result = obj.as(apt.PathLocale.wsl) ;
    end

    function result = asRemote(obj, varargin)
      % Convenience method
      result = obj.as(apt.PathLocale.remote, varargin{:}) ;
    end

    function disp(obj)
      % Display the apt.MetaPath object
      pathStr = obj.char();
      fprintf('apt.MetaPath: %s [%s:%s:%s]\n', ...
              pathStr, ...
              char(obj.locale_), ...
              char(obj.role_), ...
              char(obj.path_.platform));
    end

    function result = cat(obj, varargin)
      % Concatenate this MetaPath with multiple apt.MetaPath objects or char arrays
      %
      % Args:
      %   varargin: Variable number of apt.MetaPath objects or char arrays to concatenate
      %
      % Returns:
      %   apt.MetaPath: New MetaPath with concatenated paths, same locale and role
      %
      % Notes:
      %   - Char arrays are converted to apt.MetaPath objects with the same locale and role
      %   - apt.MetaPath arguments must have compatible locales and platforms
      
      % Convert char arrays to apt.MetaPath objects and collect apt.Path objects for underlying cat
      pathArgs = cell(size(varargin));
      for i = 1:length(varargin)
        if ischar(varargin{i})
          % Convert char array to apt.MetaPath with same locale and role, then extract path
          metaPath = apt.MetaPath(varargin{i}, obj.locale_, obj.role_);
          pathArgs{i} = metaPath.path_;
        elseif isa(varargin{i}, 'apt.MetaPath')
          % Validate locale and role compatibility
          if varargin{i}.locale_ ~= obj.locale_
            error('apt:MetaPath:LocaleMismatch', ...
              'MetaPath argument at position %d has locale %s, but this MetaPath has locale %s', ...
              i, char(varargin{i}.locale_), char(obj.locale_));
          end
          if varargin{i}.role_ ~= obj.role_
            error('apt:MetaPath:RoleMismatch', ...
              'MetaPath argument at position %d has role %s, but this MetaPath has role %s', ...
              i, char(varargin{i}.role_), char(obj.role_));
          end
          pathArgs{i} = varargin{i}.path_;
        else
          error('apt:MetaPath:InvalidArgument', 'Argument %d must be an apt.MetaPath object or char array, got %s', i, class(varargin{i}));
        end
      end
      
      % Call the apt.Path.cat() method on the underlying paths
      oldPath = obj.path_;
      newPath = oldPath.cat(pathArgs{:});
      result = apt.MetaPath(newPath, obj.locale_, obj.role_);
    end

    function result = append(obj, varargin)
      % Call the apt.Path.append() method on our .path_. Leave rest alone.
      %
      % Args:
      %   varargin: Variable number of char arrays to append as path components
      %
      % Returns:
      %   apt.MetaPath: New MetaPath with the char arrays appended as components,
      %                 same locale and role as this MetaPath
      oldPath = obj.path_;
      newPath = oldPath.append(varargin{:});
      result = apt.MetaPath(newPath, obj.locale_, obj.role_);
    end

    function [pathPart, filenamePart] = fileparts2(obj)
      % Call apt.Path.fileparts2() method and wrap results as MetaPaths
      %
      % Returns:
      %   pathPart (apt.MetaPath): Directory path portion with same locale and role
      %   filenamePart (apt.MetaPath): Filename portion (name + extension) with same locale and role
      %
      % Example:
      %   mp = apt.MetaPath('/home/user/data/movie.avi', 'wsl', 'movie');
      %   [dir, file] = mp.fileparts2();
      %   % dir will be apt.MetaPath('/home/user/data', 'wsl', 'movie')
      %   % file will be apt.MetaPath('movie.avi', 'wsl', 'movie')
      
      [pathPartAsPath, filenamePartAsPath] = obj.path_.fileparts2();
      pathPart = apt.MetaPath(pathPartAsPath, obj.locale_, obj.role_);
      filenamePart = apt.MetaPath(filenamePartAsPath, obj.locale_, obj.role_);
    end

    function result = forceRemote_(obj)
      % Force this MetaPath to have remote locale and POSIX platform
      %
      % Returns:
      %   apt.MetaPath: New MetaPath with remote locale and POSIX platform, same role
      
      assert(obj.locale_ == apt.PathLocale.wsl || obj.locale_ == apt.PathLocale.remote, ...
             'forceRemote_() can only be used on WSL or remote paths (which are already POSIX)') ;
      
      % Create new MetaPath with remote locale and same role (path is already POSIX)
      result = apt.MetaPath(obj.path_, apt.PathLocale.remote, obj.role_) ;
    end

    function result = get_property_value_(obj, name)
      % Get property value for persistence encoding
      result = obj.(name) ;
    end  % function
    
    function set_property_value_(obj, name, value)
      % Set property value for persistence decoding
      obj.(name) = value ;
    end  % function
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
      
      % On POSIX platforms, native paths are already WSL-compatible
      inputPath = inputMetaPath.path ;
      if inputPath.platform == apt.Platform.posix
        result = apt.MetaPath(inputPath, apt.PathLocale.wsl, inputMetaPath.role);
        return
      end
      
      % For a Windows path, need to convert
      if inputPath.platform == apt.Platform.windows
        newPath = inputPath.toPosix() ;      
        result = apt.MetaPath(newPath, apt.PathLocale.wsl, inputMetaPath.role);
        return
      end

      % If get here something has gone wrong
      error('Internal error: toWslFromNative_() input has unknown platform type') ;
    end

    function result = toNativeFromWsl_(inputMetaPath)
      % Convert a WSL apt.MetaPath to native locale using path operations
      %
      % Args:
      %   inputMetaPath (apt.MetaPath): WSL path to convert
      %
      % Returns:
      %   apt.MetaPath: Converted native path with same role
      %
      % Notes:
      %   - Input must be WSL locale and absolute
      %   - On Windows: converts WSL mount points (/mnt/c -> C:) and forward slashes
      %   - On POSIX: identity operation (WSL paths are same as native POSIX paths)
      %   - Uses path operations, never converts to raw strings
      
      % Validate input
      assert(isa(inputMetaPath, 'apt.MetaPath'), 'inputMetaPath must be an apt.MetaPath instance');
      assert(inputMetaPath.locale == apt.PathLocale.wsl, 'inputMetaPath must have WSL locale');
      assert(inputMetaPath.path.tfIsAbsolute, 'inputMetaPath must be an absolute path');
      
      inputPath = inputMetaPath.path;
      
      % Get current platform to determine conversion behavior
      currentPlatform = apt.Platform.current();
      
      if currentPlatform ~= apt.Platform.windows
        % On POSIX platforms, WSL paths are already native-compatible
        result = apt.MetaPath(inputPath, apt.PathLocale.native, inputMetaPath.role);
        return
      end
      
      % For Windows platform, convert WSL path to Windows path
      newPath = inputPath.toWindows();
      
      % Create new metapath with converted components
      result = apt.MetaPath(newPath, apt.PathLocale.native, inputMetaPath.role);
    end  % function
  end  % methods (Static)

  methods
    function result = encode_for_persistence_(obj, do_wrap_in_container)
      encoded_path = encode_for_persistence_(obj.path_, true) ;
      encoding = struct('path_', {encoded_path}, 'locale_', {obj.locale_}, 'role_', {obj.role_}) ;
      if do_wrap_in_container
        result = encoding_container('apt.MetaPath', encoding) ;
      else
        result = encoding ;
      end
    end
  end  % methods
  
  methods (Static)
    function result = decode_encoding(encoding)
      % Decode the encoded version of the object.  Used for loading from persistent
      % storage.
      path = decode_encoding_container(encoding.path_) ;
      result = apt.MetaPath(path, encoding.locale_, encoding.role_) ;
    end
  end  % methods (Static)
  
end  % classdef
