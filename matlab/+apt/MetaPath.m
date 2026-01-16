classdef MetaPath < apt.ShellToken
  % apt.MetaPath - Locale and role-aware path representation for multi-context execution
  %
  % This class extends apt.Path with additional metadata required for APT's
  % multi-backend execution environment. It tracks both the execution locale
  % (where the path will be used) and the file role (what type of file it
  % represents) to enable intelligent path translation between different
  % execution contexts.
  %
  % IMPORTANT: MetaPath objects always contain absolute paths, never relative
  % paths. This restriction ensures consistent path resolution across different
  % execution contexts and prevents ambiguity when translating between locales.
  %
  % Key capabilities:
  % - Automatic path translation between native, WSL, and remote contexts
  % - Role-based path mapping for different file types
  % - Validation of locale and platform compatibility within shell commands
  % - Support for prefix replacement during backend transitions
  %
  % Example usage:
  %   moviePath = apt.MetaPath('/data/movie.avi', 'native', 'movie');
  %   wslPath = moviePath.as('wsl');      % Convert to WSL context
  %   remotePath = moviePath.as('remote'); % Convert to remote backend context

  properties
    path        % apt.Path object containing the actual path
    locale      % apt.PathLocale enumeration.  Describes the 'locale' (native/wsl/remote) where the path is 
                % appropriate.
    role        % apt.FileRole enumeration.
  end

  properties (Dependent)
    platform    % Get the platform from the underlying path (derived from path.platform)
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
      if ~path.tfIsAbsolute()
        error('apt:MetaPath:RelativePath', 'MetaPath requires an absolute path');
      end

      obj.path = path;
      obj.locale = locale;
      obj.role = role;
    end

    function result = get.platform(obj)
      result = obj.path.platform;
    end

    function result = char(obj)
      % Get the path as a string, *escaped for bash*
      result = obj.path.char();
    end

    function result = charUnescaped(obj)
      % Get the path as a string, *not escaped for bash*
      result = obj.path.charUnescaped();
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
      
      result = (obj.locale == queryLocale);
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
      
      result = (obj.path.platform == queryPlatform);
    end

    function result = isequal(obj, other)
      % Check equality with another apt.MetaPath
      if ~isa(other, 'apt.MetaPath')
        result = false;
        return;
      end

      result = isequal(obj.path, other.path) && ...
               obj.locale == other.locale && ...
               obj.role == other.role;
    end

    function result = replacePrefix(obj, sourcePrefixMetaPath, targetPrefixMetaPath)
      % Replace a source prefix with a target prefix in the underlying path
      %
      % Args:
      %   sourcePrefixMetaPath (apt.MetaPath): The MetaPath prefix to replace
      %   targetPrefixMetaPath (apt.MetaPath): The replacement MetaPath prefix
      %
      % Returns:
      %   apt.MetaPath: New MetaPath with prefix replaced, using target's locale and platform
      %
      % Example:
      %   mp = apt.MetaPath('/old/base/file.txt', 'native', 'movie');
      %   sourcePrefix = apt.MetaPath('/old/base', 'native', 'movie');
      %   targetPrefix = apt.MetaPath('/new/location', 'wsl', 'movie');
      %   newMp = mp.replacePrefix(sourcePrefix, targetPrefix);
      %   % newMp will have path '/new/location/file.txt' with WSL locale and movie role
      
      % Validate arguments are MetaPaths
      assert(isa(sourcePrefixMetaPath, 'apt.MetaPath'), 'sourcePrefixMetaPath must be an apt.MetaPath');
      assert(isa(targetPrefixMetaPath, 'apt.MetaPath'), 'targetPrefixMetaPath must be an apt.MetaPath');
      
      % Validate that all roles match
      assert(obj.role == sourcePrefixMetaPath.role, 'Object and source prefix must have the same FileRole');
      assert(obj.role == targetPrefixMetaPath.role, 'Object and target prefix must have the same FileRole');
      
      % Validate that obj and source prefix have the same locale
      assert(obj.locale == sourcePrefixMetaPath.locale, 'Object and source prefix must have the same locale');
      
      % Use the underlying apt.Path replacePrefix method
      newPath = obj.path.replacePrefix(sourcePrefixMetaPath.path, targetPrefixMetaPath.path);
      
      % Create new MetaPath using target's locale and the common role
      result = apt.MetaPath(newPath, targetPrefixMetaPath.locale, obj.role);
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
      if targetLocale == obj.locale
        result = obj;
        return;
      end
      
      % Handle supported conversions
      if obj.locale == apt.PathLocale.native && targetLocale == apt.PathLocale.wsl
        % Convert native path to WSL path using static method
        result = apt.MetaPath.toWslFromNative_(obj);
      elseif obj.locale == apt.PathLocale.wsl && targetLocale == apt.PathLocale.native
        % Convert WSL path to native path using static method
        result = apt.MetaPath.toNativeFromWsl_(obj);
      else
        % Unsupported conversion
        error('apt:MetaPath:UnsupportedConversion', ...
              'Conversion from %s to %s is not supported by the MetaPath.as() method', ...
              char(obj.locale), ...
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

    % function disp(obj)
    %   % Display the apt.MetaPath object
    %   pathStr = obj.char();
    %   fprintf('apt.MetaPath: %s [%s:%s:%s]\n', ...
    %           pathStr, ...
    %           char(obj.locale), ...
    %           char(obj.role), ...
    %           char(obj.path.platform));
    % end

    function result = cat(obj, varargin)
      % Concatenate this MetaPath with multiple apt.MetaPath objects, apt.Path objects, or char arrays
      %
      % Args:
      %   varargin: Variable number of apt.MetaPath objects, apt.Path objects, or char arrays to concatenate
      %
      % Returns:
      %   apt.MetaPath: New MetaPath with concatenated paths, same locale and role
      %
      % Notes:
      %   - Char arrays are converted to apt.Path objects with the same platform
      %   - apt.Path objects are used directly as path components
      %   - apt.MetaPath arguments must have compatible locales and platforms
      
      % Convert char arrays to apt.Path objects and collect apt.Path objects for underlying cat
      pathArgs = cell(size(varargin));
      for i = 1:length(varargin)
        if ischar(varargin{i})
          % Convert char array to apt.Path
          pathArgs{i} = apt.Path(varargin{i}, obj.platform);
        elseif isa(varargin{i}, 'apt.Path')
          % Use apt.Path object directly as a path component
          pathArgs{i} = varargin{i};
        elseif isa(varargin{i}, 'apt.MetaPath')
          % Validate locale and role compatibility
          if varargin{i}.locale ~= obj.locale
            error('apt:MetaPath:LocaleMismatch', ...
              'MetaPath argument at position %d has locale %s, but this MetaPath has locale %s', ...
              i, char(varargin{i}.locale), char(obj.locale));
          end
          if varargin{i}.role ~= obj.role
            error('apt:MetaPath:RoleMismatch', ...
              'MetaPath argument at position %d has role %s, but this MetaPath has role %s', ...
              i, char(varargin{i}.role), char(obj.role));
          end
          pathArgs{i} = varargin{i}.path;
        else
          error('apt:MetaPath:InvalidArgument', 'Argument %d must be an apt.MetaPath object, apt.Path object, or char array, got %s', i, class(varargin{i}));
        end
      end
      
      % Call the apt.Path.cat() method on the underlying paths
      oldPath = obj.path;
      newPath = oldPath.cat(pathArgs{:});
      result = apt.MetaPath(newPath, obj.locale, obj.role);
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
      oldPath = obj.path;
      newPath = oldPath.append(varargin{:});
      result = apt.MetaPath(newPath, obj.locale, obj.role);
    end

    function [pathPart, filenamePart] = fileparts2(obj)
      % Call apt.Path.fileparts2() method and wrap results as a MetaPath and a Path
      %
      % Returns:
      %   pathPart (apt.MetaPath): Directory path portion with same locale and role
      %   filenamePart (apt.Path): Filename portion (name + extension) with same
      %   platform (This is an apt.Path, not and apt.MetaPath, b/c a MetaPath must
      %   be an absolute path.)
      %
      % Example:
      %   mp = apt.MetaPath('/home/user/data/movie.avi', 'wsl', 'movie');
      %   [dir, file] = mp.fileparts2();
      %   % dir will be apt.MetaPath('/home/user/data', 'wsl', 'movie')
      %   % file will be apt.Path('movie.avi', 'posix')
      
      [pathPartAsPath, filenamePart] = obj.path.fileparts2();
      pathPart = apt.MetaPath(pathPartAsPath, obj.locale, obj.role);
    end

    function result = forceRemote_(obj)
      % Force this MetaPath to have remote locale and POSIX platform
      %
      % Returns:
      %   apt.MetaPath: New MetaPath with remote locale and POSIX platform, same role
      
      assert(obj.locale == apt.PathLocale.wsl || obj.locale == apt.PathLocale.remote, ...
             'forceRemote_() can only be used on WSL or remote paths (which are already POSIX)') ;
      
      % Create new MetaPath with remote locale and same role (path is already POSIX)
      result = apt.MetaPath(obj.path, apt.PathLocale.remote, obj.role) ;
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
      assert(inputMetaPath.path.tfIsAbsolute(), 'inputPath must be an absolute path');
      
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
      assert(inputMetaPath.path.tfIsAbsolute(), 'inputMetaPath must be an absolute path');
      
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
      encoded_path = encode_for_persistence_(obj.path, true) ;
      encoding = struct('path', {encoded_path}, 'locale', {obj.locale}, 'role', {obj.role}) ;
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
      path = decode_encoding_container(encoding.path) ;
      result = apt.MetaPath(path, encoding.locale, encoding.role) ;
    end
  end  % methods (Static)
  
  methods
    function result = leafName(obj)
      % Return leaf name as char array.  The leaf name is the final element of the
      % path.
      path = obj.path ;
      result = path.leafName() ;
    end

    function result = extension(obj)
      % The file extension of the the path.  Uses same conventions as fileparts().
      path = obj.path ;
      result = path.extension() ;
    end

    function replaceExtension(obj, newExtension)
      path = obj.path ;
      path.replaceExtension(newExtension) ;
    end

    function [rest, leaf] = split(obj)
      % Return leaf name as char array, and the rest of the path as an apt.MetaPath.
      % The leaf name is the final element of the path.  Errors if obj is null.  If
      % obj holds a single-element path, rest will be the null MetaPath, leaf will be a
      % the single element as a char arrray.
      path = obj.path ;
      [restPath, leaf] = path.split() ;
      rest = apt.MetaPath(restPath, obj.locale, obj.role) ;
    end    

    function result = forceRelativeThenCat(obj, suffixPath)
      % Force suffixPath's internal apt.Path to be relative, then concatenate it
      % onto obj's internal apt.Path.  (Used to compute remote paths for movie/trx
      % files when using AWS backend.)     
      assert(isa(suffixPath, 'apt.MetaPath'), 'suffixPath must be an apt.MetaPath instance');
      assert(obj.role==suffixPath.role, 'suffixPath role must match that of obj') ;      
      oldPath = obj.path ;
      relativizedSuffixPath = suffixPath.path.forceRelative() ;
      newPath = oldPath.cat(relativizedSuffixPath) ;
      result = apt.MetaPath(newPath, obj.locale, obj.role) ;  % Note that the locale of the obj 'wins'
    end
  end  % methods
end  % classdef
