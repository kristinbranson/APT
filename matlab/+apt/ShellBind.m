classdef ShellBind < apt.ShellToken
  % apt.ShellBind - Bind mount token for shell commands
  %
  % Represents a shell bind mount that holds both a source path
  % and a destination path (e.g., for Docker bind mounts).
  
  properties
    sourcePath_  % apt.MetaPath containing the source path
    destPath_    % apt.MetaPath containing the destination path
  end
  
  methods
    function obj = ShellBind(sourcePath, destPath)
      % Constructor for apt.ShellBind
      %
      % Args:
      %   sourcePath (apt.MetaPath): The source path for the bind
      %   destPath (apt.MetaPath): The destination path for the bind
      
      if nargin < 1 || ~isa(sourcePath, 'apt.MetaPath')
        error('sourcePath must be an apt.MetaPath object');
      end
      if nargin < 2 || ~isa(destPath, 'apt.MetaPath')
        error('destPath must be an apt.MetaPath object');
      end
      
      obj.sourcePath_ = sourcePath;
      obj.destPath_ = destPath;
    end
    
    function result = char(obj)
      % Convert to string representation (type=bind,src=...,dst=...)
      result = sprintf('type=bind,src=%s,dst=%s', obj.sourcePath_.char(), obj.destPath_.char());
    end
    
    function result = tfDoesMatchLocale(obj, queryLocale)
      % ShellBinds match locale if both paths match the query locale
      result = obj.sourcePath_.tfDoesMatchLocale(queryLocale) && ...
               obj.destPath_.tfDoesMatchLocale(queryLocale);
    end
    
    function result = tfDoesMatchPlatform(obj, queryPlatform)
      % ShellBinds match platform if both paths match the query platform
      result = obj.sourcePath_.tfDoesMatchPlatform(queryPlatform) && ...
               obj.destPath_.tfDoesMatchPlatform(queryPlatform);
    end
    
    function result = isequal(obj, other)
      % Check equality with another ShellBind
      if ~isa(other, 'apt.ShellBind')
        result = false;
        return;
      end
      
      result = isequal(obj.sourcePath_, other.sourcePath_) && ...
               isequal(obj.destPath_, other.destPath_);
    end
    
    function disp(obj)
      % Display the ShellBind object
      fprintf('apt.ShellBind: "%s"\n', obj.char());
    end
  end
  
  methods
    function result = encode_for_persistence_(obj, do_wrap_in_container)
      % Encode both MetaPath properties
      encoded_sourcePath = encode_for_persistence(obj.sourcePath_, true);
      encoded_destPath = encode_for_persistence(obj.destPath_, true);
      
      encoding = struct('sourcePath_', {encoded_sourcePath}, 'destPath_', {encoded_destPath}) ;
      if do_wrap_in_container
        result = encoding_container('apt.ShellBind', encoding) ;
      else
        result = encoding ;
      end
    end
  end  % methods
  
  methods (Static)
    function result = decode_encoding(encoding)
      % Decode the encoded version of the object.  Used for loading from persistent
      % storage.
      
      % Decode both MetaPath objects
      decoded_sourcePath = decode_encoding_container(encoding.sourcePath_);
      decoded_destPath = decode_encoding_container(encoding.destPath_);
      
      result = apt.ShellBind(decoded_sourcePath, decoded_destPath) ;
    end
  end  % methods (Static)
end