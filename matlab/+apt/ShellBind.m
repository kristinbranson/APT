classdef ShellBind < apt.ShellToken
  % apt.ShellBind - Docker-style bind mount specification token
  %
  % Represents a bind mount specification that maps a source path to a destination
  % path within a container or similar execution environment. Both paths are
  % apt.MetaPath objects that can be properly translated between different
  % execution contexts (native, WSL, remote).
  %
  % The string representation follows Docker's bind mount format:
  % "type=bind,src=<source_path>,dst=<destination_path>"
  
  properties
    sourcePath  % apt.MetaPath containing the source path
    destPath    % apt.MetaPath containing the destination path
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
      
      obj.sourcePath = sourcePath ;
      obj.destPath = destPath ;
    end
    
    
    function result = char(obj)
      % Convert to string representation (type=bind,src=...,dst=...)
      result = sprintf('type=bind,src=%s,dst=%s', obj.sourcePath.char(), obj.destPath.char());
    end
    
    function result = tfDoesMatchLocale(obj, queryLocale)
      % ShellBinds match locale if both paths match the query locale
      result = obj.sourcePath.tfDoesMatchLocale(queryLocale) && ...
               obj.destPath.tfDoesMatchLocale(queryLocale);
    end
    
    function result = tfDoesMatchPlatform(obj, queryPlatform)
      % ShellBinds match platform if both paths match the query platform
      result = obj.sourcePath.tfDoesMatchPlatform(queryPlatform) && ...
               obj.destPath.tfDoesMatchPlatform(queryPlatform);
    end
    
    function result = isequal(obj, other)
      % Check equality with another ShellBind
      if ~isa(other, 'apt.ShellBind')
        result = false;
        return;
      end
      
      result = isequal(obj.sourcePath, other.sourcePath) && ...
               isequal(obj.destPath, other.destPath);
    end
    
    % function disp(obj)
    %   % Display the ShellBind object
    %   fprintf('apt.ShellBind: "%s"\n', obj.char());
    % end
  end
  
  methods
    function result = encode_for_persistence_(obj, do_wrap_in_container)
      % Encode both MetaPath properties
      encoded_sourcePath = encode_for_persistence(obj.sourcePath, true);
      encoded_destPath = encode_for_persistence(obj.destPath, true);
      
      encoding = struct('sourcePath', {encoded_sourcePath}, 'destPath', {encoded_destPath}) ;
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
      decoded_sourcePath = decode_encoding_container(encoding.sourcePath);
      decoded_destPath = decode_encoding_container(encoding.destPath);
      
      result = apt.ShellBind(decoded_sourcePath, decoded_destPath) ;
    end
  end  % methods (Static)
end