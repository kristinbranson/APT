classdef ShellLiteral < apt.ShellToken
  % apt.ShellLiteral - String literal token for shell commands
  %
  % Represents a literal string token in a shell command that doesn't
  % require path translation or locale conversion.
  
  properties
    value  % char array containing the literal string
  end
  
  methods
    function obj = ShellLiteral(value)
      % Constructor for apt.ShellLiteral
      %
      % Args:
      %   value (char or string): The literal string value
      
      if nargin < 1
        value = '';
      end
      
      obj.value = char(value);
    end
    
    
    function result = char(obj)
      % Convert to string representation
      result = obj.value ;
    end
    
    function result = tfDoesMatchLocale(obj, queryLocale)  %#ok<INUSD>
      % ShellLiterals are locale-agnostic and match any locale
      result = true;
    end
    
    function result = tfDoesMatchPlatform(obj, queryPlatform)  %#ok<INUSD>
      % ShellLiterals are platform-agnostic and match any platform
      result = true;
    end
    
    function result = isequal(obj, other)
      % Check equality with another ShellLiteral
      if ~isa(other, 'apt.ShellLiteral')
        result = false;
        return;
      end
      
      result = strcmp(obj.value, other.value);
    end
    
    % function disp(obj)
    %   % Display the ShellLiteral object
    %   fprintf('apt.ShellLiteral: "%s"\n', obj.value);
    % end
  end
  
  methods
    function result = encode_for_persistence_(obj, do_wrap_in_container)
      encoding = struct('value', {obj.value}) ;
      if do_wrap_in_container
        result = encoding_container('apt.ShellLiteral', encoding) ;
      else
        result = encoding ;
      end
    end
  end  % methods
  
  methods (Static)
    function result = decode_encoding(encoding)
      % Decode the encoded version of the object.  Used for loading from persistent
      % storage.
      result = apt.ShellLiteral(encoding.value) ;
    end
  end  % methods (Static)
end