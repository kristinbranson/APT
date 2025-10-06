classdef ShellVariableAssignment < apt.ShellToken
  % apt.ShellVariableAssignment - Environment/shell variable assignment token
  % for shell commands
  %
  % Represents a shell environment variable assignment in the form "VARIABLE=value".
  % The value can be either a plain string or an apt.MetaPath that will be properly
  % translated based on the execution context.  Variable assignments are typically
  % used to set environment variables before command execution.
  %
  % The value is automatically escaped when converting to string representation to
  % ensure proper shell parsing.
  
  properties
    identifier  % char array containing the variable identifier
    value       % apt.ShellToken containing the variable value
  end
  
  methods
    function obj = ShellVariableAssignment(identifier, value)
      % Constructor for apt.ShellVariableAssignment
      %
      % Args:
      %   identifier (char): The variable identifier/name
      %   value (char): The variable value
      
      % Assert that identifier is char and either empty or a row array
      assert(ischar(identifier), 'identifier must be a char array');
      assert(isempty(identifier) || isrow(identifier), ...
             'identifier must be either empty or a row char array');
      
      % Assert that value is char or MetaPath
      assert(ischar(value) || isa(value, 'apt.MetaPath'), 'value must be a char array or apt.MetaPath');
      if ischar(value)
        assert(isempty(value) || isrow(value), ...
               'value must be either empty or a row char array');
      end
      
      obj.identifier = identifier;
      obj.value = value;
    end
    
    
    function result = char(obj)
      % Convert to string representation (IDENTIFIER=value)
      if ischar(obj.value)
        valueStr = obj.value;
        result = sprintf('%s=%s', obj.identifier, escape_string_for_bash(valueStr)) ;
      else
        valueStr = obj.value.char();  % already escaped
        result = sprintf('%s=%s', obj.identifier, valueStr) ;
      end
    end
    
    function result = tfDoesMatchLocale(obj, queryLocale)  %#ok<INUSD>
      % ShellVariableAssignments are locale-agnostic and match any locale
      result = true;
    end
    
    function result = tfDoesMatchPlatform(obj, queryPlatform)  %#ok<INUSD>
      % ShellVariableAssignments are platform-agnostic and match any platform
      result = true;
    end
    
    function result = isequal(obj, other)
      % Check equality with another ShellVariableAssignment
      if ~isa(other, 'apt.ShellVariableAssignment')
        result = false;
        return;
      end
      
      result = strcmp(obj.identifier, other.identifier) && ...
               isequal(obj.value, other.value);
    end
    
    % function disp(obj)
    %   % Display the ShellVariableAssignment object
    %   fprintf('apt.ShellVariableAssignment: "%s"\n', obj.char());
    % end
  end
  
  methods
    function result = encode_for_persistence_(obj, do_wrap_in_container)
      % Encode the value_ property - it could be char or apt.MetaPath
      if ischar(obj.value)
        encoded_value = obj.value;
      else
        % It's an apt.MetaPath, encode it
        encoded_value = encode_for_persistence(obj.value, true);
      end
      
      encoding = struct('identifier', {obj.identifier}, 'value', {encoded_value}) ;
      if do_wrap_in_container
        result = encoding_container('apt.ShellVariableAssignment', encoding) ;
      else
        result = encoding ;
      end
    end
  end  % methods
  
  methods (Static)
    function result = decode_encoding(encoding)
      % Decode the encoded version of the object.  Used for loading from persistent
      % storage.
      
      % Decode the value - it could be a char or an encoding container for MetaPath
      if ischar(encoding.value)
        decoded_value = encoding.value;
      else
        % It should be an encoding container for a MetaPath
        decoded_value = decode_encoding_container(encoding.value);
      end
      
      result = apt.ShellVariableAssignment(encoding.identifier, decoded_value) ;
    end
  end  % methods (Static)
end