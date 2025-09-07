classdef ShellVariableAssignment < apt.ShellToken
  % apt.ShellVariableAssignment - Variable assignment token for shell commands
  %
  % Represents a shell variable assignment token that holds both an identifier
  % and a value (e.g., "VARIABLE=value").
  
  properties
    identifier_  % char array containing the variable identifier
    value_       % apt.ShellToken containing the variable value
  end
  
  properties (Dependent)
    identifier   % Get the variable identifier
    value        % Get the variable value
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
      
      obj.identifier_ = identifier;
      obj.value_ = value;
    end
    
    function result = get.identifier(obj)
      result = obj.identifier_;
    end
    
    function result = get.value(obj)
      result = obj.value_;
    end
    
    function result = char(obj)
      % Convert to string representation (IDENTIFIER=value)
      if ischar(obj.value_)
        valueStr = obj.value_;
      else
        valueStr = obj.value_.char();
      end
      result = sprintf('%s=%s', obj.identifier_, escape_string_for_bash(valueStr)) ;
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
      
      result = strcmp(obj.identifier_, other.identifier_) && ...
               isequal(obj.value_, other.value_);
    end
    
    function disp(obj)
      % Display the ShellVariableAssignment object
      fprintf('apt.ShellVariableAssignment: "%s"\n', obj.char());
    end
  end
  
  methods
    function result = encode_for_persistence_(obj, do_wrap_in_container)
      % Encode the value_ property - it could be char or apt.MetaPath
      if ischar(obj.value_)
        encoded_value = obj.value_;
      else
        % It's an apt.MetaPath, encode it
        encoded_value = encode_for_persistence(obj.value_, true);
      end
      
      encoding = struct('identifier_', {obj.identifier_}, 'value_', {encoded_value}) ;
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
      if ischar(encoding.value_)
        decoded_value = encoding.value_;
      else
        % It should be an encoding container for a MetaPath
        decoded_value = decode_encoding_container(encoding.value_);
      end
      
      result = apt.ShellVariableAssignment(encoding.identifier_, decoded_value) ;
    end
  end  % methods (Static)
end