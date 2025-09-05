classdef ShellVariableAssignment < apt.ShellToken
  % apt.ShellVariableAssignment - Variable assignment token for shell commands
  %
  % Represents a shell variable assignment token that holds both an identifier
  % and a value (e.g., "VARIABLE=value").
  
  properties
    identifier_  % char array containing the variable identifier
    value_       % apt.ShellToken containing the variable value
  end
  
  methods
    function obj = ShellVariableAssignment(identifier, value)
      % Constructor for apt.ShellVariableAssignment
      %
      % Args:
      %   identifier (char or string): The variable identifier/name
      %   value (apt.ShellToken): The variable value
      
      obj.identifier_ = char(identifier);
      obj.value_ = char(value);
    end
    
    function result = char(obj)
      % Convert to string representation (IDENTIFIER=value)
      result = sprintf('%s=%s', obj.identifier_, obj.value_.char());
    end
    
    function result = tfDoesMatchLocale(obj, queryLocale)  %#ok<INUSD>
      % ShellVariableAssignments are locale-agnostic and match any locale
      result = true;
    end
    
    function result = tfDoesMatchPlatform(obj, queryPlatform)  %#ok<INUSD>
      % ShellVariableAssignments are platform-agnostic and match any platform
      result = true;
    end
    
    function result = eq(obj, other)
      % Check equality with another ShellVariableAssignment
      if ~isa(other, 'apt.ShellVariableAssignment')
        result = false;
        return;
      end
      
      result = strcmp(obj.identifier_, other.identifier_) && ...
               obj.value_.eq(other.value_);
    end
    
    function disp(obj)
      % Display the ShellVariableAssignment object
      fprintf('apt.ShellVariableAssignment: "%s"\n', obj.char());
    end
  end
end