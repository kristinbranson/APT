classdef ShellLiteral < apt.ShellToken
  % apt.ShellLiteral - String literal token for shell commands
  %
  % Represents a literal string token in a shell command that doesn't
  % require path translation or locale conversion.
  
  properties
    value_  % char array containing the literal string
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
      
      obj.value_ = char(value);
    end
    
    function result = toString(obj)
      % Convert to string representation
      result = obj.value_;
    end
    
    function result = tfDoesMatchLocale(obj, queryLocale)  %#ok<INUSD>
      % ShellLiterals are locale-agnostic and match any locale
      result = true;
    end
    
    function result = eq(obj, other)
      % Check equality with another ShellLiteral
      if ~isa(other, 'apt.ShellLiteral')
        result = false;
        return;
      end
      
      result = strcmp(obj.value_, other.value_);
    end
    
    function disp(obj)
      % Display the ShellLiteral object
      fprintf('apt.ShellLiteral: "%s"\n', obj.value_);
    end
  end
end