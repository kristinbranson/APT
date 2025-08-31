classdef (Abstract) ShellToken
  % apt.ShellToken - Abstract base class for shell command tokens
  %
  % This abstract class defines the interface for all types of tokens that
  % can appear in shell commands. Concrete subclasses include:
  % - apt.ShellLiteral: String literals
  % - apt.MetaPath: Path tokens with locale and role information
  
  methods (Abstract)
    result = toString(obj)
    % Convert token to string representation
  end
  
  methods
    function result = isLiteral(obj)
      % Check if this token is a literal string
      result = isa(obj, 'apt.ShellLiteral');
    end
    
    function result = isPath(obj)
      % Check if this token is a path
      result = isa(obj, 'apt.MetaPath');
    end
  end
end