classdef (Abstract) ShellToken
  % apt.ShellToken - Abstract base class for shell command tokens
  %
  % This abstract class defines the interface for all types of tokens that
  % can appear in shell commands. Concrete subclasses include:
  % - apt.ShellLiteral: Plain string literals that need no translation
  % - apt.MetaPath: Path tokens with locale and role-based translation
  % - apt.ShellCommand: Nested subcommands with their own token lists
  % - apt.ShellVariableAssignment: Environment variable assignments  
  % - apt.ShellBind: Docker-style bind mount specifications
  
  methods (Abstract)
    result = char(obj)
    % Convert token to string representation
    
    result = tfDoesMatchLocale(obj, queryLocale)
    % Check if this token matches the specified locale
    % Args:
    %   queryLocale (apt.PathLocale): The locale to check against
    % Returns:
    %   logical: True if token matches locale or is locale-agnostic
    
    result = tfDoesMatchPlatform(obj, queryPlatform)
    % Check if this token matches the specified platform
    % Args:
    %   queryPlatform (apt.Platform): The platform to check against
    % Returns:
    %   logical: True if token matches platform or is platform-agnostic
  end
end
