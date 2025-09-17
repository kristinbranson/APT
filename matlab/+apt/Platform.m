classdef Platform < uint8
  % apt.Platform - Enumeration for operating system platform types
  %
  % This enumeration distinguishes between different operating system families
  % to enable platform-specific path handling, command formatting, and file
  % system operations within APT.
  %
  % Enumeration values:
  %   posix   - Unix-like systems (Linux, macOS) using POSIX conventions
  %   windows - Microsoft Windows systems with drive letters and backslashes

  enumeration
    posix (1)    % Unix-like systems (Linux, macOS, etc.)
    windows (2)  % Windows systems
  end

  methods (Static)
    function result = current()
      % Auto-detect current platform
      %
      % Returns:
      %   apt.Platform: Platform enumeration value for current platform

      if ispc()
        result = apt.Platform.windows;
      elseif ismac() || isunix()
        result = apt.Platform.posix;
      else
        error('apt:Platform:UnknownPlatform', 'Unable to determine platform');
      end
    end

    function result = fromString(platformStr)
      % Create Platform enum from string
      %
      % Args:
      %   platformStr (char): 'posix', 'linux', 'windows', or 'macos'
      %                       Note: 'linux' and 'macos' map to 'posix' for compatibility
      %
      % Returns:
      %   apt.Platform: Platform enumeration value

      switch lower(platformStr)
        case {'posix', 'linux', 'macos'}
          result = apt.Platform.posix;
        case 'windows'
          result = apt.Platform.windows;
        otherwise
          error('apt:Platform:InvalidString', ...
                'Platform must be ''posix'', ''linux'', ''windows'', or ''macos'', got ''%s''', ...
                platformStr);
      end
    end
  end

  methods
    function result = char(obj)
      % Convert Platform enum to string
      %
      % Args:
      %   platform (apt.Platform): Platform enumeration value
      %
      % Returns:
      %   char: String representation

      switch obj
        case apt.Platform.posix
          result = 'posix';
        case apt.Platform.windows
          result = 'windows';
        otherwise
          error('apt:Platform:InvalidEnum', 'Invalid Platform enumeration');
      end
    end
  end
end