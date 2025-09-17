classdef PathLocale < uint8
  % apt.PathLocale - Enumeration for path locale contexts within APT
  %
  % This enumeration defines where paths are intended to be used.
  %
  % Enumeration values:
  %   native - Paths for the native/host machine where APT frontend runs
  %   wsl    - Paths for Windows Subsystem for Linux context (On Linux/Mac, this
  %            is eqauivalent to native)
  %   remote - Paths for remote execution backends (e.g. AWS, bsub cluster).  For
  %            backends that run locally (e.g. conda), this is equivalent to
  %            wsl.

  enumeration
    native (1)
    wsl (2)
    remote (3)
  end

  methods (Static)
    function result = fromString(platformStr)
      % Create PathLocale enum from string
      %
      % Args:
      %   platformStr (char): 'native', 'wsl', or 'remote'
      %
      % Returns:
      %   apt.PathLocale: Platform enumeration value

      switch lower(platformStr)
        case 'native'
          result = apt.PathLocale.native;
        case 'wsl'
          result = apt.PathLocale.wsl;
        case 'remote'
          result = apt.PathLocale.remote;
        otherwise
          error('apt:PathLocale:InvalidString', ...
                'PathLocale must be ''native'', ''wsl'', or ''remote'', got ''%s''', ...
                platformStr);
      end
    end
  end

  methods
    function result = char(obj)
      % Convert PathLocale enum to string
      %
      % Args:
      %   pathLocale (apt.PathLocale): PathLocale enumeration value
      %
      % Returns:
      %   char: String representation

      switch obj
        case apt.PathLocale.native
          result = 'native';
        case apt.PathLocale.wsl
          result = 'wsl';
        case apt.PathLocale.remote
          result = 'remote';
        otherwise
          error('apt:PathLocale:InvalidEnum', 'Invalid path locale enumeration');
      end
    end
  end
end