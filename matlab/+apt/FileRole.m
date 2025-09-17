classdef FileRole < uint8
  % apt.FileRole - Enumeration for file role classification in APT
  %
  % This enumeration categorizes files based on their purpose and storage
  % location patterns, enabling appropriate path translation when moving
  % between different execution contexts (native, WSL, remote backends).
  %
  % File roles determine how paths are mapped when APT switches between
  % execution environments. For example, movie files are consolidated
  % into a special remote directory, while cache files follow APT's
  % standard cache directory structure.
  %
  % Used by apt.MetaPath and apt.ShellCommand classes to enable intelligent
  % path translation for different backend types.

  enumeration
    cache (1)  % For files that are in the APT cache dir
    movie (2)  % For movie files, which generally can be any darn place on the frontend machine,
    % but for backends with a remote file system are consolidated into their own
    % special folder.
    universal (3)  % Files that are accessible at the same path natively, in the local linux subsystem, and remotely.
    % Generally these are files that are used in the context of a linux-only backend for
    % which isFilesystemLocal() is true.
    source (4)  % For source code files
    immovable (5)  % For file paths that need to stay in whatever locale they started in
    torch (6)  % For PyTorch/torch cache files
    home (7)  % For files in the home directory, but not in any of the directories for the roles above
    local (8)  % For files that exist only locally (native or wsl) and should not be converted to remote
    slashhome (9)  % Special-purpose one for /home, whihc is the same on wsl and remote, but unavailable on native (in general)
  end

  methods (Static)
    function result = fromString(roleStr)
      % Create FileRole enum from string
      %
      % Args:
      %   roleStr (char): 'cache', 'movie', 'universal', 'source', 'immovable', 'torch', 'home', 'local', or 'slashhome'
      %
      % Returns:
      %   apt.FileRole: FileRole enumeration value

      switch lower(roleStr)
        case 'cache'
          result = apt.FileRole.cache;
        case 'movie'
          result = apt.FileRole.movie;
        case 'universal'
          result = apt.FileRole.universal;
        case 'source'
          result = apt.FileRole.source;
        case 'immovable'
          result = apt.FileRole.immovable;
        case 'torch'
          result = apt.FileRole.torch;
        case 'home'
          result = apt.FileRole.home;
        case 'local'
          result = apt.FileRole.local;
        case 'slashhome'
          result = apt.FileRole.slashhome;
        otherwise
          error('apt:FileRole:InvalidString', ...
                'FileRole must be ''cache'', ''movie'', ''universal'', ''source'', ''immovable'', ''torch'', ''home'', ''local'', or ''slashhome'', got ''%s''', ...
                roleStr);
      end
    end
  end

  methods
    function result = char(obj)
      % Convert FileRole enum to string
      %
      % Args:
      %   fileRole (apt.FileRole): FileRole enumeration value
      %
      % Returns:
      %   char: String representation

      switch obj
        case apt.FileRole.cache
          result = 'cache';
        case apt.FileRole.movie
          result = 'movie';
        case apt.FileRole.universal
          result = 'universal';
        case apt.FileRole.source
          result = 'source';
        case apt.FileRole.immovable
          result = 'immovable';
        case apt.FileRole.torch
          result = 'torch';
        case apt.FileRole.home
          result = 'home';
        case apt.FileRole.local
          result = 'local';
        case apt.FileRole.slashhome
          result = 'slashhome';
        otherwise
          error('apt:FileRole:InvalidEnum', 'Invalid file role enumeration');
      end
    end
  end
end