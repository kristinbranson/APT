classdef FileRole < uint8
    % apt.FileRole - Enumeration for file role types
    %
    % This enumeration defines the different file role types used in APT
    % for path classification and handling.
    %
    % Used by apt.MetaPath and apt.ShellCommand classes to categorize files
    % and determine appropriate remote path mappings.
    
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
    end
    
    methods (Static)
        function result = fromString(roleStr)
            % Create FileRole enum from string
            %
            % Args:
            %   roleStr (char): 'cache', 'movie', 'universal', 'source', or 'immovable'
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
                otherwise
                    error('apt:FileRole:InvalidString', ...
                          'FileRole must be ''cache'', ''movie'', ''universal'', ''source'', or ''immovable'', got ''%s''', ...
                          roleStr);
            end
        end
        
        function result = toString(fileRole)
            % Convert FileRole enum to string
            %
            % Args:
            %   fileRole (apt.FileRole): FileRole enumeration value
            %
            % Returns:
            %   char: String representation
            
            switch fileRole
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
                otherwise
                    error('apt:FileRole:InvalidEnum', 'Invalid file role enumeration');
            end
        end
    end
end