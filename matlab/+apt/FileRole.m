classdef FileRole < uint8
    % apt.FileRole - Enumeration for file role types
    %
    % This enumeration defines the different file role types used in APT
    % for path classification and handling.
    %
    % Used by apt.Path and apt.ShellCommand classes to categorize files
    % and determine appropriate remote path mappings.
    
    enumeration
        cache (1)
        movie (2)
    end
    
    methods (Static)
        function result = fromString(roleStr)
            % Create FileRole enum from string
            %
            % Args:
            %   roleStr (char): 'cache' or 'movie'
            %
            % Returns:
            %   apt.FileRole: FileRole enumeration value
            
            switch lower(roleStr)
                case 'cache'
                    result = apt.FileRole.cache;
                case 'movie'
                    result = apt.FileRole.movie;
                otherwise
                    error('apt:FileRole:InvalidString', ...
                          'FileRole must be ''cache'' or ''movie'', got ''%s''', ...
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
                otherwise
                    error('apt:FileRole:InvalidEnum', 'Invalid file role enumeration');
            end
        end
    end
end