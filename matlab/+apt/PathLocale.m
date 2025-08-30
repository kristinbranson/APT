classdef PathLocale < uint8
    % apt.PathLocale - Enumeration for platform types
    %
    % This enumeration defines the different platform types used in APT
    % for path and command representation.
    
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
        
        function result = toString(pathLocale)
            % Convert PathLocale enum to string
            %
            % Args:
            %   pathLocale (apt.PathLocale): PathLocale enumeration value
            %
            % Returns:
            %   char: String representation
            
            switch pathLocale
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