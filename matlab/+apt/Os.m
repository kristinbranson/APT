classdef Os < uint8
    % apt.Os - Enumeration for operating system types
    %
    % This enumeration defines the different operating system types used in APT
    % for platform-specific functionality.
    
    enumeration
        linux (1)
        windows (2)
        macos (3)
    end
    
    
    methods (Static)
        function result = current()
            % Auto-detect current operating system
            %
            % Returns:
            %   apt.Os: Operating system enumeration value for current platform
            
            if ispc()
                result = apt.Os.windows;
            elseif ismac()
                result = apt.Os.macos;
            elseif isunix()
                result = apt.Os.linux;
            else
                error('apt:Os:UnknownPlatform', 'Unable to determine operating system');
            end
        end
        
        function result = fromString(osStr)
            % Create Os enum from string
            %
            % Args:
            %   osStr (char): 'linux', 'windows', or 'macos'
            %
            % Returns:
            %   apt.Os: Operating system enumeration value
            
            switch lower(osStr)
                case 'linux'
                    result = apt.Os.linux;
                case 'windows'
                    result = apt.Os.windows;
                case 'macos'
                    result = apt.Os.macos;
                otherwise
                    error('apt:Os:InvalidString', ...
                          'Os must be ''linux'', ''windows'', or ''macos'', got ''%s''', ...
                          osStr);
            end
        end
        
        function result = toString(os)
            % Convert Os enum to string
            %
            % Args:
            %   os (apt.Os): Os enumeration value
            %
            % Returns:
            %   char: String representation
            
            switch os
                case apt.Os.linux
                    result = 'linux';
                case apt.Os.windows
                    result = 'windows';
                case apt.Os.macos
                    result = 'macos';
                otherwise
                    error('apt:Os:InvalidEnum', 'Invalid OS enumeration');
            end
        end
    end
end