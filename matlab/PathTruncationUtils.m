classdef PathTruncationUtils
    % Utility class for truncating long file paths in GUI components
    % Provides methods to calculate field widths and truncate paths with ellipses
    
    methods (Static)
        function truncated = truncateFilePath(filepath, varargin)
            % Helper function to truncate long file paths with ellipses
            % Shows beginning and end of path if it's too long
            %
            % Args:
            %   filepath: string to truncate
            %   
            % Optional parameters (name-value pairs):
            %   'maxLength': maximum character length (default 60)
            %   'component': GUI component handle to calculate max chars from its width
            %   'startFraction': fraction of available space to show from start (default 0.4)
            %                   Value should be between 0 and 1
            
            [maxLength, component, startFraction] = myparse(varargin, ...
                'maxLength', 60, ...
                'component', [], ...
                'startFraction', 0.4);
            
            if isempty(filepath)
                truncated = '';
                return;
            end
            
            % Validate startFraction
            if startFraction < 0 || startFraction > 1
                warning('startFraction should be between 0 and 1. Using default 0.4');
                startFraction = 0.4;
            end
            
            % If component handle provided, calculate maxLength from it
            if ~isempty(component)
                maxLength = PathTruncationUtils.calculateMaxCharsFromComponent(component);
            end
            
            if length(filepath) <= maxLength
                truncated = filepath;
                return;
            end
            
            % Calculate how much to show from start and end
            ellipses = '...';
            availableLength = maxLength - length(ellipses);
            startLength = floor(availableLength * startFraction);
            endLength = availableLength - startLength;
            
            % Ensure we don't go past the string boundaries
            if startLength >= length(filepath)
                truncated = filepath;
                return;
            end
            
            % Ensure we have at least 1 character for both start and end
            if startLength < 1
                startLength = 1;
                endLength = availableLength - startLength;
            end
            if endLength < 1
                endLength = 1;
                startLength = availableLength - endLength;
            end
            
            if startLength>1
              startPart = filepath(1:startLength);
            else
              startPart = '';
            end
            endPart = filepath(end-endLength+1:end);
            truncated = [startPart ellipses endPart];
        end
        
        function maxChars = calculateMaxCharsFromComponent(componentHandle)
            % Calculate maximum characters that fit in a GUI component based on its width
            
            try
                % Get component position in pixels
                % Check if component has Units property (older MATLAB components)
                try
                    originalUnits = get(componentHandle, 'Units');
                    set(componentHandle, 'Units', 'pixels');
                    pos = get(componentHandle, 'Position');
                    set(componentHandle, 'Units', originalUnits);
                catch
                    % Modern UI components (uifigure-based) don't have Units property
                    % Position is already in pixels for these components
                    pos = get(componentHandle, 'Position');
                end
                
                fieldWidthPixels = pos(3);
                
                % Get font size from the component, or use default
                try
                    try
                      origfunits = get(componentHandle,'FontUnits');
                      set(componentHandle,'FontUnits','pixels');
                    catch
                    end
                    fontSize = get(componentHandle, 'FontSize');
                    try
                      set(componentHandle,'FontUnits',origfunits);
                    catch
                    end
                catch
                    fontSize = 14; % Default font size
                end
                
                maxChars = PathTruncationUtils.calculateMaxCharsForFieldWidth(fieldWidthPixels, fontSize);
                
            catch
                % If anything fails, use default
                maxChars = 60;
            end
        end
        
        function maxChars = calculateMaxCharsForFieldWidth(fieldWidthPixels, fontSize)
            % Calculate maximum characters for a given field width and font size
            % 
            % Args:
            %   fieldWidthPixels: width of the field in pixels
            %   fontSize: font size (optional, defaults to 14)
            
            if nargin < 2
                fontSize = 14;
            end
            
            % Estimate character width based on font size
            avgCharWidthPixels = fontSize * 0.5;
            
            % Calculate max characters with padding
            maxChars = floor(fieldWidthPixels / avgCharWidthPixels * 0.95);
            
            % Set reasonable bounds
            maxChars = max(20, min(maxChars, 200));
        end
    end
end