classdef ShellCommand
    % apt.ShellCommand - A value class to represent commands with mixed literals and paths
    %
    % This class represents a command as a list of tokens, where each token
    % can be either a literal string or an apt.MetaPath object. This allows for
    % proper path translation when commands need to be executed on different
    % platforms (native, WSL, remote).
    %
    % INVARIANT: All apt.MetaPath tokens must have the same locale as the
    % ShellCommand's locale. This ensures consistency when converting between
    % different execution contexts.
    %
    % Example usage:
    %   movPath = apt.MetaPath({'data', 'movie.avi'}, 'native', 'movie');
    %   cmd = apt.ShellCommand({'python', 'script.py', '--input', movPath, '--output', 'results.txt'});
    %   wslCmd = cmd.as('wsl');
    %   cmdStr = wslCmd.toString();
    
    properties
        tokens_     % Cell array of tokens (strings and apt.MetaPath objects)
        locale_     % apt.PathLocale enumeration indicating the path locale
    end
    
    properties (Dependent)
        locale      % Get the locale type
    end
    
    methods
        function obj = ShellCommand(tokens, locale)
            % Constructor for apt.ShellCommand
            %
            % Args:
            %   tokens (cell): Cell array of command tokens (strings and apt.MetaPath objects)
            %   locale (char or apt.PathLocale): Platform context for the command
            
            if nargin < 1
                tokens = {};
            end
            if nargin < 2
                locale = apt.PathLocale.native;
            end
            
            % Convert string to enum if needed
            if ischar(locale)
                locale = apt.PathLocale.fromString(locale);
            end
            
            % Validate that all apt.MetaPath tokens have matching locale
            for i = 1:length(tokens)
                token = tokens{i};
                if isa(token, 'apt.MetaPath')
                    if token.locale ~= locale
                        error('apt:ShellCommand:LocaleMismatch', ...
                              'MetaPath token at index %d has locale %s, but ShellCommand has locale %s', ...
                              i, apt.PathLocale.toString(token.locale), apt.PathLocale.toString(locale));
                    end
                end
            end
            
            obj.tokens_ = tokens;
            obj.locale_ = locale;
        end
        
        function result = get.locale(obj)
            result = obj.locale_;
        end
        
        function result = as(obj, targetLocale)
            % Convert command to target platform by converting all path tokens
            %
            % Args:
            %   targetLocale (char or apt.PathLocale): 'native', 'wsl', 'remote', or enum
            %
            % Returns:
            %   apt.ShellCommand: New command with paths converted to target platform
            
            % Convert string to enum if needed
            if ischar(targetLocale)
                targetLocale = apt.PathLocale.fromString(targetLocale);
            end
            
            newTokens = cell(size(obj.tokens_));
            for i = 1:length(obj.tokens_)
                token = obj.tokens_{i};
                if isa(token, 'apt.MetaPath')
                    newTokens{i} = token.as(targetLocale);
                else
                    newTokens{i} = token;
                end
            end
            
            result = apt.ShellCommand(newTokens, targetLocale);
        end
        
        function result = toString(obj, varargin)
            % Convert command to string representation
            %
            % Args:
            %   varargin: Optional arguments parsed with myparse
            %     'QuotePaths' (logical): Whether to quote path tokens (default: true)
            %
            % Returns:
            %   char: String representation of the command
            
            [quotePaths] = myparse(varargin, ...
                'QuotePaths', true);
            
            stringTokens = cell(size(obj.tokens_));
            for i = 1:length(obj.tokens_)
                token = obj.tokens_{i};
                if isa(token, 'apt.MetaPath')
                    pathStr = token.toString();
                    if quotePaths && contains(pathStr, ' ')
                        stringTokens{i} = ['"' pathStr '"'];
                    else
                        stringTokens{i} = pathStr;
                    end
                else
                    stringTokens{i} = char(token);
                end
            end
            
            result = strjoin(stringTokens, ' ');
        end
        
        function result = append(obj, varargin)
            % Append additional tokens to the command
            %
            % Args:
            %   varargin: Tokens to append (strings, apt.MetaPath objects, or cell arrays)
            %
            % Returns:
            %   apt.ShellCommand: New command with tokens appended
            
            tokensToAdd = {};
            for i = 1:length(varargin)
                token = varargin{i};
                if iscell(token)
                    tokensToAdd = [tokensToAdd, token];
                else
                    tokensToAdd{end+1} = token;
                end
            end
            
            % Validate new tokens before adding
            obj.validateTokens_(tokensToAdd, 'append');
            
            newTokens = [obj.tokens_, tokensToAdd];
            result = apt.ShellCommand(newTokens, obj.locale_);
        end
        
        function result = prepend(obj, varargin)
            % Prepend tokens to the beginning of the command
            %
            % Args:
            %   varargin: Tokens to prepend (strings, apt.MetaPath objects, or cell arrays)
            %
            % Returns:
            %   apt.ShellCommand: New command with tokens prepended
            
            tokensToAdd = {};
            for i = 1:length(varargin)
                token = varargin{i};
                if iscell(token)
                    tokensToAdd = [tokensToAdd, token];
                else
                    tokensToAdd{end+1} = token;
                end
            end
            
            % Validate new tokens before adding
            obj.validateTokens_(tokensToAdd, 'prepend');
            
            newTokens = [tokensToAdd, obj.tokens_];
            result = apt.ShellCommand(newTokens, obj.locale_);
        end
        
        function result = substitute(obj, oldToken, newToken)
            % Substitute all instances of oldToken with newToken
            %
            % Args:
            %   oldToken: Token to replace (string or apt.MetaPath)
            %   newToken: Replacement token (string or apt.MetaPath)
            %
            % Returns:
            %   apt.ShellCommand: New command with substitutions made
            
            % Validate new token if it's an apt.MetaPath
            if isa(newToken, 'apt.MetaPath')
                obj.validateTokens_({newToken}, 'substitute');
            end
            
            newTokens = obj.tokens_;
            for i = 1:length(newTokens)
                if obj.tokensEqual_(newTokens{i}, oldToken)
                    newTokens{i} = newToken;
                end
            end
            
            result = apt.ShellCommand(newTokens, obj.locale_);
        end
        
        function result = getPathTokens(obj)
            % Get all apt.MetaPath tokens from the command
            %
            % Returns:
            %   cell: Cell array of apt.MetaPath objects in the command
            
            result = {};
            for i = 1:length(obj.tokens_)
                if isa(obj.tokens_{i}, 'apt.Path')
                    result{end+1} = obj.tokens_{i};
                end
            end
        end
        
        function result = getTokensByRole(obj, fileRole)
            % Get all path tokens with specific file role
            %
            % Args:
            %   fileRole (char or apt.FileRole): File role to match (e.g., 'movie', 'cache')
            %
            % Returns:
            %   cell: Cell array of apt.MetaPath objects with matching role
            
            % Convert string to enum if needed
            if ischar(fileRole)
                fileRole = apt.FileRole.fromString(fileRole);
            end
            
            result = {};
            pathTokens = obj.getPathTokens();
            for i = 1:length(pathTokens)
                if pathTokens{i}.isRole(fileRole)
                    result{end+1} = pathTokens{i};
                end
            end
        end
        
        function result = length(obj)
            % Get number of tokens in command
            result = length(obj.tokens_);
        end
        
        function result = getToken(obj, index)
            % Get token at specific index
            %
            % Args:
            %   index (numeric): 1-based index
            %
            % Returns:
            %   Token at the specified index
            
            if index < 1 || index > length(obj.tokens_)
                error('apt:ShellCommand:IndexOutOfBounds', ...
                      'Index %d is out of bounds for command with %d tokens', ...
                      index, length(obj.tokens_));
            end
            
            result = obj.tokens_{index};
        end
        
        function result = eq(obj, other)
            % Check equality with another apt.ShellCommand
            if ~isa(other, 'apt.ShellCommand')
                result = false;
                return;
            end
            
            if length(obj.tokens_) ~= length(other.tokens_) || obj.locale_ ~= other.locale_
                result = false;
                return;
            end
            
            for i = 1:length(obj.tokens_)
                if ~obj.tokensEqual_(obj.tokens_{i}, other.tokens_{i})
                    result = false;
                    return;
                end
            end
            
            result = true;
        end
        
        function disp(obj)
            % Display the apt.ShellCommand object
            fprintf('apt.ShellCommand [%s] with %d tokens:\n', ...
                    apt.PathLocale.toString(obj.locale_), length(obj.tokens_));
            for i = 1:length(obj.tokens_)
                token = obj.tokens_{i};
                if isa(token, 'apt.MetaPath')
                    fprintf('  [%d] Path: %s [%s:%s]\n', i, token.toString(), ...
                            apt.PathLocale.toString(token.locale), apt.FileRole.toString(token.fileRole));
                else
                    fprintf('  [%d] Literal: %s\n', i, char(token));
                end
            end
            fprintf('  String: %s\n', obj.toString());
        end
        
        function result = tokensEqual_(~, token1, token2)
            % Check if two tokens are equal
            if isa(token1, 'apt.MetaPath') && isa(token2, 'apt.MetaPath')
                result = token1.eq(token2);
            elseif ischar(token1) && ischar(token2)
                result = strcmp(token1, token2);
            else
                result = false;
            end
        end
        
        function validateTokens_(obj, tokens, methodName)
            % Validate that all apt.MetaPath tokens have matching locale
            %
            % Args:
            %   tokens (cell): Cell array of tokens to validate
            %   methodName (char): Name of calling method for error messages
            
            for i = 1:length(tokens)
                token = tokens{i};
                if isa(token, 'apt.MetaPath')
                    if token.locale ~= obj.locale_
                        error('apt:ShellCommand:LocaleMismatch', ...
                              'In %s: MetaPath token at index %d has locale %s, but ShellCommand has locale %s', ...
                              methodName, i, apt.PathLocale.toString(token.locale), apt.PathLocale.toString(obj.locale_));
                    end
                end
            end
        end
    end
    
    methods (Static)
        function result = fromString(cmdStr, varargin)
            % Create apt.ShellCommand from string (limited path detection)
            %
            % Args:
            %   cmdStr (char): Command string to parse
            %   varargin: Optional path hints parsed with myparse
            %     'PathHints' (cell): Cell array of {index, platform, fileRole} hints
            %     'Locale' (char or apt.PathLocale): Platform context (default: native)
            %
            % Returns:
            %   apt.ShellCommand: Command object with detected paths
            %
            % Note: This is a basic implementation. Complex path detection
            % would require more sophisticated parsing.
            
            [pathHints, locale] = myparse(varargin, ...
                'PathHints', {}, ...
                'Locale', apt.PathLocale.native);
            
            % Convert string to enum if needed
            if ischar(locale)
                locale = apt.PathLocale.fromString(locale);
            end
            
            % Simple tokenization by spaces (doesn't handle quoted strings properly)
            tokens = strsplit(cmdStr, ' ', 'CollapseDelimiters', true);
            
            % Convert specified tokens to paths based on hints
            for i = 1:size(pathHints, 1)
                idx = pathHints{i, 1};
                hintLocale = pathHints{i, 2};
                fileRole = pathHints{i, 3};
                
                % Convert string fileRole to enum if needed
                if ischar(fileRole)
                    fileRole = apt.FileRole.fromString(fileRole);
                end
                
                if idx <= length(tokens)
                    tokens{idx} = apt.MetaPath(tokens{idx}, hintLocale, fileRole);
                end
            end
            
            result = apt.ShellCommand(tokens, locale);
        end
        
        result = cat(varargin)
        % Concatenate any number of strings, apt.MetaPaths, and apt.ShellCommands
        % into a single apt.ShellCommand (defined in cat.m)
    end
end