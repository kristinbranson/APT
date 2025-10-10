classdef ShellCommand < apt.ShellToken
  % apt.ShellCommand - Type-safe shell command composition with automatic path translation
  %
  % This class represents shell commands as structured lists of tokens rather than
  % plain strings, enabling automatic translation of paths and arguments when
  % commands need to execute in different contexts (native, WSL, remote backends).
  %
  % Each command consists of a sequence of tokens that can be:
  % - apt.ShellLiteral: Plain text that requires no translation
  % - apt.MetaPath: Paths that are automatically translated between contexts
  % - apt.ShellVariableAssignment: Environment variable assignments
  % - apt.ShellBind: Docker-style bind mount specifications
  % - apt.ShellCommand: Nested subcommands (for complex shell operations)
  %
  % Key benefits over string-based commands:
  % - Automatic path translation prevents invalid cross-platform paths
  % - Type safety prevents common command construction errors
  % - Structured representation enables programmatic command manipulation
  % - Locale and platform validation ensures command consistency
  %
  % INVARIANT: All apt.MetaPath tokens must have the same locale as the
  % ShellCommand's locale. This ensures consistency when converting between
  % different execution contexts.
  %
  % Example usage:
  %   moviePath = apt.MetaPath('/data/movie.avi', 'native', 'movie');
  %   cmd = apt.ShellCommand({'python', 'script.py', '--input', moviePath, '--output', 'results.txt'});
  %   cmdStr = cmd.char();  % Returns properly formatted command string

  properties
    tokens      % Cell array of tokens (strings and apt.MetaPath objects)
    locale      % apt.PathLocale enumeration indicating the path locale
    platform    % apt.Platform enumeration indicating the target platform
  end

  methods
    function obj = ShellCommand(tokens, locale, platform)
      % Constructor for apt.ShellCommand
      %
      % Args:
      %   tokens (cell): Cell array of command tokens (strings and apt.ShellToken objects)
      %   locale (char or apt.PathLocale): Platform context for the command
      %   platform (char or apt.Platform, optional): Target platform (default: auto-detect)
      %
      % INVARIANT: All MetaPath tokens within this ShellCommand must have the same
      % platform as the ShellCommand's platform. This ensures consistency when
      % executing commands and prevents mixing paths from different platforms.

      if nargin < 1
        tokens = {};
      end
      if nargin < 2
        locale = apt.PathLocale.native;
      end
      if nargin < 3
        platform = apt.Platform.current();
      end

      % Convert string to enum if needed
      if ischar(locale)
        locale = apt.PathLocale.fromString(locale);
      end
      if ischar(platform)
        platform = apt.Platform.fromString(platform);
      end

      % INVARIANT: WSL and remote locales require POSIX platform
      if (locale == apt.PathLocale.wsl || locale == apt.PathLocale.remote) && platform ~= apt.Platform.posix
        error('apt:ShellCommand:InvalidLocaleplatformCombination', ...
          'Locale %s requires platform to be posix, but got platform %s', ...
          char(locale), char(platform));
      end

      % Convert tokens to ShellToken objects and validate
      shellTokens = cell(size(tokens));
      for i = 1:length(tokens)
        token = tokens{i};
        if isa(token, 'apt.ShellToken')
          % Already a ShellToken - validate locale compatibility
          if ~isa(token, 'apt.ShellCommand') 
            % Subcommands do not necessarily have to match the supercommand locale or
            % platform, so that e.g. a ShellCommand can hold a wsl ssh command whose
            % subcommand is a remote command.
            if ~token.tfDoesMatchLocale(locale)
              if isa(token, 'apt.MetaPath')
                error('apt:ShellCommand:LocaleMismatch', ...
                  'MetaPath token at index %d has locale %s, but ShellCommand has locale %s', ...
                  i, char(token.locale), char(locale));
              else
                error('apt:ShellCommand:LocaleMismatch', ...
                  'Token at index %d does not match ShellCommand locale %s', ...
                  i, char(locale));
              end
            end
            
            % Validate platform compatibility for all ShellToken objects (INVARIANT)
            if ~token.tfDoesMatchPlatform(platform)
              if isa(token, 'apt.MetaPath')
                error('apt:ShellCommand:PlatformMismatch', ...
                  'MetaPath token at index %d has platform %s, but ShellCommand has platform %s', ...
                  i, char(token.path.platform), char(platform));
              else
                error('apt:ShellCommand:PlatformMismatch', ...
                  'Token at index %d does not match ShellCommand platform %s', ...
                  i, char(platform));
              end
            end
          end
          
          shellTokens{i} = token;
        elseif ischar(token) || isstring(token)
          % Convert string to ShellLiteral
          shellTokens{i} = apt.ShellLiteral(char(token));
        else
          error('apt:ShellCommand:InvalidToken', ...
            'Token at index %d must be a string or apt.ShellToken, got %s', ...
            i, class(token));
        end
      end

      obj.tokens = shellTokens;
      obj.locale = locale;
      obj.platform = platform;
    end


    function result = char(obj)
      % Convert ShellCommand to char array
      %
      % Returns:
      %   char: Character array representation of the command

      % Define helper function
      function result = charAndEscapeIfSubcommand(token)
        % Convert the token to a string.  If the token is an apt.ShellCommand, escape
        % it and prepend with /bin/bash -c
        if isa(token, 'apt.ShellCommand')
          draftString = token.char() ;
          if obj.platform == apt.Platform.windows
            result = escape_string_for_cmd_dot_exe(draftString) ;
          else            
            result = escape_string_for_bash(draftString) ;
          end
        else
          result = token.char() ;
        end
      end

      % Use helper function to convert each token to a string
      stringFromTokenIndex = cellfun(@charAndEscapeIfSubcommand, obj.tokens, 'UniformOutput', false) ;

      % Join the strings together, separated by spaces
      result = strjoin(stringFromTokenIndex, ' ');
    end

    function result = tfDoesMatchLocale(obj, queryLocale)
      % Check if this ShellCommand matches the specified locale
      %
      % Args:
      %   queryLocale (char or apt.PathLocale): The locale to check against
      %
      % Returns:
      %   logical: True if ShellCommand locale matches query locale

      % Convert string to enum if needed
      if ischar(queryLocale)
        queryLocale = apt.PathLocale.fromString(queryLocale);
      end

      result = (obj.locale == queryLocale);
    end

    function result = tfDoesMatchPlatform(obj, queryPlatform)
      % Check if this ShellCommand matches the specified platform
      %
      % Args:
      %   queryPlatform (char or apt.Platform): The platform to check against
      %
      % Returns:
      %   logical: True if ShellCommand platform matches query platform

      % Convert string to enum if needed
      if ischar(queryPlatform)
        queryPlatform = apt.Platform.fromString(queryPlatform);
      end

      result = (obj.platform == queryPlatform);
    end

    function result = append(obj, varargin)
      % Append additional tokens to the command
      %
      % Args:
      %   varargin: Tokens to append (apt.ShellToken objects or char arrays)
      %
      % Returns:
      %   apt.ShellCommand: New command with tokens appended
      %
      % Notes:
      %   - Char arrays are converted to apt.ShellLiteral objects
      %   - All other argument types cause an error

      tokensToAdd = cell(1,0);
      for i = 1:length(varargin)
        token = varargin{i};
        if isa(token, 'apt.ShellToken')
          tokensToAdd{1,end+1} = token;  %#ok<AGROW>
        elseif ischar(token)
          % Convert char array to ShellLiteral
          tokensToAdd{1,end+1} = apt.ShellLiteral(token); %#ok<AGROW>
        else
          error('apt:ShellCommand:InvalidToken', ...
            'Argument %d must be an apt.ShellToken or char array, got %s', ...
            i, class(token));
        end
      end

      % Validate new tokens before adding
      obj.validateTokens_(tokensToAdd, 'append');

      newTokens = [obj.tokens, tokensToAdd];
      result = apt.ShellCommand(newTokens, obj.locale, obj.platform);
    end

    % function result = prepend(obj, varargin)
    %   % Prepend tokens to the beginning of the command
    %   %
    %   % Args:
    %   %   varargin: Tokens to prepend (strings, apt.ShellToken objects, or cell arrays)
    %   %
    %   % Returns:
    %   %   apt.ShellCommand: New command with tokens prepended
    % 
    %   tokensToAdd = {};
    %   for i = 1:length(varargin)
    %     token = varargin{i};
    %     if iscell(token)
    %       tokensToAdd = [tokensToAdd, token];  %#ok<AGROW>
    %     else
    %       tokensToAdd{end+1} = token;  %#ok<AGROW>
    %     end
    %   end
    % 
    %   % Validate new tokens before adding
    %   obj.validateTokens_(tokensToAdd, 'prepend');
    % 
    %   newTokens = [tokensToAdd, obj.tokens];
    %   result = apt.ShellCommand(newTokens, obj.locale, obj.platform);
    % end

    % function result = substitute(obj, oldToken, newToken)
    %   % Substitute all instances of oldToken with newToken
    %   %
    %   % Args:
    %   %   oldToken: Token to replace (string or apt.MetaPath)
    %   %   newToken: Replacement token (string or apt.MetaPath)
    %   %
    %   % Returns:
    %   %   apt.ShellCommand: New command with substitutions made
    % 
    %   % Validate new token if it's an apt.MetaPath
    %   if isa(newToken, 'apt.MetaPath')
    %     obj.validateTokens_({newToken}, 'substitute');
    %   end
    % 
    %   newTokens = obj.tokens;
    %   for i = 1:length(newTokens)
    %     if obj.tokensEqual_(newTokens{i}, oldToken)
    %       newTokens{i} = newToken;
    %     end
    %   end
    % 
    %   result = apt.ShellCommand(newTokens, obj.locale, obj.platform);
    % end

    % function result = getPathTokens(obj)
    %   % Get all apt.MetaPath tokens from the command
    %   %
    %   % Returns:
    %   %   cell: Cell array of apt.MetaPath objects in the command
    % 
    %   result = {};
    %   for i = 1:length(obj.tokens)
    %     if isa(obj.tokens{i}, 'apt.Path')
    %       result{end+1} = obj.tokens{i};  %#ok<AGROW>
    %     end
    %   end
    % end
    % 
    % function result = getTokensByRole(obj, fileRole)
    %   % Get all path tokens with specific file role
    %   %
    %   % Args:
    %   %   fileRole (char or apt.FileRole): File role to match (e.g., 'movie', 'cache')
    %   %
    %   % Returns:
    %   %   cell: Cell array of apt.MetaPath objects with matching role
    % 
    %   % Convert string to enum if needed
    %   if ischar(fileRole)
    %     fileRole = apt.FileRole.fromString(fileRole);
    %   end
    % 
    %   result = {};
    %   pathTokens = obj.getPathTokens();
    %   for i = 1:length(pathTokens)
    %     if pathTokens{i}.isRole(fileRole)
    %       result{end+1} = pathTokens{i};  %#ok<AGROW>
    %     end
    %   end
    % end

    function result = length(obj)
      % Get number of tokens in command
      result = length(obj.tokens);
    end

    % function result = getToken(obj, index)
    %   % Get token at specific index
    %   %
    %   % Args:
    %   %   index (numeric): 1-based index
    %   %
    %   % Returns:
    %   %   Token at the specified index
    % 
    %   if index < 1 || index > length(obj.tokens)
    %     error('apt:ShellCommand:IndexOutOfBounds', ...
    %       'Index %d is out of bounds for command with %d tokens', ...
    %       index, length(obj.tokens));
    %   end
    % 
    %   result = obj.tokens{index};
    % end

    function result = isequal(obj, other)
      % Check equality with another apt.ShellCommand
      if ~isa(other, 'apt.ShellCommand')
        result = false;
        return;
      end

      if length(obj.tokens) ~= length(other.tokens) || obj.locale ~= other.locale
        result = false;
        return;
      end

      for i = 1:length(obj.tokens)
        if ~isequal(obj.tokens{i}, other.tokens{i})
          result = false;
          return;
        end
      end

      result = true;
    end

    function prettyPrint(obj)
      obj.prettyPrintHelper(0) ;
    end
          
    function prettyPrintHelper(obj, indentDepth)
      % Pretty-print the apt.ShellCommand object
      indent = repmat(' ', [1 indentDepth]) ;
      tokenCount = length(obj.tokens);
      if tokenCount==0
        fprintf('%s<ShellCommand with 0 tokens.>\n', indent);
        return
      end
      for i = 1:tokenCount
        token = obj.tokens{i};
        if isa(token, 'apt.MetaPath')
          fprintf('%s%s\n', indent, token.char());
        elseif isa(token, 'apt.ShellLiteral')
          fprintf('%s%s\n', indent, token.char());
        elseif isa(token, 'apt.ShellCommand')
          token.prettyPrintHelper(indentDepth+4) ;
        elseif isa(token, 'apt.ShellVariableAssignment')
          fprintf('%s%s\n', indent, token.char());
        elseif isa(token, 'apt.ShellBind')
          fprintf('%s%s\n', indent, token.char());
        else
          fprintf('  [%d] Unknown token type (%s): %s\n', i, class(token), char(token));
        end
      end
    end

    function validateTokens_(obj, newTokens, methodName)
      % Validate that all ShellToken objects have compatible locale and platform.
      % Used by .append().
      %
      % Args:
      %   tokens (cell): Cell array of tokens to validate
      %   methodName (char): Name of calling method for error messages

      for i = 1:length(newTokens)
        token = newTokens{i};
        if isa(token, 'apt.ShellToken')
          % Skip locale validation for ShellCommand tokens (subcommands can have different locales)
          if ~isa(token, 'apt.ShellCommand') 
            if ~token.tfDoesMatchLocale(obj.locale)
              if isa(token, 'apt.MetaPath')
                error('apt:ShellCommand:LocaleMismatch', ...
                  'In %s: MetaPath token at index %d has locale %s, but ShellCommand has locale %s', ...
                  methodName, i, char(token.locale), char(obj.locale));
              else
                error('apt:ShellCommand:LocaleMismatch', ...
                  'In %s: Token at index %d does not match ShellCommand locale %s', ...
                  methodName, i, char(obj.locale));
              end
            end
          end
          
          % Skip platform validation for ShellCommand tokens (subcommands can have different platforms)
          if ~isa(token, 'apt.ShellCommand')
            if ~token.tfDoesMatchPlatform(obj.platform)
              if isa(token, 'apt.MetaPath')
                error('apt:ShellCommand:PlatformMismatch', ...
                  'In %s: MetaPath token at index %d has platform %s, but ShellCommand has platform %s', ...
                  methodName, i, char(token.path.platform), char(obj.platform));
              else
                error('apt:ShellCommand:PlatformMismatch', ...
                  'In %s: Token at index %d does not match ShellCommand platform %s', ...
                  methodName, i, char(obj.platform));
              end
            end
          end
        else
          % If a token is not a ShellToken, that's an error
          error('apt:ShellCommand:InvalidToken', ...
            'In %s: Token at index %d must be an apt.ShellToken, got %s', ...
            methodName, i, class(token));
        end
      end
    end  % function

    % function result = replacePrefixForFileRole_(obj, role, oldPrefix, newPrefix)
    %   % Replace prefix for all MetaPaths with the given file role
    %   %
    %   % Args:
    %   %   role (apt.FileRole): File role to match
    %   %   oldPrefix (apt.MetaPath): Prefix to replace  
    %   %   newPrefix (apt.MetaPath): Replacement prefix
    %   %
    %   % Returns:
    %   %   apt.ShellCommand: New ShellCommand with prefixes replaced
    % 
    %   function result = processToken(token)
    %     % Local function to process a single token for role-based prefix replacement
    %     if isa(token, 'apt.MetaPath') && token.role == role
    %       result = token.replacePrefix(oldPrefix, newPrefix) ;
    %     elseif isa(token, 'apt.ShellCommand')
    %       % Recurse into nested ShellCommands
    %       result = token.replacePrefixForFileRole_(role, oldPrefix, newPrefix) ;
    %     else
    %       result = token ;
    %     end
    %   end  % local function
    % 
    %   % Use cellfun to process all tokens
    %   newTokens = cellfun(@processToken, obj.tokens, 'UniformOutput', false) ;
    % 
    %   % Create new ShellCommand with the updated tokens
    %   result = apt.ShellCommand(newTokens, obj.locale, obj.platform) ;
    % end  % function
    % 
    % function result = forceRemote_(obj)
    %   % Force all contained objects to have remote locale and POSIX platform
    %   %
    %   % Returns:
    %   %   apt.ShellCommand: New ShellCommand with remote locale, POSIX platform
    % 
    %   assert(obj.locale == apt.PathLocale.wsl || obj.locale == apt.PathLocale.remote, ...
    %          'forceRemote_() can only be used on WSL or remote ShellCommands (which are already POSIX)') ;
    % 
    %   function result = processToken(token)
    %     % Local function to force each token to be remote
    %     if isa(token, 'apt.MetaPath')
    %       result = token.forceRemote_() ;
    %     elseif isa(token, 'apt.ShellCommand')
    %       % Recurse into nested ShellCommands
    %       result = token.forceRemote_() ;
    %     else
    %       result = token ;
    %     end
    %   end  % local function
    % 
    %   % Use cellfun to process all tokens
    %   newTokens = cellfun(@processToken, obj.tokens, 'UniformOutput', false) ;
    % 
    %   % Create new ShellCommand with remote locale and POSIX platform
    %   result = apt.ShellCommand(newTokens, apt.PathLocale.remote, apt.Platform.posix) ;
    % end  % function

    function result = tfIsNull(obj)
      % Check if the command has no tokens
      %
      % Returns:
      %   logical: true if the command has no tokens, false otherwise
      result = isempty(obj.tokens);
    end  % function

    function result = cat(obj, varargin)
      % Instance method to concatenate this command with additional tokens
      %
      % Args:
      %   varargin: Tokens to concatenate (strings, apt.ShellToken objects, or apt.ShellCommand objects)
      %
      % Returns:
      %   apt.ShellCommand: New command with tokens concatenated
      %
      % Notes:
      %   - Equivalent to apt.ShellCommand.concat(obj, varargin{:})
      %   - Follows the same concatenation rules as the static concat method

      % Manually implement the same logic as concat without calling it
      allTokens = cell(1,0);
      locale = obj.locale;
      
      % Start with this object's tokens
      allTokens = horzcat(allTokens, obj.tokens);
      
      % Process additional arguments
      for i = 1:length(varargin)
        arg = varargin{i};
        
        if isempty(arg)
          % Skip empty arguments
          continue;
        elseif ischar(arg) || isstring(arg)
          % String literal token
          allTokens{1,end+1} = char(arg);  %#ok<AGROW>
        elseif isa(arg, 'apt.MetaPath')
          % MetaPath token
          if locale ~= arg.locale
            error('apt:ShellCommand:LocaleMismatch', ...
                  'MetaPath argument at position %d has locale %s, but this ShellCommand has locale %s', ...
                  i, char(arg.locale), char(locale));
          end
          allTokens{1,end+1} = arg;  %#ok<AGROW>
        elseif isa(arg, 'apt.ShellCommand')
          % ShellCommand to merge
          if locale ~= arg.locale
            error('apt:ShellCommand:LocaleMismatch', ...
                  'ShellCommand argument at position %d has locale %s, but this ShellCommand has locale %s', ...
                  i, char(arg.locale), char(locale));
          end
          % Add all tokens from the ShellCommand
          allTokens = horzcat(allTokens, arg.tokens);  %#ok<AGROW>
        else
          error('apt:ShellCommand:InvalidArgument', ...
                'Argument at position %d must be a string, apt.MetaPath, or apt.ShellCommand, got %s', ...
                i, class(arg));
        end
      end
      
      result = apt.ShellCommand(allTokens, locale, obj.platform);
    end  % function
  end  % methods

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
    end  % function
  end  % methods

  methods
    function result = encode_for_persistence_(obj, do_wrap_in_container)
      % Encode the tokens array - all tokens are apt.ShellToken objects
      encoded_tokens = cellfun(@(token) encode_for_persistence(token, true), obj.tokens, 'UniformOutput', false);
      
      encoding = struct('tokens', {encoded_tokens}, 'locale', {obj.locale}, 'platform', {obj.platform}) ;
      if do_wrap_in_container
        result = encoding_container('apt.ShellCommand', encoding) ;
      else
        result = encoding ;
      end
    end
  end  % methods

  methods (Static)
    function result = decode_encoding(encoding)
      % Decode the encoded version of the object.  Used for loading from persistent
      % storage.
      
      % Decode the tokens array - all encoded tokens are encoding containers
      decoded_tokens = cellfun(@(token) decode_encoding_container(token), encoding.tokens, 'UniformOutput', false);
      
      result = apt.ShellCommand(decoded_tokens, encoding.locale, encoding.platform) ;
    end
    
    result = concat(varargin)
      % Concatenate any number of strings, apt.MetaPaths, and apt.ShellCommands
      % into a single apt.ShellCommand (defined in concat.m)
  end  % methods (Static)
end  % classdef
