function test_ShellCommand()

% Test static cat method with mixed string and MetaPath arguments
path1 = apt.MetaPath(apt.Path('/data/input.txt'), 'native', 'movie');
path2 = apt.MetaPath(apt.Path('/results/output.txt'), 'native', 'cache');
cmd = apt.ShellCommand.cat('python', 'script.py', '--input', path1, '--output', path2);
expectedStr = 'python script.py --input /data/input.txt --output /results/output.txt';
if ~strcmp(cmd.char(), expectedStr)
  error('Mixed string and MetaPath concatenation failed. Expected: %s, Got: %s', expectedStr, cmd.char());
end

% Test concatenating two ShellCommand objects
cmd1 = apt.ShellCommand({'ls', '-la'});
cmd2 = apt.ShellCommand({'grep', 'test'});
combined = apt.ShellCommand.cat(cmd1, '|', cmd2);
expectedStr2 = 'ls -la | grep test';
if ~strcmp(combined.char(), expectedStr2)
  error('ShellCommand concatenation failed. Expected: %s, Got: %s', expectedStr2, combined.char());
end

% Test concatenating only strings (should default to native locale)
stringCmd = apt.ShellCommand.cat('echo', 'hello', 'world');
expectedStr3 = 'echo hello world';
if ~strcmp(stringCmd.char(), expectedStr3)
  error('String-only concatenation failed. Expected: %s, Got: %s', expectedStr3, stringCmd.char());
end

% Test empty argument handling
emptyCmd = apt.ShellCommand.cat();
if emptyCmd.length() ~= 0
  error('Empty cat should create empty command');
end

% Test locale mismatch error
try
  nativePath = apt.MetaPath(apt.Path('/native/path'), 'native', 'movie');
  wslPath = apt.MetaPath(apt.Path('/wsl/path'), 'wsl', 'cache');
  apt.ShellCommand.cat('cmd', nativePath, wslPath);
  error('Locale mismatch should have errored');
catch ME
  if ~contains(ME.identifier, 'apt:ShellCommand:LocaleMismatch')
    error('Locale mismatch errored with wrong error type: %s', ME.identifier);
  end
end

% Test mixing ShellCommand and MetaPath with same locale
nativeCmd = apt.ShellCommand({'python', 'script.py'}, 'native');
nativePath = apt.MetaPath(apt.Path('/data/file.txt'), 'native', 'movie');
mixedCmd = apt.ShellCommand.cat(nativeCmd, '--input', nativePath);
expectedStr4 = 'python script.py --input /data/file.txt';
if ~strcmp(mixedCmd.char(), expectedStr4)
  error('Mixed ShellCommand and MetaPath concatenation failed. Expected: %s, Got: %s', expectedStr4, mixedCmd.char());
end

% Test invalid argument type error
try
  apt.ShellCommand.cat('valid', 123);
  error('Invalid argument type should have errored');
catch ME
  if ~contains(ME.identifier, 'apt:ShellCommand:InvalidArgument')
    error('Invalid argument type errored with wrong error type: %s', ME.identifier);
  end
end

% Test ShellToken hierarchy and automatic conversion
path = apt.MetaPath(apt.Path('/data/file.txt'), 'native', 'movie');
literal = apt.ShellLiteral('hello');
cmd = apt.ShellCommand({literal, 'world', path});

% Check that tokens are properly typed
token1 = cmd.getToken(1);
token2 = cmd.getToken(2);
token3 = cmd.getToken(3);

if ~isa(token1, 'apt.ShellLiteral')
  error('First token should be ShellLiteral');
end
if ~isa(token2, 'apt.ShellLiteral')
  error('Second token should be ShellLiteral (auto-converted from string)');
end
if ~isa(token3, 'apt.MetaPath')
  error('Third token should be MetaPath');
end

% Check that all tokens are ShellTokens
if ~isa(token1, 'apt.ShellToken') || ~isa(token2, 'apt.ShellToken') || ~isa(token3, 'apt.ShellToken')
  error('All tokens should inherit from ShellToken');
end

% Test polymorphic char()
expectedStr = 'hello world /data/file.txt';
if ~strcmp(cmd.char(), expectedStr)
  error('Polymorphic char failed. Expected: %s, Got: %s', expectedStr, cmd.char());
end

% Test isLiteral and isPath methods
if ~token1.isLiteral() || token1.isPath()
  error('ShellLiteral isLiteral/isPath methods failed');
end
if token3.isLiteral() || ~token3.isPath()
  error('MetaPath isLiteral/isPath methods failed');
end

% Test locale validation with tfDoesMatchLocale method
try
  nativePath = apt.MetaPath(apt.Path('/test'), 'native', 'cache');
  apt.ShellCommand({'cmd', nativePath}, 'wsl');
  error('Should have thrown locale mismatch error');
catch ME
  if contains(ME.identifier, 'LocaleMismatch')
    % Expected behavior - locale validation working correctly
  else
    error('Wrong error type: %s', ME.identifier);
  end
end

% Test that matching locales work fine
nativePath2 = apt.MetaPath(apt.Path('/test2'), 'native', 'cache');
cmd = apt.ShellCommand({'cmd', nativePath2}, 'native');
if ~strcmp(cmd.char(), 'cmd /test2')
  error('Matching locale command creation failed');
end

% Test ShellCommand inheritance from ShellToken
if ~isa(cmd, 'apt.ShellToken')
  error('ShellCommand should inherit from ShellToken');
end

% Test nested ShellCommand functionality
innerCmd = apt.ShellCommand({'echo', 'hello world'});
outerCmd = apt.ShellCommand({'bash', '-c', innerCmd});
expectedStr = 'bash -c ''echo hello world''';
if ~strcmp(outerCmd.char(), expectedStr)
  error('Nested ShellCommand failed. Expected: %s, Got: %s', expectedStr, outerCmd.char());
end

% Test ShellCommand tfDoesMatchLocale
wslCmd = apt.ShellCommand({'ls'}, 'wsl');
if ~wslCmd.tfDoesMatchLocale('wsl')
  error('ShellCommand should match its own locale');
end
if wslCmd.tfDoesMatchLocale('native')
  error('ShellCommand should not match different locale');
end

% Test singleton ShellCommand (single token that is itself a ShellCommand)
baseCmd = apt.ShellCommand({'echo', 'test'});
nestedCmd = apt.ShellCommand({'/bin/bash', '-c', baseCmd});
expectedStr = '/bin/bash -c ''echo test''';
if ~strcmp(nestedCmd.char(), expectedStr)
  error('Singleton ShellCommand failed. Expected: %s, Got: %s', expectedStr, nestedCmd.char());
end

% Verify singleton has correct token count
if nestedCmd.length() ~= 3
  error('Singleton ShellCommand should have exactly 3 tokens, got %d', nestedCmd.length());
end

% Verify the token is the original ShellCommand
token = nestedCmd.getToken(3);
if ~isa(token, 'apt.ShellCommand')
  error('Nested command third token should be a ShellCommand');
end
if ~strcmp(token.char(), 'echo test')
  error('Singleton token content incorrect');
end

% Test char() conversion
testCmd = apt.ShellCommand({'echo', 'test'});
charResult = char(testCmd);
expectedCharResult = 'echo test';
if ~strcmp(charResult, expectedCharResult)
  error('char() conversion failed. Expected: %s, Got: %s', expectedCharResult, charResult);
end

% Test ShellCommand with all possible token types
% Create all different token types
stringToken = 'hello';  % Will be converted to ShellLiteral
literalToken = apt.ShellLiteral('world');
pathToken = apt.MetaPath('/tmp/test.txt', 'native', 'cache');
varToken = apt.ShellVariableAssignment('VAR', 'value');
srcPath = apt.MetaPath('/src/path', 'native', 'source');
dstPath = apt.MetaPath('/dst/path', 'native', 'cache');
bindToken = apt.ShellBind(srcPath, dstPath);
nestedCmdToken = apt.ShellCommand({'echo', 'nested'});

% Build command with all token types
allTokensCmd = apt.ShellCommand({stringToken, literalToken, pathToken, varToken, bindToken, nestedCmdToken});

% Test that char() works for all token types
try
  result = allTokensCmd.char();
  expectedResult = 'hello world /tmp/test.txt VAR=value type=bind,src=/src/path,dst=/dst/path ''echo nested''';
  if ~strcmp(result, expectedResult)
    error('All token types test failed. Expected: %s, Got: %s', expectedResult, result);
  end
catch ME
  error('ShellCommand.char() failed with all token types: %s', ME.message);
end

% Verify individual token types
token1 = allTokensCmd.getToken(1);
token2 = allTokensCmd.getToken(2);
token3 = allTokensCmd.getToken(3);
token4 = allTokensCmd.getToken(4);
token5 = allTokensCmd.getToken(5);
token6 = allTokensCmd.getToken(6);

if ~isa(token1, 'apt.ShellLiteral')
  error('Token 1 should be ShellLiteral (converted from string)');
end
if ~isa(token2, 'apt.ShellLiteral')
  error('Token 2 should be ShellLiteral');
end
if ~isa(token3, 'apt.MetaPath')
  error('Token 3 should be MetaPath');
end
if ~isa(token4, 'apt.ShellVariableAssignment')
  error('Token 4 should be ShellVariableAssignment');
end
if ~isa(token5, 'apt.ShellBind')
  error('Token 5 should be ShellBind');
end
if ~isa(token6, 'apt.ShellCommand')
  error('Token 6 should be ShellCommand');
end

% Test persistence encoding and decoding for apt.ShellCommand objects

% Test with various ShellCommand configurations
testShellCommands = {
  % Simple command with literals only
  apt.ShellCommand({'echo', 'hello', 'world'}), ...
  
  % Command with MetaPath
  apt.ShellCommand({'python', 'script.py', '--input', apt.MetaPath('/data/file.txt', 'native', 'movie')}), ...
  
  % Command with mixed token types
  apt.ShellCommand({'cmd', apt.MetaPath('/path', 'native', 'cache'), apt.ShellVariableAssignment('VAR', 'value')}), ...
  
  % Nested command
  apt.ShellCommand({'bash', '-c', apt.ShellCommand({'ls', '-la'})}), ...
  
  % Empty command
  apt.ShellCommand({}), ...
  
  % Command with different locales and platforms
  apt.ShellCommand({'echo', 'test'}, apt.PathLocale.wsl, apt.Platform.posix), ...
  
  % Command with all possible token types (reuse from earlier in test)
  allTokensCmd ...
};

for i = 1:numel(testShellCommands)
  originalCmd = testShellCommands{i};
  
  % Encode the ShellCommand object
  encodedCmd = encode_for_persistence(originalCmd);
  
  % Test that the encoded result is an encoding container
  if ~is_an_encoding_container(encodedCmd)
    error('encode_for_persistence should return an encoding container for ShellCommand: %s', originalCmd.char());
  end
  
  % Decode the encoded ShellCommand
  decodedCmd = decode_encoding_container(encodedCmd);
  
  % Check that the decoded ShellCommand equals the original using isequal
  if ~isequal(originalCmd, decodedCmd)
    error('Persistence round-trip failed for ShellCommand: %s (locale: %s, platform: %s)', ...
          originalCmd.char(), char(originalCmd.locale), char(originalCmd.platform));
  end
end

end