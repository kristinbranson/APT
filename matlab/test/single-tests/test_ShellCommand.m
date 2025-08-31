function test_ShellCommand()

% Test static cat method with mixed string and MetaPath arguments
path1 = apt.MetaPath(apt.Path('/data/input.txt'), 'native', 'movie');
path2 = apt.MetaPath(apt.Path('/results/output.txt'), 'native', 'cache');
cmd = apt.ShellCommand.cat('python', 'script.py', '--input', path1, '--output', path2);
expectedStr = 'python script.py --input /data/input.txt --output /results/output.txt';
if ~strcmp(cmd.toString(), expectedStr)
  error('Mixed string and MetaPath concatenation failed. Expected: %s, Got: %s', expectedStr, cmd.toString());
end

% Test concatenating two ShellCommand objects
cmd1 = apt.ShellCommand({'ls', '-la'});
cmd2 = apt.ShellCommand({'grep', 'test'});
combined = apt.ShellCommand.cat(cmd1, '|', cmd2);
expectedStr2 = 'ls -la | grep test';
if ~strcmp(combined.toString(), expectedStr2)
  error('ShellCommand concatenation failed. Expected: %s, Got: %s', expectedStr2, combined.toString());
end

% Test concatenating only strings (should default to native locale)
stringCmd = apt.ShellCommand.cat('echo', 'hello', 'world');
expectedStr3 = 'echo hello world';
if ~strcmp(stringCmd.toString(), expectedStr3)
  error('String-only concatenation failed. Expected: %s, Got: %s', expectedStr3, stringCmd.toString());
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
if ~strcmp(mixedCmd.toString(), expectedStr4)
  error('Mixed ShellCommand and MetaPath concatenation failed. Expected: %s, Got: %s', expectedStr4, mixedCmd.toString());
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

% Test polymorphic toString()
expectedStr = 'hello world /data/file.txt';
if ~strcmp(cmd.toString(), expectedStr)
  error('Polymorphic toString failed. Expected: %s, Got: %s', expectedStr, cmd.toString());
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
if ~strcmp(cmd.toString(), 'cmd /test2')
  error('Matching locale command creation failed');
end

% Test ShellCommand inheritance from ShellToken
if ~isa(cmd, 'apt.ShellToken')
  error('ShellCommand should inherit from ShellToken');
end

% Test nested ShellCommand functionality
innerCmd = apt.ShellCommand({'echo', 'hello world'});
outerCmd = apt.ShellCommand({'bash', '-c', innerCmd});
expectedStr = 'bash -c /bin/bash -c ''echo hello world''';
if ~strcmp(outerCmd.toString(), expectedStr)
  error('Nested ShellCommand failed. Expected: %s, Got: %s', expectedStr, outerCmd.toString());
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
singletonCmd = apt.ShellCommand({baseCmd});
expectedStr = '/bin/bash -c ''echo test''';
if ~strcmp(singletonCmd.toString(), expectedStr)
  error('Singleton ShellCommand failed. Expected: %s, Got: %s', expectedStr, singletonCmd.toString());
end

% Verify singleton has correct token count
if singletonCmd.length() ~= 1
  error('Singleton ShellCommand should have exactly 1 token, got %d', singletonCmd.length());
end

% Verify the token is the original ShellCommand
token = singletonCmd.getToken(1);
if ~isa(token, 'apt.ShellCommand')
  error('Singleton token should be a ShellCommand');
end
if ~strcmp(token.toString(), 'echo test')
  error('Singleton token content incorrect');
end

end