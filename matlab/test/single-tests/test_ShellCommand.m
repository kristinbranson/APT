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

end