function test_Path()

for platform = enumeration('apt.Os')'
  if platform == apt.Os.windows
    nativePathAsString = 'C:\foo\bar\baz';
    correctNativePathAsList = {'C:', 'foo', 'bar', 'baz'} ;
  else
    nativePathAsString = '/foo/bar/baz';
    correctNativePathAsList = {'', 'foo', 'bar', 'baz'} ;
  end
  nativePath = apt.Path(nativePathAsString, platform) ;
  nativePathAsStringHopefully = nativePath.toString() ;
  if ~strcmp(nativePathAsString, nativePathAsStringHopefully)
    fprintf('original: ''%s''\n', nativePathAsString) ;
    fprintf('convert:  ''%s''\n', nativePathAsStringHopefully) ;
    error('Creating an apt.Path from a string and then using .toString() did not produce the same path') ;
  end
  
  if ~isequal(nativePath.list, correctNativePathAsList)
    error('Path list is wrong. Expected: %s, Got: %s', mat2str(correctNativePathAsList), mat2str(nativePath.list)) ;
  end

  % Test that constructor errors on empty string
  try
    apt.Path('', platform);
    error('Constructor should have errored on empty string but did not');
  catch ME
    if ~contains(ME.identifier, 'apt:Path:EmptyPath')
      error('Constructor errored on empty string but with wrong error type: %s', ME.identifier);
    end
  end

  % Test root path behavior: should error on Windows, succeed with empty list on Unix
  if platform == apt.Os.windows
    % On Windows, root path should error
    try
      apt.Path('/', platform);
      error('Constructor should have errored on root path "/" on Windows but did not');
    catch ME
      if ~contains(ME.identifier, 'apt:Path:EmptyPath')
        error('Constructor errored on root path but with wrong error type: %s', ME.identifier);
      end
    end
  else
    % On Linux/Mac, root path should succeed and have list with one empty string
    rootPath = apt.Path('/', platform);
    expectedRootList = {''};
    if ~isequal(rootPath.list, expectedRootList)
      error('Root path on Unix should have list {''''} but got: %s', sprintf('{''%s''}', strjoin(rootPath.list, ''', ''')));
    end
  end
  
  % Test backslash separator behavior based on platform
  backslashPath = apt.Path('C:\foo\bar\baz', platform);
  if platform == apt.Os.windows
    % On Windows, backslash separators should work as separators (4 components)
    expectedList = {'C:', 'foo', 'bar', 'baz'};
    if ~isequal(backslashPath.list, expectedList)
      error('Windows backslash path parsing failed. Expected: %s, Got: %s', mat2str(expectedList), mat2str(backslashPath.list));
    end
  else
    % On non-Windows platforms, backslashes are literal characters (1 component)
    expectedList = {'C:\foo\bar\baz'};
    if ~isequal(backslashPath.list, expectedList)
      error('Non-Windows backslash path parsing failed. Expected: %s, Got: %s', mat2str(expectedList), mat2str(backslashPath.list));
    end
  end
end  % for platform = enumeration('apt.Os')'

% Test platform property auto-detection
pathWithAutoPlatform = apt.Path('/test/path');
if ~isa(pathWithAutoPlatform.platform, 'apt.Os')
  error('Platform property should be an apt.Os enumeration');
end

% Test platform from string
pathWithStringPlatform = apt.Path('/test/path', 'macos');
if pathWithStringPlatform.platform ~= apt.Os.macos
  error('Platform from string specification failed');
end

% Test tfIsAbsolute property
% Test absolute paths
absPath1 = apt.Path('/test/path', apt.Os.linux);
if ~absPath1.tfIsAbsolute
  error('Unix absolute path should have tfIsAbsolute = true');
end

absPath2 = apt.Path('C:\Windows\System32', apt.Os.windows);
if ~absPath2.tfIsAbsolute
  error('Windows absolute path should have tfIsAbsolute = true');
end

% Test relative paths
relPath1 = apt.Path('relative/path', apt.Os.linux);
if relPath1.tfIsAbsolute
  error('Unix relative path should have tfIsAbsolute = false');
end

relPath2 = apt.Path('relative\path', apt.Os.windows);
if relPath2.tfIsAbsolute
  error('Windows relative path should have tfIsAbsolute = false');
end

% Test with cell arrays
absCellPath = apt.Path({'', 'usr', 'bin'}, apt.Os.linux);
if ~absCellPath.tfIsAbsolute
  error('Unix absolute path from cell array should have tfIsAbsolute = true');
end

relCellPath = apt.Path({'usr', 'bin'}, apt.Os.linux);
if relCellPath.tfIsAbsolute
  error('Unix relative path from cell array should have tfIsAbsolute = false');
end

winAbsCellPath = apt.Path({'C:', 'Windows', 'System32'}, apt.Os.windows);
if ~winAbsCellPath.tfIsAbsolute
  error('Windows absolute path from cell array should have tfIsAbsolute = true');
end

% Test cat2 method with apt.Path objects
basePath = apt.Path('/home/user', apt.Os.linux);
relativePath = apt.Path('docs/file.txt', apt.Os.linux);
concatenated = basePath.cat2(relativePath);
expectedPath = apt.Path('/home/user/docs/file.txt', apt.Os.linux);
if ~concatenated.eq(expectedPath)
  error('Path concatenation with apt.Path failed');
end

% Test cat2 method with string
concatenated2 = basePath.cat2('pictures/photo.jpg');
expectedPath2 = apt.Path('/home/user/pictures/photo.jpg', apt.Os.linux);
if ~concatenated2.eq(expectedPath2)
  error('Path concatenation with string failed');
end

% Test cat2 error on absolute path
try
  absolutePath = apt.Path('/absolute/path', apt.Os.linux);
  basePath.cat2(absolutePath);
  error('cat2 should have errored on absolute path');
catch ME
  if ~contains(ME.identifier, 'apt:Path:AbsolutePath')
    error('cat2 errored but with wrong error type: %s', ME.identifier);
  end
end

% Test cat2 error on platform mismatch
try
  windowsBase = apt.Path('C:\Windows', apt.Os.windows);
  linuxRelative = apt.Path('subdir/file', apt.Os.linux);
  windowsBase.cat2(linuxRelative);
  error('cat2 should have errored on platform mismatch');
catch ME
  if ~contains(ME.identifier, 'apt:Path:PlatformMismatch')
    error('cat2 errored but with wrong error type: %s', ME.identifier);
  end
end

% Test cat method with multiple arguments
basePath = apt.Path('/home/user', apt.Os.linux);
concatenated3 = basePath.cat('docs', 'projects', 'file.txt');
expectedPath3 = apt.Path('/home/user/docs/projects/file.txt', apt.Os.linux);
if ~concatenated3.eq(expectedPath3)
  error('Path concatenation with multiple strings failed');
end

% Test cat method with mixed apt.Path and string arguments
relativePath1 = apt.Path('folder1', apt.Os.linux);
concatenated4 = basePath.cat(relativePath1, 'subfolder', 'data.csv');
expectedPath4 = apt.Path('/home/user/folder1/subfolder/data.csv', apt.Os.linux);
if ~concatenated4.eq(expectedPath4)
  error('Path concatenation with mixed arguments failed');
end

% Test cat method with no arguments (should return original path)
concatenated5 = basePath.cat();
if ~concatenated5.eq(basePath)
  error('Path concatenation with no arguments should return same path');
end

% Test cat method with single argument (should be same as cat2)
concatenated6 = basePath.cat('single/path');
expectedPath6 = basePath.cat2('single/path');
if ~concatenated6.eq(expectedPath6)
  error('Path concatenation with single argument should match cat2 result');
end

end
