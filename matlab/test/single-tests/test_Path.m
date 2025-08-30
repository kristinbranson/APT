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

  % Test that constructor creates empty path for empty string
  emptyStringPath = apt.Path('', platform);
  if ~isempty(emptyStringPath.list)
    error('Empty string should create empty path');
  end
  if ~strcmp(emptyStringPath.toString(), '.')
    error('Empty string path should display as "."');
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

% Test empty path creation (with auto-detected platform)
emptyPathAuto1 = apt.Path();
emptyPathAuto2 = apt.Path('.');
if ~emptyPathAuto1.eq(emptyPathAuto2)
  error('Empty paths with auto-detected platform should be equal');
end

% Test empty path creation and cat2 behavior
for platform = enumeration('apt.Os')'
  % Test creating empty path with empty array
  emptyPath1 = apt.Path([], platform);
  if ~isempty(emptyPath1.list)
    error('Empty path created with empty array should have empty list');
  end
  if emptyPath1.tfIsAbsolute
    error('Empty path should be relative');
  end
  if ~strcmp(emptyPath1.toString(), '.')
    error('Empty path should display as "." but got: %s', emptyPath1.toString());
  end
  
  % Test creating empty path with '.'
  emptyPath2 = apt.Path('.', platform);
  if ~isempty(emptyPath2.list)
    error('Empty path created with "." should have empty list');
  end
  if emptyPath2.tfIsAbsolute
    error('Empty path created with "." should be relative');
  end
  if ~strcmp(emptyPath2.toString(), '.')
    error('Empty path created with "." should display as "." but got: %s', emptyPath2.toString());
  end
  if emptyPath2.platform ~= platform
    error('Empty path should preserve specified platform');
  end
  
  % Test that empty paths are equal when created with same platform
  if ~emptyPath1.eq(emptyPath2)
    error('Empty paths should be equal when created with same platform');
  end
  
  % Test cat2 with empty path as second argument
  if platform == apt.Os.windows
    testPath = apt.Path('C:\Users\test\docs', platform);
  else
    testPath = apt.Path('/home/user/docs', platform);
  end
  
  result1 = testPath.cat2(emptyPath2);
  if ~result1.eq(testPath)
    error('Concatenating with empty path as second argument should return first path unchanged');
  end
  
  % Test cat2 with empty path as first argument
  result2 = emptyPath2.cat2('relative/file.txt');
  expectedResult2 = apt.Path('relative/file.txt', platform);
  if ~result2.eq(expectedResult2)
    error('Concatenating empty path with relative path should return the relative path');
  end
  
  % Test cat2 with both paths empty
  emptyPath3 = apt.Path('.', platform);
  result3 = emptyPath2.cat2(emptyPath3);
  if ~result3.eq(emptyPath2)
    error('Concatenating two empty paths should return an empty path');
  end
  
  % Test fileparts2 with single component (should return empty path and filename)
  singlePath = apt.Path('foo', platform);
  [pathPart, filenamePart] = singlePath.fileparts2();
  
  % Path part should be empty path
  expectedEmptyPath = apt.Path('.', platform);
  if ~pathPart.eq(expectedEmptyPath)
    error('fileparts2 of single component should return empty path for directory part');
  end
  
  % Filename part should be the original component
  expectedFilenamePart = apt.Path('foo', platform);
  if ~filenamePart.eq(expectedFilenamePart)
    error('fileparts2 of single component should return original component as filename part');
  end
  
  % Test that '.' elements are removed from path lists
  pathWithDots = apt.Path({'foo', '.', 'bar', '.', 'baz'}, platform);
  expectedList = {'foo', 'bar', 'baz'};
  if ~isequal(pathWithDots.list, expectedList)
    error('Constructor should remove "." elements from path list');
  end
  
  % Test that path consisting only of '.' becomes empty path
  allDotsPath = apt.Path({'.'}, platform);
  if ~isempty(allDotsPath.list)
    error('Path consisting only of "." should become empty path');
  end
  if ~strcmp(allDotsPath.toString(), '.')
    error('Path consisting only of "." should display as "."');
  end
  
  % Test replacePrefix method
  if platform == apt.Os.windows
    originalPath = apt.Path('C:\old\base\file.txt', platform);
    sourcePath = apt.Path('C:\old\base', platform);
    targetPath = apt.Path('D:\new\location', platform);
    expectedResult = apt.Path('D:\new\location\file.txt', platform);
  else
    originalPath = apt.Path('/old/base/file.txt', platform);
    sourcePath = apt.Path('/old/base', platform);
    targetPath = apt.Path('/new/location', platform);
    expectedResult = apt.Path('/new/location/file.txt', platform);
  end
  
  result = originalPath.replacePrefix(sourcePath, targetPath);
  if ~result.eq(expectedResult)
    error('replacePrefix should replace matching prefix correctly');
  end
  
  % Test replacePrefix with non-matching prefix
  if platform == apt.Os.windows
    nonMatchingSource = apt.Path('C:\different\path', platform);
  else
    nonMatchingSource = apt.Path('/different/path', platform);
  end
  
  resultNoMatch = originalPath.replacePrefix(nonMatchingSource, targetPath);
  if ~resultNoMatch.eq(originalPath)
    error('replacePrefix should return original path when prefix does not match');
  end
  
  % Test replacePrefix with string arguments
  resultWithStrings = originalPath.replacePrefix(sourcePath.toString(), targetPath.toString());
  if ~resultWithStrings.eq(expectedResult)
    error('replacePrefix should work with string arguments');
  end
  
  % Test replacePrefix with empty paths
  emptySource = apt.Path('.', platform);
  emptyTarget = apt.Path('.', platform);
  resultEmptyPrefix = originalPath.replacePrefix(emptySource, emptyTarget);
  if ~resultEmptyPrefix.eq(originalPath)
    error('replacePrefix with empty source should return original path');
  end
  
  % Test replacePrefix where source equals the entire path
  exactMatchResult = originalPath.replacePrefix(originalPath, targetPath);
  if ~exactMatchResult.eq(targetPath)
    error('replacePrefix should return target path when source equals entire original path');
  end
end

% Test root path toString()
rootPath = apt.Path('/');
rootPathStr = rootPath.toString();
if ~strcmp(rootPathStr, '/')
  error('apt.Path(''/'').toString() should return ''/'' but got: %s', rootPathStr);
end

% Test toPosix() method
% Test Windows absolute path conversion
winAbsPath = apt.Path('C:\Users\data\file.txt', apt.Os.windows);
posixAbsPath = winAbsPath.toPosix();
expectedPosixPath = apt.Path('/mnt/c/Users/data/file.txt', apt.Os.linux);
if ~posixAbsPath.eq(expectedPosixPath)
  error('Windows absolute path POSIX conversion failed');
end

% Test Windows drive-only path conversion
winDrivePath = apt.Path('D:', apt.Os.windows);
posixDrivePath = winDrivePath.toPosix();
expectedPosixDrivePath = apt.Path('/mnt/d', apt.Os.linux);
if ~posixDrivePath.eq(expectedPosixDrivePath)
  error('Windows drive-only path POSIX conversion failed');
end

% Test Windows relative path conversion
winRelPath = apt.Path('relative\path\file.txt', apt.Os.windows);
posixRelPath = winRelPath.toPosix();
expectedPosixRelPath = apt.Path({'relative', 'path', 'file.txt'}, apt.Os.linux);
if ~posixRelPath.eq(expectedPosixRelPath)
  error('Windows relative path POSIX conversion failed');
end

% Test Linux path identity (should return same object)
linuxPath = apt.Path('/usr/bin/test', apt.Os.linux);
posixLinuxPath = linuxPath.toPosix();
if ~posixLinuxPath.eq(linuxPath)
  error('Linux path POSIX conversion should return identical path');
end

% Test macOS path identity (should return same object like Linux)
macPath = apt.Path('/Applications/Test.app', apt.Os.macos);
posixMacPath = macPath.toPosix();
if ~posixMacPath.eq(macPath)
  error('macOS path POSIX conversion should return identical path');
end

end
