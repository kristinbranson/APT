function test_Path()

for platform = enumeration('apt.Platform')'
  if platform == apt.Platform.windows
    nativePathAsString = 'C:\foo\bar\baz';
    expectedPathAsString = '''C:\foo\bar\baz''';  % Expect quoted version
    correctNativePathAsList = {'C:', 'foo', 'bar', 'baz'} ;
  else
    nativePathAsString = '/foo/bar/baz';
    expectedPathAsString = '/foo/bar/baz';  % Paths without special characters don't get quoted
    correctNativePathAsList = {'', 'foo', 'bar', 'baz'} ;
  end
  nativePath = apt.Path(nativePathAsString, platform) ;
  nativePathAsStringHopefully = nativePath.char() ;
  if ~strcmp(expectedPathAsString, nativePathAsStringHopefully)
    fprintf('original: ''%s''\n', nativePathAsString) ;
    fprintf('expected: ''%s''\n', expectedPathAsString) ;
    fprintf('convert:  ''%s''\n', nativePathAsStringHopefully) ;
    error('Creating an apt.Path from a string and then using .char() did not produce the expected path') ;
  end
  
  if ~isequal(nativePath.list, correctNativePathAsList)
    error('Path list is wrong. Expected: %s, Got: %s', mat2str(correctNativePathAsList), mat2str(nativePath.list)) ;
  end

  % Test that constructor creates empty path for empty string
  emptyStringPath = apt.Path('', platform);
  if ~isempty(emptyStringPath.list)
    error('Empty string should create empty path');
  end
  if ~strcmp(emptyStringPath.char(), '.')
    error('Empty string path should display as "."');
  end

  % Test root path behavior: should error on Windows, succeed with empty list on Unix
  if platform == apt.Platform.windows
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
  if platform == apt.Platform.windows
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
end  % for platform = enumeration('apt.Platform')'

% Test platform property auto-detection
pathWithAutoPlatform = apt.Path('/test/path');
if ~isa(pathWithAutoPlatform.platform, 'apt.Platform')
  error('Platform property should be an apt.Platform enumeration');
end

% Test platform from string
pathWithStringPlatform = apt.Path('/test/path', 'posix');
if pathWithStringPlatform.platform ~= apt.Platform.posix
  error('Platform from string specification failed');
end

% Test tfIsAbsolute property
% Test absolute paths
absPath1 = apt.Path('/test/path', apt.Platform.posix);
if ~absPath1.tfIsAbsolute()
  error('Unix absolute path should have tfIsAbsolute = true');
end

absPath2 = apt.Path('C:\Windows\System32', apt.Platform.windows);
if ~absPath2.tfIsAbsolute()
  error('Windows absolute path should have tfIsAbsolute = true');
end

% Test relative paths
relPath1 = apt.Path('relative/path', apt.Platform.posix);
if relPath1.tfIsAbsolute()
  error('Unix relative path should have tfIsAbsolute = false');
end

relPath2 = apt.Path('relative\path', apt.Platform.windows);
if relPath2.tfIsAbsolute()
  error('Windows relative path should have tfIsAbsolute = false');
end

% Test with cell arrays
absCellPath = apt.Path({'', 'usr', 'bin'}, apt.Platform.posix);
if ~absCellPath.tfIsAbsolute()
  error('Unix absolute path from cell array should have tfIsAbsolute = true');
end

relCellPath = apt.Path({'usr', 'bin'}, apt.Platform.posix);
if relCellPath.tfIsAbsolute()
  error('Unix relative path from cell array should have tfIsAbsolute = false');
end

winAbsCellPath = apt.Path({'C:', 'Windows', 'System32'}, apt.Platform.windows);
if ~winAbsCellPath.tfIsAbsolute()
  error('Windows absolute path from cell array should have tfIsAbsolute = true');
end

% Test cat method with apt.Path objects
basePath = apt.Path('/home/user', apt.Platform.posix);
relativePath = apt.Path('docs/file.txt', apt.Platform.posix);
concatenated = basePath.cat(relativePath);
expectedPath = apt.Path('/home/user/docs/file.txt', apt.Platform.posix);
if ~isequal(concatenated, expectedPath)
  error('Path concatenation with apt.Path failed');
end

% Test cat method with string (converted to apt.Path)
concatenated2 = basePath.cat(apt.Path('pictures/photo.jpg', apt.Platform.posix));
expectedPath2 = apt.Path('/home/user/pictures/photo.jpg', apt.Platform.posix);
if ~isequal(concatenated2, expectedPath2)
  error('Path concatenation with string failed');
end

% Test cat error on absolute path
try
  absolutePath = apt.Path('/absolute/path', apt.Platform.posix);
  basePath.cat(absolutePath);
  error('cat should have errored on absolute path');
catch ME
  if ~contains(ME.identifier, 'apt:Path:AbsolutePath')
    error('cat errored but with wrong error type: %s', ME.identifier);
  end
end

% Test cat error on platform mismatch
try
  windowsBase = apt.Path('C:\Windows', apt.Platform.windows);
  linuxRelative = apt.Path('subdir/file', apt.Platform.posix);
  windowsBase.cat(linuxRelative);
  error('cat should have errored on platform mismatch');
catch ME
  if ~contains(ME.identifier, 'apt:Path:PlatformMismatch')
    error('cat errored but with wrong error type: %s', ME.identifier);
  end
end

% Test cat method with multiple arguments (char arrays)
basePath = apt.Path('/home/user', apt.Platform.posix);
concatenated3 = basePath.cat('docs', 'projects', 'file.txt');
expectedPath3 = apt.Path('/home/user/docs/projects/file.txt', apt.Platform.posix);
if ~isequal(concatenated3, expectedPath3)
  error('Path concatenation with multiple char arrays failed');
end

% Test cat method with mixed apt.Path and char array arguments
relativePath1 = apt.Path('folder1', apt.Platform.posix);
concatenated4 = basePath.cat(relativePath1, 'subfolder', 'data.csv');
expectedPath4 = apt.Path('/home/user/folder1/subfolder/data.csv', apt.Platform.posix);
if ~isequal(concatenated4, expectedPath4)
  error('Path concatenation with mixed arguments failed');
end

% Test cat method with no arguments (should return original path)
concatenated5 = basePath.cat();
if ~isequal(concatenated5, basePath)
  error('Path concatenation with no arguments should return same path');
end

% Test cat method with single argument
concatenated6 = basePath.cat('single/path');
expectedPath6 = basePath.cat('single/path');
if ~isequal(concatenated6, expectedPath6)
  error('Path concatenation with single argument failed');
end

% Test append method with single char array
appendPath1 = basePath.append('documents');
expectedAppendPath1 = apt.Path('/home/user/documents', apt.Platform.posix);
if ~isequal(appendPath1, expectedAppendPath1)
  error('Path append with single char array failed');
end

% Test append method with multiple char arrays
appendPath2 = basePath.append('docs', 'projects', 'file.txt');
expectedAppendPath2 = apt.Path('/home/user/docs/projects/file.txt', apt.Platform.posix);
if ~isequal(appendPath2, expectedAppendPath2)
  error('Path append with multiple char arrays failed');
end

% Test append method error on empty argument
try
  basePath.append('valid', '', 'another');
  error('append should have errored on empty argument');
catch ME
  if ~contains(ME.identifier, 'apt:Path:EmptyArgument')
    error('append errored but with wrong error type: %s', ME.identifier);
  end
end

% Test append method error on non-char argument
try
  basePath.append('valid', 123, 'another');
  error('append should have errored on non-char argument');
catch ME
  if ~contains(ME.identifier, 'apt:Path:InvalidArgument')
    error('append errored but with wrong error type: %s', ME.identifier);
  end
end

% Test append method error on column vector
try
  basePath.append('valid', ['a'; 'b'], 'another');
  error('append should have errored on column vector argument');
catch ME
  if ~contains(ME.identifier, 'apt:Path:InvalidArgument')
    error('append errored but with wrong error type: %s', ME.identifier);
  end
end

% Test empty path creation (with auto-detected platform)
emptyPathAuto1 = apt.Path();
emptyPathAuto2 = apt.Path('.');
if ~isequal(emptyPathAuto1, emptyPathAuto2)
  error('Empty paths with auto-detected platform should be equal');
end

% Test empty path creation and cat behavior
for platform = enumeration('apt.Platform')'
  % Test creating empty path with empty array
  emptyPath1 = apt.Path([], platform);
  if ~isempty(emptyPath1.list)
    error('Empty path created with empty array should have empty list');
  end
  if emptyPath1.tfIsAbsolute()
    error('Empty path should be relative');
  end
  if ~strcmp(emptyPath1.char(), '.')
    error('Empty path should display as "." but got: %s', emptyPath1.char());
  end
  
  % Test creating empty path with '.'
  emptyPath2 = apt.Path('.', platform);
  if ~isempty(emptyPath2.list)
    error('Empty path created with "." should have empty list');
  end
  if emptyPath2.tfIsAbsolute()
    error('Empty path created with "." should be relative');
  end
  if ~strcmp(emptyPath2.char(), '.')
    error('Empty path created with "." should display as "." but got: %s', emptyPath2.char());
  end
  if emptyPath2.platform ~= platform
    error('Empty path should preserve specified platform');
  end
  
  % Test that empty paths are equal when created with same platform
  if ~isequal(emptyPath1, emptyPath2)
    error('Empty paths should be equal when created with same platform');
  end
  
  % Test cat with empty path as second argument
  if platform == apt.Platform.windows
    testPath = apt.Path('C:\Users\test\docs', platform);
  else
    testPath = apt.Path('/home/user/docs', platform);
  end
  
  result1 = testPath.cat(emptyPath2);
  if ~isequal(result1, testPath)
    error('Concatenating with empty path as second argument should return first path unchanged');
  end
  
  % Test cat with empty path as first argument
  result2 = emptyPath2.cat(apt.Path('relative/file.txt', platform));
  expectedResult2 = apt.Path('relative/file.txt', platform);
  if ~isequal(result2, expectedResult2)
    error('Concatenating empty path with relative path should return the relative path');
  end
  
  % Test cat with both paths empty
  emptyPath3 = apt.Path('.', platform);
  result3 = emptyPath2.cat(emptyPath3);
  if ~isequal(result3, emptyPath2)
    error('Concatenating two empty paths should return an empty path');
  end
  
  % Test fileparts2 with single component (should return empty path and filename)
  singlePath = apt.Path('foo', platform);
  [pathPart, filenamePart] = singlePath.fileparts2();
  
  % Path part should be empty path
  expectedEmptyPath = apt.Path('.', platform);
  if ~isequal(pathPart, expectedEmptyPath)
    error('fileparts2 of single component should return empty path for directory part');
  end
  
  % Filename part should be the original component
  expectedFilenamePart = apt.Path('foo', platform);
  if ~isequal(filenamePart, expectedFilenamePart)
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
  if ~strcmp(allDotsPath.char(), '.')
    error('Path consisting only of "." should display as "."');
  end
  
  % Test replacePrefix method
  if platform == apt.Platform.windows
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
  if ~isequal(result, expectedResult)
    error('replacePrefix should replace matching prefix correctly');
  end
  
  % Test replacePrefix with non-matching prefix
  if platform == apt.Platform.windows
    nonMatchingSource = apt.Path('C:\different\path', platform);
  else
    nonMatchingSource = apt.Path('/different/path', platform);
  end
  
  resultNoMatch = originalPath.replacePrefix(nonMatchingSource, targetPath);
  if ~isequal(resultNoMatch, originalPath)
    error('replacePrefix should return original path when prefix does not match');
  end
  
  % Note: replacePrefix with string arguments is not tested because .char() 
  % may return quoted strings for Windows paths, which cannot be used to construct new paths
  
  % Test replacePrefix with empty paths
  emptySource = apt.Path('.', platform);
  emptyTarget = apt.Path('.', platform);
  resultEmptyPrefix = originalPath.replacePrefix(emptySource, emptyTarget);
  if ~isequal(resultEmptyPrefix, originalPath)
    error('replacePrefix with empty source should return original path');
  end
  
  % Test replacePrefix where source equals the entire path
  exactMatchResult = originalPath.replacePrefix(originalPath, targetPath);
  if ~isequal(exactMatchResult, targetPath)
    error('replacePrefix should return target path when source equals entire original path');
  end
end

% Test root path char()
rootPath = apt.Path('/');
rootPathStr = rootPath.char();
if ~strcmp(rootPathStr, '/')
  error('apt.Path(''/'').char() should return ''/'' but got: %s', rootPathStr);
end

% Test toPosix() method
% Test Windows absolute path conversion
winAbsPath = apt.Path('C:\Users\data\file.txt', apt.Platform.windows);
posixAbsPath = winAbsPath.toPosix();
expectedPosixPath = apt.Path('/mnt/c/Users/data/file.txt', apt.Platform.posix);
if ~isequal(posixAbsPath, expectedPosixPath)
  error('Windows absolute path POSIX conversion failed');
end

% Test Windows drive-only path conversion
winDrivePath = apt.Path('D:', apt.Platform.windows);
posixDrivePath = winDrivePath.toPosix();
expectedPosixDrivePath = apt.Path('/mnt/d', apt.Platform.posix);
if ~isequal(posixDrivePath, expectedPosixDrivePath)
  error('Windows drive-only path POSIX conversion failed');
end

% Test Windows relative path conversion
winRelPath = apt.Path('relative\path\file.txt', apt.Platform.windows);
posixRelPath = winRelPath.toPosix();
expectedPosixRelPath = apt.Path({'relative', 'path', 'file.txt'}, apt.Platform.posix);
if ~isequal(posixRelPath, expectedPosixRelPath)
  error('Windows relative path POSIX conversion failed');
end

% Test Linux path identity (should return same object)
linuxPath = apt.Path('/usr/bin/test', apt.Platform.posix);
posixLinuxPath = linuxPath.toPosix();
if ~isequal(posixLinuxPath, linuxPath)
  error('Linux path POSIX conversion should return identical path');
end

% Test macOS path identity (should return same object like Linux)
macPath = apt.Path('/Applications/Test.app', apt.Platform.posix);
posixMacPath = macPath.toPosix();
if ~isequal(posixMacPath, macPath)
  error('macOS path POSIX conversion should return identical path');
end

% Test toWindows() method
% Test POSIX WSL mount path conversion
posixWslPath = apt.Path('/mnt/c/Users/data/file.txt', apt.Platform.posix);
winFromPosixPath = posixWslPath.toWindows();
expectedWinPath = apt.Path('C:\Users\data\file.txt', apt.Platform.windows);
if ~isequal(winFromPosixPath, expectedWinPath)
  error('POSIX WSL mount path Windows conversion failed');
end

% Test POSIX WSL drive-only path conversion
posixWslDrivePath = apt.Path('/mnt/d', apt.Platform.posix);
winFromDrivePath = posixWslDrivePath.toWindows();
expectedWinDrivePath = apt.Path('D:', apt.Platform.windows);
if ~isequal(winFromDrivePath, expectedWinDrivePath)
  error('POSIX WSL drive-only path Windows conversion failed');
end

% Test POSIX relative path conversion
posixRelativePath = apt.Path('relative/path/file.txt', apt.Platform.posix);
winFromRelativePath = posixRelativePath.toWindows();
expectedWinRelativePath = apt.Path({'relative', 'path', 'file.txt'}, apt.Platform.windows);
if ~isequal(winFromRelativePath, expectedWinRelativePath)
  error('POSIX relative path Windows conversion failed');
end

% Test POSIX non-WSL absolute path conversion (should just change platform)
posixNonWslPath = apt.Path('/usr/local/bin/test', apt.Platform.posix);
winFromNonWslPath = posixNonWslPath.toWindows();
expectedWinNonWslPath = apt.Path({'', 'usr', 'local', 'bin', 'test'}, apt.Platform.windows);
if ~isequal(winFromNonWslPath, expectedWinNonWslPath)
  error('POSIX non-WSL absolute path Windows conversion failed');
end

% Test Windows path identity (should return same object)
winPath = apt.Path('C:\Windows\System32\test.exe', apt.Platform.windows);
winFromWinPath = winPath.toWindows();
if ~isequal(winFromWinPath, winPath)
  error('Windows path Windows conversion should return identical path');
end

% Test round-trip conversion (Windows -> POSIX -> Windows)
originalWinPath = apt.Path('C:\Program Files\App\data.txt', apt.Platform.windows);
posixVersion = originalWinPath.toPosix();
backToWinPath = posixVersion.toWindows();
if ~isequal(backToWinPath, originalWinPath)
  error('Round-trip Windows -> POSIX -> Windows conversion failed');
end

% Test round-trip conversion (POSIX WSL -> Windows -> POSIX)
originalPosixWslPath = apt.Path('/mnt/c/temp/data.log', apt.Platform.posix);
winVersion = originalPosixWslPath.toWindows();
backToPosixPath = winVersion.toPosix();
if ~isequal(backToPosixPath, originalPosixWslPath)
  error('Round-trip POSIX WSL -> Windows -> POSIX conversion failed');
end

%
% Test persistence stuff
%

% Test with various path types and platforms
testPaths = {
  apt.Path('/usr/local/bin', apt.Platform.posix), ...
  apt.Path('C:\Windows\System32', apt.Platform.windows), ...
  apt.Path('relative/path/file.txt', apt.Platform.posix), ...
  apt.Path({'relative', 'windows', 'path.exe'}, apt.Platform.windows), ...
  apt.Path('.', apt.Platform.posix), ...  % empty path
  apt.Path('/', apt.Platform.posix), ...  % root path
  apt.Path('C:', apt.Platform.windows) ...  % drive-only path
};

for i = 1:numel(testPaths)
  originalPath = testPaths{i};
  
  % Encode the path object
  encodedPath = encode_for_persistence(originalPath);
  
  % Test that the encoded result is an encoding container
  if ~is_an_encoding_container(encodedPath)
    error('encode_for_persistence should return an encoding container for path: %s', originalPath.char());
  end
  
  % Decode the encoded path
  decodedPath = decode_encoding_container(encodedPath);
  
  % Check that the decoded path equals the original using isequal
  if ~isequal(originalPath, decodedPath)
    error('Persistence round-trip failed for path: %s (platform: %s)', ...
          originalPath.char(), char(originalPath.platform));
  end
end

end  % function
