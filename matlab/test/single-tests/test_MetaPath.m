function test_MetaPath()

% Test basic MetaPath functionality with different platforms and locales
pathObj = apt.Path('/test/path', apt.Platform.posix);
metaPath = apt.MetaPath(pathObj, apt.PathLocale.native, apt.FileRole.input);

% Test char works
pathStr = metaPath.char();
if ~strcmp(pathStr, '/test/path')
  error('MetaPath char failed. Expected: /test/path, Got: %s', pathStr);
end

% Test property accessors
if metaPath.locale ~= apt.PathLocale.native
  error('MetaPath locale property failed');
end

if metaPath.role ~= apt.FileRole.input
  error('MetaPath role property failed');
end

if metaPath.platform ~= apt.Platform.posix
  error('MetaPath platform property failed');
end

if ~isequal(metaPath.path, pathObj)
  error('MetaPath path property failed');
end

% Test constructor with string enums
metaPath2 = apt.MetaPath(pathObj, 'wsl', 'cache');
if metaPath2.locale ~= apt.PathLocale.wsl
  error('MetaPath constructor with string locale failed');
end
if metaPath2.role ~= apt.FileRole.cache
  error('MetaPath constructor with string role failed');
end

% Test constructor validates apt.Path input
try
  apt.MetaPath(123, apt.PathLocale.native, apt.FileRole.input);
  error('Constructor should have errored on non-apt.Path input');
catch ME
  if ~contains(ME.identifier, 'apt:MetaPath:InvalidPath')
    error('Constructor errored but with wrong error type: %s', ME.identifier);
  end
end

% Test constructor requires absolute path
try
  relativePath = apt.Path('relative/path', apt.Platform.posix);
  apt.MetaPath(relativePath, apt.PathLocale.native, apt.FileRole.input);
  error('Constructor should have errored on relative path');
catch ME
  if ~contains(ME.identifier, 'apt:MetaPath:RelativePath')
    error('Constructor errored on relative path but with wrong error type: %s', ME.identifier);
  end
end

% Test equality
pathObj2 = apt.Path('/test/path', apt.Platform.posix);
metaPath3 = apt.MetaPath(pathObj2, apt.PathLocale.native, apt.FileRole.input);
if ~isequal(metaPath, metaPath3)
  error('MetaPath equality test failed for identical paths');
end

% Test inequality - different locale
metaPath4 = apt.MetaPath(pathObj2, apt.PathLocale.wsl, apt.FileRole.input);
if isequal(metaPath, metaPath4)
  error('MetaPath equality test failed - should not be equal with different locale');
end

% Test inequality - different role
metaPath5 = apt.MetaPath(pathObj2, apt.PathLocale.native, apt.FileRole.cache);
if isequal(metaPath, metaPath5)
  error('MetaPath equality test failed - should not be equal with different role');
end

% Test inequality - different path
pathObj3 = apt.Path('/different/path', apt.Platform.posix);
metaPath6 = apt.MetaPath(pathObj3, apt.PathLocale.native, apt.FileRole.input);
if isequal(metaPath, metaPath6)
  error('MetaPath equality test failed - should not be equal with different path');
end

% Test asNative() method
% Test WSL to native conversion
wslPath = apt.Path('/home/user/data/file.txt', apt.Platform.posix);
wslMetaPath = apt.MetaPath(wslPath, apt.PathLocale.wsl, apt.FileRole.input);
nativeFromWsl = wslMetaPath.asNative();

% Should have same path but native locale
expectedNativeMetaPath = apt.MetaPath(wslPath, apt.PathLocale.native, apt.FileRole.input);
if ~isequal(nativeFromWsl, expectedNativeMetaPath)
  error('asNative() failed for WSL to native conversion');
end

% Test native to native (should return same object)
nativeMetaPath = apt.MetaPath(wslPath, apt.PathLocale.native, apt.FileRole.cache);
nativeFromNative = nativeMetaPath.asNative();
if ~isequal(nativeFromNative, nativeMetaPath)
  error('asNative() failed for native to native conversion (should return same object)');
end

% Test with WSL mount point path
wslMountPath = apt.Path('/mnt/c/Projects/data.csv', apt.Platform.posix);
wslMountMetaPath = apt.MetaPath(wslMountPath, apt.PathLocale.wsl, apt.FileRole.cache);
nativeFromMount = wslMountMetaPath.asNative();

% Should convert to native locale (and potentially Windows path if on Windows)
if nativeFromMount.locale ~= apt.PathLocale.native
  error('asNative() should convert locale to native');
end
if nativeFromMount.role ~= apt.FileRole.cache
  error('asNative() should preserve file role');
end

% Test round-trip conversion (native -> wsl -> native using as() method)
originalNative = apt.MetaPath(wslPath, apt.PathLocale.native, apt.FileRole.input);
wslVersion = originalNative.as(apt.PathLocale.wsl);
backToNative = wslVersion.asNative();

if ~isequal(backToNative, originalNative)
  error('Round-trip native -> WSL -> native using as() methods failed');
end

%
% Test persistence encoding and decoding for apt.MetaPath objects
%

% Test with various MetaPath types, locales, and roles
testMetaPaths = {
  apt.MetaPath('/usr/local/bin/app', apt.PathLocale.native, apt.FileRole.cache), ...
  apt.MetaPath('/home/user/movies/video.mp4', apt.PathLocale.wsl, apt.FileRole.input), ...
  apt.MetaPath('/mnt/c/Data/project.mat', apt.PathLocale.remote, apt.FileRole.cache), ...
  apt.MetaPath(apt.Path('C:\Windows\System32', apt.Platform.windows), apt.PathLocale.native, apt.FileRole.input), ...
  apt.MetaPath('/groups/data/analysis', apt.PathLocale.wsl, apt.FileRole.cache), ...
  apt.MetaPath('/', apt.PathLocale.native, apt.FileRole.input), ...  % root path
  apt.MetaPath(apt.Path('C:', apt.Platform.windows), apt.PathLocale.native, apt.FileRole.cache) ...  % drive-only path
};

for i = 1:numel(testMetaPaths)
  originalMetaPath = testMetaPaths{i};
  
  % Encode the MetaPath object
  encodedMetaPath = encode_for_persistence(originalMetaPath);
  
  % Test that the encoded result is an encoding container
  if ~is_an_encoding_container(encodedMetaPath)
    error('encode_for_persistence should return an encoding container for MetaPath: %s', originalMetaPath.char());
  end
  
  % Decode the encoded MetaPath
  decodedMetaPath = decode_encoding_container(encodedMetaPath);
  
  % Check that the decoded MetaPath equals the original using isequal
  if ~isequal(originalMetaPath, decodedMetaPath)
    error('Persistence round-trip failed for MetaPath: %s (locale: %s, role: %s)', ...
          originalMetaPath.char(), char(originalMetaPath.locale), char(originalMetaPath.role));
  end
end

end