function test_MetaPath()

% Test basic MetaPath functionality with different platforms and locales
pathObj = apt.Path('/test/path', apt.Platform.posix);
metaPath = apt.MetaPath(pathObj, apt.PathLocale.native, apt.FileRole.movie);

% Test toString works
pathStr = metaPath.toString();
if ~strcmp(pathStr, '/test/path')
  error('MetaPath toString failed. Expected: /test/path, Got: %s', pathStr);
end

% Test property accessors
if metaPath.locale ~= apt.PathLocale.native
  error('MetaPath locale property failed');
end

if metaPath.role ~= apt.FileRole.movie
  error('MetaPath role property failed');
end

if metaPath.platform ~= apt.Platform.posix
  error('MetaPath platform property failed');
end

if ~metaPath.path.eq(pathObj)
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
  apt.MetaPath(123, apt.PathLocale.native, apt.FileRole.movie);
  error('Constructor should have errored on non-apt.Path input');
catch ME
  if ~contains(ME.identifier, 'apt:MetaPath:InvalidPath')
    error('Constructor errored but with wrong error type: %s', ME.identifier);
  end
end

% Test constructor requires absolute path
try
  relativePath = apt.Path('relative/path', apt.Platform.posix);
  apt.MetaPath(relativePath, apt.PathLocale.native, apt.FileRole.movie);
  error('Constructor should have errored on relative path');
catch ME
  if ~contains(ME.identifier, 'apt:MetaPath:RelativePath')
    error('Constructor errored on relative path but with wrong error type: %s', ME.identifier);
  end
end

% Test equality
pathObj2 = apt.Path('/test/path', apt.Platform.posix);
metaPath3 = apt.MetaPath(pathObj2, apt.PathLocale.native, apt.FileRole.movie);
if ~metaPath.eq(metaPath3)
  error('MetaPath equality test failed for identical paths');
end

% Test inequality - different locale
metaPath4 = apt.MetaPath(pathObj2, apt.PathLocale.wsl, apt.FileRole.movie);
if metaPath.eq(metaPath4)
  error('MetaPath equality test failed - should not be equal with different locale');
end

% Test inequality - different role
metaPath5 = apt.MetaPath(pathObj2, apt.PathLocale.native, apt.FileRole.cache);
if metaPath.eq(metaPath5)
  error('MetaPath equality test failed - should not be equal with different role');
end

% Test inequality - different path
pathObj3 = apt.Path('/different/path', apt.Platform.posix);
metaPath6 = apt.MetaPath(pathObj3, apt.PathLocale.native, apt.FileRole.movie);
if metaPath.eq(metaPath6)
  error('MetaPath equality test failed - should not be equal with different path');
end

% Test asNative() method
% Test WSL to native conversion
wslPath = apt.Path('/home/user/data/file.txt', apt.Platform.posix);
wslMetaPath = apt.MetaPath(wslPath, apt.PathLocale.wsl, apt.FileRole.movie);
nativeFromWsl = wslMetaPath.asNative();

% Should have same path but native locale
expectedNativeMetaPath = apt.MetaPath(wslPath, apt.PathLocale.native, apt.FileRole.movie);
if ~nativeFromWsl.eq(expectedNativeMetaPath)
  error('asNative() failed for WSL to native conversion');
end

% Test native to native (should return same object)
nativeMetaPath = apt.MetaPath(wslPath, apt.PathLocale.native, apt.FileRole.cache);
nativeFromNative = nativeMetaPath.asNative();
if ~nativeFromNative.eq(nativeMetaPath)
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
originalNative = apt.MetaPath(wslPath, apt.PathLocale.native, apt.FileRole.movie);
wslVersion = originalNative.as(apt.PathLocale.wsl);
backToNative = wslVersion.asNative();

if ~backToNative.eq(originalNative)
  error('Round-trip native -> WSL -> native using as() methods failed');
end

end