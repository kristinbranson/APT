function test_MetaPath()

for platform = enumeration('apt.Os')'
  if platform == apt.Os.windows
    nativePathAsString = 'C:\foo\bar\baz';
    correctNativePathAsList = {'C:', 'foo', 'bar', 'baz'} ;
  else
    nativePathAsString = '/foo/bar/baz';
    correctNativePathAsList = {'foo', 'bar', 'baz'} ;
  end
  nativePath = apt.MetaPath(nativePathAsString, apt.PathLocale.native, apt.FileRole.movie, platform) ;
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
    apt.MetaPath('', apt.PathLocale.native, apt.FileRole.movie, platform);
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
      apt.MetaPath('/', apt.PathLocale.native, apt.FileRole.movie, platform);
      error('Constructor should have errored on root path "/" on Windows but did not');
    catch ME
      if ~contains(ME.identifier, 'apt:Path:EmptyPath')
        error('Constructor errored on root path but with wrong error type: %s', ME.identifier);
      end
    end
  else
    % On Linux/Mac, root path should succeed and have empty list
    rootPath = apt.MetaPath('/', apt.PathLocale.native, apt.FileRole.movie, platform);
    if ~isempty(rootPath.list)
      error('Root path on Unix should have empty list but got: %s', mat2str(rootPath.list));
    end
  end
  
  % Test backslash separator behavior based on platform
  backslashPath = apt.MetaPath('C:\foo\bar\baz', apt.PathLocale.native, apt.FileRole.movie, platform);
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
pathWithAutoPlatform = apt.MetaPath('/test/path', apt.PathLocale.native, apt.FileRole.movie);
if ~isa(pathWithAutoPlatform.platform, 'apt.Os')
  error('Platform property should be an apt.Os enumeration');
end

% Test platform from string
pathWithStringPlatform = apt.MetaPath('/test/path', apt.PathLocale.native, apt.FileRole.movie, 'macos');
if pathWithStringPlatform.platform ~= apt.Os.macos
  error('Platform from string specification failed');
end

end
