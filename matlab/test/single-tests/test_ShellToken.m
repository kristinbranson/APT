function test_ShellToken()

% Test ShellLiteral constructor and basic functionality
lit1 = apt.ShellLiteral('hello');
if ~strcmp(lit1.char(), 'hello')
  error('ShellLiteral toString failed');
end

lit2 = apt.ShellLiteral("world");  % Test string input
if ~strcmp(lit2.char(), 'world')
  error('ShellLiteral string input failed');
end

% Test empty constructor
emptyLit = apt.ShellLiteral();
if ~strcmp(emptyLit.char(), '')
  error('Empty ShellLiteral constructor failed');
end

% Test equality
lit3 = apt.ShellLiteral('hello');
if ~isequal(lit1, lit3)
  error('ShellLiteral equality test failed');
end

if isequal(lit1, lit2)
  error('ShellLiteral inequality test failed');
end

% Test inheritance
if ~isa(lit1, 'apt.ShellToken')
  error('ShellLiteral should inherit from ShellToken');
end

% Test isLiteral and isPath methods
if ~lit1.isLiteral()
  error('ShellLiteral isLiteral should return true');
end

if lit1.isPath()
  error('ShellLiteral isPath should return false');
end

% Test tfDoesMatchLocale method - should always return true for literals
if ~lit1.tfDoesMatchLocale(apt.PathLocale.native)
  error('ShellLiteral should match native locale');
end

if ~lit1.tfDoesMatchLocale(apt.PathLocale.wsl)
  error('ShellLiteral should match wsl locale');
end

if ~lit1.tfDoesMatchLocale(apt.PathLocale.remote)
  error('ShellLiteral should match remote locale');
end

% Test with string input to tfDoesMatchLocale
if ~lit1.tfDoesMatchLocale('native')
  error('ShellLiteral should match native locale (string input)');
end

if ~lit1.tfDoesMatchLocale('wsl')
  error('ShellLiteral should match wsl locale (string input)');
end

% Test MetaPath tfDoesMatchLocale for comparison
nativePath = apt.MetaPath(apt.Path('/test'), 'native', 'cache');
wslPath = apt.MetaPath(apt.Path('/test'), 'wsl', 'cache');

if ~nativePath.tfDoesMatchLocale(apt.PathLocale.native)
  error('Native MetaPath should match native locale');
end

if nativePath.tfDoesMatchLocale(apt.PathLocale.wsl)
  error('Native MetaPath should not match wsl locale');
end

if wslPath.tfDoesMatchLocale(apt.PathLocale.native)
  error('WSL MetaPath should not match native locale');
end

if ~wslPath.tfDoesMatchLocale(apt.PathLocale.wsl)
  error('WSL MetaPath should match wsl locale');
end

% Test MetaPath with string input
if ~nativePath.tfDoesMatchLocale('native')
  error('Native MetaPath should match native locale (string input)');
end

if nativePath.tfDoesMatchLocale('wsl')
  error('Native MetaPath should not match wsl locale (string input)');
end

end