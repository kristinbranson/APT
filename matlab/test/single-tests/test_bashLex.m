function test_bashLex()
  % Tests that the function bashLex() correctly tokenizes bash-style strings.
  % bashLex() should correctly parse strings that have been escaped for bash.

  % Seed the RNG for reproducibility.
  % Restore it afterwards.  Can't hurt...
  t = rng() ;
  rng(42) ;
  oc = onCleanup(@()(rng(t))) ;

  % Test basic tokenization
  result = bashLex('hello world') ;
  expected = {'hello', 'world'} ;
  if ~isequal(result, expected)
    error('Basic tokenization failed') ;
  end

  % Test empty string
  result = bashLex('') ;
  expected = cell(1,0) ;
  if ~isequal(result, expected)
    error('Empty string test failed') ;
  end

  % Test single token
  result = bashLex('hello') ;
  expected = {'hello'} ;
  if ~isequal(result, expected)
    error('Single token test failed') ;
  end

  % Test quoted strings
  result = bashLex('hello "world with spaces"') ;
  expected = {'hello', 'world with spaces'} ;
  if ~isequal(result, expected)
    error('Double quoted string test failed') ;
  end

  result = bashLex('hello ''world with spaces''') ;
  expected = {'hello', 'world with spaces'} ;
  if ~isequal(result, expected)
    error('Single quoted string test failed') ;
  end

  % Test escaped characters
  result = bashLex('hello\ world') ;
  expected = {'hello world'} ;
  if ~isequal(result, expected)
    error('Escaped space test failed') ;
  end

  % Test mixed quotes and escapes
  result = bashLex('echo "hello world" ''another string'' escaped\ space') ;
  expected = {'echo', 'hello world', 'another string', 'escaped space'} ;
  if ~isequal(result, expected)
    error('Mixed quotes and escapes test failed') ;
  end

  % Test particular strings that might be problematic
  testStrings = { 'a', '\', '"', '\"', '"\', 'Hello, world!\', 'foo bar \\\\"\\', 'echo foo && false ; echo $?' } ;
  for i = 1 : numel(testStrings)
    str = testStrings{i} ;
    escaped = escape_string_for_bash(str) ;
    tokens = bashLex(escaped) ;
    if numel(tokens) ~= 1 || ~strcmp(tokens{1}, str)
      error('Round-trip test failed for string ''%s''', str) ;
    end
  end

  % Test other particular strings that might be problematic
  sq = '''' ;  % a single quote
  sq_bs_sq_sq = '''\''''' ; % single quote, backslash, single quote, single quote
  testStrings = { sq_bs_sq_sq, horzcat(sq_bs_sq_sq, 'hello', sq_bs_sq_sq), horzcat('"', sq_bs_sq_sq, '"') } ;
  for i = 1 : numel(testStrings)
    str = testStrings{i} ;
    escaped = escape_string_for_bash(str) ;
    tokens = bashLex(escaped) ;
    if numel(tokens) ~= 1 || ~strcmp(tokens{1}, str)
      error('Round-trip test failed for string ''%s''', str) ;
    end
  end

  % Test multiple strings escaped and concatenated
  strs = {'hello', 'world with spaces', 'special"chars', 'back\slash'} ;
  escapedStrs = escape_cellstring_for_bash(strs) ;
  bashCommand = strjoin(escapedStrs, ' ') ;
  tokens = bashLex(bashCommand) ;
  if ~isequal(tokens, strs)
    error('Multiple escaped strings test failed') ;
  end

  % Test strings with single quotes, which are tricky
  sqStrings = {'don''t', 'it''s', 'can''t go', 'multiple''single''quotes'} ;
  for i = 1 : numel(sqStrings)
    str = sqStrings{i} ;
    escaped = escape_string_for_bash(str) ;
    tokens = bashLex(escaped) ;
    if numel(tokens) ~= 1 || ~strcmp(tokens{1}, str)
      error('Single quote test failed for string ''%s''', str) ;
    end
  end

  % Test random sequences of special characters
  specialChars = ' \t"''\\' ;
  regularChars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' ;
  allChars = [specialChars regularChars] ;
  
  len = 20 ;
  n = 50 ;
  for i = 1 : n
    indices = randi(length(allChars), [1 len]) ;
    str = allChars(indices) ;
    escaped = escape_string_for_bash(str) ;
    tokens = bashLex(escaped) ;
    if numel(tokens) ~= 1 || ~strcmp(tokens{1}, str)
      error('Random character sequence test failed for string ''%s''', str) ;
    end
  end

  % Test error conditions
  try
    bashLex('unterminated "quote') ;
    error('Should have thrown error for unterminated double quote') ;
  catch me
    if ~contains(me.message, 'Unterminated quote')
      error('Wrong error message for unterminated double quote') ;
    end
  end

  try
    bashLex('unterminated ''quote') ;
    error('Should have thrown error for unterminated single quote') ;
  catch me
    if ~contains(me.message, 'Unterminated quote')
      error('Wrong error message for unterminated single quote') ;
    end
  end
end  % function