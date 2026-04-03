function result = escape_string_for_bash(str)
% Process the string str so that when the result is passed as part of a bash
% command line, it will be interpreted as a single token, and the string
% received by the accepting executable/script will be identical to str.

% Test for known-safe characters.  If all characters are known-safe, return
% as-is.
assert(ischar(str) && (isempty(str) || isrow(str))) ;
digits = char(48:57) ;
capitals = char(65:90) ;
lowercases = char(97:122) ;
others = '/_-.' ;
safe = horzcat( lowercases, capitals, digits, others ) ;
if all(ismember(str, safe)) ,
  % Input string contains no dangerous characters, so pass through as-is
  result = str ;
else
  % The general case---escape by wrapping in single quotes.
  sq = '''' ;  % a single quote
  sq_bs_sq_sq = '''\''''' ; % single quote, backslash, single quote, single quote
  str_escaped = strrep(str, sq, sq_bs_sq_sq) ;  % replace ' with '\''
  result = horzcat(sq, str_escaped, sq) ;  % Surround with single quotes to handle all special chars besides single quote
end
