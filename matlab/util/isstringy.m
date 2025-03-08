function result = isstringy(x)
  % Tests whether x is a simple string, in the broad sense.
  % E.g. isstringy('foo') => true  % row char vector
  % E.g. isstringy("foo") => true  % string scalar
  % E.g. isstringy('') => true  % empty char array
  % E.g. isstringy("") => true  % string scalar, happens to be the empty string
  % E.g. isstringy(transpose('foo')) => false  % char vector, but not a row nor
  %                                            % empty
  % E.g. isstringy(string([])) => false  % string array, but not scalar
  % E.g  isstringy(42) => false  % double scalar, not string
  % E.g  isstringy([]) => false  % empty double matrix, not string
  % E.g  isstringy(["foo" "bar"]) => false  % string array, not scalar
  % E.g  isstringy(['foo';'bar']) => false  % string array, not scalar
  % E.g  isstringy(['foo' 'bar']) => true  % row char array
  result =  ( isstring(x) && isscalar(x) ) || ( ischar(x) && (isempty(x) || isrow(x)) ) ;
end
