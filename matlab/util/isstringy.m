function result = isstringy(x)
  % Tests whether x is a simple string, in the broad sense.
  % E.g. isstringy('foo') => true
  % E.g. isstringy("foo") => true
  % E.g. isstringy('') => true
  % E.g. isstringy("") => true
  % E.g. isstringy(transpose('foo')) => false
  % E.g. isstringy(string([])) => false
  % E.g  isstringy(42) => false
  % E.g  isstringy([]) => false
  % E.g  isstringy(["foo" bar"]) => false
  % E.g  isstringy(['foo';'bar']) => false
  % E.g  isstringy(['foo' 'bar']) => true
  result =  ( isstring(x) && isscalar(x) ) || ( ischar(x) && (isempty(x) || isrow(x)) ) ;
end
