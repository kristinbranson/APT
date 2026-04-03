function result = isOldSchoolString(x)
  % Tests whether x is a "traditional" Matlab string: An empty char array or a row char array.
  result = ( ischar(x) && (isempty(x) || isrow(x)) ) ;
end
