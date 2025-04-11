function result = relpath(path, base_path)
% E.g. if path is'/book/html/wa/foo/bar' and base_path is '/book/html',
% returns 'wa/foo/bar'.  If path does not start with base_path, path is
% returned.

pat = textBoundary('start') + pattern(base_path) ;
if contains(path, pat) ,
  result = extractAfter(path, pat) ;
else
  result = path ;
end
