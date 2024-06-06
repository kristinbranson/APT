function result = replace_prefix_path(path, base_path, new_base_path)
% E.g. if path is'/book/html/wa/foo/bar' and base_path is '/book/html',
% returns 'wa/foo/bar'.  If path does not start with base_path, path is
% returned.

pat = textBoundary('start') + pattern(base_path) ;
if contains(path, pat) ,
  suffix = extractAfter(path, pat) ;
  result = fullfile(new_base_path, suffix) ;
else
  result = path ;
end
