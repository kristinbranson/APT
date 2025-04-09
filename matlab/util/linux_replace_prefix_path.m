function result = linux_replace_prefix_path(path_or_paths, old_base, new_base)
% E.g. if path is '/book/html/wa/foo/bar' and old_base is '/book/html', and
% new_base is '/video/json',  returns '/video/json/wa/foo/bar'.  If path does
% not start with old_base, path is returned.  Also works with cellstrs of
% paths.  This version works properly with linux paths, regardless of what
% platform we're running on.

if isstringy(path_or_paths) ,
  result = linux_replace_prefix_path_helper(path_or_paths, old_base, new_base) ;
else
  % Assumed to be a cell array of paths
  result = ...
    cellfun(@(path)(linux_replace_prefix_path(path, old_base, new_base)), ...
            path_or_paths, ...
            'UniformOutput', false) ;
end

end  % function


function result = linux_replace_prefix_path_helper(path, old_base, new_base)
% E.g. if path is '/book/html/wa/foo/bar' and old_base is '/book/html', and
% new_base is '/video/json',  returns '/video/json/wa/foo/bar'.  If path does
% not start with old_base, path is returned.

pat = textBoundary('start') + pattern(old_base) ;
if contains(path, pat) ,
  suffix = extractAfter(path, pat) ;
  result = linux_fullfile(new_base, suffix) ;
else
  result = path ;
end

end  % function
