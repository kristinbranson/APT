function result = path_from_dir_struct(dis_from_index, varargin)
% For struct array of the type output by dir(), extract the full path for each
% element.  Returns a cellstring of the same shape as dis_from_index.

[do_unwrap_scalar] = ...
  myparse_nocheck(varargin, ...
                  'do_unwrap_scalar',false) ;

if isscalar(dis_from_index) && do_unwrap_scalar ,
  result = fullfile(dis_from_index.folder, dis_from_index.name) ;  
else  
  result = arrayfun(@(dis)(fullfile(dis.folder, dis.name)), dis_from_index, 'UniformOutput', false) ;
end
