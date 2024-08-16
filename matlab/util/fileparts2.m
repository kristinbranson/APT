function [parent_path, name] = fileparts2(path)
% Like filepath(), but returns only two outputs: the parent path and the
% filename (*with* extension)

[parent_path,b,c] = fileparts(path) ;
name = horzcat(b,c) ;

end
