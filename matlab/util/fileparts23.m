function result = fileparts23(path)
% Like filepath(), but returns outputs 2 and 3 of the fileparts() output: the filename (*with* extension)
% I figured calling this function filename() would get shadowed by a local
% variable name too often...

[~,b,c] = fileparts(path) ;
result = horzcat(b,c) ;

end
