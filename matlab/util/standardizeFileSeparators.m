function result = standardizeFileSeparators(path)
% Standardize all the file separators in path with the file separator used by
% the current platform.

fs = filesep() ;
temp = strrep(path,'\',fs);
result = strrep(temp,'/',fs);

end
