function result = summarizePerViewFiles(fileNameFromViewIndex, fileContentsFromViewIndex)
% Put all the per-view files (typically logs or error files) together into a
% single cellstring of lines.  (With no trailing newlines.)  Both
% fileNameFromViewIndex and fileContentsFromViewIndex should be a cell array
% of strings, with the same number of elements.
result = cell(1,0) ;
for viewIndex = 1:numel(fileNameFromViewIndex)
  logFileName = fileNameFromViewIndex{viewIndex};
  result{end+1} = sprintf('### View index %d:',viewIndex); %#ok<AGROW>
  result{end+1} = sprintf('### %s',logFileName); %#ok<AGROW>
  result{end+1} = ''; %#ok<AGROW>
  fileContents = fileContentsFromViewIndex{viewIndex} ;
  lineFromLineIndex = strsplit(fileContents,'\n') ; 
  result = horzcat(result, lineFromLineIndex) ;  %#ok<AGROW>
end
