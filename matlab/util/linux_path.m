function result = linux_path(input_path, letter_drive_parent)
% Convert a path, which might be a Windows-style path (possibly with a leading drive letter), to a linux-style
% path.  Does not handle Windows UNC paths (like '\\server\folder\file') in
% input.

if length(input_path)>=2 && isstrprop(input_path(1),'alpha') && isequal(input_path(2),':') ,
  drive_letter = input_path(1) ;
  protoresult_1 = horzcat(letter_drive_parent, '/', lower(drive_letter), '/', input_path(3:end)) ;
else
  protoresult_1 = input_path ;
end

protoresult_2 = regexprep(protoresult_1,'\','/');

result = remove_repeated_slashes(protoresult_2) ;
 
end
