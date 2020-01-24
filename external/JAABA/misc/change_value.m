%CHANGE_VALUE  Change the value assigned to a unique variable in a file
%
% Examples:
%   fail = change_value(value)
%   fail = change_value(value, variableName)
%   fail = change_value(value, variableName, filePath)
%
% Function to change the value assigned to a variable in a text file. The
% assignment must exist already, and must be on a line on its own. For
% example:
%    variableName = 'string';
%    variableName = -0.756e-8;
% Note that there must be one or more spaces either side of the = sign.
% Only the first such assignment is changed.
%
% IN:
%   value - The value to be assigned to the variable.
%   variableName - String containing the name of the variable whose value
%                  is to be set. Default: name of variable given as value.
%   filePath - Full path of the file to change. Default: path of calling
%              file.
%
% OUT:
%   fail - true if change failed, false otherwise.

function fail = change_value(value, variableName, filePath)
% Check for missing inputs
if nargin < 3
    % Get the filename of the calling function
    filePath = dbstack;
    filePath = which(filePath(2).file);
    if nargin < 2
        % Get the variable name
        variableName = inputname(1);
    end
end
fail = true;
% Read in the file
fh = fopen(filePath, 'rt');
if fh < 0
    return
end
fstrm = fread(fh, '*char')';
fclose(fh);
% Find the path
first_sec = regexp(fstrm, ['[\n\r]+ *' variableName ' += +'], 'end', 'once');
second_sec = first_sec + regexp(fstrm(first_sec+1:end), ';? *[\n\r]+', 'once');
if isempty(first_sec) || isempty(second_sec)
    return
end
% Create the changed line
if ischar(value)
    str = '''%s''';
else
    str = '%1.50g';
end
str = sprintf(str, value);
% Save the file with the value changed
fh = fopen(filePath, 'wt');
if fh < 0
    return
end
fprintf(fh, '%s%s%s', fstrm(1:first_sec), str, fstrm(second_sec:end));
fclose(fh);
fail = false;
return