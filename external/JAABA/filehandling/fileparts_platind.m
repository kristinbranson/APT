function [path, name, ext] = fileparts_platind(file)
%
%    parses paths as in windows, regardless of the os
%    renamed by KB 20160803
%    experiments directories pathes in .jab files created on windows machines 
%    are not parsed correctly
%    on unix based system because '\' is not a proper delimiter, so this
%    updated method parses dir paths on both '\' and '/' regardless of the
%    base operating system.
%    Nakul and Alice 20131107
%
%
%     ** ORIGINAL HELP MESSAGE **
%FILEPARTS Filename parts. 
%   [PATH,NAME,EXT] = FILEPARTS(FILE) returns the path, file name, and
%   file name extension for the specified FILE. The FILE input is a string
%   containing the name of a file or folder, and can include a path and
%   file name extension. The function interprets all characters following
%   the right-most path delimiter as a file name plus extension.
%
%   If the FILE input consists of a folder name only, be sure that the
%   right-most character is a path delimiter (/ or \). Othewise, FILEPARTS
%   parses the trailing portion of FILE as the name of a file and returns
%   it in NAME instead of in PATHSTR.
%
%   FILEPARTS only parses file names. It does not verify that the file or
%   folder exists. You can reconstruct the file from the parts using
%      fullfile(path,[name ext])
%
%   FILEPARTS is platform dependent. 
%
%   On Microsoft Windows systems, you can use either forward (/) or back
%   (\) slashes as path delimiters, even within the same string. On Unix
%   and Macintosh systems, use only / as a delimiter.
%
%   See also FULLFILE, PATHSEP, FILESEP.

%   Copyright 1984-2011 The MathWorks, Inc.
%   $Revision: 1.18.4.15.2.1 $ $Date: 2012/01/05 22:06:22 $

% Nothing but a row vector should be operated on.

path = '';
name = '';
ext = '';

if ~ischar(file) 
    error(message('MATLAB:fileparts:MustBeChar'));
elseif isempty(file) % isrow('') returns false, do this check first
    return;
elseif ~isrow(file)
    error(message('MATLAB:fileparts:MustBeChar'));
end


if isempty(file)
    return;
end

builtinStr = 'built-in';
if strncmp(file, builtinStr, size(builtinStr,2))
    name = builtinStr;
    return;
end

% if ispc
    ind = find(file == '/'|file == '\', 1, 'last');
    if isempty(ind)
        ind = find(file == ':', 1, 'last');
        if ~isempty(ind)       
            path = file(1:ind);
        end
    else
        if ind == 2 && (file(1) == '\' || file(1) == '/')
            %special case for UNC server
            path =  file;
            ind = length(file);
        else 
            path = file(1:ind-1);
        end
    end
    if isempty(ind)       
        name = file;
    else
        if ~isempty(path) && path(end)==':' && ...
                (length(path)>2 || (length(file) >=3 && file(3) == '\'))
                %don't append to D: like which is volume path on windows
            path = [path '\'];
        elseif isempty(deblank(path))
            path = '\';
        end
        name = file(ind+1:end);
    end
% else    % UNIX
%     ind = find(file == '/', 1, 'last');
%     if isempty(ind)
%         name = file;
%     else
%         path = file(1:ind-1); 
% 
%         % Do not forget to add filesep when in the root filesystem
%         if isempty(deblank(path))
%             path = '/';
%         end
%         name = file(ind+1:end);
%     end
% end

if isempty(name)
    return;
end

% Look for EXTENSION part
ind = find(name == '.', 1, 'last');

if isempty(ind)
    return;
else
    ext = name(ind:end);
    name(ind:end) = [];
end
