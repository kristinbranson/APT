function filepaths = win2unixpath(filepaths,unixrootdir,winrootdir)

if unixrootdir(end) ~= '/'
  unixrootdir(end+1) = '/';
end

if nargin < 3,
  filepaths = regexprep(filepaths,'^[A-Z]:\\',unixrootdir);
else

  if winrootdir(end) ~= '\',
    winrootdir(end+1) = '\';
  end
  filepaths = regexprep(filepaths,['^',strrep(strrep(winrootdir,'\','\\'),'$','\$')],unixrootdir);
end
filepaths = strrep(filepaths,'\','/');
