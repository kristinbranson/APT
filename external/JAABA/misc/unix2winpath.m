function filepaths = unix2winpath(filepaths,unixrootdir,winrootdir)

if unixrootdir(end) ~= '/'
  unixrootdir(end+1) = '/';
end
if winrootdir(end) ~= '\',
  winrootdir(end+1) = '\';
end

filepaths = regexprep(filepaths,['^',unixrootdir],winrootdir);
filepaths = strrep(filepaths,'/','\');