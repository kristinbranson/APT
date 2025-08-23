function result = localFileExistsAtWslPath(wslPath)
% Tests whether a *local* file exists at the WSL path wslPath.

escapedWslPath = escape_string_for_bash(wslPath);
command = sprintf('/usr/bin/test -e %s ; echo $?', escapedWslPath);
[st, res] = apt.syscmd(command, 'failbehavior', 'silent');
if st ~= 0 
  error('Unable to determine whether local file exists at WSL path %s', wslPath);
end
result = strcmp(strtrim(res),'0');

end
