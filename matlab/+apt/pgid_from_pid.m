function result = pgid_from_pid(pid)
% Get the process group id from the process id of a process.  Only works on
% Linux.  pid should be an old-style string, and result is an old-style
% string.  Uses system() and the ps command to get the info it needs.

command_line = sprintf('/usr/bin/ps -o pgid= -p %s', pid) ;
stdouterr = system_with_error_handling(command_line) ;
result = strtrim(stdouterr) ;
