function jobid = parseJobIDConda(response)
  % Attempt to process the (old-style) string in response to extract a process
  % ID (PID).  Then lookup the process group ID (PGID) to which that process belongs.
  % The PGID is used as the job id for conda jobs.  The PGID is returned as an
  % old-style string.  Throws an error if anything goes wrong.

  % Split response into lines, then look for the first line that cleanly
  % converts to an integer.  That integer is taken to be the process ID.
  line_from_line_index = break_string_into_lines(response) ;
  line_count = numel(line_from_line_index) ;
  for line_index = 1 : line_count ,
    line = line_from_line_index{line_index} ;
    maybe_pid = strtrim(line) ;
    maybe_pid_as_double = str2double(maybe_pid) ;
    if isfinite(maybe_pid_as_double) && round(maybe_pid_as_double)==maybe_pid_as_double ,
      pid = maybe_pid ;
      jobid = apt.pgid_from_pid(pid) ;
      return
    end
  end
  % If get here, we have failed to read a job id
  error('Could not parse job id from:\n%s', response) ;
end
