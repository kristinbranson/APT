function jobid = parseJobIDConda(res)
  % Returned jobid is an old-style string.
  % We look for the first line of res that cleanly converts to an integer. If we can't find such
  % a line, we throw a warning and return an empty string.

  %fprintf('res: %s', res) ;      
  line_from_line_index = break_string_into_lines(res) ;
  line_count = numel(line_from_line_index) ;
  for line_index = 1 : line_count ,
    line = line_from_line_index{line_index} ;
    maybe_jobid = strtrim(line) ;
    maybe_jobid_as_double = str2double(maybe_jobid) ;
    if isfinite(maybe_jobid_as_double) && round(maybe_jobid_as_double)==maybe_jobid_as_double ,
      jobid = maybe_jobid ;
      return
    end
  end
  % If get here, we have failed to read a job id
  warning('Could not parse job id from:\n%s\n',res) ;
  jobid = '' ;
end
