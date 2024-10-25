function jobid = parseJobIDConda(res)
  % We look for the first line of res that is an integer. If we can't find such
  % a line, we throw a warning and return nan.

  %fprintf('res: %s', res) ;      
  line_from_line_index = break_string_into_lines(res) ;
  line_count = numel(line_from_line_index) ;
  for line_index = 1 : line_count ,
    line = line_from_line_index{line_index} ;         
    jobid = str2double(strtrim(line)) ;
    if isfinite(jobid) && round(jobid)==jobid ,
      return
    end
  end
  % If get here, we have failed to read a job id
  warning('Could not parse job id from:\n%s\n',res);
  jobid = nan ;
end
