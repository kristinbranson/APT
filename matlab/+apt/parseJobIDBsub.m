function jobid = parseJobIDBsub(res)
  % Returned jobid is an old-style string
  PAT = 'Job <(?<jobid>[0-9]+)>';
  stoks = regexp(res,PAT,'names');
  if ~isempty(stoks)
    %jobid = str2double(stoks.jobid);
    jobid = strtrim(stoks.jobid);
  else
    jobid = '' ;
    warning('Could not parse job id from:\n%s\n',res);
  end
end
