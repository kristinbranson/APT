function jobid = parseJobIDBsub(res)
  PAT = 'Job <(?<jobid>[0-9]+)>';
  stoks = regexp(res,PAT,'names');
  if ~isempty(stoks)
    jobid = str2double(stoks.jobid);
  else
    jobid = nan;
    warning('Could not parse job id from:\n%s\n',res);
  end
end
