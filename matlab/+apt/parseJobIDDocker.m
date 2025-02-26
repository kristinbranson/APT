function jobid = parseJobIDDocker(res)
  % Returned jobID is an old-style string
  res = regexp(res,'\n','split');
  res = regexp(res,'^[0-9a-f]+$','once','match');
  l = cellfun(@numel,res);
  try
    res = res{find(l==64,1)};
    assert(~isempty(res));
    jobid = strtrim(res);
  catch ME,
    warning('Could not parse job id from:\n%s\n',res);
    disp(getReport(ME));
    jobid = '';
  end
end
