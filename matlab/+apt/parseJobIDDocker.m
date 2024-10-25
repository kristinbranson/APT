function jobID = parseJobIDDocker(res)
  res = regexp(res,'\n','split');
  res = regexp(res,'^[0-9a-f]+$','once','match');
  l = cellfun(@numel,res);
  try
    res = res{find(l==64,1)};
    assert(~isempty(res));
    jobID = strtrim(res);
  catch ME,
    warning('Could not parse job id from:\n%s\n',res);
    disp(getReport(ME));
    jobID = '';
  end
end
