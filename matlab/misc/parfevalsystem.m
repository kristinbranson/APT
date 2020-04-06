function [job,st,res] = parfevalsystem(syscmd,varargin)

job = parfeval(@system,2,syscmd,varargin{:});
res = '';
st = 0;
if strcmp(job.State,'finished'),
  [st,res] = fetchOutputs(job);
end