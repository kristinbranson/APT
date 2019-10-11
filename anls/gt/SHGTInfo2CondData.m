function [data_cond,label_cond,datatypes,labeltypes] = SHGTInfo2CondData(gtinfo,useisdup)

if nargin < 2,
  useisdup = true;
end
if useisdup,
  isnotdup = gtinfo.isduplicate==0;
else
  isnotdup = true(size(gtinfo.isduplicate));
end
nlabels = numel(gtinfo.istrain);
data_cond = zeros(nlabels,1);
data_cond(isnotdup & gtinfo.istrain==1) = 1;
data_cond(isnotdup & gtinfo.istrain==0) = 2;
datatypes = {'same fly',1
  'different fly',2
  'all',[1,2]};

label_cond = zeros(nlabels,1);
label_cond(isnotdup&~gtinfo.isactivation) = 1;
label_cond(isnotdup&gtinfo.isactivation&~gtinfo.isenriched) = 2;
label_cond(isnotdup&gtinfo.isactivation&gtinfo.isenriched) = 3;
labeltypes = {'no activation',1
  'activation, not enriched',2
  'enriched activation',3
  'activation',[2,3]
  'all',[1,2,3]};