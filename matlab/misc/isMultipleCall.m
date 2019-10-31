function flag=isMultipleCall()
flag = false;
% Get the stack
s = dbstack();
if numel(s) <= 2
  % Stack too short for a multiple call
  return
end

% How many calls to the calling function are in the stack?
names = {s(:).name};
TF = strcmp(s(2).name,names);
count = sum(TF);
if count>1
  % More than 1
  flag = true;
end
