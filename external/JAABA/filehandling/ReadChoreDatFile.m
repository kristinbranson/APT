function data = ReadChoreDatFile(filename)

nparams = 7;

data = [];

fid = fopen(filename,'r');
if fid < 1,
  error('Could not open file %s for reading',filename);
end

ids = [];
expname = '';
while true,
  l = fgetl(fid);
  if ~ischar(l),
    break;
  end
  if isempty(l),
    continue;
  end
  
  % first word is the experiment name
  [expnamecurr,count,~,nextindex] = sscanf(l,'%s',1);
  if count < 1,
    warning('Could not read experiment name from line >%s<',l);
    continue;
  end
  if ~isempty(expname) && ~strcmp(expname,expnamecurr),
    warning('Experiment name %s does not match previous experiment name %s',expnamecurr,expname);
  end
  
  % read in the rest of the data
  l = l(nextindex:end);
  [datacurr,count] = sscanf(l,'%f',nparams);
  if count < nparams,
    warning('Could not parse data from line >%s<',l);
    continue;
  end
  
  id = datacurr(1);
  timestamps = datacurr(2);
  value = datacurr(3);

  idi = find(id == ids,1);
  if isempty(idi),
    idi = numel(ids)+1;
    newdata = struct('id',id,'timestamps',timestamps,'value',value);
    if isempty(ids),
      data = newdata;
    else
      data(idi) = newdata;
    end
    ids(idi) = id;
  else
    data(idi).timestamps(end+1) = timestamps;
    data(idi).value(end+1) = value;
  end
  
end

% sort by ids
[ids,order] = sort(ids); %#ok<ASGLU>
data = data(order);

% sort by timestamps
for i = 1:numel(data),
  [data(i).timestamps,order] = sort(data(i).timestamps);
  data(i).value = data(i).value(order);
end

fclose(fid);