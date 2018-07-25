function [flyid,trial,movpath] = parseSHfullmovie(m,varargin)
noID = myparse(varargin,'noID',false);

[flyid,idx] = regexp(m,'[\\/]fly?(\d+)[\\/]','tokens','start');
if isempty(flyid)
  [flyid,idx] = regexp(m,'[\\/]fly?(\d+)[\\/_]','tokens','start');
end
if isempty(flyid)
  [flyid,idx] = regexp(m,'[\\/]fly_?(\d+)[\\/]','tokens','start');
end
if isempty(flyid)
  [flyid,idx] = regexp(m,'[\\/]fly ?(\d+)[\\/]','tokens','start');
end
if isempty(flyid)
  [flyid,idx] = regexp(m,'[\\/]Fly?(\d+)[\\/]','tokens','start');
end
if ~noID
  flyid = cat(1,flyid{:});
  flyid = unique(flyid);
  assert(isscalar(flyid));
  flyid = flyid{1};
  flyid = str2double(flyid);
  idx = max(idx);
else
  flyid = nan;
end

trl1 = regexp(m,'trial_?([0-9]+)','tokens');
if ~isempty(trl1)
  trl1 = cellfun(@(x)str2double(x{1}),trl1);
  assert(all(trl1==trl1(1)));
  trl1 = trl1(1);
  trial = trl1;
else
  trl2 = regexp(m,'C00[12]H001S00?([0-9]+)','tokens');
  trl2 = cellfun(@(x)str2double(x{1}),trl2); % always present
  assert(all(trl2==trl2(1)));
  trl2 = trl2(1);
  trial = trl2;
end
assert(m(idx)=='\' || m(idx)=='/');
movpath = m(idx+1:end);

