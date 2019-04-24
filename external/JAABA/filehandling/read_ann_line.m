function [s,param,value] = read_ann_line(fid)

s = fgetl(fid);
param = [];
value = [];

if s == -1
  return;
end
if strcmp(s,'end header'),
  return;
end

% if isspecial,
%   sz = str2num(value);
%   value = fread(fid,sz,'uint8');
% else
%   tmp = str2num(value);
%   if ~isempty(tmp),
%     value = tmp;
%   end;
% end;

i = findstr(s,':');
if isempty(i),
  return;
end;
param = s(1:i(1)-1);
value = s(i(1)+1:end);

specialparams = {'background median','background mean',...
                 'background mad','background std',...
                 'hfnorm','fracframesisback',...
                 'background center','background dev',...
                 'isarena'};
stringparams = {'bg_algorithm','version','expbgfgmodel_filename','movie_name','data format'};
pickledparams = {'roipolygons'};
maybestringparams = {'bg_type'};
isspecial = ismember(param,specialparams);
isstring = ismember(param,stringparams);
ispickled = ismember(param,pickledparams);
maybestring = ismember(param,maybestringparams);

if isspecial,
  sz = str2double(value);
  value = fread(fid,sz,'uint8');
elseif ispickled,
  sz = str2double(value);
  value = fread(fid,sz,'char');  
elseif isstring,
  % leave value as string
elseif maybestring,
  nvalue = str2double(value);
  if ~isnan(nvalue),
    value = nvalue;
  end
else
  tmp = str2double(value);
  if ~isempty(tmp),
    value = tmp;
  end;
end;
