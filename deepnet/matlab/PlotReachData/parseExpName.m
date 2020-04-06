function res = parseExpName(expname)

res = regexp(expname,'^(?<mouse>M\d+)_(?<datestr>\d{8})_(?<trialstr>v\d+)$','names','once');
if isempty(res),
  return;
end
res.datenum = datenum(res.datestr,'yyyymmdd');
res.trialnum = str2double(res.trialstr(2:end));