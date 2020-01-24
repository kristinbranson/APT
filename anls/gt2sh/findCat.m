function [lblDate,lblCat] = findCat(flyID,lblDate,lblCat)

flyID = unique(flyID);
assert(isscalar(flyID));

lblDate = cellstr(lblDate);
lblDate = unique(lblDate);
ndate = numel(lblDate);
if ndate>1
  fprintf('%d, %d dates: %s. choosing lowest, %s.\n',...
    flyID,ndate,String.cellstr2CommaSepList(lblDate),lblDate{1});
end
lblDate = lblDate(1);

lblCat = cellstr(lblCat);
catC = categorical(lblCat);
catcats = categories(catC);
catcatcnts = countcats(catC);
ncat = numel(catcats);
if ncat>1  
  [~,i] = max(catcatcnts);
  tmp = cellstr(char(catcats));
  fprintf('%d, %d cats: %s. Choosing most frequent: %s\n',...
    flyID,ncat,String.cellstr2CommaSepList(tmp),char(catcats(i)));
  lblCat = {char(catcats(i))};
else
  lblCat = {char(catcats(1))};
end
