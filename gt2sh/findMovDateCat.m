function [lblDate,lblCat,frms] = findMovDateCat(movID,lblDate,lblCat,frms)

movID = unique(movID);
assert(isscalar(movID));
movID = movID{1};

lblDate = cellstr(lblDate);
lblDate = unique(lblDate);
ndate = numel(lblDate);
if ndate>1
  fprintf('movID %s, %d dates: %s. choosing lowest, %s.\n',...
    movID,ndate,String.cellstr2CommaSepList(lblDate),lblDate{1});
end
lblDate = lblDate(1);

lblCat = cellstr(lblCat);
lblCat = unique(lblCat);
ncat = numel(lblCat);
if ncat>1
  fprintf('movID %s, %d cats. choosing lowest, %s.\n',movID,ncat,lblCat{1});
end
lblCat = lblCat(1);

frms = {frms(:)'};
