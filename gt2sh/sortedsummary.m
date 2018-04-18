function tCats = sortedsummary(c)
% Sorted category summary
% c: categorical
%
% tCats: category summary table

cats = categories(c);
cnts = countcats(c);
[~,idx] = sort(cnts,'descend');
tCats = table(cats,cnts);
tCats = tCats(idx,:);