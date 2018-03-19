function menuReorder(hParent,tags)
% reorder hParent.Children to match tags
%
% hParent: scalar parent menu handle
% tags: cellstr, some permutation of {hParent.Children.Tag}'

tag0 = {hParent.Children.Tag}';
[~,loc] = ismember(tags,tag0);
hParent.Children = hParent.Children(loc(end:-1:1));