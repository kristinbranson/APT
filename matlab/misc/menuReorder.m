function menuReorder(hParent,tags)
% reorder hParent.Children to match tags
%
% hParent: scalar parent menu handle
% tags: cellstr, some permutation of {hParent.Children.Tag}'

warning('Obsolete. You do not seem to be able to change the order of submenus in Matlab any more.');

tag0 = {hParent.Children.Tag}';
[ism,loc] = ismember(tags,tag0);
if ~all(ism),
  error('Missing child tag: %s\n',tags{~ism});
end
rest = find(~ismember(tag0,tags));
loc = [loc(:);rest(:)];

hParent.Children = hParent.Children(loc(end:-1:1));
for i = 1:numel(hParent.Children),
  hParent.Children(i).Position = i;
end
