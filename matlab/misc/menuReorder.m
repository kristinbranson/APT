function menuReorder(hParent,tags)
% reorder hParent.Children to match tags
%
% hParent: scalar parent menu handle
% tags: cellstr, some permutation of {hParent.Children.Tag}'

tag0 = {hParent.Children.Tag}';
[ism,loc] = ismember(tags,tag0);
if ~all(ism),
  error('Missing child tag: %s\n',tags{~ism});
end
rest = setdiff(1:numel(tag0),loc);
loc = [loc(:);rest(:)];

hParent.Children = hParent.Children(loc(end:-1:1));