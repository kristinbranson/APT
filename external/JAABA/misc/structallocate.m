function s = structallocate(fns,sz)

s = cell2struct(cell([numel(fns),sz]),fns,1);