function bboxes = makeBBoxes(I)
assert(iscell(I) && iscolumn(I));

sz = cellfun(@(x)[size(x,2) size(x,1)],I,'uni',0);
bboxes = cellfun(@(x)[[1 1] x],sz,'uni',0);
bboxes = cat(1,bboxes{:});