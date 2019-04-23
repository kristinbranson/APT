function snew = structprefixfieldnames(s,pfix)
assert(isstruct(s));
assert(ischar(pfix));

fnames = fieldnames(s);
pfixfnames = cellfun(@(x)[pfix x],fnames,'uni',0);

c = struct2cell(s);
snew = cell2struct(c,pfixfnames,1);

assert(isequal(size(s),size(snew)));