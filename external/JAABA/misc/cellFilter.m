function out=cellFilter(filterFunction,in)

keep=cellfun(filterFunction,in);
out=in(keep);

end
