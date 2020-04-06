function out=arrayFilter(filterFunction,in)

keep=arrayfun(filterFunction,in);
out=in(keep);

end
