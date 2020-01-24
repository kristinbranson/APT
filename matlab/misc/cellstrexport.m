function cellstrexport(c,fname)
% cellstrexport(c,fname)

assert(iscellstr(c));
if exist(fname,'file')>0
  warning('cellstrexport:overwrite','Overwriting file ''%s''.',fname);
end
fh = fopen(fname,'w');
if fh==-1
  error('cellstrexport:openfail','Failed to open file ''%s''.',fname);
end
cellfun(@(x)fprintf(fh,'%s\n',x),c);
fclose(fh);