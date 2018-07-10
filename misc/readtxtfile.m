function lines = readtxtfile(fname)
fh = fopen(fname,'r');
lines = cell(0,1);
while 1
  tline = fgetl(fh);
  if ~ischar(tline)
    break
  end
  lines{end+1,1} = tline; %#ok<AGROW>
end
fclose(fh);

