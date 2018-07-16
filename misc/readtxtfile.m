function lines = readtxtfile(fname,varargin)
discardemptylines = myparse(varargin,...
  'discardemptylines',false);

fh = fopen(fname,'r');
lines = cell(0,1);
while 1
  tline = fgetl(fh);
  if ~ischar(tline)
    break
  end
  if isempty(tline) && discardemptylines
    % none
  else
    lines{end+1,1} = tline; %#ok<AGROW>
  end
end
fclose(fh);

