function fixedtrx2ann(trx,oldannname,newannname)

% open the annotation file for reading and writing
fin = fopen(oldannname,'rb');
if exist(newannname,'file'),
  b = questdlg(sprintf('File %s exists. Overwrite?',newannname));
  if ~strcmpi(b,'yes'),
    return;
  end
end
fout = fopen(newannname,'wb');

% find the end of the header
while true,
  [s,param,value] = read_ann_line(fin);
  if ~ischar(s) && s == -1,
    error('End of file encountered before end of header found');
  end
  if ~ischar(value) && length(value) > 1,
    fprintf(fout,'%s\n',s);
    fwrite(fout,value,'uint8');
  else
    fprintf(fout,'%s\n',s);
  end
  if strcmp(s,'end header'),
    break;
  end
end

if ftell(fin) ~= ftell(fout),
  error('input and output file do not match');
end

fclose(fin);

nframes = max([trx.endframe]);
nflies = length(trx);

for i = 1:nframes,
  if mod(i,100) == 0,
    fprintf('Writing frame %d\n',i);
  end
  for j = 1:nflies,
    if trx(j).firstframe > i || trx(j).endframe < i,
      continue;
    end
    k = i - trx(j).firstframe + 1;
    fprintf(fout,'%f\t%f\t%f\t%f\t%f\t%d\t',...
      trx(j).x(k)-1,trx(j).y(k)-1,trx(j).b(k),trx(j).a(k),trx(j).theta(k),j-1);
  end
  fprintf(fout,'\n');
end

fclose(fout);

if 0,
  fclose('all');
  fin = fopen(oldannname,'rb');
  fout = fopen(newannname,'rb');
  while true,
[sin,paramin,valuein] = read_ann_line(fin);
[sout,paramout,valueout] = read_ann_line(fout);
if ~strcmp(sin,sout),
fprintf('sin does not match sout\n');
keyboard;
else
fprintf([sin,'\n']);
end
if strcmp(sin,'end header'),
   if ~strcmp(sout,'end header'),
     fprintf('end of headerin but not headerout\n');
     keyboard;
   end
   break;
elseif strcmp(sout,'end header'),
  fprintf('end of headerout but not headerin\n');
  keyboard;
  break;
end
if isempty(paramin) && isempty(paramout),
  fprintf('both empty\n');
  continue;
end
if ~strcmp(paramin,paramout),
fprintf('paramin does not match paramout\n');
keyboard;
else
fprintf([paramin,'\n']);
end
if ischar(valuein),
if ~ischar(valueout),
fprintf('ischar(valuein) but not valueout\n');
keyboard;
end
if ~strcmp(valuein,valueout),
fprintf('valuein does not match valueout\n');
end
elseif length(valuein) == 1,
if ischar(valueout),
fprintf('ischar(valueout) but not valuein\n');
keyboard;
end
if length(valueout) ~= 1,
fprintf('length(valuein) == 1, but not valueout\n');
keyboard;
end
if valuein ~= valueout,
fprintf('valuein and valueout do not match\n');
keyboard;
else
fprintf([num2str(value),'\n']);
end
else
if ischar(valueout),
fprintf('ischar(valueout) but not valuein\n');
keyboard;
end

if length(valueout) == 1,
fprintf('length(valueout) == 1 but not valuein\n');
keyboard;
end
if length(valuein) ~= length(valueout),
fprintf('lengths do not match\n');
keyboard;
end
if ~all(valuein == valueout),
fprintf('not all values match\n');
keyboard;
else
fprintf('value vectors of length %d match\n',length(valuein));
end
end
end
end