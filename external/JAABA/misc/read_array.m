
function A = read_array(filename);

fid = fopen(filename,'r');
dim = fread(fid,1,'uint');
for i = 1:dim
  d(i) = fread(fid,1,'uint');
end;
[B,count] = fread(fid,prod(d),'float');
B = reshape(B,fliplr(d));

dim = fread(fid,1,'uint');
if (~feof(fid)) %collection of arrays.
  A{1} = B;
  ct = 2;
  while (~feof(fid))
    for i = 1:dim
      d(i) = fread(fid,1,'uint');
    end;
    [B,count] = fread(fid,prod(d),'float');
    A{ct} = reshape(B,fliplr(d));
    ct = ct + 1;
    dim = fread(fid,1,'uint');
  end;
else %else a single array 
  A = B;
end;

fclose(fid);


