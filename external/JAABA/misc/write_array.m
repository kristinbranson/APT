
function write_array(filename,A);

fid = fopen(filename,'w');

if (~iscell(A))
 B = A;
 clear A;
 A{1} = B;
end;

for k = 1:length(A)
  d = size(A{k});
  dim = length(d);

  fwrite(fid,dim,'uint')
  for i = dim:-1:1,
    fwrite(fid,d(i),'uint');
  end;
%  Z = reshape(A{k}',fliplr(d));
  Z = A{k};
  fwrite(fid,Z(:),'float');
end;

fclose(fid);


