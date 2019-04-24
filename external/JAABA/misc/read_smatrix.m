function S = read_smatrix(filename)

  fid = fopen(filename,'r');

  h1 = waitbar(0,'loading sparse matrix...');
  n = fread(fid,1,'int');
  nnz = fread(fid,1,'int');
  x = zeros(nnz,1);
  y = zeros(nnz,1);
  z = zeros(nnz,1);
  ct = 1;
  for i = 1:n
    waitbar(i/n,h1);
    nz(i) = fread(fid,1,'int');
    vals = fread(fid,nz(i),'double');
    cols = fread(fid,nz(i),'int');
    x(ct:ct+nz(i)-1) = cols+1;
    y(ct:ct+nz(i)-1) = i*ones(nz(i),1);
    z(ct:ct+nz(i)-1) = vals;
    ct = ct + nz(i);
  end;
  fclose(fid);
  close(h1);
  S = sparse(x,y,z);

%fid = fopen(filename,'r');
%n = fread(fid,1,'int');
%nnz = fread(fid,1,'int');
%A = zeros(n,n);
%h1 = waitbar(0,'loading sparse matrix...');
%for i = 1:n
%  waitbar(i/n,h1);
%  nz(i) = fread(fid,1,'int');
%  vals = fread(fid,nz(i),'double');
%  cols = fread(fid,nz(i),'int');
%  A(i,cols+1) = vals';
%end;
%fclose(fid);
%close(h1);
%S = sparse(A);


