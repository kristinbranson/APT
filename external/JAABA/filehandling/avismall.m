function avismall(infilename,outfilename,nframes,scale)

for i = 1:nframes,
  fprintf('%d\n',i);
  M = aviread(infilename,i);
  data = imresize(M(1).cdata,scale);
  aviobj = addframe(aviobj,data);
end;

aviobj = close(aviobj);