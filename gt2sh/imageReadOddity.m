%%
mov = '/groups/branson/home/leea30/tmpImageReadOddity/C001H001S0006_c.avi';
frm = 295;
%%
vr = VideoReader(mov);
for i=1:frm
  im = vr.readFrame(); 
end
%%
offsets = -30:30;
[~,idx] = sort(abs(offsets));
offsets = offsets(idx);
tfFound = false;
for off=offsets
  vr = VideoReader(mov);
  im2 = vr.read(frm+off);    
  if isequal(im2,im)
    tfFound = true;
    break;
  end
  fprintf('Tried offset=%d\n',off);
end
if tfFound
  fprintf('SUCCESS, found it at offset %d!\n',off);
else
  fprintf('FAIL\n');
end