function ufmf2avi(infile,outfile,frms)

[readframe,nframes,movie_fid,headerinfo]= ...
  get_readframe_fcn(infile);

if nargin>2
  nframes = frms;
end

writerObj = VideoWriter(outfile);

writerObj.FrameRate = 30;
writerObj.Quality = 60;

open(writerObj);

for ndx = 1:nframes
  ii = readframe(ndx);
  if mod(ndx,1000) == 0,
      fprintf('Frame %d / %d, video %s\n',ndx,nframes,infile);
  end
   writeVideo(writerObj,ii);
end

close(writerObj);

if movie_fid>0
  fclose(movie_fid);
  
end
