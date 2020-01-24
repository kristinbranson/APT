function convertResults2local(infile,outdir)

remotestr = '/tier2/hantman/Jay/videos/M119_CNO_G6/';
localstr = '/home/mayank/Dropbox/AdamVideos/';
Q =load(infile);

J = struct;
count = 1;
for ndx = 1:numel(Q.moviefiles_all)
  localfile = regexprep(Q.moviefiles_all{ndx},remotestr,localstr);
  if ~exist(localfile,'file'),
    continue;
  end
  J.moviefiles_all{count} = localfile;
  J.p_all{count} = Q.p_all{ndx};
  count = count+1;
end

[~,fname,~] = fileparts(infile);
save(fullfile(outdir,fname),'-struct','J');