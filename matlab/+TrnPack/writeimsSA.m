function writeimsSA(sloc,packdir,ims)
  % Write ims for Single animal

  sdir = TrnPack.SUBDIRIM();
  if exist(fullfile(packdir,sdir),'dir')==0
    mkdir(packdir,sdir);
  end

  n=numel(sloc);
  fprintf(1,'Writing %d training images...\n',n);

  parfor i=1:n
    imgnames = sloc(i).img;
    for v=1:size(ims,2)
      imgfile = fullfile(packdir,imgnames{v});
      im = ims{i,v};
      if ~exist(imgfile,'file'),
        imwrite(im,imgfile);
      end
    end
  end
end % function
