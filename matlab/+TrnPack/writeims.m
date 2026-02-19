function writeims(sloc,packdir)
  % Currently single-view only

  sdir = TrnPack.SUBDIRIM();
  if exist(fullfile(packdir,sdir),'dir')==0
    mkdir(packdir,sdir);
  end

  imovall = [sloc.imov]';
  imovun = unique(imovall);

  fprintf(1,'Writing training images...\n');

  bufsize = 128;
  for iimov = 1:numel(imovun),
    imov=imovun(iimov);
    idx = find(imovall==imov); % indices into sloc for this mov
    % idx cannot be empty
    mov = sloc(idx(1)).mov;
    %mr.open(mov);
    fprintf(1,'Movie %d: %s (%d/%d)\n',imov,mov,iimov,numel(imovun));
    [rfcn,~,fid] = get_readframe_fcn(mov);
    frms = [sloc(idx).frm];
    filenames = arrayfun(@(i) fullfile(packdir,sdir,[sloc(i).idmovfrm '.png']),idx,'Uni',0);
    doskip = cellfun(@(x) exist(x,'file'),filenames) > 0;
    if ~all(doskip),
      curfilenames = filenames(~doskip);
      res = parforOverVideo(rfcn,frms(~doskip),@(im,frm,i) imwriteCheck(im,curfilenames{i}),'bufsize',bufsize,'verbose',true);
      assert(all(cell2mat(res)));
    end
    fprintf(1,'Wrote %d new images, %d existed previously\n',nnz(~doskip),nnz(doskip));

    if ~isempty(fid) && fid > 1,
      fclose(fid);
    end
  end

end % function
