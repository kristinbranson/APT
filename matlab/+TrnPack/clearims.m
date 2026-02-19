function clearims(packdir)
  sdir = TrnPack.SUBDIRIM();
  imdir = fullfile(packdir,sdir);
  if exist(imdir,'dir')==0
    return;
  end
  [succ,msg,mid] = rmdir(imdir,'s');
  if ~succ
    error(mid,'Failed to clear image cache: %s',msg);
  end
end % function
