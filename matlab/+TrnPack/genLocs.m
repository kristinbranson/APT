function slocg = genLocs(sagg,movInfoAll)
  assert(numel(sagg)==numel(movInfoAll));
  nmov = numel(sagg);
  slocg = [];
  for imov=1:nmov
    s = sagg(imov);
    movifo = movInfoAll{imov};
    imsz = [movifo.info.nr movifo.info.nc];
    fprintf(1,'mov %d (sz=%s): %s\n',imov,mat2str(imsz),s.mov);

    slocgI = TrnPack.genLocsGroupedI(s,imov);
    slocg = [slocg; slocgI]; %#ok<AGROW>
  end
end % function
