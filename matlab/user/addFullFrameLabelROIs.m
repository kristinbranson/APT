function addFullFrameLabelROIs(lObj)

for iMov = 1:lObj.nmovies,

  movinfo = lObj.movieInfoAll{iMov};
  nr = movinfo.info.nr;
  nc = movinfo.info.nc;

  v = [1,1
    1,nr
    nc,nr
    nc,1];

  for frm = lObj.labels{iMov}.frm(:)',

    s = lObj.labelsRoi{iMov};
    lObj.labelsRoi{iMov} = LabelROI.setF(s,v,frm);

  end

end