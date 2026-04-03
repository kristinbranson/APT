function setSegmentedLineOnAtOnlyBang(hLine,nfrm,x)
  % obj.setOnOffAt(x,true);
  setSegmentedLineOnOffAtBang(hLine,x,true) ;
  xall = 1:nfrm ;
  xcomplement = setdiff(xall,x);
  % obj.setOnOffAt(xcomp,false);
  setSegmentedLineOnOffAtBang(hLine,xcomplement,false) ;
end
