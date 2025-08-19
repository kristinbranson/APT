function setSegmentedLineOnAtOnlyBang(hLine,nfrm,x)
  yLoc = 1 ;
  % obj.setOnOffAt(x,true);
  setSegmentedLineOnOffAtBang(hLine,yLoc,x,true) ;
  xall = 1:nfrm ;
  xcomp = setdiff(xall,x);
  % obj.setOnOffAt(xcomp,false);
  setSegmentedLineOnOffAtBang(hLine,yLoc,xcomp,false) ;
end
