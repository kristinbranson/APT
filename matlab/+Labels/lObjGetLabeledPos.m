function [lpos,lposTS,lpostag] = lObjGetLabeledPos(lObj,labelsfld,gt)
  [nfrms,ntgts] = lObj.getNFrmNTrx(gt);
  nfrms = num2cell(nfrms);
  ntgts = num2cell(ntgts);
  fcn = @(zs,znfrm,zntgt)Labels.toarray(zs,'nfrm',znfrm,'ntgt',zntgt);
  [lpos,lposTS,lpostag] = cellfun(fcn,lObj.(labelsfld),nfrms(:),ntgts(:),'uni',0);
end  % function
