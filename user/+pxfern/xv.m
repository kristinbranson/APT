function [cvpart,C,corrPct] = xv(prm,tGT,kfold)

tblfldscontainsassert(tGT,MFTable.FLDSID);

cvpart = cvpartition(tGT.mov,'KFold',kfold);
C = cell(kfold,1);
corrPct = cell(kfold,1);

hWB = waitbar(0,sprintf('Fold 0/%d',kfold));
for ifold=1:kfold
  waitbar(ifold/kfold,hWB,sprintf('Fold %d/%d',ifold,kfold));
  
  fimidxs = pxfern.genfeatures(prm);
  
  tfTrn = cvpart.training(ifold);
  tblTrn = tGT(tfTrn,:);
  [fvalsTrn,fvalscTrn] = pxfern.compfeatures(prm,tblTrn,fimidxs);
  hfernTrn = pxfern.fernhist(prm,fvalsTrn,fvalscTrn);
  pfernTrn = pxfern.fernprob(prm,hfernTrn);
  
  tfTst = cvpart.test(ifold);
  tblTst = tGT(tfTst,:);
  [fvalsTst,fvalscTst] = pxfern.compfeatures(prm,tblTst,fimidxs);
  
  fvalscTstPred = pxfern.fernpred(prm,pfernTrn,fvalsTst);
  c = confusionmat(fvalscTst,fvalscTstPred,'order',1:6);
  cZ = sum(c,2);
  cnorm = c./cZ;
  corrpcts = diag(cnorm)
  
  C{ifold} = c;
  corrPct{ifold} = corrpcts;
end

delete(hWB);