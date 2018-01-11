function [cvpart,C,corrPct,tblRes] = xv(prm,tGT,kfold)

tblfldscontainsassert(tGT,MFTable.FLDSID);

cvpart = cvpartition(tGT.mov,'KFold',kfold);
C = cell(kfold,1);
corrPct = cell(kfold,1);
tblRes = cell(kfold,1);

hWB = waitbar(0,sprintf('Fold 0/%d',kfold));
for ifold=1:kfold
  waitbar(ifold/kfold,hWB,sprintf('Fold %d/%d',ifold,kfold));
  
  fimidxs = pxfern.genfeatures(prm);
  
  tfTrn = cvpart.training(ifold);
  tblTrn = tGT(tfTrn,:);
  [fvalsTrn,fmdTrn] = pxfern.compfeatures(prm,tblTrn,fimidxs);
  hfernTrn = pxfern.fernhist(prm,fvalsTrn,fmdTrn.leg);
  pfernTrn = pxfern.fernprob(prm,hfernTrn);
  
  tfTst = cvpart.test(ifold);
  tblTst = tGT(tfTst,:);
  [fvalsTst,fmdTst] = pxfern.compfeatures(prm,tblTst,fimidxs);
  
  fvalscTstPred = pxfern.fernpred(prm,pfernTrn,fvalsTst);
  c = confusionmat(fmdTst.leg,fvalscTstPred,'order',1:6);
  cZ = sum(c,2);
  cnorm = c./cZ;  
  corrpcts = diag(cnorm);  
  
  C{ifold} = c;
  corrPct{ifold} = corrpcts;
  tblRes{ifold} = [...
    fmdTst(:,{'leg' 'i' 'j'}) ...
    tblTst(fmdTst.iGT,{'mov' 'frm' 'iTgt' 'p' 'pTrx'...
                      'trxa' 'trxb' 'trxth' 'pcanon' 'idxLposLegCanon'}) ...
    table(fvalscTstPred,repmat(ifold,height(fmdTst),1),'VariableNames',...
          {'legpred','ifold'}) ...
  ];  
end

tblRes = cat(1,tblRes{:});

delete(hWB);