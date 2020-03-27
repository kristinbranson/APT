function makeBaselines
p = fileparts(mfilename('fullpath'));
lbl = fullfile(p,'pend_master.lbl');
lbl = load(lbl,'-mat');

PENDMV = 'pend_mv.lbl';
lblmv = fullfile(p,PENDMV);
lblmv = load(lblmv,'-mat');
lpos1 = SparseLabelArray.full(lbl.labeledpos{1});
lpos2 = SparseLabelArray.full(lbl.labeledpos{2});
lposmv = cat(1,lpos1,lpos2);
lposmv = SparseLabelArray.create(lposmv,'nan');
lblmv.labeledpos{1} = lposmv;
sPrmMdlOrig = lblmv.trackerData.sPrm.Model;
lblmv.trackerData.sPrm = lbl.trackerData.sPrm;
lblmv.trackerData.sPrm.Model = sPrmMdlOrig;
PRMS = {'prmTrainInit' 'prmReg' 'prmFtr'};
for ivw=1:2
  for pfld=PRMS,pfld=pfld{1}; %#ok<FXSET>
    lblmv.trackerData.trnResRC(ivw).(pfld) = lbl.trackerData.trnResRC.(pfld);
  end
end
save(PENDMV,'-mat','-struct','lblmv');
fprintf('Saved %s.\n',PENDMV);

FLDS = {'movieFilesAll' 'movieInfoAll' 'trxFilesAll' 'labeledpos' 'labeledpostag' ...
  'labeledposTS' 'labeledposMarked' 'labeledpos2' 'suspScore'};
for ivw=1:2
  lblnew = lbl;
  for f=FLDS,f=f{1}; %#ok<FXSET>
    lblnew.(f) = lblnew.(f)(ivw,:);
  end
  lblnew.trackerData.sPrm = lbl.trackerData.sPrm;
  for pfld=PRMS,pfld=pfld{1}; %#ok<FXSET>
    lblnew.trackerData.trnResRC.(pfld) = lbl.trackerData.trnResRC.(pfld);
  end
  
  lblnew.currMovie = 1;
  
  lblnew.projname = sprintf('pend_%d',ivw);  
  fname = fullfile(p,[lblnew.projname '.lbl']);
  save(fname,'-mat','-struct','lblnew');
  
  fprintf('Saved %s.\n',fname);
end
  