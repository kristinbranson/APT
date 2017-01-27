%% Check for pt 19
lpos = lObj.labeledpos{1};
lpostag = lObj.labeledpostag{1};
if size(lpos,1)==57
  idxPt19 = [19 38 57];
  lpos(idxPt19,:,:) = [];
  lpostag(idxPt19,:) = [];
  fprintf(1,'Removing pt 19 from lpos, lpostag.\n');
end

%% Check for estimated-occluded in side views; set these to nan
% The point here is that in side-views in older data, RF used est-occ to mean
% "no information, just using reconstruction from other two views" because
% there is no "fully occluded" box avail in multiview labeling. So
% side-view-est-occ points where the pt is on the "wrong side" of the fly
% we set to NaN; the labels carry no addnl info.
%
% side est-occ labels on the "right side" of the fly may carry info so we
% keep those; similarly we keep est-occ labels on the bottom as those carry
% info.
%
% for later data RF clicks far away for pure-occ labels.

[npttot,nphyspt,nview,nfrm] = RF.lposDim(lpos);
lpostag4d = reshape(lpostag,nphyspt,nview,nfrm);
tfocc = ~cellfun(@isempty,lpostag4d);
nOcc = sum(tfocc,3) % [nphysptxnview]. occluded tag, any pt, any view
%% cont
SETWRONGSIDEOCCTONAN = false;
if SETWRONGSIDEOCCTONAN
  tfPtWrongSide = RF.ptWrongSide();
  tfPtWrongSide = repmat(tfPtWrongSide,[1 1 nfrm]);
  tfOccSideSetToNan = tfocc;
  tfOccSideSetToNan(~tfPtWrongSide) = false;
  nOccSideSetToNan = sum(tfOccSideSetToNan,3)
  fprintf(1,'Setting labels for this many est-occ-side-view pts to nan.\n');
  tfOccSideSetToNan = reshape(tfOccSideSetToNan,[npttot 1 nfrm]);
  tfOccSideSetToNan = repmat(tfOccSideSetToNan,[1 2 1]);
  lpos(tfOccSideSetToNan) = nan;
end

%% check/viz outliers
OUTLIERDESC = outlier_desc_oct2916;

hFig = RF.vizLabels(lpos,lObj.movieInfoAll,OUTLIERDESC,...
  'oneBigPlot',true);
%%
[projpath,projname] = fileparts(lObj.projectfile);
hgsave(hFig,[projname '_outliers.fig']);

%% Actually rm outliers
[lpos,nRm] = RF.rmOutlierOcc(lpos,lObj.movieInfoAll,OUTLIERDESC);
fprintf(1,'nRm:\n');
disp(nRm)

%%
tFP = RF.FPtable(lpos,lpostag);

%% FROM THIS POINT ON, all operations are purely on tFP


%% Keep only rows with npts2VwLbl==nphyspts
[~,nphyspt] = RF.lposDim(lpos);
tfKeep = tFP.npts2VwLbl==nphyspt;
fprintf(1,'Removing %d/%d rows from tFP.\n',nnz(~tfKeep),numel(tfKeep));
tFP = tFP(tfKeep,:);

%% Check distro of num views
x = cellfun(@(x)sum(x,1),tFP.tfVws2VwLbl,'uni',0);
x = cat(1,x{:});
size(x)
assert(all(x(:)>=2))

%% Check that bottom view always labeled
x = cellfun(@(x)x(3,:),tFP.tfVws2VwLbl,'uni',0);
x = cat(1,x{:});
nnz(x(:)==0)
[f,j] = find(x==0)
f = tFP.frm(f)

%%
crig2 = lObj.viewCalibrationData;

%% Reconstruct
tFPaug = RF.recon3D(tFP,crig2);

%% check nan pattern of errRE. errRE will be nan if a pt is not labeled in that view
errRE = cat(1,tFPaug.err{:});
tfVwsLbl = cat(1,tFPaug.tfVws2VwLbl{:});
isequal(~tfVwsLbl,isnan(errRE))

%% Find examples of worst RE err
[npttot,nphyspt,nview,nfrm] = RF.lposDim(lpos);
[maxerr,idx] = nanmax(errRE,[],1);
rows = ceil(idx/3);
terr = table((1:nphyspt)',rows(:),tFPaug.frm(rows(:)),maxerr(:),...
  'VariableNames',{'physpt' 'rows' 'frm' 'maxerr'})

%% Log all errs greater than threshold for removal an reporting
RE_ERR_THRESH = 10; % px
tf = errRE>RE_ERR_THRESH;
[idx,ipt] = find(tf);
row = ceil(idx/3);
view = mod(idx-1,3)+1;
errre = errRE(tf);
tREOutlrs = table(tFPaug.frm(row),ipt,view,errre,...
  'VariableNames',{'frm' 'pt' 'view' 'errRecon'});
tREOutlrs = sortrows(tREOutlrs,'frm');

%%
reOutliersXls = sprintf('%s_errRecon_%dpx.xlsx',projname,RE_ERR_THRESH);
writetable(tREOutlrs,reOutliersXls);

%% remove outliers
frmsOutlr = unique(tREOutlrs.frm);
tfOutlrRow = ismember(tFPaug.frm,frmsOutlr);
tFPtrn = tFPaug(~tfOutlrRow,:);
fprintf(1,'Removed %d/%d outlier rows from tFPaug.\n',nnz(tfOutlrRow),size(tFPaug,1));

%% save training data
trnDataFile = sprintf('%s_trnData_%s.mat',projname,datestr(now,'yyyymmddTHHMMSS'));
save(trnDataFile,'-mat','crig2','tFPaug','tFPtrn','RE_ERR_THRESH');
