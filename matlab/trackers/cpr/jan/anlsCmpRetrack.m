RES0 = 'F:\DropBoxNEW\Dropbox\Tracking_KAJ\track.results\13@he@for_150902_02_001_07_v2@iTrn@lotsa1__13@he@for_150902_02_001_07_v2@iTstLbl__0225T1021\res.mat';
RES1 = 'F:\cpr\data\jan\13@he_pluslblfrms902_2_1_7@for_150902_02_001_07_v2_plus9TrnFrmsForFr1131@iTrn@lotsa1__13@he_pluslblfrms902_2_1_7@for_150902_02_001_07_v2@iTstLbl__0226T1702\res.mat';

TD0 = 'f:\cpr\data\jan\td@13@he@0217';
TD1 = 'f:\cpr\data\jan\td@13@he_pluslblfrms902_2_1_7@0226';
TDI0 = 'f:\cpr\data\jan\tdI@13@for_150902_02_001_07_v2@0224';
TDI1 = 'f:\cpr\data\jan\tdI@13@for_150902_02_001_07_v2_plus9TrnFrmsForFr1131@0226';
TDIFILEVAR = 'iTstLbl';

%%
res0 = load(RES0);
res1 = load(RES1);
tdi0 = load(TDI0);
tdi1 = load(TDI1);

%%
p0 = res0.pTstTRed(:,:,end);
p1 = res1.pTstTRed(:,:,end);
pGT = td.pGT(tdi0.(TDIFILEVAR),:);
assert(isequal(size(p0),size(p1),size(pGT)));
nTest = size(p0,1);
%%
d0 = Shape.distP(pGT,p0);
d1 = Shape.distP(pGT,p1);
d0 = d0(:,4:7);
d1 = d1(:,4:7);
dav0 = mean(d0,2);
dav1 = mean(d1,2);
ddel = dav1-dav0;

tfimp = ddel<0;
tfreg = ddel>0;
tdsam = ddel==0;
fprintf(1,'imp/same/reg: %d/%d/%d\n',nnz(tfimp),nnz(tdsam),nnz(tfreg));
%%
iTrlImp = tdi.(TDIFILEVAR)(tfimp);
iTrlReg = tdi.(TDIFILEVAR)(tfreg);
[ddelsort,idx] = sort(ddel);
iTrlSort = tdi.(TDIFILEVAR)(idx);
fImp = td.MD.frm(iTrlImp);
fReg = td.MD.frm(iTrlReg);
fSort = td.MD.frm(iTrlSort);
%%
figure('WindowStyle','docked');
plot(ddelsort)
hold on
plot([1 numel(ddelsort)],[0 0],'r');
grid on
tstr = sprintf('ddelsort: dav1-dav0. Smaller=>improved. imp/same/reg: %d/%d/%d',...
  nnz(tfimp),nnz(tdsam),nnz(tfreg));
title(tstr,'fontweight','bold');
%% djump
iTrl = tdi.(TDIFILEVAR);
frms = td.MD.frm(iTrl);
dfrm = unique(diff(frms));
assert(isscalar(dfrm));

[~,~,~,~,~,djump47av0,~,repMad47av0] = janResults(res0);
[~,~,~,~,~,djump47av1,~,repMad47av1] = janResults(res1);
djumpS0 = sort(djump47av0);
djumpS1 = sort(djump47av1);
deldjump = djump47av1-djump47av0;
deldjumpS = sort(deldjump);
figure('Windowstyle','docked');
subplot(3,1,1);
x = 1:nTest;
plot(x,djumpS0,x,djumpS1);
grid on;
legend('djumpS0','djumpS1');
tstr = sprintf('djumpS: delta-frm=%d',dfrm);
title('djumpS','fontweight','bold');
% subplot(3,1,2);
% plot(x,deldjumpS);
% grid on;
% legend('djump1 - 0');

subplot(3,1,2);
[cnt0,edge0] = histcounts(djump47av0);
cnt1 = histcounts(djump47av1,edge0);
edgemid = (edge0(1:end-1)+edge0(2:end))/2;
plot(edgemid,cnt0,edgemid,cnt1);
grid on;
legend('jump0','jump1');
fprintf('jump0->jump1 imp/same/reg: %d/%d/%d\n',nnz(deldjump<0),nnz(deldjump==0),nnz(deldjump>0));
tstr = sprintf('L2 between successive red-lbls, av over 4-7. dfrm: %d',dfrm);
title(tstr,'fontweight','bold');

subplot(3,1,3);
[cnt0,edge0] = histcounts(repMad47av0);
cnt1 = histcounts(repMad47av1,edge0);
edgemid = (edge0(1:end-1)+edge0(2:end))/2;
plot(edgemid,cnt0,edgemid,cnt1);
grid on;
legend('repMad47Av0','repMad47Av1');
tstr = 'Median L2 from rep to repCent; aved over 4-7';
title(tstr,'fontweight','bold');

%%
pFull0 = res0.pTstT(:,:,end);
% xyFull0 = reshape(



%%
subplot(1,2,1);

hist(djump47av0)

xyLpos = permute(lObj.labeledpos{1},[3 1 2]);
xyLpos(isinf(xyLpos)) = nan; % fully-occluded
assert(isequal(size(xyLpos),size(xyPTstTRedPadfull),[nf Dlbl/2 2]));
pLpos = reshape(xyLpos,[nf Dlbl]);
pTstTRedPadfull = reshape(xyPTstTRedPadfull,[nf Dlbl]);
d = Shape.distP(pLpos,pTstTRedPadfull);
assert(isequal(size(d),[nf Dlbl/2]));
dav = nanmean(d,2);
lObj.setSuspScore({dav});
