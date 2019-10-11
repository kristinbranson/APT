%% Overlay/Plot training data
NFRMPLT = 40;
NPTS = 7;
tMD = td.MD;
iTrlAll = find(strncmp('150730_02_',tMD.lblFile,10) & td.isFullyLabeled);
nTrlAll = numel(iTrlAll);
fprintf(1,'nTrl=%d\n',nTrlAll);

pGTTrl = td.pGT(iTrlAll,:);
pGTTrl = reshape(permute(pGTTrl,[2 1]),[7 2 nTrlAll]);

figure('windowstyle','docked');
imshow(td.I{iTrlAll(1)});
hIm = findall(gca,'type','image');
hold on;
hPt = gobjects(NFRMPLT,NPTS);
clrs = jet(NPTS);
for i=1:NFRMPLT
  for j=1:NPTS
   hPt(i,j) = plot(nan,nan,'o','Color',clrs(j,:),'MarkerFaceColor',clrs(j,:));    
  end
end

iCurr = 1;
MRKRSIZES = [12 12 12 12 12 8 8 8 8 8 4 4 4 4 4 4 4 4 4 4 repmat(3,1,NFRMPLT-20)]; 
for iiT = 500:nTrlAll
  iTrl = iTrlAll(iiT);

  hIm.CData = td.I{iTrl};
  
  xy = pGTTrl(:,:,iiT);
  for j = 1:NPTS
    hPt(iCurr,j).XData = xy(j,1);
    hPt(iCurr,j).YData = xy(j,2);
  end
  for i=0:NFRMPLT-1    
    ii = mod(iCurr-i-1,NFRMPLT)+1;
    for j = 1:NPTS
      hPt(ii,j).MarkerSize = MRKRSIZES(i+1);    
    end
  end
  
  iCurr = mod(iCurr,NFRMPLT)+1;
  
  tstr = sprintf('%s: frm%04d\n',tMD.lblFile{iTrl},tMD.frm(iTrl));
  title(tstr,'fontweight','bold','interpreter','none','fontsize',24);
  pause(0.05);
end


%% PCA: dims
p = reshape(pGTTrl,14,[])';
[coef,score,latent,tsq,expl,mu] = pca(p);
%% 