imi = 1;
nfids = D/2;
colors = jet(nfids);
clf;
maxint = prctile(Is{imi}(Is{imi}>=1000),95);
imagesc(Is{imi},[0,maxint]);
axis image;
hold on;
for fid = 1:nfids,
  plot(squeeze(pAll(imi,fid,max(1,t-5):t+1)),squeeze(pAll(imi,nfids+fid,max(1,t-5):t+1)),'.--','Color',colors(fid,:));
  plot(squeeze(pAll(imi,fid,t+1)),squeeze(pAll(imi,nfids+fid,t+1)),'wo','MarkerFaceColor',colors(fid,:));
  plot(squeeze(pGt(imi,fid)),squeeze(pGt(imi,nfids+fid)),'x','Color',colors(fid,:),'LineWidth',2);
end
drawnow;