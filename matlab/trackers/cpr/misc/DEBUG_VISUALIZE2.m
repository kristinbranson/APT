iplot = 1;
nfids = D/2;
colors = jet(nfids);
clf;
nplot = 5;
hax = createsubplots(5,1,.01);
isplot = randsample(size(pAll,1),nplot);
dtplot = 20;

for ii = 1:nplot,
  iplot = isplot(ii);
  imi = imgIds(iplot);
  %axes(hax(ii));
  imagesc(Is{imi},'Parent',hax(ii));
  axis(hax(ii),'image');
  axis(hax(ii),'off');
  hold(hax(ii),'on');
  for fid = 1:nfids,
    plot(hax(ii),squeeze(pAll(iplot,fid,max(1,t-dtplot):t+1)),squeeze(pAll(iplot,nfids+fid,max(1,t-dtplot):t+1)),'.--','Color',colors(fid,:));
    plot(hax(ii),squeeze(pAll(iplot,fid,t+1)),squeeze(pAll(iplot,nfids+fid,t+1)),'wo','MarkerFaceColor',colors(fid,:));
    plot(hax(ii),squeeze(pGt(iplot,fid)),squeeze(pGt(iplot,nfids+fid)),'x','Color',colors(fid,:),'LineWidth',2);
  end
end
drawnow;