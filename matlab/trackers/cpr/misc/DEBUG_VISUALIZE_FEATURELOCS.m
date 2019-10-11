for landmarki = 1:nfids,
clf;
imagesc(img);
hold on;
axis image;
idxplot = find(any(ftrData.xs(:,1:2)==landmarki,2));
landmarkj = ftrData.xs(idxplot,1);
landmarkj(ftrData.xs(idxplot,1)==landmarki) = ftrData.xs(idxplot(ftrData.xs(idxplot,1)==landmarki),2);
colorsplot = jet(nfids);
for tmpj = 1:nfids,
  idxplotcurr = idxplot(landmarkj==tmpj);
  plot(cs1(n,idxplotcurr),rs1(n,idxplotcurr),'.','Color',colorsplot(tmpj,:));
  for tmpk = idxplotcurr(:)',
    plot([poscs(n,tmpj),cs1(n,tmpk)],[posrs(n,tmpj),rs1(n,tmpk)],'-','Color',colorsplot(tmpj,:));
  end
  plot(poscs(n,tmpj),posrs(n,tmpj),'o','Color',colorsplot(tmpj,:)*.75+.25);
end
plot(poscs(n,landmarki),posrs(n,landmarki),'ws','MarkerFaceColor',colorsplot(landmarki,:)*.75+.25,'MarkerSize',12);
input(num2str(landmarki));
end