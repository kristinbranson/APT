m1L=sqrt((pT(:,1)-pT(:,2)).^2+(pT(:,5)-pT(:,6)).^2);
m2L=sqrt((pT(:,3)-pT(:,4)).^2+(pT(:,7)-pT(:,8)).^2);

m1Cx=(pT(:,1)+pT(:,2))/2; m1Cy=(pT(:,5)+pT(:,6))/2;
m2Cx=(pT(:,3)+pT(:,4))/2; m2Cy=(pT(:,7)+pT(:,8))/2;
mD=sqrt((m1Cx-m2Cx).^2+(m1Cy-m2Cy).^2);


fh=figure;
set(fh,'Position',[-260 1300 1600 800])

subplot(2,3,[1,2,4,5]);
him=imagesc(IsT{1},[509 2500]);
colormap fire_colormap
axis image
axis off
hold on
p1h=plot(pT(1,1:2),pT(1,5:6),'xb','MarkerSize',15,'LineWidth',3);
p2h=plot(pT(1,3:4),pT(1,7:8),'xg','MarkerSize',15,'LineWidth',3);
p12h=plot([m1Cx(1),m2Cx(1)],[m1Cy(1),m2Cy(1)],'k','LineWidth',3);


subplot(2,3,3);
axis([1 size(pT,1) min([m1L;m2L])*0.95 max([m1L;m2L])*1.05])
hold on
pLh=plot(1,m1L(1),'b',1,m2L(1),'g');
set(pLh,'LineWidth',2)
title('Muscle length','FontSize',14)
xlabel('Frame','FontSize',12)
ylabel('Length (Px)','FontSize',12)

subplot(2,3,6);
axis([1 size(pT,1) min(mD)*0.95 max(mD)*1.05])
hold on
pDh=plot(1,mD(1),'k','LineWidth',2);
title('Muscle distance','FontSize',14)
xlabel('Frame','FontSize',12)
ylabel('Distance (Px)','FontSize',12)


for i=1:size(pT,1)
    set(him,'CData',IsT{i})
    set(p1h,'XData',pT(i,1:2),'YData',pT(i,5:6))
    set(p2h,'XData',pT(i,3:4),'YData',pT(i,7:8)) 
    set(p12h,'XData',[m1Cx(i),m2Cx(i)],'YData',[m1Cy(i),m2Cy(i)])
    set(p2h,'XData',pT(i,3:4),'YData',pT(i,7:8))
    set(pLh(1),'XData',1:i,'YData',m1L(1:i))
    set(pLh(2),'XData',1:i,'YData',m2L(1:i))
	set(pDh,'XData',1:i,'YData',mD(1:i))

    export_fig('-nocrop','-transparent','filename', ['../larva/vimg(', num2str(i),').tif'])
end