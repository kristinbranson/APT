function centerOnParentFigure(fig,parentFig)

unitsOrig = get(fig,'Units');
set(fig,'Units',get(parentFig,'Units'));
pos = get(fig,'position');
%offset = pos(1:2);
sz = pos(3:4);

parentPos = get(parentFig,'position');
parentOffset = parentPos(1:2);
parentSz = parentPos(3:4);

newOffset = parentOffset + (parentSz-sz)/2;
set(fig,'position',[newOffset sz]);
set(fig,'Units',unitsOrig);
