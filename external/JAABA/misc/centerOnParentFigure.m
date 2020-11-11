function centerOnParentFigure(fig,parentFig,varargin)
[setParentFigUnitsPx] = myparse(varargin,...
  'setParentFixUnitsPx',false ... % for dealing with UIFigure/appdesigner
  );

if setParentFigUnitsPx
  parentUnitsOrig = get(parentFig,'Units');
  set(parentFig,'Units','pixels');
end

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

if setParentFigUnitsPx
  set(parentFig,'Units',parentUnitsOrig);
end