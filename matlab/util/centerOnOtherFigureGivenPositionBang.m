function centerOnOtherFigureGivenPositionBang(fig, otherPosition, otherUnits)
% Center fig on another figure, given the other figure's offset and size, and
% the units of the other figure's offset and size.

unitsOrig = get(fig,'Units');
set(fig,'Units',otherUnits);
pos = get(fig,'position');
sz = pos(3:4);

otherOffset = otherPosition(1:2) ;
otherSize = otherPosition(3:4) ;
newOffset = otherOffset + (otherSize-sz)/2;
set(fig,'position',[newOffset sz]);
set(fig,'Units',unitsOrig);

end
