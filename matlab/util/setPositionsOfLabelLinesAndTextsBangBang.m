function setPositionsOfLabelLinesAndTextsBangBang(hPts, hTxt, xy, txtOffset)

nPoints = size(xy,1);
assert(size(xy,2)==2);
assert(isequal(nPoints,numel(hPts),numel(hTxt)));

for i = 1:nPoints
  oldx = get(hPts(i),'XData');
  oldy = get(hPts(i),'YData');
  if isnan(oldx) && isnan(xy(i,1)) && isnan(oldy) && isnan(xy(i,2)),
    continue;
  end
  if oldx==xy(i,1) && oldy==xy(i,2),
    continue;
  end
  set(hPts(i),'XData',xy(i,1),'YData',xy(i,2));
  set(hTxt(i),'Position',[xy(i,1)+txtOffset xy(i,2)+txtOffset 1]);
end

end % function
