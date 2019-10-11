function tf = figIsOffScreen(hFig)
% tf = figIsOffScreen(hFig) 
%
% tf: true if at least one corner of hFig is off-screen

assert(isa(hFig,'matlab.ui.Figure'));

gr = groot;
mposes = gr.MonitorPositions;
mposUnits = gr.Units;
figUnits0 = hFig.Units;
oc = onCleanup(@()set(hFig,'Units',figUnits0));
hFig.Units = mposUnits;
figpos = hFig.Position;

[xloM,xhiM,yloM,yhiM] = figPos2Lims(mposes); % each output is [ndispx1]
[xlo,xhi,ylo,yhi] = figPos2Lims(figpos);
% corners(iCorner,:) gives [x y] for corner iCorner
corners = [ xlo ylo; xhi ylo; xhi yhi; xlo yhi ];
for iCorner=1:4
  xy = corners(iCorner,:);
  tfInDisp = xloM<=xy(1) & xy(1)<=xhiM & yloM<=xy(2) & xy(2)<=yhiM;
  if ~any(tfInDisp)
    tf = true;
    return;
  end
end

tf = false;
