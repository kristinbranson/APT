function [uniquedays,sxperday,hax,hfig] = PlotLabeledReachPosXStdPerMouse(labeldata,Sperday,uniquedays,varargin)

[hax,hfig,mousename] = myparse(varargin,'hax',[],'hfig',[],'mousename','');

if isempty(hax),
  if isempty(hfig),
    hfig = figure;
  else
    figure(hfig);
  end
  hax = gca;
end

for i = 1:numel(labeldata.labeledpos),
  if ~isempty(labeldata.labeledpos),
    [nlandmarks,nd,~] = size(labeldata.labeledpos{i});
    assert(nd == 2,'Not implemented: can only plot 2d data');
    break;
  end
end

ndays = numel(uniquedays);
sxperday = reshape(sqrt(Sperday(1,1,:,:)),[nlandmarks,ndays]);
colors = [0,0,0;
  .7,0,0];
linestyles = {'-','--'};
h = nan(1,nlandmarks);
legends = cell(1,nlandmarks);
for landmarki = 1:nlandmarks,
  h(landmarki) = plot(hax,1:ndays,sxperday(landmarki,:),'o-',...
    'Color',colors(landmarki,:),'MarkerFaceColor',colors(landmarki,:),...
    'LineStyle',linestyles{landmarki},'LineWidth',2);
  hold(hax,'on');
  legends{landmarki} = sprintf('View %d',landmarki);
end
set(hax,'XTick',1:ndays,'XTickLabel',uniquedays,'Box','off');
legend(hax,h,legends);
ylabel(hax,'Standard deviation in x (px)');
if ~isempty(mousename),
  title(hax,mousename,'Interpreter','none');
end