function hFig = easymontage(I,p,nr,nc,varargin)

[mrkr,mrkrsz,clr,idstr,hFig,doroi,roixrad,roiyrad,axisimage] = myparse(varargin,...
  'marker','.',...
  'markerSize',10,...
  'color',[1 1 0],... % either [1x3], or [nptx3]
  'idstr',[],...
  'hFig',[],...
  'doroi',false,... % if true, zoom in on label centroid
  'roixrad',75,...
  'roiyrad',75,...
  'axisimage',false);

[N,D] = size(p);
assert(numel(I)==N);
tfID = ~isempty(idstr);
if tfID
  assert(numel(idstr)==N);
end

npts = D/2;
nax = nr*nc;
npage = ceil(N/nax);

if isequal(size(clr),[1 3])
  clr = repmat(clr,npts,1);
end
szassert(clr,[npts 3]);

if isempty(hFig)
  hFig = figure;
end
hTG = uitabgroup('Parent',hFig);
  
axsall = [];
for ipage=1:npage
  hT = uitab(hTG,'Title',sprintf('Page%02d',ipage));
  axs = mycreatesubplots(nr,nc,.01,hT);
  axsall = [axsall axs(:)'];

  idx = (1:nax) + nax*(ipage-1);
  for iax=1:nax
    ax = axs(iax);
    axes(ax);

    i = idx(iax);
    if i>N
      break;
    end
    
    imagesc(I{i});
    colormap(ax,'gray');  
    hold(ax,'on');
    
    px = p(i,1:D/2);
    py = p(i,D/2+1:end);
    for ipt=1:npts
      plot(ax,px(ipt),py(ipt),mrkr,'markersize',mrkrsz,'color',clr(ipt,:));
    end
              
    if doroi
      xcent = nanmean(px);
      ycent = nanmean(py);
      roi = [xcent-roixrad xcent+roixrad ycent-roiyrad ycent+roiyrad];
      axis(ax,roi);
    elseif axisimage
      axis(ax,'image');
    end
    
    lims = axis(ax);
    dx = lims(2)-lims(1);
    dy = lims(4)-lims(3);
    TEXTOFFSETFAC = 15;
    text(lims(1)+dx/TEXTOFFSETFAC,lims(3)+dy/TEXTOFFSETFAC,idstr{i},...
      'color',[1 1 0],'fontweight','bold','interpreter','none');
    set(ax,'XTick',ax.XTick(end),'YTick',ax.YTick(end),...
           'XTickLabel',ax.XTickLabel(end),'YTickLabel',ax.YTickLabel(end));


  end
end
