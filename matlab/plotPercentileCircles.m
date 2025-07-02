function islight = plotPercentileCircles(im,prcs,labels,prc_vals,hpar,txtOffset,ntotal)

if nargin<5
  hpar = figure;
end
if nargin<6
  txtOffset = 5;
end
if nargin<7,
  ntotal = [];
end

nviews = size(prcs,3);

if numel(hpar) == 1 && strcmpi(hpar.Type,'figure'),
  hfig = hpar;
  clf(hfig);
  hax = createsubplots(1,nviews,.01,hfig);
else
  hax = hpar;
end

nprctiles = size(prcs,1);
nlandmarks = size(prcs,2);
% make sure we get red
colors = flipud(jet((nprctiles-1)*5+1));
colors = colors(1:5:end,:);
kpcolor = [.833,0,.833];

% guess max-value of the image
maxv = max(cellfun(@(x) max(x(:)), im));
if isa(im{1},'uint8'),
  maxv = 255;
elseif isa(im{1},'uint16'),
  maxv = 2^16-1;
elseif maxv <= 1,
  maxv = 1;
else
  maxv = 2^(ceil(log2(double(max(im(:))))/8)*8)-1;
end

% determine if it is a light image or a dark image
meanv = mean(cellfun(@(x) mean(double(x(:))), im));
islight = meanv/maxv > .5;

% make colors darker if it is a light image
if islight,
  colors = colors*.75;
  kpcolor = kpcolor*.75;
  tcol = 'k';
else
  tcol = 'w';
end

hcirc = gobjects(nprctiles+1,1);

for viewi = 1:nviews
  curim = im{viewi};
  if ndims(curim)<=2
    curim = repmat(curim(:,:,1),[1,1,3]);
  end
  image(curim,'Parent',hax(viewi));
  axis(hax(viewi),'image','off');
  hold(hax(viewi),'on');
  hcirc(1) = plot(hax(viewi),labels(:,1,viewi),labels(:,2,viewi),'+','Color',kpcolor);
  for l=1:nlandmarks
    text(labels(l,1,viewi)+txtOffset,labels(l,2,viewi)+txtOffset,num2str(l),'Parent',hax(viewi),'Color',tcol);
  end
  for p = 1:nprctiles
    for l = 1:nlandmarks
      hcirc(p+1) = drawellipse(labels(l,1,viewi),labels(l,2,viewi),0,...
        prcs(p,l,viewi),prcs(p,l,viewi),'Color',colors(p,:),'Parent',hax(viewi));
    end
  end
  s = {};
  if nviews > 1,
    s{end+1} = sprintf('View %d',viewi);
  end
  if numel(ntotal) >= viewi && ~isnan(ntotal(viewi)),
    s{end+1} = sprintf('N examples = %d',ntotal(viewi));
  end
  if ~isempty(s),
    text(5,5,s,'Color',tcol,'Parent',hax(viewi),...
      'VerticalAlignment','top');
  end
end


legends = cell(nprctiles+1,1);
legends{1} = 'Label';
for p = 1:nprctiles,
  legends{p+1} = sprintf('%sth %%ile',num2str(prc_vals(p)));
end
hl = legend(hcirc,legends,'Location','NorthEastOutside');
if ~islight,
  set(hl,'Color','k','TextColor','w');
end
