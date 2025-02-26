function plotPercentileHist(im,prcs,labels,prc_vals,h,txtOffset)

if nargin<5
  h = figure;
end
if nargin<6
  txtOffset = 5;
end

nviews = size(prcs,3);
nprctiles = size(prcs,1);
nlandmarks = size(prcs,2);
colors = winter(nprctiles);

hax = createsubplots(1,nviews,.01);

for viewi = 1:nviews
  curim = im{viewi};
  if ndims(curim)==2
    curim = repmat(curim(:,:,1),[1,1,3]);
  end
  image(curim,'Parent',hax(viewi));
  axis(hax(viewi),'image','off');
  hold(hax(viewi),'on');
  plot(hax(viewi),labels(:,1,viewi),labels(:,2,viewi),'m+');
  tcol = round(1-squeeze(mean(curim,[1,2]))/255);
  for l=1:nlandmarks
    text(labels(l,1,viewi)+txtOffset,labels(l,2,viewi)+txtOffset,num2str(l),'Parent',hax(viewi),'Color',tcol);
  end
  for p = 1:nprctiles
    for l = 1:nlandmarks
      h(p) = drawellipse(labels(l,1,viewi),labels(l,2,viewi),0,...
        prcs(p,l,viewi),prcs(p,l,viewi),'Color',colors(p,:),'Parent',hax(viewi));
    end
  end
  text(5,5,sprintf('view %d',viewi),'Color','w','Parent',hax(viewi),...
      'VerticalAlignment','top');
end


legends = cell(1,nprctiles);
for p = 1:nprctiles,
  legends{p} = sprintf('%sth %%ile',num2str(prc_vals(p)));
end
hl = legend(h,legends);
set(hl,'Color','k','TextColor','w');
