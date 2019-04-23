function cminterp = logscale_colormap(cm,clim,off)

 ncm = size(cm,1);
 if ~exist('off','var'),
   off = (clim(2)-clim(1))/ncm;
 end
 x = logspace(log10(off),log10(clim(2)-clim(1)+off),ncm) + clim(1) - off;
 xinterp = linspace(clim(1),clim(2),ncm);
 cminterp = nan(size(cm));
 for i = 1:3,
   cminterp(:,i) = interp1(x,cm(:,i),xinterp,'linear','extrap');
 end
cminterp = min(1,max(0,cminterp));