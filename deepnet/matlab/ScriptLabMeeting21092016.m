Q = load('~/temp/stephen_fly325/Dropbox__fly325__0020_front.mat');
P = load('~/Dropbox/talks/labMeetingSep21_extra/basepredfly325.mat');
readfcn = get_readframe_fcn(Q.expname);

%%
fnum = 406;
im = readfcn(fnum);

llocs = [394 358; 332 336; 407 321; 351 294; 357 374];
f = figure('position',[675 171 1000 400]);
axis('tight')
cc = jet(5);
subplot('position',[0.05 0.05 0.4 0.9]);
imshow(im);
axis('tight');
% hold on;
% scatter(Q.locs(fnum,:,1,1),Q.locs(fnum,:,1,2),50,cc,'.');
subplot('position',[0.55 0.05 0.4 0.9]);
imshow(im);
hold on;
scatter(Q.locs(fnum,:,1,1),Q.locs(fnum,:,1,2),50,cc,'+');
for ndx = 1:5
  plot([Q.locs(1,ndx,1,1) llocs(ndx,1)],[Q.locs(fnum,ndx,1,2) llocs(ndx,2)],...
    'Color',cc(ndx,:),'linewidth',0.5)
end
axis('tight');

%%

% Part detector scores
f = figure('position',[600,500,1000,400]);
titles = {'Right bottom','Left bottom','Right up','Left up','Proboscis'};
pp = [];
for ndx = 1:5
  ii = squeeze(P.basepred(1,:,:,ndx));
  ii(1,1) = -1;
  ii(end,end) = 0.5;
  xx = 1-floor( (ndx-1)/3);
  yy = mod(ndx-1,3);
  pp(ndx) = subplot('position',[yy*0.3+0.05,xx*0.5+0.05,0.28,0.4]);
  imagesc(ii); axis equal; axis tight; axis off;
  title(titles{ndx})
  colorbar('EastOutside');
end

%%

% mrf detector scores
f = figure('position',[600,500,1000,400]);
titles = {'Right bottom','Left bottom','Right up','Left up','Proboscis'};
pp = [];
for ndx = 1:5
  ii = squeeze(Q.scores(fnum,:,33:end-32,ndx));
  ii(1,1) = 0;
  ii(end,end) = 0.5;
  xx = 1-floor( (ndx-1)/3);
  yy = mod(ndx-1,3);
  pp(ndx) = subplot('position',[yy*0.3+0.05,xx*0.5+0.05,0.28,0.4]);
  imagesc(ii); axis equal; axis tight; axis off;
  title(titles{ndx})
  colorbar('EastOutside');
end

