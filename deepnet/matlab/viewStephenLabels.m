addpath(genpath('/groups/branson/home/bransonk/tracking/code/piotr_toolbox_V3.02'));
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;

folder = '../headTracking';
file = 'FlyHeadStephenCuratedData_Janelia.mat';

ld = load(fullfile(folder,file));

% view training data

ld.expdirs = ld.vid2files;
if ndims(ld.pts)>3,
  ld.pts = squeeze(ld.pts(:,2,:,:));
  ld.pts = permute(ld.pts,[2 1 3]);
end


hfig = 1;
figure(hfig);
clf;
hax = gca;

npts = size(ld.pts,1);
colors = jet(npts);
colors = colors(randperm(npts),:);

expdirprev = '';
fid = -1;

for i = 1:numel(ld.expidx),

  expdir = ld.expdirs{ld.expidx(i)};
  t = ld.ts(i);
  if ~strcmp(expdir,expdirprev),
    if fid > 0,
      fclose(fid);
    end
    count = 1;
    [readframe,nframes,fid] = get_readframe_fcn(expdir);
    expdirprev = expdir;
  else
    if count > 5
      continue;
    end
  end
  im = readframe(ld.ts(i));
  hold(hax,'off');
  imagesc(im,'Parent',hax,[0,255]);
  axis(hax,'image','off');
  hold(hax,'on');
  colormap gray;
  for j = 1:npts,
    plot(hax,ld.pts(j,1,i),ld.pts(j,2,i),'wo','MarkerFaceColor',colors(j,:));
  end
  title(sprintf('%d/%d Frame:%d/%d Exp:%d',i,numel(ld.expidx),t,nframes,ld.expidx(i)));
  count = count + 1;
  pause(0.1);
%     text(ax(1)+5,ax(3)+5,num2str(t),'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax);

end

if fid > 0,
  fclose(fid);
end

