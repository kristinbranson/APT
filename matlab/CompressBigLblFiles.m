function CompressBigLblFiles(varargin)

if exist('get_readframe_fcn','file') == 0,
  p = fileparts(mfilename('fullpath'));
  addpath(fullfile(p,'..'));
  APT.setpath;
end  

[matfile,outfile,lObj,i0,i1,DEBUG,doexit,newprojectfile,isgrayscale,sourceinfoonly] = ...
  myparse(varargin,'matfile','','outfile','','lObj',[],...
  'i0',1,'i1',inf,'debug',false,'doexit',false,...
  'newprojectfile','','isgrayscale',true,'sourceinfoonly',false);

if ischar(i0),
  i0 = str2double(i0);
end
if ischar(i1),
  i1 = str2double(i1);
end
if ischar(doexit),
  doexit = str2double(doexit)~=0;
end

% find movies that have labels
if isempty(matfile),
  tblMF = lObj.labelGetMFTableLabeled();
  movfiles = cell(size(lObj.movieFilesAll));
  for i = 1:numel(movfiles),
    movfiles{i} = lObj.projLocalizePath(lObj.movieFilesAll{i});
  end
  isCrop = lObj.cropProjHasCrops;
  
  %nlabels = hist(tblMF.mov,1:size(movfiles,1));
  [nmovies,nviews] = size(lObj.movieFilesAll);
  npts = lObj.nPhysPoints;
  
  %
  % outmovdirs = cell(1,nviews);
  % vidobjs = cell(1,nviews);
  % for i = 1:nviews,
  %   outmovdirs{i} = fullfile(outdir,sprintf('%s_view%d',name,i));
  %   if ~exist(outmovdirs{i},'dir'),
  %     mkdir(outmovdirs{i});
  %   end
  % end

  if isCrop,
    cropROIs = lObj.cropGetAllRois;
    nc = reshape(cropROIs(:,2,:)-cropROIs(:,1,:)+1,[nmovies,nviews]);
    nr = reshape(cropROIs(:,4,:)-cropROIs(:,3,:)+1,[nmovies,nviews]);
    assert(all(all(nr==nr(1,:))) && all(all(nc==nc(1,:))));
    nr = nr(1,:);
    nc = nc(1,:);
  else
    nr = nan;
    nc = nan;
  end
else
  load(matfile);
end
i1 = min(i1,nmovies);

newTblMF = tblMF;
off = numel(find(tblMF.mov < i0));
if DEBUG
  imcrops = cell(1,nviews);
end
isset = false(size(tblMF,1),nviews);
isbad = false(size(tblMF,1),1);
allims = cell(size(tblMF,1),nviews);
sourceTblMF = tblMF;
for i = i0:i1,
% could not read video 445 for stephen
%   if i == 445,
%     isbad(i) = true;
%     continue;
%   end
  fprintf('Movie %d = %d / %d\n',i,i-i0+1,i1-i0+1);
  idx = find(tblMF.mov == i);
  if isempty(idx),
    continue;
  end
  newTblMF(off+1:off+numel(idx),:) = tblMF(idx,:);
  sourceTblMF(off+1:off+numel(idx),:) = tblMF(idx,:);
  if ~sourceinfoonly,
    for vwi = 1:nviews,
      try
        [readframe] = get_readframe_fcn(movfiles{i,vwi});
      catch ME,
        isbad(off+off:numel(idx)) = true;
        warning('Could not open movie %s:\n%s',movfiles{i,vwi},getReport(ME));
        continue;
      end
      for j = 1:numel(idx),
        counter = off + j;
        k = idx(j);
        f = tblMF.frm(k);
        im = readframe(f);
        if isCrop,
          im = im(cropROIs(i,3,vwi):cropROIs(i,4,vwi),cropROIs(i,1,vwi):cropROIs(i,2,vwi),:);
        end
        if DEBUG,
          imcrops{vwi} = im;
        end
        allims{counter,vwi} = im;
        %imwrite(im,fullfile(outmovdirs{vwi},sprintf('%05d.png',counter)),'png');
        %writeVideo(vidobjs{vwi},im);
        newTblMF.mov(counter) = 1;
        newTblMF.frm(counter) = counter;
        p = reshape(tblMF.p(k,:),[npts,nviews,2]); % p is npts x nviews x 2
        p = reshape(p(:,vwi,:),[npts,2]);
        if isCrop,
          p(:,1) = p(:,1) - cropROIs(i,1,vwi) + 1;
          p(:,2) = p(:,2) - cropROIs(i,3,vwi) + 1;
        end
        for d = 1:2,
          newTblMF.p(counter,sub2ind([npts,nviews,2],1:npts,vwi+zeros(1,npts),d+zeros(1,npts))) = p(:,d)';
        end
        isset(counter,vwi) = true;
      end
      clear readframe;
    end
    if DEBUG,
      j = numel(idx);
      counter = off + j;
      clf;
      p = reshape(newTblMF.p(counter,:),[npts,nviews,2]);
      for vwi = 1:nviews,
        subplot(1,nviews,vwi);
        imagesc(imcrops{vwi}); axis image; colormap gray; hold on;
        plot(p(:,vwi,1),p(:,vwi,2),'r.');
      end
      input(num2str(counter));
    end
  end
  off = off + numel(idx);
end

newTblMF(off+1:end,:) = [];
sourceTblMF(off+1:end,:) = [];

if ~isempty(outfile),
  if sourceinfoonly,
    save(outfile,'sourceTblMF','movfiles');
  else
    save(outfile,'sourceTblMF','movfiles','newTblMF','isset','isbad','allims','i0','i1');
  end
end
% 
% for i = 1:nviews,
%   close(vidobjs{i});
% end

if doexit,
  exit;
end

if sourceinfoonly,
  return;
end

if isempty(lObj),
  return;
end

% call a bunch of times on subsets of the videos within different matlabs
if false,
  matlabcmd = '/misc/local/matlab-2018b/bin/matlab';
  nperjob = 10;
  ncores = 1;
  outdir = '/groups/branson/home/bransonk/tracking/code/APT/CompressStuff_sh_trn4992_gtcomplete_cacheddata_updated20200317';
  if ~exist(outdir,'dir'),
    mkdir(outdir);
  end
  for i0 = 1:nperjob:nmovies,
    i1 = min(i0+nperjob-1,nmovies);
    cmd1 = sprintf('%s -nodisplay -r \\"CompressBigLblFiles matfile /groups/branson/home/bransonk/tracking/code/APT/CompressStuff_sh_trn4992_gtcomplete_cacheddata_updated20200317.mat i0 %d i1 %d outfile %s/res%03dto%03d.mat\\"',matlabcmd,i0,i1,outdir,i0,i1);
    cmd2 = sprintf('bsub -n %d -J CBLF%03d -o %s/CBLF%03d.txt ''%s''',ncores,i0,outdir,i0,cmd1);
    cmd3 = sprintf('ssh login1 "cd /groups/branson/home/bransonk/tracking/code/APT/matlab; %s"',cmd2);
    cmd = sprintf('%s -nodisplay -r "CompressBigLblFiles matfile /groups/branson/home/bransonk/tracking/code/APT/CompressStuff_sh_trn4992_gtcomplete_cacheddata_updated20200317.mat i0 %d i1 %d outfile %s/res%03dto%03d.mat doexit 1"',matlabcmd,i0,i1,outdir,i0,i1);
    disp(cmd);
  end
  
end

if false,
  % load in and combine results
  isset = false(size(tblMF,1),nviews);
  isbad = false(size(tblMF,1),1);
  for i0 = 1:nperjob:nmovies,
    i1 = min(i0+nperjob-1,nmovies);
    resfile = fullfile(outdir,sprintf('res%03dto%03d.mat',i0,i1));
    rd = load(resfile);
    idx = find(all(rd.isset,2));
    assert(all(all(~isset(idx,:))));
    assert(all(~isbad(idx)));
    allims(idx,:) = rd.allims(idx,:);
    isbad(idx) = rd.isbad(idx);
    isset(idx,:) = true;
    newTblMF(idx,:) = rd.newTblMF(idx,:);
  end
  idx = find(all(isset,2));
  nims = numel(idx);
  naxr = 20; naxc = 20; npages = ceil(nims/naxr/naxc);
end

% plot 
if false,
  isfirst = true(1,npages);
  
  for ii = 1:numel(idx),
    i = idx(ii);
    [axc,axr,fig] = ind2sub([naxc,naxr,npages],ii);
    figs = (fig-1)*nviews + (1:nviews);

    if isfirst(fig),
      for j = 1:nviews,
        if ishandle(figs(j)),
          clf(figs(j));
        end
        figure(figs(j));
        set(figs(j),'Units','pixels','Position',[10,10,1000,1000]);
        hax{fig,j} = reshape(createsubplots(naxr,naxc,0),[naxr,naxc]);
      end
      isfirst(fig) = false;
    end

    p = reshape(newTblMF.p(i,:),[npts,nviews,2]);
    for j = 1:nviews,
      haxcurr = hax{fig,j}(axr,axc);
      imagesc(allims{i,j},'Parent',haxcurr);
      hold(haxcurr,'on'); axis(haxcurr,'image', 'off'); 
      colormap(haxcurr,'gray');
      plot(haxcurr,p(:,j,1),p(:,j,2),'r.');
    end
    drawnow;
  end
end

[outdir,name] = fileparts(lObj.projectfile);
if isempty(newprojectfile),
  newprojectfile = fullfile(outdir,sprintf('%s_compress%s.lbl',name,datestr(now,'yyyymmdd')));
end
[~,newname] = fileparts(newprojectfile);

vidobjs = cell(1,nviews);
outmovfiles = cell(1,nviews);
for i = 1:nviews,
  outmovfiles{i} = fullfile(outdir,sprintf('%s_view%d.avi',newname,i));
  if isgrayscale,
    format = {'Grayscale AVI'};
  else
    format = {};
  end
  vidobjs{i} = VideoWriter(outmovfiles{i},format{:});
  open(vidobjs{i});
end

idx = find(all(isset,2));
nims = numel(idx);
tblMF1 = newTblMF(idx,:);
tblMF1.frm = (1:nims)';
for j = 1:nviews,
  for ii = 1:nims,
    i = idx(ii);
    imcurr = allims{i,j};
    if isgrayscale && (size(imcurr,3) > 1),
      %assert(all(all(all(imabsdiff(repmat(imcurr(:,:,1),[1,1,3]),imcurr)<=2,1),2),3));
      imcurr = imcurr(:,:,1);
    end
    writeVideo(vidobjs{j},imcurr);
  end
end
for j = 1:nviews,
  close(vidobjs{j});
end

lObj.projSaveAs(newprojectfile);
lObj.clearAllTrackers();
lObj.movieRmAll();
lObj.movieSetAdd(outmovfiles);
tblMF1 = removevars(tblMF1,'mov');
lObj.labelPosBulkImportTblMov(tblMF1);
lObj.projSaveSmart();