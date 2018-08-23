

local = true;
if ~local
  ldir = '/localhome/kabram/Dropbox (HHMI)/PoseEstimation/Stephen/18012017_trainingData/justHead/';
  Q = load('/groups/branson/bransonlab/mayank/PoseTF/headTracking/FlyHeadStephenCuratedData.mat');
else
  Q = load('/home/mayank/work/poseTF/headTracking/FlyHeadStephenCuratedData.mat');
  ldir = '/home/mayank/Dropbox/PoseEstimation/Stephen/18012017_trainingData/justHead/';
end
dd = dir([ldir '*.lbl']);
P = load(fullfile(ldir,'fly90.lbl'),'-mat');


%%
% in Q.pts, view is the second co-ordinate. i.e., Q.pts(:,1,:,:)
% corresponds to side view


J = struct;

nexp = numel(Q.vid1files);
movf = cell(nexp,2);
lpos = cell(nexp,1);
lmarked = cell(nexp,1);
lothers = cell(nexp,1);
for ndx = 1:nexp
  if ~local,
    v1f = ['/groups/branson/bransonlab/mayank/' Q.vid1files{ndx}(19:end)];
    v2f = ['/groups/branson/bransonlab/mayank/' Q.vid2files{ndx}(19:end)];
  else
    v1f = Q.vid1files{ndx};
    v2f = Q.vid2files{ndx};
  end
  
  movf{ndx,1} = v1f;
  movf{ndx,2} = v2f;
  
  [rfn,nframes,fid,hinfo] = get_readframe_fcn(v1f);
  nframes = int64(hinfo.nframes);
  curidx = find(Q.expidx==ndx);
  pside = permute(Q.pts(:,1,:,curidx),[3,1,4,2]);  
  pfront = permute(Q.pts(:,2,:,curidx),[3,1,4,2]);
  curpos = nan(10,2,nframes);
  curt = Q.ts(curidx);
  curpos(1:5,:,curt) = pside;
  curpos(6:10,:,curt) = pfront;
  lpos{ndx} = curpos;
  curm = false(10,nframes);
  curm(:,curidx) = true;
  lmarked{ndx} = curm;
  
  if fid>0,fclose(fid); end
end

%%
for ndx = 1:numel(dd)
  P = load(fullfile(ldir,dd(ndx).name),'-mat');
  nexp = numel(P.labeledpos);  
  
  K = cell(nexp,2);
  for ne = 1:nexp
    for vv = 1:2
        kk = strrep(P.movieFilesAll{ne,vv},'\','/');
        if P.movieFilesAll{ne,1}(1) == '$',
          K{ne,vv} = ['/groups/huston/hustonlab/' kk(10:end)];
        else
          K{ne,vv} = ['/groups/huston/hustonlab/' kk(4:end)];
        end
        
        if strfind(K{ne,vv},'/fly_450_to_452_26_9_16/')
          K{ne,vv} = [K{ne,vv}(1:50) 'fly_450_to_452_26_9_16norpAkirPONMNchr' K{ne,vv}(73:end)];
        end
        if strfind(K{ne,vv},'/fly_453_to_457_27_9_16/')
          K{ne,vv} = [K{ne,vv}(1:50) 'fly_453_to_457_27_9_16_norpAkirPONMNchr'  K{ne,vv}(73:end)];
        end

    end    
%     l1 = any(~isnan(P.labeledpos{ne}(:,:,:)),2);
%     l2 = all(P.labeledposMarked{ne},1);
%     if ~all(l1==l2),
%       fprintf('Marked and labels dont match for %d,%d\n',ndx,nexp);
%     end
  end
  movf = [movf;K];
  lpos = [lpos; P.labeledpos];
  lmarked = [lmarked; P.labeledposMarked];
  
end


%%
J.movieFilesAll = movf;
J.labeledpos = lpos;
J.labeledposMarked = lmarked;
J.cfg = P.cfg;

%%

rfn = get_readframe_fcn(Q.vid1files{1});
ii = rfn(Q.ts(1));
figure; imshow(ii);
hold on;
scatter(Q.pts(1,1,:,1),Q.pts(2,1,:,1),'.');
hold off;

%%

pp = [];
for ndx = 1:numel(J.labeledpos)
  ff = ~isnan(J.labeledpos{ndx}(1,1,:));
  pp = cat(3,pp,J.labeledpos{ndx}(:,:,ff));
  
end


%%
f = figure;

nc = 5; nr = 4;
for ndx = 1:10
  for zz = 2
    subplot(nc,nr,ndx);
    hist(squeeze(pp(ndx,zz,:)));
  end  
end


%% label for 318

kk = [];
for ndx = 1:numel(Q.expdirs)
%   if ~isempty(strfind(Q.expdirs{ndx},'138'));
%     kk(end+1) = ndx;
%   end
  fprintf('%s\n',Q.expdirs{ndx}(46:end));
end

%%

ts = [];
expidx = [];
for ndx = 1:numel(J.labeledpos)
  ff = find(~isnan(J.labeledpos{ndx}(1,1,:)));
  nn = numel(ff);
  expidx(end+1:end+nn) = ndx;
  ts(end+1:end+nn) = ff;
  
end


%%

dx = []; dy = [];
for ndx= 1:size(qq,3)
  dx(ndx) = max(qq(:,1,ndx))-min(qq(:,1,ndx));
  dy(ndx) = max(qq(:,2,ndx))-min(qq(:,2,ndx));
end

prctile(dx,99)
prctile(dy,99)

%%

p1 = 1;
p2 = 7;
ss1 = pp(:,:,p1);
ss2 = pp(:,:,p2);
ss = bsxfun(@minus,pp,ss1);

gg = sqrt( sum( (ss).^2,2));
figure; hist(gg(:,:,p2),30)

%%

fly_num = [];
for i = 1:size(J.movieFilesAll,1)
  vv = regexpi(J.movieFilesAll{i,1},'fly_*(\d+)','tokens');
  fly_num(i) = str2double(vv{end});
end

jj = unique(fly_num);
n_labels = zeros(1,numel(jj));
for ndx = 1:numel(expidx)
  curndx = find(jj== fly_num(expidx(ndx)));
  n_labels(curndx) = n_labels(curndx)+1;
end

for ndx = 1:numel(jj)
  if n_labels(ndx)==0, continue; end
  fprintf('Fly: %d, labels:%d \n',jj(ndx), n_labels(ndx));
end

%%

aa = unique(J.movieFilesAll(:,1));
