%% Joining old training data with new one

datadir = '/home/mayank/work/poseEstimation/headTracking/'; 
Q = load(fullfile(datadir,'FlyHeadStephenCuratedData_Janelia.mat'));


tt = dir(fullfile(datadir,'trainingData_fly*.lbl'));

newfiles = {};
for ndx = 1:numel(tt)
  newfiles{ndx} = fullfile(datadir,tt(ndx).name);
end

%%

for ndx = 1:numel(Q.expdirs)
  Q.expdirs{ndx} = fullfile('/home/mayank/Dropbox/PoseEstimation',Q.expdirs{ndx}(54:end));
  Q.vid1files{ndx} = fullfile('/home/mayank/Dropbox/PoseEstimation',Q.vid1files{ndx}(54:end));
  Q.vid2files{ndx} = fullfile('/home/mayank/Dropbox/PoseEstimation',Q.vid2files{ndx}(54:end));
end

tpts = 0;
for ndx = 1:numel(newfiles)
  P = load(newfiles{ndx},'-mat');
  nmov = size(P.movieFilesAll,1);
  expidx_start = numel(Q.expdirs);
  Q.expdirs(end+1:end+nmov) = P.movieFilesAll(:,1);
  for endx = 1:nmov
    [epath,~,~] = fileparts(P.movieFilesAll{endx,1});
    [epath,~,~] = fileparts(epath);
    [ename,~,~] = fileparts(epath);
    Q.expnames{end+1} = ename;
    smov = strrep(P.movieFilesAll{endx,1}(42:end),'\','/');
    Q.vid1files{end+1} = fullfile('/home/mayank/Dropbox/PoseEstimation/Stephen',smov);
    fmov = strrep(P.movieFilesAll{endx,2}(42:end),'\','/');
    Q.vid2files{end+1} = fullfile('/home/mayank/Dropbox/PoseEstimation/Stephen',fmov);
    lbl_frm = find(~isnan(P.labeledpos{endx,1}(1,1,:)));
    nlbl = numel(lbl_frm);
    Q.ts(end+1:end+nlbl) = lbl_frm;
    Q.expidx(end+1:end+nlbl) = numel(Q.expnames);
    curpts1 = P.labeledpos{endx,1}(1:5,:,lbl_frm);
    curpts2 = P.labeledpos{endx,1}(11:15,:,lbl_frm);
    curpts1 = permute(curpts1,[2 1 3]);
    curpts2 = permute(curpts2,[2 1 3]);
    Q.pts(:,1,:,end+1:end+nlbl) = curpts1;
    Q.pts(:,2,:,end-nlbl+1:end) = curpts2;
    tpts = tpts + nlbl;
  end
end

save('/home/mayank/work/poseEstimation/headTracking/FlyHeadStephen_curatedData_withnewlabels.mat','-struct','Q');

%%

curmov = P.movieFilesAll{1,2};
curmov = strrep(curmov,'\','/');
curmov = fullfile('/home/mayank/Dropbox/PoseEstimation/Stephen',curmov(42:end));
a = get_readframe_fcn(curmov);
im = a(1);
figure; 
imshow(im);
hold on
curpts = P.labeledpos{1,1}(:,:,1);
scatter(curpts(11:15,1),curpts(11:15,2));

%%

a = get_readframe_fcn(fullfile('/home/mayank/Dropbox/PoseEstimation/',Q.vid2files{1}(53:end)));

im = a(60);
figure; 
imshow(im);
hold on
curpts = Q.pts(:,:,:,1);
scatter(curpts(1,2,:),curpts(2,2,:))

%%

curmov = 0;
fig = figure;
for ndx = 2660:10:numel(Q.expidx)
  if Q.expidx(ndx)~=curmov
    curmov = Q.expidx(ndx);
    cur = Q.vid2files{Q.expidx(ndx)};
    if strcmp(cur(1:3), '/gr'),
      cur = fullfile('/home/mayank/Dropbox/PoseEstimation',cur(54:end));
    end
    readf = get_readframe_fcn(cur);
  end
  im = readf(Q.ts(ndx));
  imshow(im); hold on;
  scatter(Q.pts(1,2,:,ndx),Q.pts(2,2,:,ndx),'.');
  hold off;
  title(sprintf('%d',ndx));
%   pause(0.1);
end