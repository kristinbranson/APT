
psz = 20;
mimage = {};
count = [];
for lid = 1:nfids;
  mimage{lid} = zeros(2*psz+1,2*psz+1);
  count(lid) = 0;
end

for lid = 1:nfids;
  for ndx = 1:numel(IsT)
    lmarks = round(reshape(phisT(ndx,:),[nfids,3]));
    if (lmarks(lid,3)>0), continue; end
    curp = padgrab(IsT{ndx},128,lmarks(lid,2)-psz,lmarks(lid,2)+psz,lmarks(lid,1)-psz,lmarks(lid,1)+psz);
    mimage{lid} = mimage{lid} + double(curp)/255;
    count(lid) = count(lid)+1;
  end
end
figure;

for lid = 1:nfids
  subplot(5,6,lid); 
  imshow(mimage{lid}/count(lid));
  title(sprintf('%d',lid));
end


%%

psz = 20;
mimage = {};
count = [];
for lid = 1:nfids;
  mimage{lid} = [];
  count(lid) = 0;
end

for lid = 1:nfids;
  for ndx = 1:numel(IsT)
    lmarks = round(reshape(phisT(ndx,:),[nfids,3]));
    if (lmarks(lid,3)>0), continue; end
    curp = padgrab(IsT{ndx},128,lmarks(lid,2)-psz,lmarks(lid,2)+psz,lmarks(lid,1)-psz,lmarks(lid,1)+psz);
    mimage{lid}(:,:,end+1) = double(curp)/255;
    count(lid) = count(lid)+1;
  end
end
figure;

lid = 2;
qq = randsample(count(lid),80);
for lix = 1:numel(qq)
  subplot(8,10,lix); 
  imshow(mimage{lid}(:,:,qq(lix)));
end


%% correlation between different points 

f = figure;
x1 = 23;
for x2 = 1:29;
  p1_all = [];
  p2_all = [];
  cc = [];
  for ndx = 1:numel(regModel.regs)
    p1 = []; p2 = [];
    for k = 1:numel(regModel.regs(ndx).regInfo)
      p1 = [p1; regModel.regs(ndx).regInfo{k}.ysFern(:,x1)];
      p2 = [p2; regModel.regs(ndx).regInfo{k}.ysFern(:,x2)];
      
    end
    p1_all = [p1_all; p1];
    p2_all = [p2_all; p2];  
    gg = corrcoef(p1,p2);
    cc(ndx) = gg(2);
  end
  gg = corrcoef(p1_all,p2_all);
  figure(f); subplot(5,6,x2); plot(cc); title(sprintf('%d %.2f',x2,gg(2)));
  ylim([0 1]);
end

%% convert labels into output for visualization.

O1 = load('/home/mayank//Work/Tracking/data/mouse/Data/mouselabelerm127.mat');
O2 = load('/home/mayank//Work/Tracking/data/mouse/Data/mouselabelerm127b.mat');

P1 = struct; 
count = 1;
for ndx = 1:5
  if(~isequal(O1.expdirs{ndx},O2.expdirs{ndx})), continue; end;
  cure = O1.expdirs{ndx};
  [~,fname ] = fileparts(cure);
  P1.moviefiles_all{count} = ['/home/mayank//Dropbox/AdamVideos/multiPoint/M127_20140909/' fname '/movie_comb.avi'];
  curp1 = O1.labeledpos_perexp{ndx};
  curp2 = O2.labeledpos_perexp{ndx};
  curp = cat(1,curp1,curp2);
  curp = permute(curp,[3 1 2]);
  curp = reshape(curp,[],8);
  curp = curp(1:10:end,:);
  curp = permute(curp,[3 1 2]);
  curp = repmat(curp,[10 1 1]);
  curp = reshape(curp,[],8);

  P1.p_all{count} = curp;
  count = count + 1;

end

P1.maxv = 255;
P1.minv = 0;
P.npoints = 4;
save('../../temp/M127_labels.mat','-struct','P1');


%%

O1 = load('/home/mayank//Work/Tracking/data/mouse/Data/M118_all_local.mat');
localdir = '/home/mayank/Dropbox/AdamVideos/multiPoint/';

P1 = struct; 
count = 1;
for ndx = 1:numel(O1.expdirs)
  cure = O1.expdirs{ndx};
  [~,fname ] = fileparts(cure);
  [~,~,~,dat] = regexp(cure,'\d{8}');
  dat = dat{1};
  localname = fullfile(localdir,[name '_' dat],fname,'movie_comb.avi');
  P1.moviefiles_all{count} = localname;
  curp = O1.labeledpos_perexp{ndx};
  curp = permute(curp,[3 1 2]);
  curp = reshape(curp,[],12);
  curp = curp(1:10:end,:);
  curp = permute(curp,[3 1 2]);
  curp = repmat(curp,[10 1 1]);
  curp = reshape(curp,[],12);

  P1.p_all{count} = curp;
  count = count + 1;

end

P1.maxv = 255;
P1.minv = 0;
P.npoints = 4;
save('../../temp/M118_multipoints.mat','-struct','P1');

%% Keep only local in label files

name = 'M127';
O1 = load(['/home/mayank/Work/Tracking/data/mouse/Data/' name '_all.mat']);
localdir = '/home/mayank/Dropbox/AdamVideos/multiPoint/';
keep = false(1,numel(O1.expdirs));
for ndx = 1:numel(O1.expdirs)
  cure = O1.expdirs{ndx};
  [~,fname] = fileparts(cure);
  [~,~,~,dat] = regexp(cure,'\d{8}');
  dat = dat{1};
  localname = fullfile(localdir,[name '_' dat],fname);
  if ~exist(localname,'dir'),continue;end
  O1.expdirs{ndx} = localname;
  keep(ndx) = true;

end
O1.expdirs = O1.expdirs(keep);
O1.labeledpos_perexp = O1.labeledpos_perexp(keep);
save(['../../data/mouse/Data/' name '_all_local.mat'],'-struct','O1');


%% copy training dir to dropbox

name = 'M130';
Q = load(['/home/mayank/Work/Tracking/data/mouse/Data/' name '_all.mat']);
ff = fopen(['../../temp/copy' name '.txt'],'w');
localdir = '/home/mayank/Dropbox/AdamVideos/multiPoint/';
remotedir = '/groups/branson/home/kabram/Dropbox/AdamVideos/multiPoint/';
for ndx = 1:numel(Q.expdirs)
  cure = Q.expdirs{ndx};
  [~,fname] = fileparts(cure);
  [~,~,~,dd] = regexp(Q.expdirs{ndx},'\d{8}');
  dd = dd{1};
  localname = fullfile(localdir,[name '_' dd],fname);
  if exist(localname,'dir'), continue; end
  remotename = fullfile(remotedir,[name '_' dd]);
  fprintf(ff,'cp -r %s %s\n',cure,remotename);
  [ddname] = fileparts(localname);
  if ~exist(ddname,'dir'), mkdir(ddname); end
  
end

fclose(ff);
%% combine multiple mouse files

names = {'M118','M119','M122','M127','M130'};
A.expdirs = {};
A.labeledpos_perexp = {};
datfile = '../../data/mouse/Data/multiPoint_All_local.mat';
localdir = '/home/mayank/Dropbox/AdamVideos/multiPoint/';
for ndx = 1:numel(names);
  infile = ['../../data/mouse/Data/' names{ndx} '_all.mat'];
  Q = load(infile);
  for ix = 1:numel(Q.expdirs)
    cure = Q.expdirs{ix};
    [~,fname] = fileparts(cure);
    [~,~,~,dd] = regexp(cure,'\d{8}');
    dd = dd{1};
    localname = fullfile(localdir,[names{ndx} '_' dd],fname);

    A.expdirs{end+1} = localname;
    A.labeledpos_perexp{end+1} = Q.labeledpos_perexp{ix};
  end
end
save(datfile,'-struct','A');

all_trainoutfile = '../../data/mouse/Data/multiPoint_All_train.mat';
all_testoutfile = '../../data/mouse/Data/multiPoint_All_test.mat';
model_file = '../../data/mouse/Data/multiPoint_all_model.mat';

cvp = cvpartition(numel(A.expdirs),'Holdout',0.2);
prepareTrainingFiles(datfile,all_trainoutfile,find(cvp.training),500);
prepareTrainingFiles(datfile,all_testoutfile,find(cvp.test),500);
[dname,fname,ext] = fileparts(all_trainoutfile);
[regModel,regPrm,prunePrm,H0] = RCPR_simple(all_trainoutfile,...
  fullfile(dname,[fname '_Is' ext]),'mouse_paw_multi');
save(model_file,'regModel','regPrm','prunePrm','H0');


%% train a regressor.
name = 'M127';
infile = ['../../data/mouse/Data/' name '_all_local.mat'];
train_outfile = ['../../data/mouse/Data/multiPoint_' name '_local_part.mat'];
test_outfile =  ['../../data/mouse/Data/multiPoint_' name '_local_part_test.mat'];
model_file = ['../../data/mouse/Data/' name '_local_part_model.mat'];
[dname,fname,ext] = fileparts(train_outfile);
Q = load(infile);
moviefile = fullfile(Q.expdirs{end},'movie_comb.avi');
prepareTrainingFiles(infile,train_outfile,1:round(numel(Q.expdirs)-2),500);
[regModel,regPrm,prunePrm,H0] = RCPR_simple(train_outfile,...
  fullfile(dname,[fname '_Is' ext]),'mouse_paw_multi');
save(model_file,'regModel','regPrm','prunePrm','H0');

%% test the regressor.
[p,Y,lossT,lossY,bad,pR] = track_video('moviefile',moviefile,'model',model_file,'H0_file',model_file);
figure; plot(bad);
Labeler_fix({p},{moviefile});

%%
fnum = 331;

[readframe,nframes] = get_readframe_fcn(moviefile);
zz = readframe(fnum);
figure(452); imshow(zz);
hold on;
scatter(p(fnum,1:6),p(fnum,7:12),'r.');

%%

fail = 1;
count = 1;
p_act = Q.labeledpos_perexp{end}(:,:,fnum);
p_init = [];
p_init(1,:,1) = p_act(:)' + 10*randn(1,12);
p_init(1,:,2) = p_act(:)' + 10*randn(1,12);
p_init(1,:,3) = p_act(:)' + 10*randn(1,12);



while(~isempty(fail)) && count < 20
[p_cur,pR,~,fail] = test_rcpr([],[1 1 size(zz,2) size(zz,1)],{zz},regModel,regPrm,prunePrm,p_init); 
count = count + 1;
end
figure(452); 
hold off;
imshow(zz);
hold on;
scatter(p_act(:,1),p_act(:,2),'r.');
scatter(p_cur(:,1:6),p_cur(:,7:12),'b.');
if isempty(fail), title('success'); else title('fail'); end

%% dist vs th

fail = 1;
p_act = Q.labeledpos_perexp{end}(:,:,fnum);

all_fail = [];
all_dist = [];
for th = 5:10
  prunePrm.th = th;
  for ndx = 1:100
    p_init = [];
    p_init(1,:,1) = p_act(:)' + 10*randn(1,12);
    p_init(1,:,2) = p_act(:)' + 10*randn(1,12);
    p_init(1,:,3) = p_act(:)' + 10*randn(1,12);
    
    [p_cur,pR,~,fail] = test_rcpr([],[1 1 size(zz,2) size(zz,1)],{zz},regModel,regPrm,prunePrm,p_init);
    if isempty(fail), fail = 0; end;
    all_fail(th-4,ndx) = fail;
    all_dist(th-4,ndx) = shapeGt('dist',regModel.model,reshape(p_act(1:2:end,:),1,[]),reshape(p_cur(1:2:end),1,[]));
  end
end

figure(453);
plot(5:10,mean(all_dist,2));
xlabel('Pruning threshold');
ylabel('Avg distance between predicted and labeled pts');
title('Prediction error vs pruning threshold');

figure(454);

all_dist_nan = all_dist;
all_dist_nan(all_fail>0) = nan;
plot(5:10,nanmean(all_dist_nan,2));

xlabel('Pruning threshold');
ylabel('Avg distance between predicted and labeled pts');
title('Prediction error vs pruning threshold');


%% point cloud of convergence..

% % paw is open. Only M118 trained model does alright, but sometimes fails
% % partially.
% fnum = 331;
% moviefile = '/home/mayank/Dropbox/AdamVideos/multiPoint/M118_20140829/M118_20140829_v001/movie_comb.avi';
% model_file = '../../data/mouse/Data/M118_local_part_model.mat';


% % Test image where it doesn't converge.
% fnum = 268;
% moviefile = '/home/mayank/Dropbox/AdamVideos/multiPoint/M127_20140911/M127_20140911_v002/movie_comb.avi';
% model_file = '../../data/mouse/Data/multiPoint_all_model.mat';

% % Train image which is similar to above but tracking looks fine.
% fnum = 269;
% moviefile = '/home/mayank/Dropbox/AdamVideos/multiPoint/M127_20140911/M127_20140911_v001/movie_comb.avi';
% model_file = '../../data/mouse/Data/multiPoint_all_model.mat';

% % Train image just some other frame.
% fnum = 230;
% moviefile = '/home/mayank/Dropbox/AdamVideos/multiPoint/M127_20140911/M127_20140911_v001/movie_comb.avi';
% model_file = '../../data/mouse/Data/multiPoint_all_model.mat';

% Test image where there is left-right paw confusion.
fnum = 75;
moviefile = '/home/mayank/Dropbox/AdamVideos/multiPoint/M130_20140910/M130_20140910_v001/movie_comb.avi';
model_file = '../../data/mouse/Data/multiPoint_all_model.mat';


M = load(model_file);

[readframe,nframes] = get_readframe_fcn(moviefile);
I = readframe(fnum);
I = histeq(rgb2gray_cond( I),M.H0);
Is =repmat({I},300);

bboxes=repmat([1 1 fliplr(size(I))],300,1);


M.prunePrm.th = 30;
[p,pR,~,fail,p_t]=...
  test_rcpr([],bboxes,Is,M.regModel,M.regPrm,M.prunePrm);

figure(401); imshow(I); hold on; scatter(p(:,5),p(:,11),'.'); scatter(p(:,6),p(:,12),'.');

%% Plot success points evolution.

q_t = p_t(1:300,:,:);
q_t(fail,:,:) = [];
figure;  
for ptn = 1:6
subplot(3,2,ptn);
imshow(I); axis on; hold on;
cc = jet(5);
for ndx = 1:5
  scatter(q_t(:,ptn,20*ndx-1),q_t(:,ptn+6,20*ndx-1),15,cc(ndx,:),'.');
end

end

%% all training labels.

all_labels = [];
idx = [];
for ndx = find(cvp.training');
  all_labels = cat(3,all_labels,A.labeledpos_perexp{ndx});
  npts = size(A.labeledpos_perexp{ndx},3);
  idx = [idx ndx*ones(1,npts)];
end

%invalid = any(any(isnan(all_labels),1),2);
invalid = [];
all_labels(:,:,invalid) = [];


%% 3d reconstruction

calibrationdata = load('/home/mayank/Dropbox/AdamVideos/multiPoint/CameraCalibrationParams20150217.mat');
J = load('../../data/mouse/Data/multiPoint_All_local.mat');
movie_file = fullfile(J.expdirs{1},'movie_comb.avi');
readframe = get_readframe_fcn(movie_file);
ex_frame = readframe(10);
sz = size(ex_frame,2);
for expi = 1:numel(J.expdirs)
  xL = permute(J.labeledpos_perexp{expi}(1,:,:),[2 3 1]);
  xR = permute(J.labeledpos_perexp{expi}(2,:,:),[2 3 1]);
  xR(:,1) = xR(:,1) - sz/2; % compensate for joint frame.
  [~,~,~,mouseid] = regexp(J.expdirs{expi},'M\d\d\d_');
  mouseid = mouseid{1}(1:end-1);
  mousendx = find(strcmp(calibrationdata.mice,mouseid));
  omcurr = calibrationdata.ompermouse(:,mousendx);
  Tcurr = calibrationdata.Tpermouse(:,mousendx);
  [X3d,X3d_right]  = stereo_triangulation(xL,xR,omcurr,Tcurr,calibrationdata.fc_left,...
         calibrationdata.cc_left,calibrationdata.kc_left,calibrationdata.alpha_c_left,...
         calibrationdata.fc_right,calibrationdata.cc_right,calibrationdata.kc_right,...
         calibrationdata.alpha_c_right);
  X3d = bsxfun(@minus,X3d,calibrationdata.origin(:,mousendx));       
  J.data3d{expi} = X3d;
end

%% correlation between 3-d coordinates and x,y coordinates

all2d = [];
all3d = [];
for expi = 1:numel(J.expdirs)
  all2d = cat(3,all2d,J.labeledpos_perexp{expi});
  all3d = cat(3,all3d,J.data3d{expi});
  
end

figure; 

for lr = 1:2
  for xy = 1:2
    for dd = 1:3
      subplot( 3,4, [(lr-1)*2+xy,dd]);
      scatter(all2d(lr,xy,:),all3d(dd,:),'.');
    end
  end
end