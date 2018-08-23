%%

projfile = '/groups/branson/bransonlab/mayank/temp/fly245_trk_2017212.lbl';
J = load(projfile,'-mat');
for ndx = 1:size(J.movieFilesAll,1)
  trk = [];
  for view = 1:2
    J.movieFilesAll{ndx,view} = strrep(J.movieFilesAll{ndx,view},'\','/');
    % for 220
%     J.movieFilesAll{ndx,view} = ['/groups/branson/bransonlab/mayank/temp/' J.movieFilesAll{ndx,view}(84:end)];
    % for 229 and 245
    J.movieFilesAll{ndx,view} = ['/groups/branson/bransonlab/mayank/temp/' J.movieFilesAll{ndx,view}(86:end)];
  % for 325
%     J.movieFilesAll{ndx,view} = ['/groups/branson/bransonlab/mayank/temp/' J.movieFilesAll{ndx,view}(78:end)];
    % for 421 and 
%     J.movieFilesAll{ndx,view} = ['/groups/branson/bransonlab/mayank/temp/' J.movieFilesAll{ndx,view}(127:end)];
    S = load([J.movieFilesAll{ndx,view}(1:end-3) 'trk'],'-mat');
    trk = cat(1,trk,S.pTrk);
  end
  J.labeledpos2{ndx} = trk;
end

save([projfile(1:end-4) '_local.lbl'],'-struct','J');



%%

projfile = '/groups/branson/bransonlab/mayank/temp/fly229_trk_2017212_local.lbl';
J = load(projfile,'-mat');
idx = {};
ll = [];
for ndx = 1:numel(J.labeledpos2)
  aa = abs(J.labeledpos2{ndx}(:,:,2:end)-J.labeledpos2{ndx}(:,:,1:end-1));
  ff = find(aa>30);
  [xx,yy,zz] = ind2sub(size(aa),ff);
  idx{ndx,1} = zz;
  idx{ndx,2} = xx;
  idx{ndx,3} = yy;
  fprintf('%d:%d\n',ndx,numel(zz));
  ll = [ll aa(:)];
end
%%
Q = load('/groups/branson/bransonlab/mayank/temp/stephenOut/temp__fly229_trial9_125fps__0001_side.mat');
f = figure();
nc = 5;
nr = 8;
thresh_perc = 99.5;
r_nonmax = 2;
start = 1120;
stop = 1140;
pt = 5;
for ndx = start:stop
  subplot(nc,nr,ndx-start+1);
  scorescurr = squeeze(Q.scores(ndx,:,:,pt));
%         threshcurr = thresh_nonmax_front;
  tscores = scorescurr(r_nonmax+1:end-r_nonmax,r_nonmax+1:end-r_nonmax); % safeguard against boundaries
  threshcurr = prctile(tscores(:),thresh_perc);

  imagesc(scorescurr);%>threshcurr);
  
end

%%

ll = [];
pfrontbest = nan(2,nlandmarks,T);
psidebest = nan(2,nlandmarks,T);
start = 130;
stop = 150;
for poslambdafixed = logspace(-1,5,5)
  X = permute(Psample(:,:,i,:),[1,4,2,3]);
  appearancecost = permute(w(:,i,:),[3,1,2]);
  [Xbest,idxcurr,totalcost,poslambdacurr] = ChooseBestTrajectory(X,-log(appearancecost),'dampen',dampen,'poslambda',poslambdafixed);
  for t = 1:T,
    pfrontbest(:,i,t) = psample_front(:,idxcurr(t),i,t);
    psidebest(:,i,t) = psample_side(:,idxcurr(t),i,t);
  end
  ll(:,end+1) = squeeze(pfrontbest(1,i,:));
  
end
ll(start:stop,:)

%%

