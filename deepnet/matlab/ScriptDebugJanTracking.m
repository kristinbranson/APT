A = load('/nrs/branson/mayank/janOut/151106_03_016_02_C001H001S.mat');
B = load('~/temp/delete_10.mat');

l1 = permute(A.locs(:,:,1,:),[1 4 2 3]);
l2 = permute(B.pTrk,[2 3 1]);

%
rfn = get_readframe_fcn(A.expname);

% %%
% figure; imshow(rfn(30));
% hold on;
% id = 4;
% scatter(l1(30,1,id),l1(30,2,id),'.');
%
gg = sqrt(sum( (l1-l2).^2,2));

%%


[sortd,sortid] = sort(gg(:),'descend');
[fr,cc,i] = ind2sub(size(gg),sortid(1:1000));

done = false(size(gg));
n2show = 10;
f = figure(1);
for ndx = 1:1000
  if done(fr(ndx),1,i(ndx)), continue; end
%   if i(ndx)==3; continue; end
  clf(f);
  ix = i(ndx);
  for idx = -n2show:n2show-1
    fx = fr(ndx)+idx;
    if fx<0; continue; end
    if fx>size(gg,1), continue; end
    
    subplot(5,5,idx+n2show+1);
    imshow(rfn(fx)); axis on;
    hold on;
    scatter(l1(fx,1,ix),l1(fx,2,ix),'b.');
    scatter(l2(fx,1,ix),l2(fx,2,ix),'r.');
    hold off;
    done(fx,1,ix) = true;
  end
  title(sprintf('id:%d fr:%d dist:%.2f',i(ndx),fr(ndx),sortd(ndx)))
  pause;
end

%%

tt = 496;
i = 3;
f = figure(3);
for idx = -10:9
  subplot(5,4,idx+11);
  curs = squeeze(rdf.scores(tt+idx,:,:,i));
  imagesc(curs);
  hold on;
  scatter(Xbest(1,tt+idx)/4,Xbest(2,tt+idx)/4,100,'.')
  
end