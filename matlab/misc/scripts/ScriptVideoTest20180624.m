movfile = '/groups/huston/hustonlab/aviFrameStamped/frameNumAdded_Xvid.avi';

nreads = 3;
nframestest = 200;

%% read in frames using get_readframe_fcn, but in order, so calling read(f)
[readframe,nframes0,fid,headerinfo] = get_readframe_fcn(movfile);
nr = headerinfo.nr;
nc = headerinfo.nc;
nframes = min(nframestest,nframes0);

ims_seq = nan([nr*nc,nframes]);
times_seq = nan(2,nframes);
for f = 1:nframes,
  [im, times_seq(1,f)] = readframe(f);
  ims_seq(:,f) = vectorize(rgb2gray(im));
  times_seq(2,f) = get(headerinfo.readerobj,'CurrentTime');
end

%% read in frames in order using readFrame, so no seeking

ims_seq2 = nan([nr*nc,nframes]);
times_seq2 = nan(2,nframes);
readerobj = VideoReader(movfile);
for f = 1:nframes,
  times_seq2(1,f) = get(readerobj,'CurrentTime');
  ims_seq2(:,f) = vectorize(rgb2gray(readFrame(readerobj)));
  times_seq2(2,f) = get(readerobj,'CurrentTime');
  if ~hasFrame(readerobj),
    break;
  end
end

%% read in frames in random order

[readframe,~,~,headerinfo] = get_readframe_fcn(movfile);

ims_rar = nan([nr*nc,nreads,nframes]);
times_rar = nan(2,nreads,nframes);

readorder = nan(nreads,nframes);
for readi = 1:nreads,
  readorder(readi,:) = randperm(nframes);
end

for readi = 1:nreads,
  for i = 1:nframes,
    f = readorder(readi,i);
    [im, times_rar(1,readi,f)] = readframe(f);
    ims_rar(:,readi,f) = vectorize(rgb2gray(im));
    times_rar(2,readi,f) = get(headerinfo.readerobj,'CurrentTime');
  end
end

%% compare random order reads

D = zeros([nreads*(nreads-1)/2,nframes]);
for i = 1:nframes,
  D(:,i) = pdist(ims_rar(:,:,i)');
end

readij = nan(nreads*(nreads-1)/2,2);
tmp = squareform(1:nreads*(nreads-1)/2);
for i = 1:nreads*(nreads-1)/2,
  [readij(i,1),readij(i,2)] = find(tmp==i,1);
end
[idx,fs] = find(D>0);
fprintf('Non-zero differences for the following reads, frames:\n');
for i = 1:numel(fs),
  fprintf('%d vs %d, frame %d\n',readij(idx(i),1),readij(idx(i),2),fs(i));
end

%% compare sequential reads to random order reads

D_seq_rar = max(abs( ims_seq - reshape(ims_rar(:,1,:),[nr*nc,nframes]) ),[],1);
D_seq2_rar = max(abs( ims_seq2 - reshape(ims_rar(:,1,:),[nr*nc,nframes]) ),[],1);
D_seq_seq2 = max(abs( ims_seq2 - ims_seq ),[],1);

%% plot stuff

figure(1);
clf;
subplot(3,1,1);
plot(1:nframes,D>0,'-');
title('Difference between random reads');
fprintf('Max difference between reads, random access: %f\n',max(D(:)));
set(gca,'YLim',[-.05,1.05]);

subplot(3,1,2);
plot(1:nframes,D_seq_rar>0,'.');
hold on;
plot(1:nframes,D_seq2_rar>0,'.');
title('Frame offset between sequential and first random read');
set(gca,'YLim',[-.05,1.05]);
legend('seq, get_readframe_fcn','seq, readFrame');

subplot(3,1,3);
plot(D_seq_seq2 > 0,'.');
title('Difference between sequential1 and sequential2');
set(gca,'YLim',[-.05,1.05]);
