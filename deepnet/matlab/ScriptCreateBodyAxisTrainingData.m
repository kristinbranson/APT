fid = fopen('~/work/PoseEstimationData/Stephen//bodyAxisTrainingData.csv');
K = textscan(fid,'%s %s %s',inf,'Delimiter','"');
fclose(fid);

%%

npermovie = 100;
cc = jet(5);
J = struct;
ts = zeros(1,0);
pts = zeros(2,2,5,0);
expidx = zeros(1,0);
vid1files = {};
vid2files = {};
for ndx = 1:numel(K{1})
%%
  kfile = K{1}{ndx}(44:end-1);
  local_kfile = fullfile('/home/mayank/Dropbox/PoseEstimation/Stephen',kfile);
  kd = load(local_kfile);
  dlt_side = kd.data.cal.coeff.DLT_1;
  dlt_front = kd.data.cal.coeff.DLT_2;
  eval_str = ['ppts = kd.' K{2}{ndx} ';'];
  eval(eval_str);
  fpts = nan(2,size(ppts,1));
  spts = nan(2,size(ppts,1));
  [fpts(1,:),fpts(2,:)] = dlt_3D_to_2D(dlt_front,ppts(:,1),ppts(:,2),ppts(:,3));
  [spts(1,:),spts(2,:)] = dlt_3D_to_2D(dlt_side,ppts(:,1),ppts(:,2),ppts(:,3));
  
  bdir = fileparts(local_kfile);
  bdir = fileparts(bdir);
  [bdir,fname] = fileparts(bdir);
  movf = fullfile(bdir,fname,[fname '_trial1'],'C002H001S0001','C002H001S0001_c.avi');
  if ~exist(movf,'file'),
    movf = fullfile(bdir,fname,[fname '_300ms_stimuli'],'C002H001S0001','C002H001S0001_c.avi');
  end
  movs = [movf(1:end-33) '/C001H001S0001/C001H001S0001_c.avi'];
  if ~exist(movf,'file'),
    fprintf('%d movie %s doesnt exist',ndx,movf);
  end
  [readf,ffrms,fid] = get_readframe_fcn(movf);
  frf = readf( str2double(K{3}{ndx}(2:end)));

  [reads,sfrms,fid] = get_readframe_fcn(movs);
  frs = reads( str2double(K{3}{ndx}(2:end)));
  
%   if ndx>1,pause; end
%%    
  figure(1); clf;
  subplot(1,2,1);
  hold off;
  imshow(frf); hold on;
  scatter(fpts(1,:),fpts(2,:),50,cc,'.');
  title(sprintf('%s',fname));
  subplot(1,2,2);
  hold off;
  imshow(frs); hold on;
  scatter(spts(1,:),spts(2,:),50,cc,'.');
  
%%
  pts(:,1,:,((ndx-1)*npermovie+1):(ndx*npermovie)) = repmat(reshape(spts,[2 1 5]),[1 1 1 npermovie]);
  pts(:,2,:,((ndx-1)*npermovie+1):(ndx*npermovie)) = repmat(reshape(fpts,[2 1 5]),[1 1 1 npermovie]);
  J.expdirs{ndx} = local_kfile;
  vid1files{ndx} = movs;
  vid2files{ndx} = movf;
  expidx(((ndx-1)*npermovie+1):(ndx*npermovie)) = ndx;
  ts(((ndx-1)*npermovie+1):(ndx*npermovie)) = linspace(1,ffrms,npermovie);
  
end

J.vid1files = vid1files;
J.vid2files = vid2files;
J.expidx = expidx;
J.ts = ts;
J.pts = pts;


%% Test the labels


for ndx = 1:numel(J.vid1files)
  [readf,ffrms,fid] = get_readframe_fcn(J.vid2files{ndx});
  [reads,sfrms,fid] = get_readframe_fcn(J.vid1files{ndx});

  ii = find(J.expidx == ndx);
  bdir = fileparts(J.vid1files{ndx});
  bdir = fileparts(bdir);
  bdir = fileparts(bdir);
  [bdir,fname] = fileparts(bdir);
  
  for idx = 1:20:numel(ii)
    frf = readf(J.ts(ii(idx)));
    frs = reads(J.ts(ii(idx)));
    fpts = squeeze(J.pts(:,2,:,ii(idx)));
    spts = squeeze(J.pts(:,1,:,ii(idx)));    
    if idx ==1,
      fprintf('%f,%f\n',size(frf,1),size(frf,2));
    end
  %%    
    figure(1); clf;
    subplot(1,2,1);
    hold off;
    imshow(frf); hold on;
    scatter(fpts(1,:),fpts(2,:),50,cc,'.');
    plot([128 128],[0 512],'-');
    plot([640 640],[0 512],'-');
    title(sprintf('%s',fname));
    subplot(1,2,2);
    hold off;
    imshow(frs); hold on;
    scatter(spts(1,:),spts(2,:),50,cc,'.');
    plot([128 128],[0 512],'-');
    plot([640 640],[0 512],'-');
    hold off;
  end  
end