addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;

addpath /groups/branson/bransonlab/projects/flyHeadTracking/code/
addpath /groups/branson/home/bransonk/tracking/code/Ctrax/matlab/netlab

%%

J = load(fullfile('/groups/branson/bransonlab/mayank/PoseTF/RomainLeg','RomainCombined_fixed_fixedbabloo_20170410.lbl'),'-mat');

outdir = '/nrs/branson/mayank/romain/out';
resdir = '/nrs/branson/mayank/romain/results';

%%
for ndx = 2:size(J.movieFilesAll,1)

  view = 1;
  moviefile = J.movieFilesAll{ndx,view};
  [~,b,c] = fileparts(moviefile);
  scorefile = fullfile(outdir,[b c '_side1.h5']);

  outfile = fullfile(outdir,[b '.mat']);
  
  aa = h5info(scorefile);
  for ix = 1:numel(aa.Datasets)
    if strcmp(aa.Datasets(ix).Name,'scores')
      sz = aa.Datasets(ix).Dataspace.Size;
      nframes = sz(5);
      npts = sz(2);
      xsz = sz(3);
      ysz = sz(4);
    end
  end

  t_locs = cell(1,3);
  in_locs = cell(1,3);
  mrf_scores = cell(1,3);
  final_scores = cell(1,3);
  for view = 1:3
   t_locs{view} = zeros(nframes,npts,2);
   moviefile = J.movieFilesAll{ndx,view};
   [~,b,c] = fileparts(moviefile);
   
   if view == 1
     extra_str = '_side1';
   elseif view == 2
     extra_str = '_side2';
   elseif view == 3
     extra_str = '_bottom';
   end
   
   scorefile = fullfile('/nrs/branson/mayank/romain/out',[b c extra_str '.h5']);
   cur_locs = h5read(scorefile,'/locs');
   
   in_locs{view} = permute(cur_locs,[4,3,2,1]);
   mrf_scores{view} = nan(nframes,npts);
   final_scores{view} = nan(nframes,npts);
  end

  chunkSize = 1000;
  nblocks = ceil(nframes/chunkSize);
  for curblk = 1:nblocks

    start = (curblk-1)*chunkSize+1;
    stop = min(nframes,curblk*chunkSize);
    readsize = stop-start+1;

    scores = cell(1,3);
    for view = 1:3
      moviefile = J.movieFilesAll{ndx,view};
      [~,b,c] = fileparts(moviefile);

      if view == 1
        extra_str = '_side1';
      elseif view == 2
        extra_str = '_side2';
      elseif view == 3
        extra_str = '_bottom';
      end

      scorefile = fullfile('/nrs/branson/mayank/romain/out',[b c extra_str '.h5']);

      cur_scores = h5read(scorefile,'/scores',[1,1,1,1,start],[1,inf,inf,inf,readsize]);
      cur_scores = permute(cur_scores,[5,4,3,2,1]);
      scores{view} = cur_scores;
    end

    curTracks = compute3Dfrom2DRomain(scores,J.viewCalibrationData{ndx});
    
    for v = 1:3
      t_locs{v}(start:stop,:,:) = permute(curTracks.pbest_re{v},[3,2,1]);

      for tm = start:stop
        tndx = tm-start+1;
        for curp = 1:npts
          xloc = round(in_locs{v}(tm,curp,1,1)/4);
          yloc = round(in_locs{v}(tm,curp,1,2)/4);
          if xloc>0 && yloc>0 && xloc<=size(scores{v},3) && yloc<=size(scores{v},2)
            mrf_scores{v}(tndx,curp) = scores{v}(tndx,yloc,xloc,curp);
          end
          xloc = max(1,round(t_locs{v}(tndx,curp,1)/4));
          yloc = max(1,round(t_locs{v}(tndx,curp,2)/4));
          if xloc>0 && yloc>0 && xloc<=size(scores{v},3) && yloc<=size(scores{v},2)
            final_scores{v}(tndx,curp) = scores{v}(tndx,yloc,xloc,curp);
          end
        end
      end      
    end
    fprintf('.');
  end
  fprintf('\n');
  R = struct;
  R.R = cell(1,3);
  for v = 1:3
    R.R{v}.pd_locs = permute(in_locs{v}(:,:,2,:),[1,2,4,3]);
    R.R{v}.mrf_locs = permute(in_locs{v}(:,:,1,:),[1,2,4,3]);
    R.R{v}.final_locs = t_locs{v};
    R.R{v}.mrf_scores = mrf_scores{v};
    R.R{v}.final_scores = final_scores{v};
    R.R{v}.labels = permute(J.labeledpos{ndx}( (1:18)+(v-1)*18,:,:),[3,1,2]);
  end
save(outfile,'-struct','R','-v7.3');

end


% %%
% 
% rfn = get_readframe_fcn(moviefile);
% startat = 1001;
% K = h5read(scorefile,'/scores',[1,1,1,1,startat],[1,18,64,168,1000]);
% K = permute(K,[5,4,3,2,1]);
% %%
% f = figure(1);
% for ndx = 1:100:1000
%   figure(f);
%   ii = rfn(ndx+startat);
%   subplot(4,5,1);
%   imshow(ii);
%   for idx = 1:18
%     subplot(4,5,idx+1);
%     sim = squeeze(K(ndx,:,:,idx));
%     sim(1,1)= 1;
%     sim(end,end) = -1;
%     sim(sim>1) = 1;
%     sim(sim<-1) = -1;
%     imagesc(sim); axis image;
%   end
%   
%   pause;
%   
% end

%% create results without tracking


J = load(fullfile('/groups/branson/bransonlab/mayank/PoseTF/RomainLeg','RomainCombined_fixed_fixedbabloo_20170410.lbl'),'-mat');

outdir = '/nrs/branson/mayank/romain/out';
resdir = '/nrs/branson/mayank/romain/results';

%%
for ndx = 2:size(J.movieFilesAll,1)

  view = 1;
  moviefile = J.movieFilesAll{ndx,view};
  [~,b,c] = fileparts(moviefile);

  outfile = fullfile(resdir,[b '.mat']);

  in_locs = cell(1,3);
  for view = 1:3
   t_locs{view} = zeros(nframes,npts,2);
   moviefile = J.movieFilesAll{ndx,view};
   [~,b,c] = fileparts(moviefile);
   
   if view == 1
     extra_str = '_side1';
   elseif view == 2
     extra_str = '_side2';
   elseif view == 3
     extra_str = '_bottom';
   end
   
   scorefile = fullfile('/nrs/branson/mayank/romain/out',[b c extra_str '.h5']);
   cur_locs = h5read(scorefile,'/locs');
   in_locs{view} = permute(cur_locs,[4,3,2,1]);
   
  end

  R = struct;
  R.R = cell(1,3);
  for v = 1:3
    R.R{v}.pd_locs = permute(in_locs{v}(:,:,2,:),[1,2,4,3]);
    R.R{v}.mrf_locs = permute(in_locs{v}(:,:,1,:),[1,2,4,3]);
    R.R{v}.labels = permute(J.labeledpos{ndx}( (1:18)+(v-1)*18,:,:),[3,1,2]);
  end
save(outfile,'-struct','R','-v7.3');

end
