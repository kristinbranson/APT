%% test_choose_orientations

MINVEL = 3;
MINANGLEDIFF = 3*pi/4;
MINANGLECHANGE = 3*pi/4;
MININTERVALLENGTH = 5;
velocity_angle_weight = .05;
max_velocity_angle_weight = .25;
WINDOWRADIUS = 100;

tag = 'movie20071007_165047';
path = '/home/kristin/FLIES/data/walking_arena/highres/';
ext = '.fmf';
moviename = ['/media/data2/alice_data/groundturthing/',tag,ext];
addpath ../../exploremtraxresults
matname = [path,tag,'.mat'];
annname = [path,tag,ext,'.ann'];
load(matname);

x_pos = x_pos + 1;
y_pos = y_pos + 1;
[dataperfly,x_pos_norm,y_pos_norm,maj_ax_norm,min_ax_norm,identity] = createdata_perfile(moviename,matname,annname,false);

newdataperfly = dataperfly;
%for i = 1:length(dataperfly),
%  newdataperfly(i).theta = choose_orientations(dataperfly(i).x,dataperfly(i).y,...
%    dataperfly(i).theta,velocity_angle_weight,max_velocity_angle_weight);
%end

%% find frames, flies where the velocity and the orientation are very different
nframes = max(getstructarrayfield(newdataperfly,'endframe'));
isreverse = false(length(newdataperfly),nframes);
for i = 1:length(newdataperfly),
  dx = [0,diff(newdataperfly(i).x)];
  dy = [0,diff(newdataperfly(i).y)];
  velang = atan2(dy,dx);
  velmag = sqrt(dx.^2+dy.^2);
  isreverse(i,newdataperfly(i).firstframe:newdataperfly(i).endframe) = ...
    (velmag >= MINVEL) & (abs(angledist(velang,newdataperfly(i).theta)) > MINANGLEDIFF);
end

%% show these frames, flies
if strcmp(ext,'.fmf')
  [header_size, version, nr, nc, bytes_per_chunk, max_n_frames, data_format] = fmf_read_header( moviename );
  frame2file = bytes_per_chunk * (0:max_n_frames-1) + header_size;
  fid = fopen(moviename,'r');
  readframe = @() fmf_read_frame(fid,nr,nc,bytes_per_chunk,data_format);
else
  [nr,nc,max_n_frames,bgcenter,bgstd,frame2file,version,differencemode] = sbfmf_read_header(filename);
  fid = fopen(moviename,'r');
  readframe = @() sbfmf_read_frame(fid,bgcenter);
end

[starts,ends] = get_interval_ends(any(isreverse,1));
intervallength = ends - starts + 1;
isshort = intervallength < MININTERVALLENGTH;
starts(isshort) = [];
ends(isshort) = [];
if isempty(starts),
  fprintf('No sequences where fly moves backwards\n');
else
  fprintf('Showing sequences where fly moves backwards\n');
end
for j = 1:length(starts),
  clf;
  flies = find(any(isreverse(:,starts(j):ends(j)),2));
  startcurr = max(1,starts(j)-20);
  endcurr = min(ends(j)+20,nframes);
  
  fprintf('Showing frames %d to %d for flies: ',startcurr,endcurr);
  for i = 1:length(flies),
    fprintf('%d ',flies(i));
  end
  fprintf('\n');
  
  while 1,
    for i = startcurr:endcurr,

      showframe(fid,frame2file,newdataperfly,i,readframe,flies);
      zoominonflies(newdataperfly,flies,i,WINDOWRADIUS);
      title(sprintf('Frame %d',i));
      drawnow;
    end
    v = input('Type 1 when done');
    if ~isempty(v) && v == 1,
      break;
    end
  end
  
end

%% find sequences where there are temporal discontinuities in the
% orientation
ischange = false(length(newdataperfly),nframes);
for i = 1:length(newdataperfly),
  d = [0,abs(modrange(diff(newdataperfly(i).theta),-pi,pi))];
  ischange(i,newdataperfly(i).firstframe:newdataperfly(i).endframe) = d > MINANGLECHANGE;
end

[starts,ends] = get_interval_ends(any(ischange,1));
if isempty(starts),
  fprintf('No sequences with temporal discontinuities in orientation\n');
else
  fprintf('Showing sequences where there are temporal discontinuities in the orientation\n');
end
for j = 1:length(starts),
  clf;
  flies = find(any(ischange(:,starts(j):ends(j)),2));
  startcurr = max(1,starts(j)-20);
  endcurr = min(ends(j)+20,nframes);
  
  fprintf('Showing frames %d to %d for flies: ',startcurr,endcurr);
  for i = 1:length(flies),
    fprintf('%d ',flies(i));
  end
  fprintf('\n');
  
  while 1,
    for i = startcurr:endcurr,

      showframe(fid,frame2file,newdataperfly,i,readframe,flies);
      zoominonflies(newdataperfly,flies,i,WINDOWRADIUS);
      title(sprintf('Frame %d',i));
      drawnow;
    end
    v = input('Type 1 when done');
    if ~isempty(v) && v == 1,
      break;
    end
  end
  
end

fprintf('Showing everything\n');
for i = 1:nframes,
  showframe(fid,frame2file,newdataperfly,i,readframe);
  drawnow;
end;