function tracks = readmtrax(filename)

load(filename);

% because matlab is retarded
load(filename,'angle');

D = 8;

% total number of frames
nframes = length(ntargets);

% data. tracks{t} are the positions of the targets in frame t
% tracks{t} is D x ntargets(t)
tracks = cell(1,nframes);

j0 = 1;
for i = 1:nframes,
  
  j1 = j0 + ntargets(i) - 1;
  tracks{i} = zeros(D,ntargets(i));
  tracks{i}(1,:) = identity(j0:j1);
  tracks{i}(2,:) = x_pos(j0:j1);
  tracks{i}(3,:) = y_pos(j0:j1);
  tracks{i}(4,:) = maj_ax(j0:j1);
  tracks{i}(5,:) = min_ax(j0:j1);
  tracks{i}(6,:) = angle(j0:j1);

  j0 = j1 + 1;
  
  if i == 1,
    continue;
  end;
  
  % compute velocity
  for j = 1:ntargets(i),
    id = tracks{i}(1,j);
    k = find(tracks{i-1}(1,:)==id);
    if isempty(k),
      continue
    end;
    tracks{i}(7,j) = tracks{i}(2,j) - tracks{i-1}(2,k);
    tracks{i}(8,j) = tracks{i}(3,j) - tracks{i-1}(3,k);
  end;
  
end;