infile = '~/bransonlab/PoseEstimationData/Stephen/folders2track.txt';
fid = fopen(infile,'r');

X = textscan(fid,'%s');
Xf = X{1}(2:2:end);
Xs = X{1}(1:2:end);
fclose(fid);


%%

for ndx = 1:numel(Xf)
  matfile = [Xf{ndx} '_3Dres.mat'];
  if ~exist(matfile,'file'),
    fprintf('3D mat file dont exist for %s\n',Xf{ndx})
    continue;
  end
  [~,fname] = fileparts(Xf{ndx});
  ftrx = fullfile(Xf{ndx}, [fname '_c.trk']);
  
  [~,sname] = fileparts(Xs{ndx});
  strx = fullfile(Xs{ndx}, [sname '_c.trk']);
  
  convertResultsToTrx(matfile,ftrx,strx);
  fprintf('Done conversion for %d %s\n',ndx,Xf{ndx});
end