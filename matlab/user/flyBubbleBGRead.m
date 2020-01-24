function [bg,bgdev,n_bg_std_thresh_low] = flyBubbleBGRead(movfile,movifo)

annfile = fullfile(fileparts(movfile),'movie.ufmf.ann');
[bg,bgdev,n_bg_std_thresh_low] = ...
  read_ann(annfile,'background_center','background_dev','n_bg_std_thresh_low');
imsz = [movifo.nr movifo.nc];
bg = reshape(bg,imsz);
bgdev = reshape(bgdev,imsz);
bg = bg'; % annfile
bgdev = bgdev'; 
