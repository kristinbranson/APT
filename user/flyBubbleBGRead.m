function [bg,bgdev] = flyBubbleBGRead(movfile,movifo)

annfile = fullfile(fileparts(movfile),'movie.ufmf.ann');
[bg,bgdev] = read_ann2(annfile,'background_center','background_dev');
imsz = [movifo.nr movifo.nc];
bg = reshape(bg,imsz);
bgdev = reshape(bgdev,imsz);
