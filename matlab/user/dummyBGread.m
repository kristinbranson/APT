function [bg,bgdev] = dummyBGread(movfile,movifo)
bg = zeros(movifo.nr,movifo.nc);
bgdev = ones(size(bg));
