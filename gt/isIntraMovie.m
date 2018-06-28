function [tf,nonintramov] = isIntraMovie(mov)
% tf: true if mov is an intra-movie
% nonintramov: if tf, corresponding regular movie 

[p,f,e] = fileparts(mov);
tf = strcmp(f(end-1:end),'_i');
if tf
  nonintramov = fullfile(p,[f(1:end-2) e]);
else
  nonintramov = [];
end


