function [tf,nonintramov] = isIntraMovie(mov)
% tf: true if mov is an intra-movie
% nonintramov: if tf, corresponding regular movie. 
%   WARNING: separator in nonintramov could be changed relative to mov
%   depending on platform

[p,f,e] = fileparts(mov);
tf = strcmp(f(end-1:end),'_i');
if tf
  nonintramov = fullfile(p,[f(1:end-2) e]); % could change separator
else
  nonintramov = [];
end


