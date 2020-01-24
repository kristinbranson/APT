function frms = bouts2frames(bouts)
% Convert bouts to vector of frames.
%
% bouts: [nbout x 2]. col1: bout starts. col2: bout one-past-ends.
% 
% frms: vector of frames included in bouts.

[nbout,d] = size(bouts);
assert(d==2);
frms = zeros(1,0);
for ibout=1:nbout
  frms = [frms bouts(ibout,1):bouts(ibout,2)-1]; %#ok<AGROW>
end