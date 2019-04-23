% mtrax2fixedtrx(inmatname,outmatname)
% inmatname: name of a matfile created by mtrax
% outmatname: name of a matfile to save results to
% results will be saved in the format returned by fixerrorsgui
function mtrax2fixedtrx(inmatname,outmatname)

load(inmatname);
load(inmatname,'angle'); % because matlab is retarded :)
idscurr = unique(identity);
x_pos = x_pos + 1; %#ok<NODEF>
y_pos = y_pos + 1; %#ok<NODEF>

% frame number
framenumber = zeros(size(x_pos));
j = 0;
for i = 1:length(ntargets),
  framenumber(j+(1:ntargets(i))) = i;
  j = j + ntargets(i);
end;

nflies = length(idscurr);
tmp = cell(1,nflies);
trx = struct('x',tmp,'y',tmp,'theta',tmp,'a',tmp,'b',tmp,'id',tmp,'firstframe',tmp,'arena',tmp,'nframes',tmp,'endframe',tmp);

for i = 1:nflies,
  id = idscurr(i);
  idx = identity == id;
  trx(i).x = x_pos(idx);
  trx(i).y = y_pos(idx);
  trx(i).theta = angle(idx);
  trx(i).a = maj_ax(idx);
  trx(i).b = min_ax(idx);
  trx(i).id = id;
  trx(i).firstframe = framenumber(find(idx,1));
  trx(i).arena.x = nan;
  trx(i).arena.y = nan;
  trx(i).arena.r = nan;
  trx(i).nframes = length(trx(i).x);
  trx(i).endframe = trx(i).nframes + trx(i).firstframe - 1;
end

save(outmatname,'trx');