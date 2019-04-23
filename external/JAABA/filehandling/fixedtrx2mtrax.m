% fixedtrx2mtrax(inmatname,outmatname)
% inmatname: name of a matfile created by fixerrorsgui
% outmatname: name of a matfile to save results to
% results will be saved in the format returned by mtrax
function fixedtrx2mtrax(inmatname,outmatname)

load(inmatname);
nflies = length(trx);
nframes = 0;
for fly = 1:nflies,
  nframes = max(nframes,trx(fly).endframe);
end
x_pos = nan(nflies,nframes);
y_pos = nan(nflies,nframes);
angle = nan(nflies,nframes);
maj_ax = nan(nflies,nframes);
min_ax = nan(nflies,nframes);
identity = repmat((1:nflies)',[1,nframes]);
for fly = 1:nflies,
  x_pos(fly,trx(fly).firstframe:trx(fly).endframe) = trx(fly).x;
  y_pos(fly,trx(fly).firstframe:trx(fly).endframe) = trx(fly).y;
  angle(fly,trx(fly).firstframe:trx(fly).endframe) = trx(fly).theta;
  maj_ax(fly,trx(fly).firstframe:trx(fly).endframe) = trx(fly).a;
  min_ax(fly,trx(fly).firstframe:trx(fly).endframe) = trx(fly).b;
end
keep = ~isnan(x_pos);
x_pos = x_pos(keep)'-1;
y_pos = y_pos(keep)'-1;
angle = angle(keep)';
maj_ax = maj_ax(keep)';
min_ax = min_ax(keep)';
identity = identity(keep)' - 1;
ntargets = sum(double(keep),1);

save(outmatname,'x_pos','y_pos','angle','maj_ax','min_ax','identity','ntargets');