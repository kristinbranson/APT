
% load ppdata.mat;
% if ndims(X) >= 5,
%   X = permute(X,[1,2,3,5,4]);
% end

load ppdata_sh.mat;
mov = unique(tblMFT.mov);
assert(numel(mov)==1);
caldata = viewCalibrationData{mov};
assert(~isempty(caldata));

projfile = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4523_gt080618_made20180627.lbl';

%% 
ppobj = PostProcess();
ppobj.SetUseGeometricError(false);
ppobj.SetCalibrationData(caldata);
ppobj.SetSampleData(X);
starttime = tic;
ppobj.algorithm = 'median';
ppobj.run();
fprintf('Time to run median: %f\n',toc(starttime));
starttime = tic;
ppobj.SetJointSamples(false);
ppobj.algorithm = 'maxdensity';
ppobj.run();
fprintf('Time to run maxdensity_indep: %f\n',toc(starttime));
starttime = tic;
ppobj.SetJointSamples(true);
ppobj.algorithm = 'maxdensity';
ppobj.run();
fprintf('Time to run maxdensity_joint: %f\n',toc(starttime));

postdata = ppobj.postdata;

starttime = tic;
ppobj.SetJointSamples(false);
ppobj.algorithm = 'viterbi';
ppobj.SetViterbiParams('poslambda',.002,'dampen',.25,'misscost',inf);
ppobj.run();
fprintf('Time to run viterbi_indep_nomiss: %f\n',toc(starttime));
postdata.viterbi_indep_nomiss = ppobj.postdata.viterbi_indep;


starttime = tic;
ppobj.SetJointSamples(true);
ppobj.algorithm = 'viterbi';
ppobj.SetViterbiParams('poslambda',.002,'dampen',.25,'misscost',inf);
ppobj.run();
fprintf('Time to run viterbi_joint_nomiss: %f\n',toc(starttime));
postdata.viterbi_joint_nomiss = ppobj.postdata.viterbi_joint;

starttime = tic;
ppobj.SetJointSamples(false);
ppobj.algorithm = 'viterbi';
ppobj.SetViterbiParams('poslambda',.002,'dampen',.25,'misscost',5);
ppobj.run();
fprintf('Time to run viterbi_indep_miss: %f\n',toc(starttime));
postdata.viterbi_indep_miss = ppobj.postdata.viterbi_indep;


starttime = tic;
ppobj.SetJointSamples(true);
ppobj.algorithm = 'viterbi';
ppobj.SetViterbiParams('poslambda',.002,'dampen',.25,'misscost',5);
ppobj.run();
fprintf('Time to run viterbi_joint_miss: %f\n',toc(starttime));
postdata.viterbi_joint_miss = ppobj.postdata.viterbi_joint;


%%

figure(2);
clf;
[N,K,npts,d] = size(ppobj.sampledata.x);

hax = createsubplots(npts,d,[.02,.02;.03,.01]);
hax = reshape(hax,[npts,d]);
fns = fieldnames(postdata);
colors = lines(numel(fns));
h = nan(1,numel(fns));

for ipt = 1:npts,
  for id = 1:d,
    for i = 1:numel(fns),
      h(i) = plot(hax(ipt,id),1:N,postdata.(fns{i}).x(:,ipt,id),'LineWidth',2);
      hold(hax(ipt,id),'on');
    end
    %h(numel(fns)+1) = plot(hax(ipt,id),1:N,tmp2.pTrk(:,sub2ind([npts,d],ipt,id)),':','LineWidth',2);
    axisalmosttight([],hax(ipt,id));
    box(hax(ipt,id),'off');
  end
end
set(hax,'XLim',[0,N+1]);%,'YLim',[min(ppobj.sampledata.x(:)),max(ppobj.sampledata.x(:))]);
set(hax(1:end-1,:),'XTickLabel',{});%,'YTickLabel',{});
legend(h,fns,'Interpreter','none');
linkaxes(hax,'x');