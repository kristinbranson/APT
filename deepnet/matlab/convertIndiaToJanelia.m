J = load('~/bransonlab/PoseTF/headTracking/FlyHeadStephenCuratedData.mat');
for ndx = 1:numel(J.vid1files),
J.vid1files{ndx} = ['/groups/branson/bransonlab/mayank/' J.vid1files{ndx}(19:end)];
J.expdirs{ndx} = ['/groups/branson/bransonlab/mayank/' J.expdirs{ndx}(19:end)];
J.vid2files{ndx} = ['/groups/branson/bransonlab/mayank/' J.vid2files{ndx}(19:end)];
if ~exist(J.expdirs{ndx},'file'),fprintf('%s\n',J.expdirs{ndx});end
if ~exist(J.vid1files{ndx},'file'),fprintf('%s\n',J.vid1files{ndx});end
if ~exist(J.vid2files{ndx},'file'),fprintf('%s\n',J.vid2files{ndx});end

end

save('~/bransonlab/PoseTF/headTracking/FlyHeadStephenCuratedData_Janelia.mat','-struct','J','-v7.3');