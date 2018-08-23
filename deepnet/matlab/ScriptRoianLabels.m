Q = load('/groups/branson/bransonlab/roian/apt_testing/head_tail.lbl','-mat');

%%

nmov = numel(Q.labeledpos);
newnames = cell(nmov,1);
nlabels = zeros(nmov,1);
ddir = '/groups/branson/bransonlab/roian';
expnum = nan(nmov,1);
mousenum = nan(nmov,1);
sz = {};
for ndx = 1:nmov
  curmov = Q.movieFilesAll{ndx,1};
  curmov = strrep(curmov,'\','/');
  curmov = [ddir curmov(9:end)];
  if ~exist(curmov,'file')
    fprintf('%d %s doesnt exist\n',ndx,curmov);
  end
  newnames{ndx,1} = curmov;
  nlabels(ndx) = nnz(all(all(~isnan(Q.labeledpos{ndx}),1),2));
  path_parts = strsplit(curmov,'/');
  expname = path_parts{end};
  [~,~,~,~,mm] = regexp(expname,'experiment_(\d+)_.*_Mouse(\d).avi');
%   expnum(ndx,1) = str2double(mm{1}{1});
%   mousenum(ndx,1) = str2double(mm{1}{2});
%   [rfn,a,b,c] = get_readframe_fcn(newnames{ndx});
%   sz{ndx} = [c.Width,c.Height];
end
Q.movieFilesAll = newnames;

%%
dstr = datestr(now,'yyyymmdd');
save(['/groups/branson/bransonlab/mayank/PoseTF/data/roian/head_tail_' dstr '.lbl'],'-struct','Q','-v7.3');


%%

dd = [];
for ndx =1:9,
  aa = cellfun(@(x) ~isempty(x),Q.labeledpostag{1});
  dd(ndx) = nnz(aa(:));
end