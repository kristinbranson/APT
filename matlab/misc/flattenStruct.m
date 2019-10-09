function sdst = flattenStruct(ssrc,sdst,path)

if nargin < 2,
  sdst = struct;
end
if nargin < 3,
  path = {};
end

fns = fieldnames(ssrc);
for i = 1:numel(fns),
  if isstruct(ssrc.(fns{i})),
    sdst = flattenStruct(ssrc.(fns{i}),sdst,[path,fns(i)]);
  else
    fn = fns{i};
    if isfield(sdst,fn),
      for i = 1:numel(path),
        fn = [path{end-i+1},'_',fn];
        if ~isfield(sdst,fn),
          break;
        end
      end
    end        
    assert(~isfield(sdst,fn));
    sdst.(fn) = ssrc.(fns{i});
  end
end