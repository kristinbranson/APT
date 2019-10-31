function [s,nsetfn,ts_struct] = setStructLeaf(s,fn,v,varargin)

[ts_struct,ts_set,warnfun] = myparse(varargin,'ts_struct',[],'ts_set',0,...
  'warnfun',@(fn) sprintf('Collision collapsing parameter %s, using most recent value',fn));

if isempty(ts_struct),
  ts_struct = struct;
end

fns = fieldnames(s);
nsetfn = 0;
for i = 1:numel(fns),
  if isstruct(s.(fns{i})),
    if ~isfield(ts_struct,fns{i}),
      ts_struct.(fns{i}) = struct;
    end
    [s.(fns{i}),nsetcurr,ts_struct.(fns{i})] = setStructLeaf(s.(fns{i}),fn,v,'ts_struct',ts_struct.(fns{i}),'ts_set',ts_set,'warnfun',warnfun);
    nsetfn = nsetfn + nsetcurr;
  else
    if ~isfield(ts_struct,fns{i}),
      ts_struct.(fns{i}) = -1;
    end
    if strcmp(fn,fns{i}),
      if ts_struct.(fns{i}) >= 0 && (numel(s.(fn)) ~= numel(v) || ~all(s.(fn)(:) == v(:))),
        warningNoTrace(warnfun(fn));
      end
      if ts_set > ts_struct.(fns{i}),
        ts_struct.(fns{i}) = ts_set;
        s.(fn) = v;
      end
      nsetfn = nsetfn + 1;
    end
  end
end