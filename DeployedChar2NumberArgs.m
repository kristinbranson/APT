function args = DeployedChar2NumberArgs(args,varargin)

assert(mod(numel(args),2) == 0);
idxnumeric = find(ismember(args(1:2:end-1),varargin));
for i = idxnumeric(:)',
  if ischar(args{2*i}),
    args{2*i} = str2double(args{2*i});
  end
end