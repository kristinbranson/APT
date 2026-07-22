function [isOk,msgs] = paramChecker(sPrm)

isOk = init(sPrm,true);
msgs = {};

if isSubField(sPrm,{'ROOT','MultiAnimal','TargetCrop','ManualRadius'}) && ...
    sPrm.ROOT.MultiAnimal.TargetCrop.ManualRadius <= 0,
  isOk.ROOT.MultiAnimal.TargetCrop.ManualRadius = false;
  msgs{end+1} = 'Multitarget crop radius must be at least 1.';
end

function out = init(in,val)

if isstruct(in),
  fns = fieldnames(in);
  for i = 1:numel(fns),
    out.(fns{i}) = init(in.(fns{i}),val);
  end
else
  out = val;
end
