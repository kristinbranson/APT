function [isOk,msgs] = paramChecker(sPrm)

isOk = init(sPrm,true);
msgs = {};

% if using background subtraction, bgreadfcn must be set
if isSubField(sPrm,{'ROOT','ImageProcessing','BackSub','Use'}) && ...
    sPrm.ROOT.ImageProcessing.BackSub.Use && ...
    isSubField(sPrm,{'ROOT','ImageProcessing','BackSub','BgReadFcn'}) && ...
    isempty(sPrm.ROOT.ImageProcessing.BackSub.BGReadFcn),
  isOk.ROOT.ImageProcessing.BackSub.Use = false;
  isOk.ROOT.ImageProcessing.BackSub.BGReadFcn = false;
  msgs{end+1} = 'If background subtraction is enabled, Background Read Function must be set';
end

% can either use background subtraction of histogram equalization, but not
% both
if isSubField(sPrm,{'ROOT','ImageProcessing','BackSub','Use'}) && ...
    sPrm.ROOT.ImageProcessing.BackSub.Use && ...
    isSubField(sPrm,{'ROOT','ImageProcessing','HistEq','Use'}) && ...
    sPrm.ROOT.ImageProcessing.HistEq.Use,
  isOk.ROOT.ImageProcessing.BackSub.Use = false;
  isOk.ROOT.ImageProcessing.HistEq.Use = false;
  msgs{end+1} = 'Background subtraction and histogram equalization cannot both be enabled.';
end

% if sPrm.ROOT.ImageProcessing.HistEq.NSampleH0 <= 0,
%   isOk.ROOT.ImageProcessing.HistEq.NSampleH0 = false;
%   msgs{end+1} = 'Histogram equalization: Num frames sample must be at least 1.';
% end

if isSubField(sPrm,{'ROOT','MultiAnimal','TargetCrop','ManualRadius'}) && ...
    sPrm.ROOT.MultiAnimal.TargetCrop.ManualRadius <= 0,
  isOk.ROOT.MultiAnimal.TargetCrop.ManualRadius = false;
  msgs{end+1} = 'Multitarget crop radius must be at least 1.';
end

% if sPrm.ROOT.MultiAnimal.NeighborMask.Use && ...
%     isempty(sPrm.ROOT.ImageProcessing.BackSub.BGReadFcn),
%   isOk.ROOT.MultiAnimal.NeighborMask.Use = false;
%   isOk.ROOT.ImageProcessing.BackSub.BGReadFcn = false;
%   msgs{end+1} = 'If masking neighbors is enabled, Background Read Function must be set';
% end

% if sPrm.ROOT.MultiAnimal.NeighborMask.FGThresh < 0,
%   isOk.ROOT.MultiAnimal.NeighborMask.FGThresh = false;
%   msgs{end+1} = 'Mask Neighbors Foreground Threshold must be non-negative.';
% end

% if sPrm.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta && ...
%     strcmp(sPrm.ROOT.CPR.RotCorrection.OrientationType,'fixed')
%   msgs{end+1} = 'CPR OrientationType cannot be ''fixed'' if aligning target crops using trx.theta.';
%   isOk.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta = false;
%   isOk.ROOT.CPR.RotCorrection.OrientationType = false;
% end

function out = init(in,val)

if isstruct(in),
  fns = fieldnames(in);
  for i = 1:numel(fns),
    out.(fns{i}) = init(in.(fns{i}),val);
  end
else
  out = val;
end
