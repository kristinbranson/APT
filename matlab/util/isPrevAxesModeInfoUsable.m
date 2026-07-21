function result = isPrevAxesModeInfoUsable(modeInfoStruct)
% Returns true iff modeInfoStruct carries a usable frozen prev-axes target spec.
% Legacy projects saved in FROZEN mode can have a ModeInfo struct whose identity
% fields (iMov, frm, iTgt) are empty (and which carry only the obsolete im/isrotated
% fields).  Such a struct cannot be turned into a valid CorePrevAxesTargetSpec, so we
% treat it as unset rather than letting the constructor assert.
result = isstruct(modeInfoStruct) && ...
         isfield(modeInfoStruct, 'iMov') && isScalarFiniteNonneg(modeInfoStruct.iMov) && ...
         isfield(modeInfoStruct, 'frm') && isScalarFiniteNonneg(modeInfoStruct.frm) && ...
         isfield(modeInfoStruct, 'iTgt') && isScalarFiniteNonneg(modeInfoStruct.iTgt) ;
end  % function
