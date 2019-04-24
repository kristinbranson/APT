function trxGraphicParams=cookTrxGraphicParams(trxGraphicParamsRaw)

trxGraphicParams=cookMarkerParams(trxGraphicParamsRaw);
if isfield(trxGraphicParamsRaw,'colormap') && ~isempty(trxGraphicParamsRaw.colormap)
  trxGraphicParams.colormap = trxGraphicParamsRaw.colormap;  
else
  trxGraphicParams.colormap = 'jet';
end
if isfield(trxGraphicParamsRaw,'colormap_multiplier') && ~isempty(trxGraphicParamsRaw.colormap_multiplier)
  trxGraphicParams.colormap_multiplier = trxGraphicParamsRaw.colormap_multiplier;  
else
  trxGraphicParams.colormap_multiplier = 0.7;
end

if isfield(trxGraphicParamsRaw,'assignment') && ~isempty(trxGraphicParamsRaw.assignment)
  trxGraphicParams.assignment = trxGraphicParamsRaw.assignment;  
else
  trxGraphicParams.assignment = 'random';
end


end
