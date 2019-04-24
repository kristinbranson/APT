function trxGraphicParams=trxGraphicParamsFromAnimalType(animalType)
% Given the animalType, a string, load the animalList.xml file and get the
% markerParams out of it.  Append a few other things to make it into a
% trxGraphicParams.

if isdeployed,
  filename = deployedRelative2Global('params/animalList.xml');
else
  thisFileNameAbs=mfilename('fullpath');
  thisFileDirAbs=fileparts(thisFileNameAbs);
  jaabaDirAbs=fileparts(thisFileDirAbs);
  filename = fullfile(jaabaDirAbs,'perframe','params','animalList.xml');
end
animalParamsFromAnimalType = ReadXMLParams(filename);

markerParamsRaw=struct([]);
if isfield(animalParamsFromAnimalType,animalType)
  thing1=animalParamsFromAnimalType.(animalType);
  if isfield(thing1,'plot')
    thing2=thing1.plot;
    if isfield(thing2,'trx')
      markerParamsRaw=thing2.trx;
    end
  end
end
markerParams=cookMarkerParams(markerParamsRaw);
trxGraphicParams=markerParams;
trxGraphicParams.colormap = 'jet';
trxGraphicParams.colormap_multiplier = 0.7;

end
