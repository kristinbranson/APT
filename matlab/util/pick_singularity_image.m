function result = pick_singularity_image(backend, netmode)
  % Returns the correct singularity image path to use, based on the backend and
  % netmode.
  if iscell(netmode),
    netmode = netmode{end};
  end
  if netmode.isObjDet ,
    result = backend.singularity_detection_image_path ;
  else
    result = backend.singularity_image_path ;
  end
end
