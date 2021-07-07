function test_quick(proj_file,varargin)
tobj = testAPT('name','dummy');

[backend,net,aws_params] = myparse(varargin,...
  'backend','docker',...
  'net','mdn',...
  'aws_params',{});

tobj.setup_lbl(proj_file);
lObj = tobj.lObj;
% s=lObj.trackGetParams;
% s.ROOT.DeepTrack.DataAugmentation.rrange = 10;
% s.ROOT.DeepTrack.DataAugmentation.trange = 5;
% s.ROOT.DeepTrack.DataAugmentation.scale_factor_range = 1.1;
% s.ROOT.DeepTrack.ImageProcessing.scale = 1.;
% lObj.trackSetParams(s);

tobj.setup_alg(net);
tobj.setup_backend(backend,aws_params);


