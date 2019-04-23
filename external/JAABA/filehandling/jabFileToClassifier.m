function jabFileToClassifier(injabfile,outmatfile)

jd = loadAnonymous(injabfile);
inClassifierParams = jd.classifierStuff.params;
fns = fieldnames(inClassifierParams);
classifierParams = struct;
for i = 1:numel(fns),
  fn = fns{i};
  classifierParams.(fn) = reshape([inClassifierParams.(fn)],size(inClassifierParams));
end
save(outmatfile,'-struct','classifierParams','-v7.3');