function [featureLexiconNameList, ...
          featureLexiconFileNameList, ...
          featureLexiconAnimalTypeList] = ...
  getFeatureLexiconListsFromXML()

% Figure out the name of the XML file holding the list of the possible
% feature lexicons
if isdeployed,
  filename = deployedRelative2Global('params/featureConfigList.xml');
else
  filename = 'featureConfigList.xml';
end
% Read in the XML file
featureLexiconListThing = ReadXMLParams(filename);
% break out that information into three cell arrays, each of length equal
% to the number of feature lexicons
%   featureLexiconNameList{i} holds the name of the ith lexicon, e.g.
%   'flies', 'larvae', 'vivek_mice'.
%   featureLexiconFileNameList{i} holds the file name of the XML file
%     describing the ith lexicon
%   featureLexiconAnimalTypeList{i} holds the name of the animal that the
%     ith lexicon is designed to be used with.  E.g. 'fly', or 'mouse', or
%     'larva'.  This should be singular, not plural.
featureLexiconNameList=fieldnames(featureLexiconListThing);
nFeatureLexicons=length(featureLexiconNameList);
featureLexiconFileNameList=cell(nFeatureLexicons,1);
featureLexiconAnimalTypeList=cell(nFeatureLexicons,1);
for i=1:nFeatureLexicons
  featureLexiconName=featureLexiconNameList{i};
  featureLexiconFileNameList{i}=featureLexiconListThing.(featureLexiconName).file;
  featureLexiconAnimalTypeList{i}=featureLexiconListThing.(featureLexiconName).animal;
end

end
