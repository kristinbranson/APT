function outTable = convertToTableCore(featureLexicon,inParams)

% deal with optional arguments
if nargin<2 || isempty(inParams)
  inParams=struct();
end

%featureLexicon = ReadXMLParams(featureconfigfile);
allpf = fieldnames(featureLexicon.perframe);
pftypeList = {};
for pfndx = 1:numel(allpf)
  curpf = allpf{pfndx};
  curtypes  = featureLexicon.perframe.(curpf).type; 
  if ischar(curtypes)
    curT = curtypes;
    if ~any(strcmp(pftypeList,curT))
      pftypeList{end+1} = curT;
    end
  else    
    for tndx = 1:numel(curtypes)
      curT = curtypes{tndx};
      if ~any(strcmp(pftypeList,curT))
        pftypeList{end+1} = curT;
      end
    end
  end
end

types = fieldnames(inParams);
outTable = cell(numel(pftypeList),3);
for ndx = 1:numel(pftypeList)
  matches = strcmp(types,pftypeList{ndx});
  if any(matches)
    mndx = find(matches,1);
    outTable{ndx,1} = types{mndx};
    outTable{ndx,2} = inParams.(types{mndx}).mode{1};
    outTable{ndx,3} = inParams.(types{mndx}).selection{1};  
  else
    outTable{ndx,1} = pftypeList{ndx};
    outTable{ndx,2} = 'None';
    outTable{ndx,3} = 'normal';  
  end
end
