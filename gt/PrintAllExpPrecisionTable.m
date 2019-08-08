function PrintAllExpPrecisionTable(Ptbls,allexptypes,savename,varargin)

exptypedict = {...
  'FlyBubble','10 flies'
  'SHView0','Side'
  'SHView1','Front'
  'SH3D','3D'
  'RFView0','View 1'
  'RFView1','View 2'
  'RF3D','3D'
  'Roian','2 mice'
  'BSView0x','Bottom'
  'BSView1x','Front'
  'BSView2x','Back'
  'Larva','larva'
  'FlyBubbleMDNvsDLC','Diff'};

[annfns,labelallonly,dataallonly,dorotateheader,exptypedict] = myparse(varargin,'annfns',{'intra','inter'},'labelallonly',false,'dataallonly',false,'dorotateheader',true,...
  'exptypedict',exptypedict);

allonly = labelallonly && dataallonly;

if nargin < 3 || isempty(savename),
  fid = 1;
else
  fid = fopen(savename,'w');
end

nexptypes = numel(allexptypes);
netnames = {};
for eti = 1:nexptypes,
  for i = 1:numel(Ptbls{eti}),
    ism = ismember(Ptbls{eti}{i}.Properties.RowNames,netnames);
    netnames = [netnames;Ptbls{eti}{i}.Properties.RowNames(~ism)];
  end
end
nnets = numel(netnames);
allndatatypes = nan(1,nexptypes);
allnlabeltypes = nan(1,nexptypes);
for eti = 1:nexptypes,
  [allndatatypes(eti),allnlabeltypes(eti)] = size(Ptbls{eti});
end
if allonly,
  ncolsperexptype(:) = ones(1,nexptypes);
elseif labelallonly,
  ncolsperexptype = allndatatypes;
elseif dataallonly,
  ncolsperexptype = allnlabeltypes;
else
  ncolsperexptype = allndatatypes.*allnlabeltypes;
end


ncols = sum(ncolsperexptype);
fprintf(fid,['\\begin{tabular}{|c||',repmat('c|',[1,ncols]),'}\\hline\n']);

% exptype column headers
fprintf(fid,'Network');
for eti = 1:nexptypes,
  j = find(strcmp(allexptypes{eti},exptypedict(:,1)));
  if ~isempty(j),
    etname = exptypedict{j,2};
  else
    etname = allexptypes{eti};
  end
  if ncolsperexptype(eti) > 1,
    fprintf(fid,' & \\multicolumn{%d}{c|}{%s}',ncolsperexptype(eti),etname);
  else
    fprintf(fid,' & %s',etname);
  end
end
if ~dataallonly,
  % datatype column headers
  fprintf(fid,'\\\\\n');
  for eti = 1:nexptypes,
    [ndatatypes,nlabeltypes] = size(Ptbls{eti});
    if ndatatypes > 1,
      labeli = nlabeltypes;
      for datai = 1:ndatatypes,
        tbl = Ptbls{eti}{datai,labeli};
        m = regexp(tbl.Properties.Description,'^(.+)\/(.+)$','tokens','once');
        datatype = m{1};
        if ~labelallonly && nlabeltypes > 1,
          fprintf(fid,' & \\multicolumn{%d}{c|}{%s}',nlabeltypes,datatype);
        else
          if ~dorotateheader || ~labelallonly,
            fprintf(fid,' & %s',datatype);
          else
            fprintf(fid,' & \\rot{%s}',datatype);
          end
        end
      end
    else
      fprintf(fid,' &');
    end
  end
end

if ~labelallonly,
  fprintf(fid,'\\\\\n');
  % labeltype column headers
  for eti = 1:nexptypes,
    [ndatatypes,nlabeltypes] = size(Ptbls{eti});
    if dataallonly,
      nreps = 1;
    else
      nreps = ndatatypes;
    end
    if nlabeltypes > 1,
      datai = ndatatypes;
      for repi = 1:nreps,
        for labeli = 1:nlabeltypes,
          tbl = Ptbls{eti}{datai,labeli};
          m = regexp(tbl.Properties.Description,'^(.+)\/(.+)$','tokens','once');
          labeltype = m{2};
          if dorotateheader,
            fprintf(fid,' & \\rot{%s}',labeltype);
          else
            fprintf(fid,' & %s',labeltype);
          end
        end
      end
    else
      fprintf(fid,' &');
    end
  end
end
fprintf(fid,'\\\\\\hline\\hline\n');

bestval = cell(1,nexptypes);
for eti = 1:nexptypes,
  [ndatatypes,nlabeltypes] = size(Ptbls{eti});
  bestval{eti} = nan(ndatatypes,nlabeltypes);
  for datai = 1:ndatatypes,
    for labeli = 1:nlabeltypes,
      tbl = Ptbls{eti}{datai,labeli};
      idxcurr = ~ismember(tbl.Properties.RowNames,annfns);
      bestval{eti}(datai,labeli) = max(tbl.AWP(idxcurr));
    end
  end
end

% AWP
roundprecision = 2;
for ndx = 1:nnets,
  fprintf(fid,'%s',netnames{ndx});
  for eti = 1:nexptypes,
    [ndatatypes,nlabeltypes] = size(Ptbls{eti});
    for datai = 1:ndatatypes,
      for labeli = 1:nlabeltypes,
        tbl = Ptbls{eti}{datai,labeli};
        
        m = regexp(tbl.Properties.Description,'^(.+)\/(.+)$','tokens','once');
        if ndatatypes > 1 && dataallonly,
          if ~strcmpi(m{1},'all'),
            continue;
          end
        end
        
        if nlabeltypes > 1 && labelallonly,
          if ~strcmpi(m{2},'all'),
            continue;
          end
        end
        
        j = find(strcmp(tbl.Properties.RowNames,netnames{ndx}));
        
        if isempty(j),
          fprintf(fid,' &');
        else
          isbestval = round(tbl.AWP(j),roundprecision) >= round(bestval{eti}(datai,labeli),roundprecision);
          if isbestval,
            fprintf(fid,' & \\textbf{%.2f}',tbl.AWP(j));
          else
            fprintf(fid,' & %.2f',tbl.AWP(j));
          end
        end
      end
    end
  end
  fprintf(fid,'\\\\\\hline\n');
end
fprintf(fid,'\\end{tabular}\n');