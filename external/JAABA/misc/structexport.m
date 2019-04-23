function structexport(s,fname,varargin)
% structexport(s,fname,varargin)
% Optional PVs:
%   - format. One of {'csv' 'xls'}. Defaults to csv.

format = myparse(varargin,'format','csv');
assert(ismember(format,{'csv' 'xls'}),'Invalid ''format''.');

pvargs = {};
if strcmp(format,'xls')
  pvargs = {'nansAsStrings' true};
end
r = struct2rowsexport(s,pvargs{:});

switch format
  case 'csv'
    [fh,msg] = fopen(fname,'w');
    if isequal(fh,-1)
      error('structexport:cannotOpen','Cannot open file ''%s'' for writing: %s',...
        fname,msg);
    end
    
    for i = 1:size(r,1)
      row = r(i,:);
      for j = 1:numel(row)
        el = row{j};
        if ischar(el)
          fprintf(fh,'%s',el);
        elseif isscalar(el) && (isnumeric(el) || islogical(el))
          if isequal(floor(double(el)),double(el))
            fprintf(fh,'%d',el);
          else
            fprintf(fh,'%.3f',el);
          end
        elseif isequal(el,[])
          % Special case for []: don't print anything
        else
          fprintf(fh,'<unk>');
        end
        if j<numel(row)
          fprintf(fh,',');
        end
      end
      
      fprintf(fh,'\n');
    end
    
    fclose(fh);
  case 'xls'
    
    [status,msg] = xlswrite(fname,r);
    %status = xlwrite(fname,r);
    if ~status
      error('structexport:failedXlsWrite','Exporting to Excel failed: %s.',msg.message);
    end    
end

      
      