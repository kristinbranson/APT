function result = WriteYaml(filename, data, flowstyle)
import yaml.*;
if ~exist('flowstyle','var')
        flowstyle = 0;
    end;
    if ~ismember(flowstyle, [0,1])
        error('Flowstyle must be 0,1 or empty.');
    end;
    result = [];
    [pth,~,~] = fileparts(mfilename('fullpath'));
    try
        import('org.yaml.snakeyaml.*');
        javaObject('Yaml');
    catch
        dp = [pth filesep 'external' filesep 'snakeyaml-1.9.jar'];
        if not(ismember(dp, javaclasspath ('-dynamic')))
        	javaaddpath(dp); % javaaddpath clears global variables!?
        end
        import('org.yaml.snakeyaml.*');
    end;
    javastruct = scan(data);
    dumperopts = DumperOptions();
    dumperopts.setLineBreak(        javaMethod('getPlatformLineBreak',        'org.yaml.snakeyaml.DumperOptions$LineBreak'));
    if flowstyle
        classes = dumperopts.getClass.getClasses;
        flds = classes(3).getDeclaredFields();
        fsfld = flds(1);
        if ~strcmp(char(fsfld.getName), 'FLOW')
            error(['Accessed another field instead of FLOW. Please correct',            'class/field indices (this error maybe caused by new snakeyaml version).']);
        end;
        dumperopts.setDefaultFlowStyle(fsfld.get([]));
    end;
    yaml = Yaml(dumperopts);
    output = yaml.dump(javastruct);
    if ~isempty(filename)
        fid = fopen(filename,'w');
        fprintf(fid,'%s',char(output) );
        fclose(fid);
    else
        result = output;
    end;
end
function result = scan(r)
import yaml.*;
if ischar(r)
        result = scan_char(r);
    elseif iscell(r)
        result = scan_cell(r);
    elseif isord(r)
        result = scan_ord(r);
    elseif isstruct(r)
        result = scan_struct(r);                
    elseif isnumeric(r)
        result = scan_numeric(r);
    elseif islogical(r)
        result = scan_logical(r);
    elseif isa(r,'DateTime')
        result = scan_datetime(r);
    else
        error(['Cannot handle type: ' class(r)]);
    end
end
function result = scan_numeric(r)
import yaml.*;
if isempty(r)
        result = java.util.ArrayList();
    elseif(isinteger(r))
        result = java.lang.Integer(r);
    else
        result = java.lang.Double(r);
    end
end
function result = scan_logical(r)
import yaml.*;
if isempty(r)
        result = java.util.ArrayList();
    else
        result = java.lang.Boolean(r);
    end
end
function result = scan_char(r)
import yaml.*;
if isempty(r)
        result = java.util.ArrayList();
    else
        result = java.lang.String(r);
    end
end
function result = scan_datetime(r)
import yaml.*;
[Y, M, D, H, MN,S] = datevec(double(r));            
	result = java.util.GregorianCalendar(Y, M-1, D, H, MN,S);
	result.setTimeZone(java.util.TimeZone.getTimeZone('UTC'));
end
function result = scan_cell(r)
import yaml.*;
if(isrowvector(r))  
        result = scan_cell_row(r);
    elseif(iscolumnvector(r))
        result = scan_cell_column(r);
    elseif(ismymatrix(r))
        result = scan_cell_matrix(r);
    elseif(issingle(r));
        result = scan_cell_single(r);
    elseif(isempty(r))
        result = java.util.ArrayList();
    else
        error('Unknown cell content.');
    end;
end
function result = scan_ord(r)
import yaml.*;
if(isrowvector(r))
        result = scan_ord_row(r);
    elseif(iscolumnvector(r))
        result = scan_ord_column(r);
    elseif(ismymatrix(r))
        result = scan_ord_matrix(r);
    elseif(issingle(r))
        result = scan_ord_single(r);
    elseif(isempty(r))
        result = java.util.ArrayList();
    else
        error('Unknown ordinary array content.');
    end;
end
function result = scan_cell_row(r)
import yaml.*;
result = java.util.ArrayList();
    for ii = 1:size(r,2)
        result.add(scan(r{ii}));
    end;
end
function result = scan_cell_column(r)
import yaml.*;
result = java.util.ArrayList();
    for ii = 1:size(r,1)
        tmp = r{ii};
        if ~iscell(tmp)
            tmp = {tmp};
        end;
        result.add(scan(tmp));
    end;    
end
function result = scan_cell_matrix(r)
import yaml.*;
result = java.util.ArrayList();
    for ii = 1:size(r,1)
        i = r(ii,:);
        result.add(scan_cell_row(i));
    end;
end
function result = scan_cell_single(r)
import yaml.*;
result = java.util.ArrayList();
    result.add(scan(r{1}));
end
function result = scan_ord_row(r)
import yaml.*;
result = java.util.ArrayList();
    for i = r
        result.add(scan(i));
    end;
end
function result = scan_ord_column(r)
import yaml.*;
result = java.util.ArrayList();
    for i = 1:size(r,1)
        result.add(scan_ord_row(r(i)));
    end;
end
function result = scan_ord_matrix(r)
import yaml.*;
result = java.util.ArrayList();
    for i = r'
        result.add(scan_ord_row(i'));
    end;
end
function result = scan_ord_single(r)
import yaml.*;
result = java.util.ArrayList();
    for i = r'
        result.add(r);
    end;
end
function result = scan_struct(r)
import yaml.*;
result = java.util.LinkedHashMap();
    for i = fields(r)'
        key = i{1};
        val = r.(key);
        result.put(key,scan(val));
    end;
end
