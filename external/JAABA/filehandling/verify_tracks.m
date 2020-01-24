function [isvalid msg]  = verify_tracks(filename,targettype)
% function [isvalid msg] = verify_tracks(filename,targettype)
% Check integrity of tracks file (see published doc as a guide).
%
% filename: full path to trx file
% targettype: JLabelData targetType
%
% isvalid: true if trx file passes all checks, false otherwise
% msg: if isvalid is false, string containing reason(s) for failure. 

TRX_FLDS_REQ = {'x' 'y' 'theta' 'a' 'b' 'nframes' 'firstframe' 'endframe' ...
  'off' 'id' 'x_mm' 'y_mm' 'theta_mm' 'a_mm' 'b_mm' 'sex' 'dt' 'fps'};
TRX_FLDS_OPT = {'timestamps'};
TRX_FLDS_LAR = {'area' 'xcontour' 'ycontour' 'xspine' 'yspine' 'area_mm' ...
  'xspine_mm' 'yspine_mm'};

try
  mat = load(filename','-mat');
catch %#ok<CTCH>
  isvalid = false;
  msg = sprintf('Can''t load MAT file ''%s''.',filename);
  return;
end

if ~isfield(mat,'trx')
  isvalid = false;
  msg = 'MAT file does not contain variable ''trx''.';
  return;
end
trx = mat.trx;

% check trx fields
tf = isfield(trx,TRX_FLDS_REQ);
if ~all(tf)
  isvalid = false;
  msg = sprintf('trx struct array missing required fields: %s',...
    civilizedStringFromCellArrayOfStrings(TRX_FLDS_REQ(~tf)));
  return;
end

unkFlds = setdiff(fieldnames(trx),[TRX_FLDS_REQ TRX_FLDS_OPT TRX_FLDS_LAR]);
if ~isempty(unkFlds)
  warning('verify_tracks:unknownFields','Unknown fields in trx struct array: %s',...
    civilizedStringFromCellArrayOfStrings(unkFlds));
end

Ntrx = numel(trx);
issues = cell(0,1);
NFRAMES_ROW_VECS = {'x' 'y' 'theta' 'a' 'b' 'x_mm' 'y_mm' 'theta_mm' 'a_mm' 'b_mm'};
VALID_SEX_VALS = {'M' 'F' '?'};
% Check each trx element. This is not a 100% comprehensive check.
for iTrx = 1:Ntrx
  
  trxEl = trx(iTrx);
  nf = trxEl.nframes;

  % sizes of row vectors
  for fld = NFRAMES_ROW_VECS, fld = fld{1}; %#ok<*FXSET>
    val = trxEl.(fld);
    issues = lclChk(issues,iTrx,isrow(val) && numel(val)==nf,...
      sprintf('trx field ''%s'' has unexpected size',fld));
  end

  % firstframe/endframe/nframes/off
  issues = lclChk(issues,iTrx,(trxEl.endframe-trxEl.firstframe+1)==trxEl.nframes,...
    'inconsistent values for nframes/firstframe/endframe');
  issues = lclChk(issues,iTrx,trxEl.off==1-trxEl.firstframe,...
    'inconsistent values for firstframe/off');

  val = trxEl.sex;
  issues = lclChk(issues,iTrx,...
    (isscalar(val) || isvector(val) && iscellstr(val) && numel(val)==nf) ...
      && all(ismember(val,VALID_SEX_VALS)),'invalid value for field ''sex''');

  val = trxEl.dt;
  issues = lclChk(issues,iTrx,isrow(val) && numel(val)==nf-1,...
      'field ''dt'' has unexpected size');
end

isvalid = isempty(issues);
msg = sprintf('%s\n',issues{:}); 
msg = msg(1:end-1);

function issues = lclChk(issues,iTrx,tf,str)
if ~tf
  issues{end+1,1} = sprintf('Trx %d: %s',iTrx,str);
end
  