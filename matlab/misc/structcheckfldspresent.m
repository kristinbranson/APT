function structcheckfldspresent(s0,s1,varargin)
% Check for equality among fields present in (possibly-nested) structs
% 
% Every field is compared (recursively on structs) with isequaln.

[pathstr,pathexceptions,assertmsg] = myparse(varargin,...
  'pathstr','',... % structure 'path' to s0,s1
  'pathexceptions',cell(0,1),... % path cellstr of "exceptions" that are allowed to differ
  'assertmsg','Structure mismatch: %s.'...
  );

assert(isstruct(s0));
assert(isstruct(s1));

s0 = structrestrictflds(s0,s1);
s1 = structrestrictflds(s1,s0);
s1 = orderfields(s1,s0);
fns = fieldnames(s0);

% recurse 
tfstructval0 = structfun(@isstruct,s0);
tfstructval1 = structfun(@isstruct,s1);
assert(isequal(tfstructval0,tfstructval1),assertmsg);
fnsstructval = fns(tfstructval0);
for f=fnsstructval(:)',f=f{1}; %#ok<FXSET>
  pathstrnew = [pathstr '.' f];
  structcheckfldspresent(s0.(f),s1.(f),'pathstr',pathstrnew,...
    'pathexceptions',pathexceptions);
end

% check
s0 = rmfield(s0,fnsstructval);
s1 = rmfield(s1,fnsstructval);
fnsnonstructval = fieldnames(s0);
for f=fnsnonstructval(:)',f=f{1}; %#ok<FXSET>
  pathstrnew = [pathstr '.' f];
  if ~any(strcmp(pathstrnew,pathexceptions))
    assert(isequaln(s0.(f),s1.(f)),assertmsg,pathstrnew);
  end
end
