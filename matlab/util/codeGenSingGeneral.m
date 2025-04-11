function codestr = codeGenSingGeneral(basecmd,varargin)

% Take a base command and run it in a sing img
DFLTBINDPATH = {
  '/groups/branson/bransonlab'
  '/groups/branson/home'
  '/nrs/branson'
  '/scratch'};      
[bindpath,singimg] = ...
  myparse(varargin,...
          'bindpath',DFLTBINDPATH,...
          'singimg','');
assert(~isempty(singimg)) ;
bindpath = cellfun(@(x)['"' x '"'],bindpath,'uni',0);      
Bflags = [repmat({'-B'},1,numel(bindpath)); bindpath(:)'];
Bflagsstr = sprintf('%s ',Bflags{:});
codestr = sprintf('singularity exec --nv %s %s bash -c "%s"',...
  Bflagsstr,singimg,basecmd);

end
