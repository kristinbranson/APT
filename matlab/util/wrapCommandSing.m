function cmdout = wrapCommandSing(cmdin, varargin)

DFLTBINDPATH = {
  '/groups'
  '/nrs'
  '/scratch'};
[bindpath,singimg] = ...
  myparse(varargin,...
          'bindpath',DFLTBINDPATH,...
          'singimg','');
assert(~isempty(singimg)) ;
quotedbindpath = cellfun(@escape_string_for_bash,bindpath,'uni',0);   
path_count = numel(quotedbindpath) ;
Bflags = [repmat({'-B'},1,path_count); quotedbindpath(:)'];
Bflagsstr = space_out(Bflags);
quotedsingimg = escape_string_for_bash(singimg) ;
quotedcmd = escape_string_for_bash(cmdin) ;
cmdout = sprintf('singularity exec --nv %s %s bash -c %s',...
                 Bflagsstr,quotedsingimg,quotedcmd);

