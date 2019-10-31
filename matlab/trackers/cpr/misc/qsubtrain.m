function qsubtrain(tpfile,tdIfile,varargin)

SCRIPT = '/groups/flyprojects/home/leea30/git/cpr.build/current/trainAL/for_testing/run_trainAL.sh';
MCR = '/groups/flyprojects/home/leea30/mlrt/v90/';

[tdIfileVar,datatype] = myparse(varargin,...
  'tdIfileVar','iTrn',...
  'datatype','jan');    % for computeIpp

nowstr = datestr(now,'yyyymmddTHHMMSS');
cmd = sprintf('qsub -pe batch 12 -j y -b y -cwd -A bransonk -o qsub/trn.%s.%s.log -N cpr-trn-%s ''%s %s td@30@he@0329.mat %s tdIfile %s tdIfileVar %s datatype %s''',...
      tdIfile,nowstr,nowstr,SCRIPT,MCR,tpfile,tdIfile,tdIfileVar,datatype);

fprintf(1,'Running:\n%s\n',cmd);
system(cmd);
