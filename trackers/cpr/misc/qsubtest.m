function qsubtest(tpfile,trfile,tdIfile,varargin)

SCRIPT = '/groups/flyprojects/home/leea30/git/cpr.build/current/testAL/for_testing/run_testAL.sh';
MCR = '/groups/flyprojects/home/leea30/mlrt/v90/';

[tdfile,tdIfileVar,datatype] = myparse(varargin,...
  'tdfile','td@30@he@0329.mat',...
  'tdIfileVar','iTstLbl',...
  'datatype','jan'); % for computeIpp
  
nowstr = datestr(now,'yyyymmddTHHMMSS');
cmd = sprintf('qsub -pe batch 6 -j y -b y -cwd -A bransonk -o qsub/test.%s.%s.log -N cpr-test-%s ''%s %s %s %s %s tdIfile %s tdIfileVar %s datatype %s skipLoss 1''',...
      tdIfile,nowstr,nowstr,SCRIPT,MCR,tpfile,tdfile,trfile,tdIfile,tdIfileVar,datatype);
fprintf(1,'Running:\n%s\n',cmd);
system(cmd);
