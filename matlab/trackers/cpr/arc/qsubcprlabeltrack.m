function qsubcprlabeltrack(tObjFile,resBase,movFile,varargin)

SCRIPT = '/groups/flyprojects/home/leea30/git/cpr.build/current/CPRLabelTrackerTrack/for_testing/run_CPRLabelTrackerTrack.sh';
MCR = '/groups/flyprojects/home/leea30/mlrt/v90/';

df = myparse(varargin,...
  'df',5);

% ./CPRLabelTrackerTrack/for_redistribution_files_only/run_CPRLabelTrackerTrack.sh 
% /groups/flyprojects/home/leea30/mlrt/v90/
% /groups/flyprojects/home/leea30/cpr/jan/data/tObj@35exps_rc@20160429.mat 
% /groups/flyprojects/home/leea30/cpr/jan/data/testcprlabeltrackertrack 
% /groups/flyprojects/home/leea30/cpr/jan/data/movs.txt 5

nowstr = datestr(now,'yyyymmddTHHMMSS');
cmd = sprintf('qsub -pe batch 4 -j y -b y -cwd -A bransonk -o qsub/cprtrk.%s.log -N cprtrk-%s ''%s %s %s %s %s %d''',...
      nowstr,nowstr,SCRIPT,MCR,tObjFile,resBase,movFile,df);
fprintf(1,'Running:\n%s\n',cmd);
system(cmd);
