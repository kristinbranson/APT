%strips user labels off a batch of APT projects.  Useful when you have
%labelled points as training data already but want to generate a new
%training data batch but not get it mixed with the old one.  PLEASE BACKUP
%projects before using


%% get filenames of projects to strip labels from
[f,p]=uigetfile(['*.lbl'],'multiSelect','on');

%%
button = questdlg('Are you sure you want to remove all labels from these projects?');
if ~strcmp('Yes',button)
    return
end

%%

for proj = 1:length(p)%for each project
  
    %load project
    ad = load([p,f{proj}],'-mat');
    adnew = ad;
    
    %save backup of project in case mess this up
    time = clock;
    if ~exist([p,'backup_',num2str(time(1)),num2str(time(2)),num2str(time(3))])
        mkdir(p,['backup_',num2str(time(1)),num2str(time(2)),num2str(time(3))])
    end
    save([p,'backup_',num2str(time(1)),num2str(time(2)),num2str(time(3)),filesep,f{proj}],'-struct','ad','-mat');
    
    %re-creating empty labeledPos data
    for expi = 1:length(adnew.labeledpos)
%         adnew.labeledposMarked{expi,1} = false(size(adnew.labeledposMarked{expi,1}));
        adnew.labeledpos{expi,1} = nan(size(adnew.labeledpos{expi,1}));
        adnew.labeledpostag{expi,1} = cell(size(adnew.labeledpostag{expi,1}));%initializing manual label tags as empty because no manual labels currently exist
        % AL 20171110: .labeledpostag{expi,1} is now a logical array rather
        % than a cell array. The legacy format (cell array) will be
        % converted upon project load however so using a cell here should
        % still work.
        adnew.labeledposTS{expi,1} = inf(size(adnew.labeledposTS{expi,1}));%initializing manual label timestamps as -Inf because no manual labels currently exist
    end
    
    save([p,filesep,f{proj}],'-struct','adnew','-mat');
    
end

disp(['Labels removed.  Backup with labels still present saved in: backup_',num2str(time(1)),num2str(time(2)),num2str(time(3))])