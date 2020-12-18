%ScriptGTDataSetParameters

% info is nlabels x 1 x 3, where last 3 indices correspond to 
% movie, frm and trx_id

assert(exist('exptype','var')>0);

nets = {'openpose','leap','leap_orig','deeplabcut','deeplabcut_orig',...
  'dpk','unet','resnet_unet','mdn_joint_fpn','mdn_joint','mdn','mdn_unet','cpropt'};
legendnames = {'OpenPose','LEAP', 'LEAP 0','DeepLabCut','DeepLabCut 0',...
  'DeepPoseKit','U-net','Res-U-net','GRONe','MDN Joint','MDN','MDN U-net','CPR'};

% nets = {'openpose','leap','deeplabcut','unet','resnet_unet','mdn','cpropt'};
% legendnames = {'OpenPose','LEAP','DeepLabCut','U-net','Res-U-net','MDN','CPR'};


nnets = numel(nets);
colors = [
  0         0.4470    0.7410 % openpose
  0.4660    0.6740    0.1880 % leap  
  0.7330    0.8370    0.5940 % leap original (blend of white and leap)
  0.8500    0.3250    0.0980 % dlc 
  0.8875    0.4938    0.3235 % dlc original (blend of white and dlc)
  0.9290    0.6940    0.1250 % DPK
  0.7263    0.3085    0.3880 % unet (blend of resunet and white)
  0.6350    0.0780    0.1840 % resnet_unet
  0.4940    0.1840    0.5560 % GRONe
  0.6205    0.3880    0.6670 % mdn joint (blend of mdn joint and white, 3:1)
  0.7470    0.5920    0.7780 % mdn (blend of mdn joint and white, 1:1) 
  0.3293    0.1227    0.3707 % mdn unet (blend of mdn joint and black) 
  0.3010    0.7450    0.9330 % cpr
  ];
prcs = [50,75,90,95,97];

% idxnet = [1 2 3 7 4 5 6];
% nets = nets(idxnet);
% colors = colors(idxnet,:);
% legendnames = legendnames(idxnet);
vwi = 1;
doAlignCoordSystem = false;
annoterrdata = [];
is3d = false;

gtfileinfo = GTFileInfo(exptype);
gtfile_trainsize_cpr = gtfileinfo.trainsize_cpr;
gtfile_cpr = gtfileinfo.cpr;
gtfile_traintime_cpr = gtfileinfo.traintime_cpr;
gtfile_trainsize = gtfileinfo.trainsize;
gtfile_traintime = gtfileinfo.traintime;
gtfile_final = gtfileinfo.final;
annoterrfile = gtfileinfo.annoterrfile;
condinfofile = gtfileinfo.condinfofile;
lblfile = gtfileinfo.lblfile;
gtimagefile = gtfileinfo.gtimagefile;

switch exptype,
  case {'SHView0','SHView1'}
    if strcmp(exptype,'SHView0'),
      vwi = 1;
    else
      vwi = 2;
    end

    incondinfo = load(condinfofile);
    conddata = struct;
    % conditions:
    % enriched + activation
    % not enriched + activation
    % not activation
    % data types:
    % train
    % not train
    [conddata.data_cond,conddata.label_cond,datatypes,labeltypes] = SHGTInfo2CondData(incondinfo.gtinfo,true);
    pttypes = {'L. antenna tip',1
      'R. antenna tip',2
      'L. antenna base',3
      'R. antenna base',4
      'Proboscis roof',5};
    %     labeltypes = {'all',1};
    %     datatypes = {'all',1};
    maxerr = 60;
    freezeInfo = struct;
    freezeInfo.iMov = 502;
    freezeInfo.iTgt = 1;
    freezeInfo.frm = 746;
    doplotoverx = true;
    %doplotoverx = false;
    gtimdata = struct;
    gtimdata.ppdata = incondinfo.ppdata;
    gtimdata.tblPGT = incondinfo.tblPGT;
    
  case 'SH3D'    
    vwi = 1;
    
    incondinfo = load(condinfofile);
    conddata = struct;
    % conditions:
    % enriched + activation
    % not enriched + activation
    % not activation
    % data types:
    % train
    % not train
    [conddata.data_cond,conddata.label_cond,datatypes,labeltypes] = SHGTInfo2CondData(incondinfo.gtinfo,true);
    pttypes = {'L. antenna tip',1
      'R. antenna tip',2
      'L. antenna base',3
      'R. antenna base',4
      'Proboscis roof',5};
    %     labeltypes = {'all',1};
    %     datatypes = {'all',1};
    maxerr = 0.4123;
    freezeInfo = [];
    doplotoverx = true;
    gtimdata = struct;
    gtimdata.ppdata = incondinfo.ppdata;
    gtimdata.tblPGT = incondinfo.tblPGT;
    is3d = true;
    
  case 'FlyBubble'
    % there are some networks in final that are not in trainsize
    gtimdata = load(gtimagefile);
    conddata = load(condinfofile);
    
    pttypes = {'head',[1,2,3]
      'body',[4,5,6,7]
      'middle leg joint 1',[8,10]
      'middle leg joint 2',[9,11]
      'front leg tarsi',[12,17]
      'middle leg tarsi',[13,16]
      'back leg tarsi',[14,15]};
    labeltypes = {'moving',1
      'grooming',2
      'close',3
      'all',[1,2,3]};
    datatypes = {'same fly',1
      'same genotype',2
      'different genotype',3
      'all',[1,2,3]};
    maxerr = 30;
    doplotoverx = true;
    %doplotoverx = false;
    
  case {'RFView0','RFView1'}
    if strcmp(exptype,'RFView0'),
      vwi = 1;
    else
      vwi = 2;
    end
    
    conddata = [];
    labeltypes = {};
    datatypes = {};
    
    maxerr = [];
    freezeInfo = struct;
    freezeInfo.iMov = 1;
    freezeInfo.iTgt = 1;
    freezeInfo.frm = 302;
    freezeInfo.clim = [0,192];
    doplotoverx = false;
    gtimdata = load(gtimagefile);
    
    pttypes = {'abdomen',19
      'front leg joint 1',[13,16]
      'front leg joint 2',[7,10]
      'front leg tarsi',[1,4]
      'middle leg joint 1',[14,17]
      'middle leg joint 2',[8,11]
      'middle leg tarsi',[2,5]
      'back leg joint 1',[15,18]
      'back leg joint 2',[9,12]
      'back leg tarsi',[3,6]};
    
    
  case 'RF3D'
    vwi = 1;
    
    conddata = [];
    maxerr = [];
    freezeInfo = [];
    is3d = true;
    doplotoverx = false;
    gtimdata = load(gtimagefile);
    
    pttypes = {'abdomen',19
      'front leg joint 1',[13,16]
      'front leg joint 2',[7,10]
      'front leg tarsi',[1,4]
      'middle leg joint 1',[14,17]
      'middle leg joint 2',[8,11]
      'middle leg tarsi',[2,5]
      'back leg joint 1',[15,18]
      'back leg joint 2',[9,12]
      'back leg tarsi',[3,6]};
    
  case 'Larva',
        
    conddata = [];
    labeltypes = {};
    datatypes = {};
    pttypes = {'outside',[1:2:13,16:2:28]
      'inside',[2:2:14,15:2:27]};
    maxerr = [];
    gtimdata = load(gtimagefile);
    
    freezeInfo = struct;
    freezeInfo.iMov = 4;
    freezeInfo.iTgt = 1;
    freezeInfo.frm = 5;
    freezeInfo.axes_curr.XLim = [745,1584];
    freezeInfo.axes_curr.YLim = [514,1353];
    doplotoverx = false;
    
  case 'Roian'
    vwi = 1;
    
    conddata = [];
    labeltypes = {};
    datatypes = {};
    maxerr = [];
    freezeInfo = struct;
    freezeInfo.iMov = 1;
    freezeInfo.iTgt = 1;
    freezeInfo.frm = 1101;
    doplotoverx = false;
    gtimdata = load(gtimagefile);
    
    pttypes = {'nose',1
      'tail',2
      'ear',[3,4]};
    
  case {'BSView0x','BSView1x','BSView2x'}
    vwi = 1;
    if strcmp(exptype,'BSView0x'),
      pttypes = {'Front foot',[1,2]
        'Back foot',[3,4]
        'Tail',5};
    elseif strcmp(exptype,'BSView1x'),
      pttypes = {'Front foot',[1,2]};
    else
      pttypes = {'Back foot',[1,2]
        'Tail',3};
    end
    
    conddata = [];
    labeltypes = {};
    datatypes = {};
    maxerr = [];
    
    freezeInfo = struct;
    freezeInfo.i = 1;
    doplotoverx = false;
    gtimdatain = load(gtimagefile);
    realvwi = str2double(regexp(exptype,'View(\d+)x','once','tokens'))+1;
    gtimdata = struct;
    gtimdata.cvi = gtimdatain.cvidx{realvwi};
    gtimdata.ppdata = gtimdatain.ppdatas{realvwi};
    gtimdata.tblPGT = gtimdatain.tblPGTs{realvwi};
    gtimdata.frame = gtimdata.tblPGT.frm;
    gtimdata.movieidx = gtimdata.tblPGT.mov;
    gtimdata.movies = gtimdatain.trnmovies{realvwi};
    gtimdata.target = gtimdata.tblPGT.iTgt;
  case 'FlyBubbleMDNvsDLC',
    conddata = [];
    gtimdata = load(gtimagefile);
    
    nets = {'DLC','MDN'};
    legendnames = {'DeepLabCut','MDN'};
    nnets = numel(nets);
    colors = [
      0.8500    0.3250    0.0980
      0.4940    0.1840    0.5560
      ];
    labeltypes = {};
    datatypes = {};
    pttypes = {'head',[1,2,3]
      'body',[4,5,6,7]
      'middle leg joint 1',[8,10]
      'middle leg joint 2',[9,11]
      'front leg tarsi',[12,17]
      'middle leg tarsi',[13,16]
      'back leg tarsi',[14,15]};
    maxerr = [];
    doplotoverx = false;
    doAlignCoordSystem = true;
    
    
  otherwise
    error('Unknown exp type %s',exptype);
    
end
