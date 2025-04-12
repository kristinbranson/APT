classdef TestAPT < handle
  % Class to perform tests of APT functionality.

  % Simplest way to test:
  % testObj = TestAPT('name','alice'); 
  % testObj.test_full('nets',{'deeplabcut'});
  
  % If you want to interact with GUI before training:
  % testObj = TestAPT('name','alice');
  % testObj.test_setup();
  % Mess with GUI
  % testObj.test_train('net_type','mdn_joint_fpn',...
  %                    'backend','docker','niters',1000,'test_tracking',true)
  
  % MA/roian
  % testObj = TestAPT('name','roianma');
  % testObj.test_setup('simpleprojload',1);
  % testObj.test_train('net_type',[],'params',-1,'niters',1000);  
  
  % MA/roian, Kristin's suggestion:
  % testObj = TestAPT('name','roianma');  
  % testObj.test_full('nets',{},'setup_params',{'simpleprojload',1,'jrcgpuqueue','gpu_rtx8000','jrcnslots',4},'backend','bsub');
  % testObj.test_full('nets',{'Stg1tddobj_Stg2tdpobj','magrone','Stg1tddht_Stg2tdpht','maopenpose'},'setup_params',{'simpleprojload',1,'jrcgpuqueue','gpu_rtx8000','jrcnslots',4},'backend','docker');
  % empty nets means test all nets
  
  % Carmen/GT workflow (proj on JRC/dm11)
  % testObj = TestAPT('name','carmen');
  % testObj.test_setup('simpleprojload',1);
  % testObj.test_train('backend','bsub');
  % testObj.test_track('backend','bsub');
  % testObj.test_gtcompute('backend','bsub');
  
  properties
    labeler  % a Labeler object, or empty
    controller  % a Controller object, or empty
    info
      % A struct of some kind, set by a call to the set_info_() method.
      % One field is ref_lbl, containing a path to an APT .lbl file.
      % (Or empty.)
    old_lbl  
      % A struct that holds the output of a call to loadLbl(), or empty.  Set by a
      % call to the load_lbl_() method.
    % path_setup_done = false
  end
  
  methods
    
    function testObj = TestAPT(varargin)
      %testObj.setup_path_();
      [name] = myparse(varargin,'name','alice');
      if ~strcmp(name,'dummy')
        testObj.set_info_(name);      
      end
    end
    
    function delete(obj)
      if ~isempty(obj.controller) && isvalid(obj.controller) ,
        delete(obj.controller) ;
      end
      if ~isempty(obj.labeler) && isvalid(obj.labeler) ,
        delete(obj.labeler) ;
      end      
    end

    function create_bundle(obj)
      % Creates a .tar.gz to upload to Dropbox
      % Presumably used to capture a test configuration, for sending to the APT
      % devs?  -- ALT, 2024-10-02
      info = obj.info;
      ref_lbl = info.ref_lbl;
      exp_dir_base = info.exp_dir_base;
      
      proj_name = info.proj_name;
      old_lbl = loadLbl(ref_lbl);
      old_lbl.movieFilesAll = FSPath.macroReplace(old_lbl.movieFilesAll,struct('dataroot',exp_dir_base));
      old_lbl.movieFilesAllGT = FSPath.macroReplace(old_lbl.movieFilesAllGT,struct('dataroot',exp_dir_base));

      tdir = fullfile(tempname,proj_name);
      mkdir(tdir);
      all_m = old_lbl.movieFilesAll;
      for ndx = 1:size(all_m,1)
        for iview = 1:info.nviews
          exp_name = TestAPT.get_exp_name(obj.info, all_m{ndx,iview});
          outdir = fullfile(tdir,exp_name);
          if ~exist(outdir,'dir'),
            mkdir(outdir);
          end
          copyfile(all_m{ndx,iview},outdir);
          if info.has_trx
            trx_file_name = old_lbl.trxFilesAll{ndx};
            trx_file_name = strrep(trx_file_name,'$movdir',fullfile(exp_dir_base,exp_name));
            copyfile(trx_file_name,outdir);
          end
        end
      end
      copyfile(info.ref_lbl,tdir);
      tar(fullfile(tempdir,[proj_name '_data.tar.gz']),tdir,tdir);
      rmdir(tdir);
    end  % function
    
%     function setup_path_(obj)
%       if ~obj.path_setup_done , 
%         APT.setpathsmart() ;
%         %addpath('..');
%         %APT.setpath;
%         obj.path_setup_done = true;
%       end      
%     end

    function set_info_(obj,name)
      if ispc() ,
        % We assume /groups/branson/bransonlab is mounted on Z:
        bransonlab_path = 'Z:/' ;
      else
        bransonlab_path = '/groups/branson/bransonlab' ;
      end
      unittest_dir_path = fullfile(bransonlab_path, 'apt/unittest') ;
      info = struct() ;
      if strcmp(name,'alice')      
        % info.ref_lbl = '/work/mayank/work/FlySpaceTime/multitarget_bubble_expandedbehavior_20180425_fixederrors_fixed.lbl';
        % info.ref_lbl = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/multitarget_bubble_expandedbehavior_20180425_allGT_MK_MDN04182019.lbl';
        info.ref_lbl = fullfile(unittest_dir_path, 'alice/multitarget_bubble_expandedbehavior_20180425_allGT_MK_MDN04182019.lbl') ;
        info.exp_dir_base = fullfile(unittest_dir_path, 'alice/data') ;
        info.nviews = 1;
        info.npts = 17;
        info.has_trx = true;
        info.proj_name = 'alice_test';
        info.sz = 90;
        info.bundle_link = 'https://www.dropbox.com/s/u5p27rdi7kczv78/alice_test_data.tar.gz?dl=1';
        info.op_graph = [1 3; 1 2; 3 17; 2 12; 3 4; 4 10; 10 11; 11 16; 10 6; 8 6; 5 8; 8 9; 9 13;6 14;6 15; 6 7];

      % elseif strcmp(name,'stephen')
      %   %info.ref_lbl = '/groups/branson/bransonlab/mayank/APT_projects/sh_test_lbl_20200310.lbl';
      %   info.ref_lbl = fullfile(unittest_dir_path, 'sh_test_lbl_20200310_modded_resaved_tweaked.lbl') ;
      %   info.exp_dir_base = fullfile(unittest_dir_path, 'stephen_test') ;
      %   info.nviews = 2;
      %   info.npts = 5;
      %   info.has_trx = false;
      %   info.proj_name = 'stephen_test';
      %   info.sz = [];
      %   info.bundle_link = 'https://www.dropbox.com/s/asl1f3ssfgtdwmc/stephen_test_data.tar.gz?dl=1';
      %   info.op_graph = [1 5; 1 3; 3 4; 4 2];
        
      elseif strcmp(name,'stephen_training')
        %info.ref_lbl = '/groups/branson/bransonlab/mayank/APT_projects/sh_test_lbl_20200310.lbl';
        info.ref_lbl = fullfile(unittest_dir_path, 'sh_test_lbl_20200310_modded_resaved_tweaked_20240122.lbl') ;
        info.exp_dir_base = fullfile(unittest_dir_path, 'stephen_test') ;
        info.nviews = 2;
        info.npts = 5;
        info.has_trx = false;
        info.proj_name = 'stephen_test';
        info.sz = [];
        info.bundle_link = 'https://www.dropbox.com/s/asl1f3ssfgtdwmc/stephen_test_data.tar.gz?dl=1';
        info.op_graph = [1 5; 1 3; 3 4; 4 2];
        
      elseif strcmp(name,'stephen_tracking')
        %info.ref_lbl = '/groups/branson/bransonlab/mayank/APT_projects/sh_test_lbl_20200310.lbl';
        info.ref_lbl = fullfile(unittest_dir_path, 'sh_test_lbl_20200310_modded_resaved_tweaked_lightly_trained_20240122.lbl') ;
        info.exp_dir_base = fullfile(unittest_dir_path, 'stephen_test') ;
        info.nviews = 2;
        info.npts = 5;
        info.has_trx = false;
        info.proj_name = 'stephen_test';
        info.sz = [];
        info.bundle_link = 'https://www.dropbox.com/s/asl1f3ssfgtdwmc/stephen_test_data.tar.gz?dl=1';
        info.op_graph = [1 5; 1 3; 3 4; 4 2];
        
      elseif strcmp(name,'carmen')
        % info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/pez7_al.lbl';
        info.ref_lbl = fullfile(unittest_dir_path, 'pez7_al_updated_20241015.lbl') ;
        info.exp_dir_base = fullfile(bransonlab_path, 'card_lab_apt_data/Data_pez3000') ;
        info.nviews = 1;
        info.npts = 10;
        info.has_trx = false;
        info.proj_name = 'carmen_test';
        info.sz = [];
        info.bundle_link = '';
        info.op_graph = [];   
        
      elseif strcmp(name,'carmen_tracking')
        % info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/pez7_al.lbl';
        info.ref_lbl = fullfile(unittest_dir_path, 'pez7_al_updated_20241015_lightly_trained.lbl') ;
        info.exp_dir_base = fullfile(bransonlab_path, 'card_lab_apt_data/Data_pez3000') ;
        info.nviews = 1;
        info.npts = 10;
        info.has_trx = false;
        info.proj_name = 'carmen_track_test';
        info.sz = [];
        info.bundle_link = '';
        info.op_graph = [];   
        
      % elseif strcmp(name,'roianma')
      %   info.ref_lbl = ...
      %     strcatg('/groups/branson/bransonlab/apt/unittest/', ...
      %             'four_points_180806_ma_bothmice_extra_labels_re_radius_150_ds2_gg_add_movie_UT_20210929_trunc_20241016.lbl');
      %   info.exp_dir_base = '';
      %   info.nviews = nan;
      %   info.npts = nan;
      %   info.has_trx = false;
      %   info.proj_name = 'test';
      %   info.sz = 250;
      %   info.bundle_link = '';
      %   info.op_graph = [];   
      % 
      % elseif strcmp(name,'roianmammpose1')
      %   info.ref_lbl = ...
      %     strcatg('/groups/branson/bransonlab/apt/unittest/', ...
      %             'four_points_180806_ma_bothmice_extra_labels_re_radius_150_ds2_gg_add_movie_UT_20210929_trunc_20241016_mmpose1.lbl') ;
      %   info.exp_dir_base = '';
      %   info.nviews = nan;
      %   info.npts = nan;
      %   info.has_trx = false;
      %   info.proj_name = 'test';
      %   info.sz = 250;
      %   info.bundle_link = '';
      %   info.op_graph = [];   
        
      % elseif strcmp(name,'argrone')
      %   info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/flybubble_grone_20210523_allGT_KB_20210626_UT_20210823.lbl';
      %   info.exp_dir_base = '';
      %   info.nviews = nan;
      %   info.npts = nan;
      %   info.has_trx = true;
      %   info.proj_name = 'test';
      %   info.sz = 90;
      %   info.bundle_link = '';
      %   info.op_graph = [];   
        
      elseif strcmp(name,'argroneSA')
        info.ref_lbl = fullfile(unittest_dir_path, 'multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_2.lbl');
        info.exp_dir_base = fullfile(unittest_dir_path, 'alice/data') ;
        info.nviews = nan;
        info.npts = nan;
        info.has_trx = true;
        info.proj_name = 'test';
        info.sz = 90;
        info.bundle_link = '';
        info.op_graph = [];   
        
      % elseif strcmp(name,'sam2view')
      %   %info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/2011_mouse_cam13.lbl';
      %   info.ref_lbl = fullfile(unittest_dir_path, '2011_mouse_cam13_updated_movie_paths_20241111_modded.lbl') ;
      %   info.exp_dir_base = fullfile(bransonlab_path, 'DataforAPT/JumpingMice') ;
      %   info.nviews = 2;
      %   info.npts = nan;
      %   info.has_trx = false;
      %   info.proj_name = 'test';
      %   info.sz = 100; % dont set this to empty even if it is not used
      %   info.bundle_link = '';
      %   info.op_graph = [];   
        
      elseif strcmp(name,'sam2view_training')
        %info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/2011_mouse_cam13.lbl';
        info.ref_lbl = fullfile(unittest_dir_path, '2011_mouse_cam13_updated_movie_paths_20241111_modded.lbl') ;
        info.exp_dir_base = fullfile(bransonlab_path, 'DataforAPT/JumpingMice') ;
        info.nviews = 2;
        info.npts = nan;
        info.has_trx = false;
        info.proj_name = 'test';
        info.sz = 100; % dont set this to empty even if it is not used
        info.bundle_link = '';
        info.op_graph = [];   
        
      elseif strcmp(name,'sam2view_tracking')
        %info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/2011_mouse_cam13.lbl';
        info.ref_lbl = fullfile(unittest_dir_path, '2011_mouse_cam13_updated_movie_paths_20241111_modded_lightly_trained.lbl') ;
        info.exp_dir_base = fullfile(bransonlab_path, 'DataforAPT/JumpingMice') ;
        info.nviews = 2;
        info.npts = nan;
        info.has_trx = false;
        info.proj_name = 'test';
        info.sz = 100; % dont set this to empty even if it is not used
        info.bundle_link = '';
        info.op_graph = [];   
        
      elseif strcmp(name,'roianma2')
        %info.ref_lbl = '/groups/branson/bransonlab/taylora/apt/four-points/four-points-testing-2024-11-19-with-gt-added.lbl';
        info.ref_lbl = fullfile(unittest_dir_path, 'four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl') ;
        info.exp_dir_base = fullfile(unittest_dir_path, 'four-points-reduced-movies') ;
        info.nviews = nan;
        info.npts = nan;
        info.has_trx = false;
        info.proj_name = 'roianma2-test';
        info.sz = 250 ; % dont set this to empty even if it is not used
        info.bundle_link = '';
        info.op_graph = [];           

      elseif strcmp(name,'roianma2_tracking')
        %info.ref_lbl = '/groups/branson/bransonlab/taylora/apt/four-points/four-points-testing-2024-11-19-with-gt-added.lbl';
        info.ref_lbl =  fullfile(unittest_dir_path, 'four-points-testing-2025-04-12-with-rois-added-and-fewer-smaller-avi-movies-lightly-trained.lbl') ;
        info.exp_dir_base = fullfile(unittest_dir_path, 'four-points-reduced-movies') ;
        info.nviews = nan;
        info.npts = nan;
        info.has_trx = false;
        info.proj_name = 'roianma2-test';
        info.sz = 250 ; % dont set this to empty even if it is not used
        info.bundle_link = '';
        info.op_graph = [];           

      elseif strcmp(name,'roianma2mmpose1')
        %info.ref_lbl = '/groups/branson/bransonlab/taylora/apt/four-points/four-points-testing-2024-11-19-with-gt-added-mmpose1.lbl';
        info.ref_lbl = fullfile(unittest_dir_path, 'four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies-mmpose1.lbl') ;
        info.exp_dir_base = fullfile(unittest_dir_path, 'four-points-reduced-movies') ;
        info.nviews = nan;
        info.npts = nan;
        info.has_trx = false;
        info.proj_name = 'roianma2-mmpose1-test';
        info.sz = 250 ; % dont set this to empty even if it is not used
        info.bundle_link = '';
        info.op_graph = [];           

      else
        error('Unrecognized test name');
      end
      info.name = name;
      obj.info = info;
    end  % function
        
    function result = do_files_exist_(obj)
      old_lbl = obj.old_lbl;
      info = obj.info;
      result = true;
      for ndx = 1:numel(old_lbl.movieFilesAll),
        if ~exist(old_lbl.movieFilesAll{ndx},'file')
            result = false;
            return
        end
        if info.has_trx && ~exist(old_lbl.trxFilesAll{ndx},'file')
            result = false;
            return
        end        
      end      
    end  % function
    
    function load_lbl_(obj)
      info = obj.info;
      [data_dir, lbl_file] = TestAPT.get_file_paths(obj.info);
      if ~exist(lbl_file,'file')
        obj.setup_data_();
      end
      old_lbl = loadLbl(lbl_file);
      old_lbl.movieFilesAll = FSPath.macroReplace(old_lbl.movieFilesAll,old_lbl.projMacros);
      old_lbl.movieFilesAllGT = FSPath.macroReplace(old_lbl.movieFilesAllGT,old_lbl.projMacros);
      all_m = old_lbl.movieFilesAll;
      for ndx = 1:size(old_lbl.movieFilesAll,1)
        for iview = 1:info.nviews
          [~,mov_name,mext] = fileparts(all_m{ndx,iview});
          exp_name = TestAPT.get_exp_name(obj.info, all_m{ndx,iview});
          old_lbl.movieFilesAll{ndx,iview} = fullfile(data_dir,exp_name,[mov_name mext]);
          if info.has_trx
            trx_file_name = old_lbl.trxFilesAll{ndx};
            trx_file_name = strrep(trx_file_name,'$movdir',fullfile(data_dir,exp_name));
            old_lbl.trxFilesAll{ndx} = trx_file_name;
          end
        end
      end
      all_m = old_lbl.movieFilesAllGT;
      for ndx = 1:size(all_m,1)
        for iview = 1:info.nviews
          [~,mov_name,mext] = fileparts(all_m{ndx,iview});
          exp_name = TestAPT.get_exp_name(obj.info, all_m{ndx,iview});
          old_lbl.movieFilesAllGT{ndx,iview} = fullfile(data_dir,exp_name,[mov_name mext]);
          if info.has_trx
            trx_file_name = old_lbl.trxFilesAllGT{ndx};
            trx_file_name = strrep(trx_file_name,'$movdir',fullfile(data_dir,exp_name));
            old_lbl.trxFilesAllGT{ndx} = trx_file_name;
          end
        end
      end
      obj.old_lbl = old_lbl;
    end  % function
    
    function setup_data_(obj)
      info = obj.info;
      cacheDir = APT.getdotaptdirpath();
      out_file = fullfile(tempdir(), strcatg(info.proj_name, '_data.tar.gz')); 
            
      try
        untar(out_file,cacheDir);
        return
      catch exception  
        % none, fallthru
        fprintf('Untarring failed, falling through to websave...\n') ;
        fprintf('%s\n', exception.getReport()) ;
      end

      % fallback to websave
      try
        websave(out_file,info.bundle_link);
      catch exception
        if endsWith(exception.identifier,'SSLConnectionSystemFailure')
          % JRC cluster
          wo = weboptions('CertificateFilename','/etc/ssl/certs/ca-bundle.crt');
          websave(out_file,info.bundle_link,wo);
        else
          rethrow(exception);
        end
      end
      
      untar(out_file,cacheDir);          
    end  % function
    
    function test_full(obj,varargin)
      [all_nets,backend,params,backend_params,setup_params,niters] = ...
        myparse(varargin,...
                'nets',{'mdn_joint_fpn'},...
                'backend','docker',...
                'params',{},...
                'backend_params',struct(),...
                'setup_params',{}, ...
                'niters',1000);
      obj.test_setup(setup_params{:});  

      if ischar(all_nets) || (isscalar(all_nets) && ~iscell(all_nets)),
        all_nets = {all_nets};
      end
      if isempty(all_nets),
        %all_nets = num2cell(1:numel(obj.labeler.trackersAll));
        all_nets = cellfun(@(tracker)(tracker.algorithmName), ...
                           obj.labeler.trackersAll, ...
                           'UniformOutput', false) ;
      end
      if isnumeric(all_nets),
        all_nets = num2cell(all_nets);
      end
      for nndx = 1:numel(all_nets)
        obj.test_train('net_type',all_nets{nndx},...
                       'backend',backend,...
                       'niters', niters, ...
                       'params',params,...
                       'backend_params',backend_params);
      end
    end  % function
    
    function test_setup(obj,varargin)
      %obj.setup_path_();
      [target_trk,simpleprojload] = ...
        myparse(varargin,...
                'target_trk',MFTSetEnum.CurrMovTgtNearCurrFrame,...
                'simpleprojload',false) ;  % if true, just load the proj; use when proj on local filesys with all deps
      
      if simpleprojload
        [labeler, controller] = StartAPT();
        labeler.projLoadGUI(obj.info.ref_lbl);
        obj.labeler = labeler;
        obj.controller = controller ;
        obj.old_lbl = [];
      else
        obj.load_lbl_();
        if ~obj.do_files_exist_()
          obj.setup_data_();
        end      
        [labeler, controller] = obj.create_project_();
        obj.labeler = labeler;
        obj.controller = controller ;
        obj.add_movies_();
        obj.add_labels_quick_();
      end
      
      % Set the labeler tracking mode to target_trk
      if labeler.hasTrx
        trkTypes = MFTSetEnum.TrackingMenuTrx;
      else
        trkTypes = MFTSetEnum.TrackingMenuNoTrx;
      end
      trk_pum_ndx = find(trkTypes == target_trk );      
      obj.labeler.trackModeIdx = trk_pum_ndx ;

      % Set the skeleton edges to match obj.info.op_graph
      if ~isempty(obj.info.op_graph) ,
        obj.labeler.setSkeletonEdges(obj.info.op_graph);
      end
    end  % function
    
        
    function [labeller, controller] = create_project_(obj)
     % Create the new project
      info = obj.info;
      cfg = yaml.ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      cfg.NumViews = info.nviews;
      cfg.NumLabelPoints = info.npts;
      cfg.Trx.HasTrx = info.has_trx;
      cfg.ViewNames = {};
      cfg.LabelPointNames = {};
      cfg.Track.Enable = true;
      cfg.ProjectName = info.proj_name;
      FIELDS2DOUBLIFY = {'Gamma' 'FigurePos' 'AxisLim' 'InvertMovie' 'AxFontSize' 'ShowAxTicks' 'ShowGrid'};
      cfg.View = repmat(cfg.View,cfg.NumViews,1); 
      for i=1:numel(cfg.View)
        cfg.View(i) = ProjectSetup('structLeavesStr2Double',cfg.View(i),FIELDS2DOUBLIFY);
      end

      [labeller,controller] = StartAPT() ;
      %labeller = Labeler() ;
      labeller.initFromConfig(cfg);
      labeller.projNew(cfg.ProjectName);
      labeller.notify('projLoaded');
    end
    
    function add_movies_(obj)
      % Add movies
      labeller = obj.labeler;
      old_lbl = obj.old_lbl;
      info = obj.info;
      nmov = size(old_lbl.movieFilesAll,1);
      PROPS = labeller.gtGetSharedProps();
      for ndx = 1:nmov        
          if info.has_trx
              labeller.movieAdd(old_lbl.movieFilesAll{ndx,1},old_lbl.trxFilesAll{ndx,1});
              if ndx == 1
                labeller.movieSet(1,'isFirstMovie',true);
              else
                labeller.movieSet(ndx);
              end

          else
              labeller.movieSetAdd(old_lbl.movieFilesAll(ndx,:));
              if ndx == 1
                labeller.movieSet(1,'isFirstMovie',true);
              else
                labeller.movieSet(ndx);
              end
              if ~isempty(old_lbl.viewCalibrationData{ndx})
                crObj = old_lbl.viewCalibrationData{ndx};                
                labeller.viewCalSetCurrMovie(crObj);
              end
              for iview = 1:info.nviews
                if ~isempty(old_lbl.movieFilesAllCropInfo{ndx}(iview))
                  roi = old_lbl.movieFilesAllCropInfo{ndx}(iview).roi;
                  % Set the crops
                  labeller.(PROPS.MFACI){ndx}(iview).roi = roi;
                end
              end
              labeller.notify('cropCropsChanged'); 
          end
      end
    end  % function
    
    function add_labels_quick_(obj)
      old_lbl = obj.old_lbl;
      labeller = obj.labeler;
      %info = obj.info;

      % create the table      
      for ndx = 1:numel(old_lbl.labeledpos)
        cur_l = SparseLabelArray.full(old_lbl.labeledpos{ndx});
        cur_o = SparseLabelArray.full(old_lbl.labeledpostag{ndx});
        ss = size(cur_l);
        tgt = zeros(0,1); frm = zeros(0,1);
        p = zeros(0,ss(1)*ss(2));
        occ = false(0,ss(1));
        for tndx = 1:size(cur_l,4)
          cc = reshape(cur_l(:,:,:,tndx),[ss(1)*ss(2) ss(3)]);
          cur_idx = find(any(~isnan(cc),1));
          for lndx = 1:numel(cur_idx)
            tgt(end+1,:) = tndx; %#ok<AGROW> 
            p(end+1,:) = cc(:,cur_idx(lndx))'; %#ok<AGROW> 
            occ(end+1,:) = cur_o(:,cur_idx(lndx))'; %#ok<AGROW> 
            frm(end+1,:) = cur_idx(lndx); %#ok<AGROW> 
          end
        end
        tbl = table(frm,tgt,p,occ,'VariableNames',{'frm','iTgt','p','tfocc'});
        labeller.labelPosBulkImportTblMov(tbl,ndx);
      end            
    end  % function
    
%     function add_labels_(obj)
%       % add the labels
%       old_lbl = obj.old_lbl;
%       labeller = obj.labeller;
%       info = obj.info;
% 
%       lc = labeller.lblCore;
%       nmov = size(old_lbl.movieFilesAll,1);
%       for ndx = 1:nmov
%           labeller.movieSet(ndx);
%           old_labels = SparseLabelArray.full(old_lbl.labeledpos{ndx});
%           ntgts = size(old_labels,4);
%           for itgt = 1:ntgts
% 
%               frms = find(squeeze(any(any(~isnan(old_labels(:,:,:,itgt)),1),2)));
%               if isempty(frms), continue; end        
%               for fr = 1:numel(frms)
%                   cfr = frms(fr);
%                   labeller.setFrameAndTarget(cfr,itgt);
%                   for pt = 1:info.npts
%                       lc.hPts(pt).XData = old_labels(pt,1,cfr,itgt);
%                       lc.hPts(pt).YData = old_labels(pt,2,cfr,itgt);
%                   end
%                   lc.acceptLabels()
%               end        
%           end    
%       end
%     end  % function
        
%     function [labeller, controller] = setup_lbl(obj,ref_lbl)
%       % Start from label file
% 
%       [labeller, controller] = StartAPT() ;
%       %labeller = Labeler();
%       labeller.projLoad(ref_lbl);
%     end

    function setup_alg_for_training_(obj,alg)
      % Set the algorithm.

      if isempty(alg) ,
        % Just leave the algorithm alone
        return
      end

      labeler = obj.labeler;

      if isnumeric(alg)
        trackerIndex = alg;
        assert(trackerIndex > 0, sprintf('No algorithm named %s', alg)) ;
        labeler.trackMakeNewTrackerCurrent(trackerIndex);
      else
        algName = alg ;
        labeler.trackMakeNewTrackerCurrentByName(algName) ;
          % Make a virgin tracker for training
      end
    end  % function
    
    function setup_alg_for_tracking_(obj,alg)
      % Set the algorithm.  Since we're tracking, we use a trained tracker from the
      % history

      if isempty(alg) ,
        % Just leave the algorithm alone
        return
      end

      labeler = obj.labeler;

      if isnumeric(alg)
        trackerIndex = alg;
        assert(trackerIndex > 0, sprintf('No algorithm named %s', alg)) ;
        labeler.trackMakeOldTrackerCurrent(trackerIndex);
      else
        algName = alg ;
        labeler.trackMakeOldTrackerCurrentByName(algName) ;
      end
    end  % function
    
    function set_params_base_(obj, has_trx, dl_steps, manual_radius, batch_size)
      labeller = obj.labeler;
      sPrm = labeller.trackGetTrainingParams();      
      
      sbase = struct() ;
      sbase.AlignUsingTrxTheta = has_trx;
      sbase.dl_steps = dl_steps;
      sbase.ManualRadius = manual_radius;
        % Note 'ManualRadius' by itself may not do anything since
        % 'AutoRadius' is on by default
      if ~isempty(batch_size) ,
        sbase.batch_size = batch_size;
      end
      sPrm2 = structsetleaf(sPrm,sbase,'verbose',true);

      labeller.trackSetTrainingParams(sPrm2);
    end  % function
        
    function set_backend_(obj, backend_type_as_string, raw_backend_params)
      % backend_params: structure (or cell array) containing name-value pairs to be set on the backend
      if iscell(raw_backend_params) ,
        backend_params = struct_from_key_value_list(raw_backend_params) ;
      elseif isstruct(raw_backend_params)
        backend_params = raw_backend_params ;
      else
        error('raw_backend_params must be a cell array or a struct') ;
      end
      labeller = obj.labeler;
      % Set the Backend
      backend_type = DLBackEndFromString(backend_type_as_string) ;
      labeller.set_backend_property('type', backend_type);
      name_from_field_index = fieldnames(backend_params) ;
      for field_index = 1 : numel(name_from_field_index) ,
        name = name_from_field_index{field_index} ;
        value = backend_params.(name) ;
        labeller.set_backend_property(name, value) ;
      end
    end  % function
    
    function test_train(obj,varargin)
      [net_type,...
       backend,...
       niters,...
       test_tracking,...
       block,...
       ~, ...
       batch_size, ...
       params, ...
       backend_params] = ...
        myparse(varargin,...
                'net_type','',...
                'backend','docker',...
                'niters',1000,...
                'test_tracking',false,...
                'block',true,...
                'serial2stgtrain',true,...
                'batch_size',[],...  % used to be 8, but seems better to leave it to the project
                'params',[],... % optional, struct; see structsetleaf
                'backend_params',struct());
          
      if ~isempty(net_type)
        obj.setup_alg_for_training_(net_type)
      end
      fprintf('Training with tracker %s\n',obj.labeler.tracker.algorithmNamePretty);
      obj.set_params_base_(obj.info.has_trx, niters, obj.info.sz, batch_size);
      if ~isempty(params)
        sPrm = obj.labeler.trackGetTrainingParams();
        sPrm = structsetleaf(sPrm,params,'verbose',true);
        obj.labeler.trackSetTrainingParams(sPrm);
      end
      obj.set_backend_(backend,backend_params);

      labeler = obj.labeler;
      labeler.silent = true;
      labeler.train();
      
      if block
        % block while training
        
        % Alternative to polling:
        % ho = HGsetgetObj;
        % ho.data = false;
        % tObj = labeller.tracker;
        % tObj.addlistener('trainEnd',@(s,e)set(ho,'data',true));        
        % waitfor(ho,'data');
        
        pause(2);
        while labeler.bgTrnIsRunning
          pause(10);
        end
        pause(10);
        if test_tracking
          obj.test_track('block', block, 'net_type', net_type, 'backend', backend, 'backend_params', backend_params);
        end
      end      
    end  % function
    
    function test_track(obj,varargin)
      [block,net_type,backend,backend_params] = ...
        myparse(varargin,...
                'block',true,...
                'net_type','',...
                'backend','',...
                'backend_params',struct());
      
      if ~isempty(net_type)
        obj.setup_alg_for_tracking_(net_type)
      end
      if ~isempty(backend),
        obj.set_backend_(backend, backend_params) ;
      end
      labeler = obj.labeler ;
      %kk = LabelerGUI('get_local_fn','pbTrack_Callback');
      %kk(obj.labeler.gdata.pbTrack,[],obj.labeler.gdata);      
      labeler.track() ;
      if block,
        pause(2);
        while labeler.bgTrkIsRunning
          pause(10);
        end
        pause(10);
      end
    end  % function
    
    function test_track_export(obj)
      labeller = obj.labeler;
      iMov = labeller.currMovie;
      nvw = labeller.nview;
      tfiles = arrayfun(@(x)tempname(),1:nvw,'uni',0);
      labeller.trackExportResults(iMov,'trkfiles',tfiles);
      
      for ivw=1:nvw
        trk = TrkFile.load(tfiles{ivw});
        fprintf(1,'Exported and re-loaded trkfile!\n');
        if nvw>1
          fprintf(1,'  View %d:\n',ivw);
        end
        disp(trk);
      end
    end  % function
    
    function test_gtcompute(obj,varargin)
      [block, backend, backend_params] = ...
       myparse(varargin, ...
               'block',true, ...
               'backend','', ...
               'backend_params',struct());
      if ~isempty(backend),
        obj.set_backend_(backend,backend_params);
      end
      %kk = LabelerGUI('get_local_fn','pbTrack_Callback');
      %kk(obj.labeller.gdata.pbTrack,[],obj.labeller.gdata);
      obj.labeler.gtSetGTMode(1);
      drawnow;
      %obj.labeler.tracker.trackGT();  % Seems this is a deprecated way to run
                                       % tracking for GT.  -- ALT, 2024-11-04
      obj.labeler.gtComputeGTPerformance() ;
      if block,
        pause(2);
        while obj.labeler.bgTrkIsRunning
          pause(10);
        end
        pause(10);
      end
    end  % function

    % function test_quick(obj, proj_file, net, backend, backend_params) 
    %   obj.setup_lbl(proj_file);
    %   %lObj = tobj.lObj;
    %   % s=lObj.trackGetParams;
    %   % s.ROOT.DeepTrack.DataAugmentation.rrange = 10;
    %   % s.ROOT.DeepTrack.DataAugmentation.trange = 5;
    %   % s.ROOT.DeepTrack.DataAugmentation.scale_factor_range = 1.1;
    %   % s.ROOT.DeepTrack.ImageProcessing.scale = 1.;
    %   % lObj.trackSetTrainingParams(s);
    %   obj.setup_alg_(net);
    %   obj.setup_backend_(backend,backend_params);
    % end  % function
  end  % methods
  
  methods (Static) 
    function runCITestSuite(varargin)
      % testAPT.sh interface
      % for triggering tests from unix commandline, CI, etc    
      
      disp('### TestAPT/runCITestSuite ###');
      disp(varargin);
      
%       APTPATH = '/groups/branson/home/leea30/git/apt.param';
%       addpath(APTPATH);
%       APT.setpath
      
      if isempty(varargin) ,
        varargin = {'full', 'roianma2', 'backend', 'conda', 'nets', {'multi_cid'} } ;
      end
      action = varargin{1};
      switch action
        case 'train'          
          name = varargin{2};
          iTracker = varargin{3};
          iTracker = str2double(iTracker);
          dotrack = varargin{4};
          dotrack = str2double(dotrack);
          be = varargin{5};
          
          TRNITERS = 500;
          
          testObj = TestAPT('name',name);
          testObj.test_setup('simpleprojload',1);
          testObj.labeller.projTempDirDontClearOnDestructor = true;
          testObj.labeller.trackAutoSetParams = false;
          testObj.test_train(...
            'net_type',iTracker,...
            'niters',TRNITERS,...
            'params',struct('batch_size',2),...
            'backend',be,...
            'test_tracking',false);
          
          disp('Train done!');          
          trackerObj = testObj.labeller.tracker;
          trackerInfo = trackerObj.trackerInfo;
          iters1 = trackerInfo.iterFinal;
          if iters1==TRNITERS
            fprintf(1,'Final iteration (%d) matches expected (%d)!\n',...
              iters1,TRNITERS);
          else
            error('apttest:missedfinaliter',...
              'Final iteration (%d) is not expected (%d)!',iters1,TRNITERS);
          end
          
          if dotrack
            pause(10);
            testObj.test_track(...
              'net_type',iTracker,...
              'backend',be,...
              'block',true ...
              );
            
            disp('Track done!');
            testObj.test_track_export();
          end
          
        case 'track'          
          name = varargin{2};
          iTracker = varargin{3};
          iTracker = str2double(iTracker);
                    
          testObj = TestAPT('name',name);
          testObj.test_setup('simpleprojload',1);
          testObj.labeller.projTempDirDontClearOnDestructor = true;
          testObj.labeller.trackAutoSetParams = false;
          testObj.test_track(...
            'net_type',iTracker,...
            'block',true ...
            );
          
          disp('Track done!');
          %pause(10);
          testObj.test_track_export();
          
        case 'full'
          name = varargin{2};
          rest_of_args = varargin(3:end) ;
          testObj = TestAPT('name',name);
          %testObj.test_setup('simpleprojload',true);
          testObj.test_full('setup_params', {'simpleprojload',true}, rest_of_args{:});
          testObj.labeller.projTempDirDontClearOnDestructor = true;
          testObj.labeller.trackAutoSetParams = false;
          
        case 'hello'
          disp('hello!');
          
        case 'testerr'
          error('TestAPT:testerr','Test error!');
          
        otherwise
          error('TestAPT:testerr','Unrecognized action: %s',action);
      end
    end  % function

    function exp_name = get_exp_name(info, pin)
      % For alice we care only about the directory one level above eg.
      % GMRxx
      % For Stephen we care about the whole path from flp-xxx
      exp_dir_base = info.exp_dir_base;
      if strcmp(info.name,'alice')
        [~,exp_name] = fileparts(fileparts(pin));
      elseif strcmp(info.name,'stephen')
        exp_name = fileparts(strrep(pin,exp_dir_base,''));
      end      
    end  % function
            
    function [data_dir, lbl_file] = get_file_paths(info)
      cacheDir = APT.getdotaptdirpath();
      data_dir = fullfile(cacheDir,info.proj_name);
      [~,lbl_name,ext] = fileparts(info.ref_lbl);
      lbl_file = fullfile(data_dir,strcatg(lbl_name,ext));
    end  % function
    
%     function create_lbl_sh()
%       % For Stephen's projects, we select 5 movies from the label file, add
%       % and create a new project based on them. The labels are selected
%       % from trk files. This project then becomes the old_lbl. As usual
%       % the naming old_lbl becomes non-sensical, but we will continue to
%       % use it to expose the limitations of our foresight. I say "ours"
%       % because I refuse to believe I'm alone in this.
%       A = loadLbl('/groups/branson/bransonlab/apt/experiments/data/sh_trn5017_20200121.lbl');
% 
%       info = struct();
%       info.npts = 5;
%       info.nviews = 2;
%       info.has_trx = false;
%       info.proj_name = 'sh_test';
%       cfg = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
%       cfg.NumViews = info.nviews;
%       cfg.NumLabelPoints = info.npts;
%       cfg.Trx.HasTrx = info.has_trx;
%       cfg.ViewNames = {};
%       cfg.LabelPointNames = {};
%       cfg.Track.Enable = true;
%       cfg.ProjectName = info.proj_name;
%       FIELDS2DOUBLIFY = {'Gamma' 'FigurePos' 'AxisLim' 'InvertMovie' 'AxFontSize' 'ShowAxTicks' 'ShowGrid'};
%       cfg.View = repmat(cfg.View,cfg.NumViews,1); 
%       for i=1:numel(cfg.View)
%         cfg.View(i) = ProjectSetup('structLeavesStr2Double',cfg.View(i),FIELDS2DOUBLIFY);
%       end
% 
%       labeller = Labeler();
%       labeller.initFromConfig(cfg);
%       labeller.projNew(cfg.ProjectName);
%       labeller.notify('projLoaded');
% 
%       PROPS = labeller.gtGetSharedProps();
%       % select 7 movie sets from > 512, because movies < 512 don't have ortho
%       % cal calibrations
%       mov_lbl = [571 653 609 634 729]; 
%       lbl_space = 20;
%       for ndx = 1:numel(mov_lbl)
%         cur_ndx = mov_lbl(ndx);
%         cur_movs = {};
%         trk_files = {};
%         for mndx = 1:info.nviews
%           in_mov = A.movieFilesAll{cur_ndx,mndx};
%           cur_movs{mndx} = FSPath.macroReplace(in_mov,A.projMacros);
%           trk_files{mndx} = strrep(cur_movs{mndx},'.avi','.trk');
%         end
%         labeller.movieSetAdd(cur_movs);  
%         if ndx == 1
%             labeller.movieSet(1,'isFirstMovie',true);
%         else
%             labeller.movieSet(ndx);    
%         end
%         crObj = A.viewCalibrationData{cur_ndx};
%         labeller.viewCalSetCurrMovie(crObj);
%         for iview = 1:info.nviews
%           roi = A.movieFilesAllCropInfo{cur_ndx}(iview).roi;
%           % Set the crops
%           labeller.(PROPS.MFACI){ndx}(iview).roi = roi;
%         end
%         labeller.notify('cropCropsChanged'); 
% 
%         % Add the labels from trk file.
%         cur_l = labeller.labeledpos{ndx};
%         ss = size(cur_l);
%         all_trk = {};
%         for mndx = 1:info.nviews
%           trk_in = load(trk_files{mndx},'-mat');
%           cc = trk_in.pTrk;
%           cur_idx = 1:lbl_space:size(cc,3);
%           all_trk{end+1} = cc(:,:,cur_idx);
%         end
%         p = cat(1,all_trk{:});
%         p = reshape(p,[], size(p,3))';
%         npts = numel(cur_idx);
%         tgt = ones(npts,1);
%         frm = cur_idx';
%         occ = false(npts,ss(1));
%         tbl = table(frm,tgt,p,occ,'VariableNames',{'frm','iTgt','p','tfocc'});
%         labeller.labelPosBulkImportTblMov(tbl,ndx);
% 
%       end
% 
%       % Two movies without labels.
%       mov_nlbl = [545 694]; 
%       for ndx = 1:numel(mov_nlbl)
%         cur_ndx = mov_nlbl(ndx);
%         cur_movs = {};
%         trk_files = {};
%         for mndx = 1:info.nviews
%           in_mov = A.movieFilesAll{cur_ndx,mndx};
%           cur_movs{mndx} = FSPath.macroReplace(in_mov,A.projMacros);
%           trk_files{mndx} = strrep(cur_movs{mndx},'.avi','.trk');
%         end
%         labeller.movieSetAdd(cur_movs);  
%         labeller.movieSet(ndx+numel(mov_lbl));    
%         crObj = A.viewCalibrationData{cur_ndx};
%         labeller.viewCalSetCurrMovie(crObj);
%         for iview = 1:info.nviews
%           roi = A.movieFilesAllCropInfo{cur_ndx}(iview).roi;
%           % Set the crops
%           labeller.(PROPS.MFACI){ndx+numel(mov_lbl)}(iview).roi = roi;
%         end
%         labeller.notify('cropCropsChanged'); 
% 
%       end
% 
%       uc_fn = LabelerGUI('get_local_fn','set_use_calibration');
%       uc_fn(labeller.gdata,true);
%       dstr = datestr(now,'YYYYmmDD');
%       out_file = fullfile('/groups/branson/bransonlab/mayank/APT_projects/',...
%         sprintf('sh_test_lbl_%s.lbl',dstr));
%       labeller.projSaveRaw(out_file);
%     end    
    
%     function cdir = get_cache_dir()
%       cdir = APT.getdotaptdirpath() ;
%     end
        
  end  % methods (Static)
end  % classdef
