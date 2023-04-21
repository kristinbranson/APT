classdef testAPT < handle
  
  % Simplest way to test:
  % testObj = testAPT('name','alice'); 
  % testObj.test_full('nets',{'mdn','deeplabcut'});
  
  % If you want to interact with GUI before training:
  % testObj = testAPT('name','alice');
  % testObj.test_setup();
  % Mess with GUI
  % testObj.test_train('net_type','mdn',...
  %        'backend','docker','niters',1000,'test_tracking',true)
  
  % MA/roian
  % testObj = testAPT('name','roianma');
  % testObj.test_setup('simpleprojload',1);
  % testObj.test_train('net_type',[],'params',-1,'niters',1000);  
  
  % MA/roian, Kristin's suggestion:
  % testObj = testAPT('name','roianma');  
  % testObj.test_full('nets',{},'setup_params',{'simpleprojload',1,'jrcgpuqueue','gpu_rtx8000','jrcnslots',4},'backend','bsub');
  % testObj.test_full('nets',{'Stg1tddobj_Stg2tdpobj','magrone','Stg1tddht_Stg2tdpht','maopenpose'},'setup_params',{'simpleprojload',1,'jrcgpuqueue','gpu_rtx8000','jrcnslots',4},'backend','docker');
  % empty nets means test all nets
  
  % Carmen/GT workflow (proj on JRC/dm11)
  % testObj = testAPT('name','carmen');
  % testObj.test_setup('simpleprojload',1);
  % testObj.test_train('backend','bsub');
  % testObj.test_track('backend','bsub');
  % testObj.test_gtcompute('backend','bsub');
  
  
  properties
    lObj = [];
    info = [];
    old_lbl = [];
    path_setup_done = false;
  end
  
  methods (Static)
            
    function create_lbl_sh()
      % For Stephen's projects, we select 5 movies from the label file, add
      % and create a new project based on them. The labels are selected
      % from trk files. This project then becomes the old_lbl. As usual
      % the naming old_lbl becomes non-sensical, but we will continue to
      % use it to expose the limitations of our foresight. I say "ours"
      % because I refuse to believe I'm alone in this.
      A = loadLbl('/groups/branson/bransonlab/apt/experiments/data/sh_trn5017_20200121.lbl');

      info = struct();
      info.npts = 5;
      info.nviews = 2;
      info.has_trx = false;
      info.proj_name = 'sh_test';
      cfg = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
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

      lObj = Labeler;
      lObj.initFromConfig(cfg);
      lObj.projNew(cfg.ProjectName);
      lObj.notify('projLoaded');

      PROPS = lObj.gtGetSharedProps();
      % select 7 movie sets from > 512, because movies < 512 don't have ortho
      % cal calibrations
      mov_lbl = [571 653 609 634 729]; 
      lbl_space = 20;
      for ndx = 1:numel(mov_lbl)
        cur_ndx = mov_lbl(ndx);
        cur_movs = {};
        trk_files = {};
        for mndx = 1:info.nviews
          in_mov = A.movieFilesAll{cur_ndx,mndx};
          cur_movs{mndx} = FSPath.macroReplace(in_mov,A.projMacros);
          trk_files{mndx} = strrep(cur_movs{mndx},'.avi','.trk');
        end
        lObj.movieSetAdd(cur_movs);  
        if ndx == 1
            lObj.movieSet(1,'isFirstMovie',true);
        else
            lObj.movieSet(ndx);    
        end
        crObj = A.viewCalibrationData{cur_ndx};
        lObj.viewCalSetCurrMovie(crObj);
        for iview = 1:info.nviews
          roi = A.movieFilesAllCropInfo{cur_ndx}(iview).roi;
          % Set the crops
          lObj.(PROPS.MFACI){ndx}(iview).roi = roi;
        end
        lObj.notify('cropCropsChanged'); 

        % Add the labels from trk file.
        cur_l = lObj.labeledpos{ndx};
        ss = size(cur_l);
        all_trk = {};
        for mndx = 1:info.nviews
          trk_in = load(trk_files{mndx},'-mat');
          cc = trk_in.pTrk;
          cur_idx = 1:lbl_space:size(cc,3);
          all_trk{end+1} = cc(:,:,cur_idx);
        end
        p = cat(1,all_trk{:});
        p = reshape(p,[], size(p,3))';
        npts = numel(cur_idx);
        tgt = ones(npts,1);
        frm = cur_idx';
        occ = false(npts,ss(1));
        tbl = table(frm,tgt,p,occ,'VariableNames',{'frm','iTgt','p','tfocc'});
        lObj.labelPosBulkImportTblMov(tbl,ndx);

      end

      % Two movies without labels.
      mov_nlbl = [545 694]; 
      for ndx = 1:numel(mov_nlbl)
        cur_ndx = mov_nlbl(ndx);
        cur_movs = {};
        trk_files = {};
        for mndx = 1:info.nviews
          in_mov = A.movieFilesAll{cur_ndx,mndx};
          cur_movs{mndx} = FSPath.macroReplace(in_mov,A.projMacros);
          trk_files{mndx} = strrep(cur_movs{mndx},'.avi','.trk');
        end
        lObj.movieSetAdd(cur_movs);  
        lObj.movieSet(ndx+numel(mov_lbl));    
        crObj = A.viewCalibrationData{cur_ndx};
        lObj.viewCalSetCurrMovie(crObj);
        for iview = 1:info.nviews
          roi = A.movieFilesAllCropInfo{cur_ndx}(iview).roi;
          % Set the crops
          lObj.(PROPS.MFACI){ndx+numel(mov_lbl)}(iview).roi = roi;
        end
        lObj.notify('cropCropsChanged'); 

      end

      uc_fn = LabelerGUI('get_local_fn','set_use_calibration');
      uc_fn(lObj.gdata,true);
      dstr = datestr(now,'YYYYmmDD');
      out_file = fullfile('/groups/branson/bransonlab/mayank/APT_projects/',...
        sprintf('sh_test_lbl_%s.lbl',dstr));
      lObj.projSaveRaw(out_file);
    end    
    
    function cdir = get_cache_dir()
      cdir = APT.getdlcacheroot;
    end
        
  end
  
  methods
    
    function testObj = testAPT(varargin)
      testObj.setup_path();
      [name] = myparse(varargin,'name','alice');
      if ~strcmp(name,'dummy')
        testObj.get_info(name);      
      end
    end
    
    function exp_name = get_exp_name(self,pin)
      % For alice we care only about the directory one level above eg.
      % GMRxx
      % For Stephen we care about the whole path from flp-xxx
      info = self.info;
      exp_dir_base = info.exp_dir_base;
      if strcmp(info.name,'alice')
        [~,exp_name] = fileparts(fileparts(pin));
      elseif strcmp(info.name,'stephen')
        exp_name = fileparts(strrep(pin,exp_dir_base,''));
      end      
    end
    
    function create_bundle(self)
      % creates a targz to upload to Drobox
      info = self.info;
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
          exp_name = self.get_exp_name(all_m{ndx,iview});
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
    end

    
    function setup_path(self)
      if ~self.path_setup_done && ~isdeployed
        addpath('..');
        APT.setpath;
        self.path_setup_done = true;
      end      
    end

    function get_info(self,name)
      info = struct;
      if strcmp(name,'alice')      
%         info.ref_lbl = '/work/mayank/work/FlySpaceTime/multitarget_bubble_expandedbehavior_20180425_fixederrors_fixed.lbl';
        info.ref_lbl = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/multitarget_bubble_expandedbehavior_20180425_allGT_MK_MDN04182019.lbl';
        info.exp_dir_base = '/groups/branson/home/robiea/Projects_data/Labeler_APT';
        info.nviews = 1;
        info.npts = 17;
        info.has_trx = true;
        info.proj_name = 'alice_test';
        info.sz = 90;
        info.bundle_link = 'https://www.dropbox.com/s/u5p27rdi7kczv78/alice_test_data.tar.gz?dl=1';
        info.op_graph = [1 3; 1 2; 3 17; 2 12; 3 4; 4 10; 10 11; 11 16; 10 6; 8 6; 5 8; 8 9; 9 13;6 14;6 15; 6 7];

      elseif strcmp(name,'stephen')
        info.ref_lbl = '/groups/branson/bransonlab/mayank/APT_projects/sh_test_lbl_20200310.lbl';
        info.exp_dir_base = '/groups/huston/hustonlab/flp-chrimson_experiments';
        info.nviews = 2;
        info.npts = 5;
        info.has_trx = false;
        info.proj_name = 'stephen_test';
        info.sz = [];
        info.bundle_link = 'https://www.dropbox.com/s/asl1f3ssfgtdwmc/stephen_test_data.tar.gz?dl=1';
        info.op_graph = [1 5; 1 3; 3 4; 4 2];
        
      elseif strcmp(name,'carmen')
        info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/pez7_al.lbl';
        info.exp_dir_base = '';
        info.nviews = 1;
        info.npts = 10;
        info.has_trx = false;
        info.proj_name = 'carmen_test';
        info.sz = [];
        info.bundle_link = '';
        info.op_graph = [];   
        
      elseif strcmp(name,'roianma')
        info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/four_points_180806_ma_bothmice_extra_labels_re_radius_150_ds2_gg_add_movie_UT_20210929_trunc.lbl';
        info.exp_dir_base = '';
        info.nviews = nan;
        info.npts = nan;
        info.has_trx = false;
        info.proj_name = 'test';
        info.sz = 250;
        info.bundle_link = '';
        info.op_graph = [];   
        
      elseif strcmp(name,'argrone')
        info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/flybubble_grone_20210523_allGT_KB_20210626_UT_20210823.lbl';
        info.exp_dir_base = '';
        info.nviews = nan;
        info.npts = nan;
        info.has_trx = true;
        info.proj_name = 'test';
        info.sz = 90;
        info.bundle_link = '';
        info.op_graph = [];   
        
      elseif strcmp(name,'argroneSA')
        info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT.lbl';
        info.exp_dir_base = '';
        info.nviews = nan;
        info.npts = nan;
        info.has_trx = true;
        info.proj_name = 'test';
        info.sz = 90;
        info.bundle_link = '';
        info.op_graph = [];   
        
      elseif strcmp(name,'sam2view')
        info.ref_lbl = '/groups/branson/bransonlab/apt/unittest/2011_mouse_cam13.lbl';
        info.exp_dir_base = '';
        info.nviews = 2;
        info.npts = nan;
        info.has_trx = false;
        info.proj_name = 'test';
        info.sz = 100; % dont set this to empty even if it is not used
        info.bundle_link = '';
        info.op_graph = [];   
        
        
        
      else
        error('Unrecognized test name');
      end
      info.name = name;
      self.info = info;
    end
        
    function [data_dir, lbl_file] = get_file_paths(self)
      info = self.info;
      cacheDir = testAPT.get_cache_dir();
      data_dir = fullfile(cacheDir,info.proj_name);
      [~,lbl_name,ext] = fileparts(info.ref_lbl);
      lbl_file = fullfile(data_dir,[lbl_name ext]);
    end
    
    function ok = exist_files(self)
      old_lbl = self.old_lbl;
      info = self.info;
      ok = true;
      for ndx = 1:numel(old_lbl.movieFilesAll),
        if ~exist(old_lbl.movieFilesAll{ndx},'file')
            ok = false;
            return;          
        end
        if info.has_trx && ~exist(old_lbl.trxFilesAll{ndx},'file')
            ok = false;
            return;          
        end        
      end
      
    end
    
    function load_lbl(self)
      info = self.info;
      [data_dir, lbl_file] = self.get_file_paths();
      if ~exist(lbl_file,'file')
        self.setup_data();
      end
      old_lbl = loadLbl(lbl_file);
      old_lbl.movieFilesAll = FSPath.macroReplace(old_lbl.movieFilesAll,old_lbl.projMacros);
      old_lbl.movieFilesAllGT = FSPath.macroReplace(old_lbl.movieFilesAllGT,old_lbl.projMacros);
      all_m = old_lbl.movieFilesAll;
      for ndx = 1:size(old_lbl.movieFilesAll,1)
        for iview = 1:info.nviews
          [tt,mov_name,mext] = fileparts(all_m{ndx,iview});
          exp_name = self.get_exp_name(all_m{ndx,iview});
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
          [tt,mov_name,mext] = fileparts(all_m{ndx,iview});
          exp_name = self.get_exp_name(all_m{ndx,iview});
          old_lbl.movieFilesAllGT{ndx,iview} = fullfile(data_dir,exp_name,[mov_name mext]);
          if info.has_trx
            trx_file_name = old_lbl.trxFilesAllGT{ndx};
            trx_file_name = strrep(trx_file_name,'$movdir',fullfile(data_dir,exp_name));
            old_lbl.trxFilesAllGT{ndx} = trx_file_name;
          end
        end
      end
      self.old_lbl = old_lbl;
    end
    
    function setup_data(self)
      info = self.info;
      cacheDir = testAPT.get_cache_dir();
      out_file = fullfile(tempdir,[info.proj_name '_data.tar.gz']); 
            
      try
        untar(out_file,cacheDir);
        return;
      catch ME
        % none, fallthru
      end

      % fallback to websave
      try
        websave(out_file,info.bundle_link);
      catch ME
        if endsWith(ME.identifier,'SSLConnectionSystemFailure')
          % JRC cluster
          wo = weboptions('CertificateFilename','/etc/ssl/certs/ca-bundle.crt');
          websave(out_file,info.bundle_link,wo);
        else
          rethrow(ME);
        end
      end
      
      untar(out_file,cacheDir);          
    end
    
    function test_full(self,varargin)
      [all_nets,backend,params,aws_params,setup_params] = myparse(varargin,...
        'nets',{'mdn'},'backend','docker','params',{},...
        'aws_params',struct(),'setup_params',{});
      self.test_setup(setup_params{:});

      if ischar(all_nets) || isscalar(all_nets),
        all_nets = {all_nets};
      end
      if isempty(all_nets),
        all_nets = num2cell(1:numel(self.lObj.trackersAll));
      end
      if isnumeric(all_nets),
        all_nets = num2cell(all_nets);
      end
      for nndx = 1:numel(all_nets)
        self.test_train('net_type',all_nets{nndx},'backend',backend,...
          'params',params,'aws_params',aws_params);
      end
    end
    
    function test_setup(self,varargin)
      self.setup_path();
      [target_trk,simpleprojload,jrcgpuqueue,jrcnslots] = myparse(varargin,...
        'target_trk',MFTSetEnum.CurrMovTgtNearCurrFrame,...
        'simpleprojload',false, ... % if true, just load the proj; use when proj on local filesys with all deps
        'jrcgpuqueue','',... % override gpu queue
        'jrcnslots',[]... % override nslots
        );
      
      if simpleprojload
        lObj = StartAPT();
        lObj.projLoad(self.info.ref_lbl);
        self.lObj = lObj;
        self.old_lbl = [];
      else
        self.load_lbl();
        old_lbl = self.old_lbl;
        if ~self.exist_files()
          self.setup_data();
        end      
        lObj = self.create_project();
        self.lObj = lObj;
        self.add_movies();
        self.add_labels_quick();
      end
      
      if lObj.hasTrx
        trkTypes = MFTSetEnum.TrackingMenuTrx;
      else
        trkTypes = MFTSetEnum.TrackingMenuNoTrx;
      end
      trk_pum_ndx = find(trkTypes == target_trk );      
      set(self.lObj.gdata.pumTrack,'Value',trk_pum_ndx);
      self.lObj.setSkeletonEdges(self.info.op_graph);
      
      if ~isempty(jrcgpuqueue),
        for i = 1:numel(lObj.trackersAll),
          if isprop(lObj.trackersAll{i},'jrcgpuqueue'),
            lObj.trackersAll{i}.setJrcgpuqueue(jrcgpuqueue);
          end
        end
      end
      if ~isempty(jrcnslots),
        for i = 1:numel(lObj.trackersAll),
          if isprop(lObj.trackersAll{i},'jrcnslots'),
            lObj.trackersAll{i}.setJrcnslots(jrcnslots);
          end
        end
      end
    end
    
        
    function lObj = create_project(self)
     % Create the new project
      info = self.info;
      cfg = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
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

      lObj = Labeler;
      lObj.initFromConfig(cfg);
      lObj.projNew(cfg.ProjectName);
      lObj.notify('projLoaded');
    end
    
    function add_movies(self)
      % Add movies
      lObj = self.lObj;
      old_lbl = self.old_lbl;
      info = self.info;
      nmov = size(old_lbl.movieFilesAll,1);
      PROPS = lObj.gtGetSharedProps();
      for ndx = 1:nmov        
          if info.has_trx
              lObj.movieAdd(old_lbl.movieFilesAll{ndx,1},old_lbl.trxFilesAll{ndx,1});
              if ndx == 1
                lObj.movieSet(1,'isFirstMovie',true);
              else
                lObj.movieSet(ndx);
              end

          else
              lObj.movieSetAdd(old_lbl.movieFilesAll(ndx,:));
              if ndx == 1
                lObj.movieSet(1,'isFirstMovie',true);
              else
                lObj.movieSet(ndx);
              end
              if ~isempty(old_lbl.viewCalibrationData{ndx})
                crObj = old_lbl.viewCalibrationData{ndx};                
                lObj.viewCalSetCurrMovie(crObj);
              end
              for iview = 1:info.nviews
                if ~isempty(old_lbl.movieFilesAllCropInfo{ndx}(iview))
                  roi = old_lbl.movieFilesAllCropInfo{ndx}(iview).roi;
                  % Set the crops
                  lObj.(PROPS.MFACI){ndx}(iview).roi = roi;
                end
              end
              lObj.notify('cropCropsChanged'); 
          end
      end
    end
    
    function add_labels_quick(self)
      old_lbl = self.old_lbl;
      lObj = self.lObj;
      info = self.info;

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
            tgt(end+1,:) = tndx;
            p(end+1,:) = cc(:,cur_idx(lndx))';
            occ(end+1,:) = cur_o(:,cur_idx(lndx))';
            frm(end+1,:) = cur_idx(lndx);
          end
        end
        tbl = table(frm,tgt,p,occ,'VariableNames',{'frm','iTgt','p','tfocc'});
        lObj.labelPosBulkImportTblMov(tbl,ndx);
      end      
      
    end
    
    function add_labels(self)
      % add the labels
      old_lbl = self.old_lbl;
      lObj = self.lObj;
      info = self.info;

      lc = lObj.lblCore;
      nmov = size(old_lbl.movieFilesAll,1);
      for ndx = 1:nmov
          lObj.movieSet(ndx);
          old_labels = SparseLabelArray.full(old_lbl.labeledpos{ndx});
          ntgts = size(old_labels,4);
          for itgt = 1:ntgts

              frms = find(squeeze(any(any(~isnan(old_labels(:,:,:,itgt)),1),2)));
              if isempty(frms), continue; end        
              for fr = 1:numel(frms)
                  cfr = frms(fr);
                  lObj.setFrameAndTarget(cfr,itgt);
                  for pt = 1:info.npts
                      lc.hPts(pt).XData = old_labels(pt,1,cfr,itgt);
                      lc.hPts(pt).YData = old_labels(pt,2,cfr,itgt);
                  end
                  lc.acceptLabels()
              end        
          end    
      end
    end
        
    function lObj = setup_lbl(self,ref_lbl)
      % Start from label file

      lObj = Labeler;
      lObj.projLoad(ref_lbl);
    end

    function setup_alg(self,alg)
      % Set the algorithm.

      lObj = self.lObj;

      if isnumeric(alg)
        tndx = alg;
      else
        nalgs = numel(lObj.trackersAll);
        tndx = 0;
        for ix = 1:nalgs
          if strcmp(lObj.trackersAll{ix}.algorithmName,alg)
            tndx = ix;
          end
        end
      end
      assert(tndx > 0)
      lObj.trackSetCurrentTracker(tndx);
    end
    
    function set_params_base(self,has_trx,dl_steps,sz, batch_size)
      lObj = self.lObj;
      sPrm = lObj.trackGetParams();      
      
      sbase.AlignUsingTrxTheta = has_trx;
      sbase.dl_steps = dl_steps;
      sbase.ManualRadius = sz;
      sbase.batch_size = batch_size;
      sPrm = structsetleaf(sPrm,sbase,'verbose',true);

      % AL: Note 'ManualRadius' by itself may not do anything since
      % 'AutoRadius' is on by default
      lObj.trackSetParams(sPrm);
      
%       sPrm.ROOT.DeepTrack.GradientDescent.dl_steps = dl_steps;
%       sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.Radius = sz;
%       if has_trx
%         sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.AlignUsingTrxTheta = has_trx;
%       end
%       for ndx = 1:2:numel(params)
%         sPrm = setfield(sPrm,params{ndx}{:},params{ndx+1});
%       end
    end
        
    function set_backend(self,backend,aws_params)
      % aws_params: can be a pre-configured AWSec2 instance
      
      lObj = self.lObj;

      % Set the Backend
      if strcmp(backend,'docker')
        beType = DLBackEnd.Docker;
      elseif strcmp(backend,'bsub')
        beType = DLBackEnd.Bsub;
      elseif strcmp(backend,'conda')
        beType = DLBackEnd.Conda;
      elseif strcmp(backend,'aws')
        beType = DLBackEnd.AWS;
      end
      be = DLBackEndClass(beType,lObj.trackGetDLBackend);
      lObj.trackSetDLBackend(be);
      if strcmp(backend,'aws')
        if isa(aws_params,'AWSec2')
          be.shutdown();
          be.awsec2 = aws_params;
        else
          aobj = lObj.trackDLBackEnd.awsec2;
          aobj.setPemFile(aws_params.pemFile);
          aobj.setKeyName(aws_params.keyName);
          if isfield(aws_params,'instanceID') && ...
              isfield(aws_params,'instanceType') &&...
              ~isempty(aws_params.instanceID) && ...
              ~isempty(aws_params.instanceType)
            aobj.setInstanceID(...
              aws_params.instanceID,aws_params.instanceType);
          else
            tf = aobj.launchInstance();
            if ~tf
              reason = 'Could not launch AWS EC2 instance.';
              return;
            end
            instanceID = aobj.instanceID;
            instanceType = aobj.instanceType;
            aobj.setInstanceID(instanceID,instanceType);

          end
        end        
      end
    end
    
    
    function test_train(self,varargin)
      [net_type,backend,niters,test_tracking,block,serial2stgtrain, ...
        batch_size, params, aws_params] = myparse(varargin,...
            'net_type','mdn','backend','docker',...
            'niters',1000,'test_tracking',true,'block',true,...
            'serial2stgtrain',true,...
            'batch_size',8,...
            'params',[],... % optional, struct; see structsetleaf
            'aws_params',struct());
          
      if ~isempty(net_type)
        self.setup_alg(net_type)
      end
      fprintf('Training with tracker %s\n',self.lObj.tracker.algorithmNamePretty);
      self.set_params_base(self.info.has_trx,niters,self.info.sz, batch_size);
      if ~isempty(params)
        sPrm = self.lObj.trackGetParams();
        sPrm = structsetleaf(sPrm,params,'verbose',true);
        self.lObj.trackSetParams(sPrm);
      end
      self.set_backend(backend,aws_params);

      lObj = self.lObj;
      handles = lObj.gdata;
      %oc1 = onCleanup(@()ClearStatus(handles));
      wbObj = WaitBarWithCancel('Training');
      oc2 = onCleanup(@()delete(wbObj));
      centerOnParentFigure(wbObj.hWB,handles.figure);
      tObj = lObj.tracker;
      tObj.skip_dlgs = true;
      lObj.silent = true;
      if lObj.trackerIsTwoStage && ~strcmp(backend,'bsub')
        tObj.forceSerial = serial2stgtrain;
      end      
      lObj.trackRetrain('retrainArgs',{'wbObj',wbObj});
      if wbObj.isCancel
        msg = wbObj.cancelMessage('Training canceled');
        msgbox(msg,'Train');
      end      
      
      if block
        % block while training
        
        % Alternative to polling:
        % ho = HGsetgetObj;
        % ho.data = false;
        % tObj = lObj.tracker;
        % tObj.addlistener('trainEnd',@(s,e)set(ho,'data',true));        
        % waitfor(ho,'data');
        
        pause(2);
        while tObj.bgTrnIsRunning
          pause(10);
        end
        pause(30);
        if test_tracking
          self.test_track('block',block);
        end
      end      
    end
    
    function test_track(self,varargin)
      [block,net_type,backend,aws_params] = myparse(varargin,...
        'block',true,...
        'net_type',[],...
        'backend','','aws_params',struct);
      
      if ~isempty(net_type)
        self.setup_alg(net_type)
      end
      if ~isempty(backend),
        self.set_backend(backend,aws_params);
      end
      kk = LabelerGUI('get_local_fn','pbTrack_Callback');
      kk(self.lObj.gdata.pbTrack,[],self.lObj.gdata);      
      if block,
        pause(2);
        while self.lObj.tracker.bgTrkIsRunning
          pause(10);
        end
        pause(10);
      end
    end
    
    function test_track_export(self)
      lObj = self.lObj;
      iMov = lObj.currMovie;
      nvw = lObj.nview;
      tfiles = arrayfun(@(x)tempname(),1:nvw,'uni',0);
      lObj.trackExportResults(iMov,'trkfiles',tfiles);
      
      for ivw=1:nvw
        trk = TrkFile.load(tfiles{ivw});
        fprintf(1,'Exported and re-loaded trkfile!\n');
        if nvw>1
          fprintf(1,'  View %d:\n',ivw);
        end
        disp(trk);
      end
    end
    
    function test_gtcompute(self,varargin)
      [block,backend,aws_params] = myparse(varargin,'block',true,...
        'backend','','aws_params',struct);
      if ~isempty(backend),
        self.set_backend(backend,aws_params);
      end
      %kk = LabelerGUI('get_local_fn','pbTrack_Callback');
      %kk(self.lObj.gdata.pbTrack,[],self.lObj.gdata);
      self.lObj.gtSetGTMode(1);
      drawnow;
      self.lObj.tracker.trackGT();
      if block,
        pause(2);
        while self.lObj.tracker.bgTrkIsRunning
          pause(10);
        end
        pause(10);
      end
    end
    
  end
  
  methods (Static) 
    % testAPT.sh interface
    % for triggering tests from unix commandline, CI, etc
    
    function CIsuite(varargin)
      
      disp('### testAPT/CIsuite ###');
      disp(varargin);
      
%       APTPATH = '/groups/branson/home/leea30/git/apt.param';
%       addpath(APTPATH);
%       APT.setpath
      
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
          
          testObj = testAPT('name',name);
          testObj.test_setup('simpleprojload',1);
          testObj.lObj.projTempDirDontClearOnDestructor = true;
          testObj.lObj.trackAutoSetParams = false;
          testObj.test_train(...
            'net_type',iTracker,...
            'niters',TRNITERS,...
            'params',struct('batch_size',2),...
            'backend',be,...
            'test_tracking',false);
          
          disp('Train done!');          
          tObj = testObj.lObj.tracker;
          tinfo = tObj.trackerInfo;
          iters1 = tinfo.iterFinal;
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
                    
          testObj = testAPT('name',name);
          testObj.test_setup('simpleprojload',1);
          testObj.lObj.projTempDirDontClearOnDestructor = true;
          testObj.lObj.trackAutoSetParams = false;
          testObj.test_track(...
            'net_type',iTracker,...
            'block',true ...
            );
          
          disp('Track done!');
          %pause(10);
          testObj.test_track_export();
          
        case 'full'
          name = varargin{2};
          be = varargin{3};          
          testObj = testAPT('name',name);
          testObj.test_full('backend',be);            
          testObj.lObj.projTempDirDontClearOnDestructor = true;
          testObj.lObj.trackAutoSetParams = false;
          
        case 'hello'
          disp('hello!');
          
        case 'testerr'
          error('testAPT:testerr','Test error!');
          
        otherwise
          error('testAPT:testerr','Unrecognized action: %s',action);
      end
    end
  end
    
end
