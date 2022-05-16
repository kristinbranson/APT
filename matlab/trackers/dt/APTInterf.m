classdef APTInterf
  % CodeGen methods for APT_interface.py
  
  methods (Static)

    function basecmd = trainCodeGen(modelChainID,dllbl,cache,errfile,...
        netType,netMode,varargin)
      isMA = netType.isMultiAnimal;
      isNewStyle = isMA || ...
        (netMode~=DLNetMode.singleAnimal && netMode~=DLNetMode.multiAnimalTDPoseTrx);
      
      if isNewStyle
        basecmd = APTInterf.maTrainCodeGenTrnPack(modelChainID,dllbl,cache,errfile,...
          netType,netMode,varargin{:});
      else
        basecmd = APTInterf.regTrainCodeGen(modelChainID,dllbl,cache,errfile,...
          netType,varargin{:});
      end
    end
    
    function [codestr,code] = maTrainCodeGenTrnPack(trnID,dllbl,cache,...
        errfile,netType,netMode,varargin)
      % Wrapper for matrainCodeGen, assumes standard trnpack structure.
      % Reads some files from trnpack
      % 
      % netType/netMode: for TopDown trackers, these are currently ALWAYS
      % stage2. pass stage1 in the varargin. Yes, this is a little weird if
      % maTopDownStage=='first'.
      
       [maTopDown,maTopDownStage,maTopDownStage1NetType,...
         maTopDownStage1NetMode,leftovers] = ...
           myparse_nocheck(varargin,...
         'maTopDown',false, ...
         'maTopDownStage',[], ... % '-stage' flag: {'multi','first','second'}
         'maTopDownStage1NetType',[], ...
         'maTopDownStage1NetMode',[] ...
         );
      
      [trnpack,dllblID] = fileparts(dllbl);
      trnjson = fullfile(trnpack,'loc.json');
      dllbljson = fullfile(trnpack,[dllblID '.json']);
      dlj = readtxtfile(dllbljson);
      dlj = jsondecode(dlj{1});
      
      assert(maTopDown == ~isscalar(dlj.TrackerData));
      
      if maTopDown
        isObjDet = netMode.isObjDet;
        [codestr,code] = APTInterf.matdTrainCodeGen(trnID,dllbl,cache,...
          errfile,isObjDet,maTopDownStage1NetType,netType,trnjson,maTopDownStage,...
          leftovers{:});
      else
        assert(netMode==DLNetMode.multiAnimalBU);
        [codestr,code] = APTInterf.mabuTrainCodeGen(trnID,dllbl,cache,errfile,...
          netType,trnjson,leftovers{:});
      end
    end
    
    function [codestr,code] = mabuTrainCodeGen(trnID,dllbl,cache,errfile,...
        netType,trnjson,varargin)
      % Simplified relative to trainCodeGen
      
      [deepnetroot,fs,filequote,confparamsextra,confparamsfilequote,...
        prev_model,torchhome,val_split,augOnly,augOut,...
        clsfyIsClsfy,clsfyOut] = ...
        myparse_nocheck(varargin,...
        'deepnetroot',APT.getpathdl,...
        'filesep','/',...
        'filequote','\"',... % quote char used to protect filenames/paths.
        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
        ... % is expected to wrap in enclosing regular double-quotes " !!
        'confparamsextra',{},...
        'confparamsfilequote','\"', ...
        'prev_model',[],...
        'torchhome',APT.torchhome, ...
        'val_split',[],...
        'augOnly',false, ...
        'augOut','', ... % used only if augOnly==true        
        'clsfyIsClsfy',false,... % if true, classify not train
        'clsfyOut',[] ...        
        );
      
      aptintrf = [deepnetroot fs 'APT_interface.py'];

      % MK 20220128 -- db_format should come from params_deeptrack_net.yaml
%       confParams = { ... %        'is_multi' 'True' ...    'max_n_animals' num2str(maxNanimals) ...
%         'db_format' [confparamsfilequote 'coco' confparamsfilequote] ... % when the job is submitted the double quote need to escaped. This is tested fro cluster. Not sure for AWS etc. MK 20210226
%         confparamsextra{:} ...
%         };
      confParams = confparamsextra;
      
      code = cat(2,{ ...
        ['TORCH_HOME=' filequote torchhome filequote] ...
        'python' ...
        [filequote aptintrf filequote] ...
        dllbl ...
        '-name' trnID ...
        '-err_file' [filequote errfile filequote] ... 
        '-json_trn_file' trnjson ...
        '-conf_params'}, ...
        confParams, ...
        {'-type' netType}); ...

      if ~isempty(prev_model)
        code = [code {'-model_files' [filequote prev_model filequote]}];
      end
      
      code = [code {'-cache' [filequote cache filequote]}];
      if clsfyIsClsfy
        assert(~isempty(clsfyOut));
        code = [code {'classify' '-out' clsfyOut}];
      else
        code = [code {'train' '-use_cache'}];
      end
      if augOnly
        code = [code ...
          {'-only_aug' ...
          '-aug_out' augOut}];
      end
      dosplit = ~isempty(val_split) && ~clsfyIsClsfy;
      if dosplit
        code = [code {'-val_split' num2str(val_split)}];
      end      

      codestr = String.cellstr2DelimList(code,' ');
    end
            
    function [codestr,code] = matdTrainCodeGen(trnID,dllbl,cache,errfile,...
        isObjDet,netTypeStg1,netTypeStg2,trnjson,stage,varargin)
      
      [deepnetroot,fs,filequote,confparamsfilequote,...
        prev_model,prev_model2,torchhome,augOnly,augOut] = ...
        myparse_nocheck(varargin,...
        'deepnetroot',APT.getpathdl,...
        'filesep','/',...
        'filequote','\"',... % quote char used to protect filenames/paths.
        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
        ... % is expected to wra% TODO: ht paramsp in enclosing regular double-quotes " !!
        'confparamsfilequote','\"', ...
        'prev_model',[],...
        'prev_model2',[],...
        'torchhome',APT.torchhome, ...
        'augOnly',false,...
        'augOut','' ... % used only if augOnly==true
        );
      
      aptintrf = [deepnetroot fs 'APT_interface.py'];
      
      STAGEFLAGS = {'multi' 'first' 'second'};
      
      code = { ...
        ['TORCH_HOME=' filequote torchhome filequote] ...
        'python' ...
        [filequote aptintrf filequote] ...
        dllbl ...
        '-name' trnID ...
        '-err_file' [filequote errfile filequote] ...
        '-json_trn_file' trnjson ...
        '-stage' STAGEFLAGS{stage+1} ...
        };
              
      % set -conf_params, -type
      if stage==0 || stage==1 % inc stg1/detect
        code = [code ...
          {
          ... %'-conf_params' ...
          ... %'db_format' [confparamsfilequote 'coco' confparamsfilequote] ... % when the job is submitted the double quote need to escaped. This is tested fro cluster. Not sure for AWS etc. MK 20210226
          ... %'mmdetect_net' [confparamsfilequote 'frcnn' confparamsfilequote] ...
          '-type' netTypeStg1 ...
          } ];
      end
      
      % if nec, set -conf_params2, -type2
      if stage==2
        % single stage 2 
        
        tfaddstg2 = true;
        confparamsstg2 = '-conf_params';
        typestg2 = '-type';
        if ~isempty(prev_model)
          code = [code {'-model_files' [filequote prev_model filequote]}];
        end
      elseif stage==0
        tfaddstg2 = true;
        confparamsstg2 = '-conf_params2';
        typestg2 = '-type2';
        if ~isempty(prev_model)
          code = [code {'-model_files' [filequote prev_model filequote]}];
        end
        if ~isempty(prev_model2)
          code = [code {'-model_files2' [filequote prev_model2 filequote]}];
        end
        
      else
        tfaddstg2 = false;
        if ~isempty(prev_model)
          code = [code {'-model_files' [filequote prev_model filequote]}];
        end

      end
      if tfaddstg2
        confparams2 = { ...
          confparamsstg2 ...
          ... %'db_format' [confparamsfilequote 'tfrecord' confparamsfilequote] ...
          };
        if isObjDet
          confparams2 = [confparams2 {'use_bbox_trx' 'True'}];
        end
        confparams2 = [confparams2 {typestg2 netTypeStg2}];
        code = [code confparams2];
      end
      
      code = [code ...
        { ...
        '-cache' [filequote cache filequote] ... % String.escapeSpaces(cache),...
        'train' ...
        '-use_cache' ...
        } ];
      if augOnly
        code = [code ...
          {'-only_aug' ...
          '-aug_out' augOut}];
      end
      codestr = String.cellstr2DelimList(code,' ');
    end
    
    function codestr = regTrainCodeGen(trnID,dllbl,cache,errfile,netType,...        
        varargin)
      % "reg" => regular, from pre-MA APT
      
      [view,deepnetroot,splitfile,classify_val,classify_val_out,val_split,...
        trainType,fs,prev_model,filequote,augOnly,confparamsfilequote,augOut] = myparse(varargin,...
        'view',[],... % (opt) 1-based view index. If supplied, train only that view. If not, all views trained serially
        'deepnetroot',APT.getpathdl,...
        'split_file',[],...
        'classify_val',false,... % if true, split_file must be spec'd
        'classify_val_out',[],... % etc
        'val_split',[],...
        'trainType',DLTrainType.New,...
        'filesep','/',...
        'prev_model',[],...
        'filequote','\"',... % quote char used to protect filenames/paths.
        'augOnly',false,...
        'confparamsfilequote','\"',... % this is used in other train code functions, adding here to remove warning
        'augOut','' ... % used only if augOnly==true
        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
        ... % is expected to wrap in enclosing regular double-quotes " !!
        );
      torchhome = APT.torchhome;
      
      tfview = ~isempty(view);
      [trnpack,dllblID] = fileparts(dllbl);
      trnjson = fullfile(trnpack,'loc.json');
      dllbljson = fullfile(trnpack,[dllblID '.json']);
      dlj = readtxtfile(dllbljson);
      dlj = jsondecode(dlj{1});
  
      aptintrf = [deepnetroot fs 'APT_interface.py'];
      
      switch trainType
        case DLTrainType.New
          continueflags = '';
        case DLTrainType.Restart
          continueflags = '-continue -skip_db';
        case DLTrainType.RestartAug
          continueflags = '-continue';
        otherwise
          assert(false);
      end
      
      dosplit = ~isempty(val_split);
      if dosplit
        splitfileargs = sprintf('-val_split %d',val_split);
        if classify_val
          splitfileargs = [splitfileargs sprintf(' -classify_val -classify_val_out %s',classify_val_out)];
        end
      else
        splitfileargs = '';
      end
      
      code = { ...
        ['TORCH_HOME=' filequote torchhome filequote] ...
        'python' ...
        [filequote aptintrf filequote] ...
        '-name' ...
        trnID ...
        };
      if tfview
        code = [code {'-view' num2str(view)}];
      end
      code = [code { ...
        '-cache' ...
        [filequote cache filequote] ... % String.escapeSpaces(cache),...
        '-err_file' ...
        [filequote errfile filequote] ...
        '-json_trn_file' ...
        [filequote trnjson filequote]}
        ]; ... % String.escapeSpaces(errfile),...
      if ~isempty(prev_model)
        code = [code {'-model_files' [filequote prev_model filequote]}];
      end
      code = [code ...
        {'-type' ...
        netType ...
        [filequote dllbl filequote] ... % String.escapeSpaces(dllbl),...
        'train' ...
        '-use_cache' ...
        continueflags ...
        splitfileargs} ];
      if augOnly
        code = [code ...
          {'-only_aug' ...
          '-aug_out' augOut}];
      end
      codestr = String.cellstr2DelimList(code,' ');
    end
        
    function splitTrainValCodeGenCmdfile(codefname,classifyOutmat,...
          trnID,dllbl,cache,errfile,netType,netMode,valSplit,varargin)
      % write cmdfile to disk for train/val of given split
      % 
      % if not successful, throws
        
      [fh,msg] = fopen(codefname,'w');
      if isequal(fh,-1)
        error('Filed to open %s: %s',codefname,msg);
      end
      
      argsTrain = [varargin {'filequote' '"' 'val_split' valSplit}];
      argsVal = [varargin ...
        {'filequote' '"' 'clsfyIsClsfy' true 'clsfyOut' classifyOutmat }];
      
      s1 = APTInterf.trainCodeGen(trnID,dllbl,cache,errfile,...
        netType,netMode,argsTrain{:});
      s2 = APTInterf.trainCodeGen(trnID,dllbl,cache,errfile,...
        netType,netMode,argsVal{:});
      
      fprintf(fh,'#!/usr/bin/env bash\n');
      fprintf(fh,'\n%s\n',s1);
      fprintf(fh,'\n%s\n',s2);
      fclose(fh);
    end    
    
    function [codestr,code] = trackCodeGenBase(fileinfo,...
        frm0,frm1,... % (opt) can be empty. these should prob be in optional P-Vs
        varargin)
      
      % Serial mode: 
      % - movtrk is [nmov] array
      % - outtrk is [nmov] array
      % - trxtrk is unsupplied, or [nmov] array
      % - view is a *scalar* and *must be supplied*
      % - croproi is unsupplied, or [xlo1 xhi1 ylo1 yhi1 xlo2 ... yhi_nmov] or row vec of [4*nmov]
      % - model_file is unsupplied, or [1] cellstr, or [nmov] cellstr      

      trnID = fileinfo.trnID;
      dllbl = fileinfo.dllbl;
      errfile = fileinfo.errfile;
      nettype = fileinfo.nettype; % for 2-stage, this is the stage2 nettype
      netmode = fileinfo.netmode; % " netmode
      % either char or [nviewx1] cellstr; or [nmov] in "serial mode" (see below)
      movtrk = fileinfo.movtrk; 
      % save as movtrk, except for 2 stage, this will be [nviewx2] or [nmovx2]
      outtrk = fileinfo.outtrk; 
      
      [listfile,cache,trxtrk,trxids,view,croproi,hmaps,deepnetroot,model_file,log_file,...
        updateWinPaths2LnxContainer,lnxContainerMntLoc,fs,filequote,...
        confparamsfilequote,tfserialmode,...
        track_id] = ...
        myparse_nocheck(varargin,...
        'listfile','',...
        'cache',[],... % (opt) cachedir
        'trxtrk','',... % (opt) trxfile for movtrk to be tracked 
        'trxids',[],... % (opt) 1-based index into trx structure in trxtrk. empty=>all trx
        'view',[],... % (opt) 1-based view index. If supplied, track only that view. If not, all views tracked serially 
        'croproi',[],... % (opt) 1-based [xlo xhi ylo yhi] roi (inclusive). can be [nview x 4] for multiview
        'hmaps',false,...% (opt) if true, generate heatmaps
        'deepnetroot',APT.getpathdl,...
        'model_file',[], ... % can be [nview] cellstr
        'log_file',[],... (opt)
        'updateWinPaths2LnxContainer',ispc, ... % if true, all paths will be massaged from win->lnx for use in container 
        'lnxContainerMntLoc','/mnt',... % used when updateWinPaths2LnxContainer==true
        'filesep','/',...
        'filequote','\"',... % quote char used to protect filenames/paths.
                        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
                        ... % is expected to wrap in enclosing regular double-quotes " !!
        'confparamsfilequote','\"', ...
        'serialmode',false, ...  % see serialmode above
        'track_id',false ... % Track id over ride json conf setting
        );
      
     
      tflistfile = ~isempty(listfile);
      tffrm = ~tflistfile && ~isempty(frm0) && ~isempty(frm1);
      if tffrm, % ignore frm if it doesn't limit things
        if all(frm0 == 1) && all(isinf(frm1)),
          tffrm = false;
        end
      end
      tfcache = ~isempty(cache);
      tftrx = ~tflistfile && ~isempty(trxtrk);
      tftrxids = ~tflistfile && ~isempty(trxids);
      tfview = ~isempty(view);
      tfcrop = ~isempty(croproi) && ~all(any(isnan(croproi),2),1);
      tflog = ~isempty(log_file);
      tf2stg = netmode.isTwoStage;
      nstage = 1+double(tf2stg);
      tfmodel = ~isempty(model_file) && ~tf2stg; % -model_file arg not working in Py for 2stg yet
      
      torchhome = APT.torchhome;
                
      movtrk = cellstr(movtrk);
      outtrk = cellstr(outtrk);
      if tftrx
        trxtrk = cellstr(trxtrk);
      end
      if tfmodel
        model_file = cellstr(model_file);
      end
      
      if tfserialmode
        nmovserialmode = numel(movtrk);
        szassert(outtrk,[nmovserialmode nstage]);
        if tftrx
          assert(numel(trxtrk)==nmovserialmode);
        end
        assert(isscalar(view),'A scalar view must be specified for serial-mode.');
        if tfcrop
          szassert(croproi,[nmovserialmode 4]);
        end
        if tfmodel
          if isscalar(model_file)
            model_file = repmat(model_file,nmovserialmode,1);
          else
            assert(numel(model_file)==nmovserialmode);
          end
        end        
      else
        if tfview % view specified. track a single movie
          nview = 1;
          assert(isscalar(view));
          if tftrx
            assert(isscalar(trxtrk));
          end
        else
          nview = numel(movtrk);
          if nview>1
            assert(~tftrx && ~tftrxids,'Trx not supported for multiple views.');
          end
        end
        assert(nview==numel(movtrk));
        szassert(outtrk,[nview nstage]);
        if tfmodel
          assert(numel(model_file)==nview);
        end
        if tfcrop
          szassert(croproi,[nview 4]);
        end      
      end
      
      assert(~(tftrx && tfcrop));
      aptintrf = [deepnetroot fs 'APT_interface.py'];

      conf_str = '';
      
      if updateWinPaths2LnxContainer
        fcnPathUpdate = @(x)DeepTracker.codeGenPathUpdateWin2LnxContainer(x,lnxContainerMntLoc);
        aptintrf = fcnPathUpdate(aptintrf);

        movtrk = cellfun(fcnPathUpdate,movtrk,'uni',0);
        outtrk = cellfun(fcnPathUpdate,outtrk,'uni',0);
        if tftrx
          trxtrk = cellfun(fcnPathUpdate,trxtrk,'uni',0);
        end
        if tfmodel
          model_file = cellfun(fcnPathUpdate,model_file,'uni',0);
        end
        if tflog
          log_file = fcnPathUpdate(log_file);
        end
        if tfcache
          cache = fcnPathUpdate(cache);
        end
        errfile = fcnPathUpdate(errfile);
        dllbl = fcnPathUpdate(dllbl);
      end      

      code = { ...
        ['TORCH_HOME=' filequote torchhome filequote] ...
        'python' [filequote aptintrf filequote] ...
        '-name' trnID ...
        };

      if tfview
        code = [code {'-view' num2str(view)}]; % view: 1-based for APT_interface
      end
      if tfcache
        code = [code {'-cache' [filequote cache filequote]}];
      end
      code = [code {'-err_file' [filequote errfile filequote]}];
      if tfmodel
        code = [code {'-model_files' ...
          DeepTracker.cellstr2SpaceDelimWithQuote(model_file,filequote)}];
      end
      if tflog
        code = [code {'-log_file' [filequote log_file filequote]}];
      end
      if track_id
        conf_str = sprintf('%s link_id True',conf_str);
      end
      if tf2stg
        if fileinfo.netmodeStage1.isObjDet
          use_bbox_trx_val = 'True';
        else
          use_bbox_trx_val = 'False';
        end
         conf_str = sprintf('%s use_bbox_trx %s',conf_str,use_bbox_trx_val);
      end
      if ~isempty(conf_str)
        code = [code {sprintf('-conf_params %s', conf_str)}];
      end
      
      if tf2stg
        %szassert(outtrk,[1 nstage],...
        %  'Multiview or serial multimovie unsupported for two-stage trackers.');
        if fileinfo.netmodeStage1.isObjDet
          use_bbox_trx_val = 'True';
        else
          use_bbox_trx_val = 'False';
        end
        code = [code {'-stage' 'multi' ...
                      '-type' char(fileinfo.nettypeStage1) ...
                      '-type2' char(nettype) ...
                      [filequote dllbl filequote] ...
                      'track' ...
                      '-out' DeepTracker.cellstr2SpaceDelimWithQuote(outtrk(:,2),filequote) }];        
         conf_str = sprintf('%s use_bbox_trx %s',conf_str,use_bbox_trx_val);
      else
        code = [code {'-type' char(nettype) ...
                      [filequote dllbl filequote] ...
                      'track' ...
                      '-out' DeepTracker.cellstr2SpaceDelimWithQuote(outtrk,filequote) }];
      end
      if tflistfile
        code = [code {'-list_file' [filequote listfile filequote]}];
      else
        code = [code {'-mov' DeepTracker.cellstr2SpaceDelimWithQuote(movtrk,filequote)}];
      end
      if tffrm
        frm0(isnan(frm0)) = 1;
        frm1(isinf(frm1)|isnan(frm1)) = -1;
        frm0 = round(frm0); % fractional frm0/1 errs in APT_interface due to argparse type=int
        frm1 = round(frm1); % just round silently for now        
        sfrm0 = sprintf('%d ',frm0); sfrm0 = sfrm0(1:end-1);
        sfrm1 = sprintf('%d ',frm1); sfrm1 = sfrm1(1:end-1);
        code = [code {'-start_frame' sfrm0 '-end_frame' sfrm1}];
      end
      if tftrx
        code = [code {'-trx' DeepTracker.cellstr2SpaceDelimWithQuote(trxtrk,filequote)}];
        if tftrxids
          if ~iscell(trxids),
            trxids = {trxids};
          end
          for i = 1:numel(trxids),
            trxidstr = sprintf('%d ',trxids{i});
            trxidstr = trxidstr(1:end-1);
            code = [code {'-trx_ids' trxidstr}]; %#ok<AGROW>
          end
        end
      elseif tf2stg
        code = [code {'-trx' DeepTracker.cellstr2SpaceDelimWithQuote(outtrk(:,1),filequote)}];
      end
      if tfcrop
        croproi = round(croproi);
        croproirowvec = croproi';
        croproirowvec = croproirowvec(:)'; % [xlovw1 xhivw1 ylovw1 yhivw1 xlovw2 ...] OR [xlomov1 xhimov1 ylomov1 yhimov1 xlomov2 ...] in serialmode
        roistr = mat2str(croproirowvec);
        roistr = roistr(2:end-1);
        code = [code {'-crop_loc' roistr}];
      end
      if hmaps
        code = [code {'-hmaps'}];
      end
      
      codestr = String.cellstr2DelimList(code,' ');
    end
    
  end
  
end