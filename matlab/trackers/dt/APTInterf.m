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
      
      [trnpack,dllblID] = fileparts(dllbl);
      trnjson = fullfile(trnpack,'loc.json');
      dllbljson = fullfile(trnpack,[dllblID '.json']);
      dlj = readtxtfile(dllbljson);
      dlj = jsondecode(dlj{1});
      
      madet = dlj.TrackerData.sPrmAll.ROOT.MultiAnimalDetection;
      maxNanimals = madet.max_n_animals;
      
      switch netMode
        case DLNetMode.multiAnimalBU 
          [codestr,code] = APTInterf.mabuTrainCodeGen(trnID,dllbl,cache,errfile,...
            maxNanimals,netType,trnjson,varargin{:});          

        case DLNetMode.multiAnimalTDDetectHT
          htpts = [1 2];
          fprintf(2,'TODO: htpts into codegen\n');
          %htpts = [ma.head_point ma.tail_point];
          [codestr,code] = APTInterf.mabuTrainCodeGen(trnID,dllbl,cache,errfile,...
            maxNanimals,netType,trnjson,varargin{:},'htonly',true,'htpts',htpts);
          
        case DLNetMode.multiAnimalTDPoseHT
          htpts = [1 2];
          fprintf(2,'TODO: htpts into codegen\n');
          %htpts = [ma.head_point ma.tail_point];
          [codestr,code] = APTInterf.htPoseTrainCodeGen(trnID,dllbl,cache,errfile,...
            netType,trnjson,htpts,varargin{:});
          
          
        otherwise
          assert(false);
      end
    end
    
    function [codestr,code] = mabuTrainCodeGen(trnID,dllbl,cache,errfile,...
        maxNanimals,netType,trnjson,varargin)
      % Simplified relative to trainCodeGen
      
      [deepnetroot,fs,filequote,confparamsfilequote,torchhome,htonly,htpts] = ...
        myparse_nocheck(varargin,...
        'deepnetroot',APT.getpathdl,...
        'filesep','/',...
        'filequote','\"',... % quote char used to protect filenames/paths.
        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
        ... % is expected to wrap in enclosing regular double-quotes " !!
        'confparamsfilequote','\"', ...
        'torchhome',APT.torchhome, ...
        'htonly',false, ...
        'htpts',[] ... % used only if htonly==True
        );
      
      aptintrf = [deepnetroot fs 'APT_interface.py'];
      
      confParams = { ...
        'is_multi' 'True' ...
        'max_n_animals' num2str(maxNanimals) ...
        'db_format' [confparamsfilequote 'coco' confparamsfilequote] ... % when the job is submitted the double quote need to escaped. This is tested fro cluster. Not sure for AWS etc. MK 20210226
        };
      if htonly
        warningNoTrace('HT params TODO');
        confParams = [confParams ...
          {'multi_use_mask' 'False' ...
          'multi_loss_mask' 'True' ...
          'multi_crop_ims' 'True' ...
          'rrange' '180' ... % TODO: ht params
          'trange' '30' ... % TODO: ht params
          'ht_pts' sprintf('\\(%d,%d\\)',htpts(1),htpts(2)) ...
          'multi_only_ht' 'True' ...
          'rescale' num2str(4) ... % TODO: ht params
          }];
      end
      
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
        {'-type' netType ...
        '-cache' [filequote cache filequote] ... 
        'train' ...
        '-use_cache' ...
        });
      
      codestr = String.cellstr2DelimList(code,' ');
    end
    
%     function [codestr,code] = htDetTrainCodeGen(trnID,dllbl,cache,errfile,...
%         maxNanimals,netType,trnjson,htpts,varargin)
%       % Simplified relative to trainCodeGen
%       
%       [deepnetroot,fs,filequote,confparamsfilequote,torchhome] = ...
%         myparse_nocheck(varargin,...
%         'deepnetroot',APT.getpathdl,...
%         'filesep','/',...
%         'filequote','\"',... % quote char used to protect filenames/paths.
%         ... % *IMPORTANT*: Default is escaped double-quote \" => caller
%         ... % is expected to wra% TODO: ht paramsp in enclosing regular double-quotes " !!
%         'confparamsfilequote','\"', ...
%         'torchhome',APT.torchhome ...
%         );
%       
%       aptintrf = [deepnetroot fs 'APT_interface.py'];
%       
%       code = { ...
%         ['TORCH_HOME=' filequote torchhome filequote] ...
%         'python' ...
%         [filequote aptintrf filequote] ...
%         dllbl ...
%         '-name' trnID ...
%         '-err_file' [filequote errfile filequote] ...
%         '-json_trn_file' trnjson ...
%         '-conf_params' ...
%    
%         'is_multi' 'True' ...
%         'max_n_animals' num2str(maxNanimals) ...
%         'db_format' [confparamsfilequote 'coco' confparamsfilequote] ... % when the job is submitted the double quote need to escaped. This is tested fro cluster. Not sure for AWS etc. MK 20210226
%         '-type' netType ...
%         '-cache' [filequote cache filequote] ... % String.escapeSpaces(cache),...
%         'train' ...
%         '-use_cache' ...
%         };
%       codestr = String.cellstr2DelimList(code,' ');
%     end
    
    function [codestr,code] = htPoseTrainCodeGen(trnID,dllbl,cache,errfile,...
        netType,trnjson,htpts,varargin)
      % Simplified relative to trainCodeGen
      
      [deepnetroot,fs,filequote,confparamsfilequote,torchhome] = ...
        myparse_nocheck(varargin,...
        'deepnetroot',APT.getpathdl,...
        'filesep','/',...
        'filequote','\"',... % quote char used to protect filenames/paths.
        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
        ... % is expected to wra% TODO: ht paramsp in enclosing regular double-quotes " !!
        'confparamsfilequote','\"', ...
        'torchhome',APT.torchhome ...
        );
      
      aptintrf = [deepnetroot fs 'APT_interface.py'];
      
      warningNoTrace('HT params TODO');
      code = { ...
        ['TORCH_HOME=' filequote torchhome filequote] ...
        'python' ...
        [filequote aptintrf filequote] ...
        dllbl ...
        '-name' trnID ...
        '-err_file' [filequote errfile filequote] ...
        '-json_trn_file' trnjson ...
        '-conf_params' ...
        'mmpose_net' [confparamsfilequote 'higherhrnet' confparamsfilequote] ...
        'rrange' '30' ... % TODO: ht params
        'trange' '20' ... % TODO: ht params
        'imsz' '\(192,192\)' ...
        'trx_align_theta' 'True' ...
        'use_ht_trx' 'True' ...
        'img_dim' '1' ...        
        'ht_pts' sprintf('\\(%d,%d\\)',htpts(1),htpts(2)) ...
        'db_format' [confparamsfilequote 'tfrecord' confparamsfilequote] ... % when the job is submitted the double quote need to escaped. This is tested fro cluster. Not sure for AWS etc. MK 20210226
        '-type' netType ...
        '-cache' [filequote cache filequote] ... % String.escapeSpaces(cache),...
        'train' ...
        '-use_cache' ...
        };
      codestr = String.cellstr2DelimList(code,' ');
    end
    
    function codestr = regTrainCodeGen(trnID,dllbl,cache,errfile,netType,...        
        varargin)
      % "reg" => regular, from pre-MA APT
      
      [view,deepnetroot,splitfile,classify_val,classify_val_out,...
        trainType,fs,filequote] = myparse(varargin,...
        'view',[],... % (opt) 1-based view index. If supplied, train only that view. If not, all views trained serially
        'deepnetroot',APT.getpathdl,...
        'split_file',[],...
        'classify_val',false,... % if true, split_file must be spec'd
        'classify_val_out',[],... % etc
        'trainType',DLTrainType.New,...
        'filesep','/',...
        'filequote','\"'... % quote char used to protect filenames/paths.
        ... % *IMPORTANT*: Default is escaped double-quote \" => caller
        ... % is expected to wrap in enclosing regular double-quotes " !!
        );
      torchhome = APT.torchhome;
      
      tfview = ~isempty(view);
      
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
      
      dosplit = ~isempty(splitfile);
      if dosplit
        splitfileargs = sprintf('-split_file %s',[filequote splitfile filequote]);
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
        [filequote errfile filequote] ... % String.escapeSpaces(errfile),...
        '-type' ...
        netType ...
        [filequote dllbl filequote] ... % String.escapeSpaces(dllbl),...
        'train' ...
        '-use_cache' ...
        continueflags ...
        splitfileargs} ];
      codestr = String.cellstr2DelimList(code,' ');
    end
    
  end
  
end