classdef VideoTest
  methods (Static)
    
    function test1(mov,varargin)
      % Test1 is:
      % - Read frames 1..nmax sequentially.
      % - In a new read session (get_readframe_fcn), read nsamp randomly 
      % selected frames (in 1..nmax), passing over each one npass times.
      %
      % Expected result:
      % 1. Random-access-read (RAR) frames to match sequentially-read (SR) 
      % frames as appropriate.
      % 2. RAR frames to match themselves as appropriate regardless of read 
      % order
      %
      % Saved:
      % - SR frame stack
      % - RAR frame stack
      % - Metadata: movname, matlab ver, computer, host, timestamp, readfcn
      
      [nmax,nsamp,npass,frms,outdir,ISR3p,ISR3pname,dispmod] = ...
        myparse(varargin,...
        'nmax',200,... % max framenum to read up to. If supplied as empty, read from mov
        'nsamp',60,... % number of random frames to sample in 1..nmax
        'npass',3,... % each frame in nsamp will be read 3 times
        'frms',[],... % random-access frames. all must be less than nmax. if supplied, supercedes nsamp/npass
        'outdir',[],... % filesys location to put output
        'ISR3p',[],... % optional, externally-generated ISR for comparison.
        'ISR3pname','',... % optional, name for display
        'dispmod',50 ...
        );
      
      [rf1,nf1,fid1,info1] = get_readframe_fcn(mov);
      
      % nmax
      nmaxOrig = nmax;
      if isempty(nmax)
        fprintf('nmax not supplied. Using %d as apparent num frames in mov.\n',...
          nf1);
        nmax = nf1;
      end
      
      % nsamp/npass/frms
      if isempty(frms)
        frms = randsample(nmax,nsamp);
        frms = repmat(frms,npass,1);
        p = randperm(numel(frms));
        frms = frms(p);
        nfrms = numel(frms);
        fprintf('Generated %d total random frames (%d samps, %d passes) to read.\n',...
          nfrms,nsamp,npass);
      else
        frms = frms(:);
        assert(all(frms<=nmax),'Random-access frames exceed nmax.');
        nfrms = numel(frms);
        fprintf('%d random frames specified.\n',nfrms);
      end
      
      % outdir
      txtinfo = struct();
      matinfo = struct();
      txtinfo.nmaxsupplied = ~isempty(nmaxOrig);
      txtinfo.nmaxused = nmax;
      txtinfo.host = strtrim(getHostName());
      matinfo.computer = computer;
      matinfo.ver = ver('matlab');
      txtinfo.nowstr = datestr(now,'yyyymmddTHHMMSS');      
      txtinfo.mov = mov;
      txtinfo.frms = frms;
      matinfo.get_readframe_fcn = which('get_readframe_fcn'); %#ok<STRNU>

      if isempty(outdir)
        [movP,movF,movE] = fileparts(mov);
        release = matinfo.ver.Release(2:end-1);
        outdir = fullfile(movP,sprintf('VideoTest_%s_%s_%s_%s_%s',...
          movF,matinfo.computer,txtinfo.host,release,txtinfo.nowstr));
      end
      
      % read SR
      ISR = cell(nmax,1);
      for i=1:nmax
        ISR{i} = rf1(i);
        if mod(i,dispmod)==0
          fprintf('... read SR frame %d\n',i);
        end
      end      
      if fid1
        fclose(fid1);
      end

      [rf2,nf2,fid2,info2] = get_readframe_fcn(mov);
      assert(nf1==nf2);
      assert(isequal(info1,info2));
      
      IRAR = cell(nfrms,1);
      for i=1:nfrms
        f = frms(i);
        IRAR{i} = rf2(f);
        if mod(i,dispmod)==0
          fprintf('... read RAR frame %d\n',i);
        end
      end      
      if fid2  
        fclose(fid2);
      end

      % Test A: internal consistency
      VideoTest.test1core(ISR,IRAR,frms);
      if ~isempty(ISR3p)
        % Test B: (skip) check ISR against ISR3p if supplied. 
        % Test C: check IRAR against ISR3p if supplied. 
        % If Test A and Test C pass, then Test B would have passed.
        % If Test A fails, we expect Test C to fail. The results of Test B
        % might be interesting.
        % If Test A passes, but Test C fails, that is interesting.
        fprintf(1,'Performing checks against external results: %s\n',...
          ISR3pname);
        if numel(ISR3p)~=numel(ISR)
          fprintf(2,'External SR has different number of frames (%d vs %d)\n',...
            numel(ISR3p),numel(ISR));
        end
        VideoTest.test1core(ISR3p,IRAR,frms);
      end
      
      if exist(outdir,'dir')==0
        fprintf(1,'Directory ''%s'' does not exist. Creating...\n',outdir);
        [succ,msg] = mkdir(outdir);
        if succ==0
          error('Could not create output directory: %s\n',msg);
        end
      end
      
      for i=1:numel(ISR)
        fname = sprintf('sr_%06d.png',i);
        fname = fullfile(outdir,fname);
        imwrite(ISR{i},fname);
        if mod(i,dispmod)==0
          fprintf(' ... wrote SR frame %d\n',i);
        end
      end
      
      for i=1:numel(IRAR)
        fname = sprintf('rar_%06d_%06d.png',i,frms(i));
        fname = fullfile(outdir,fname);
        imwrite(IRAR{i},fname);
        if mod(i,dispmod)==0
          fprintf(' ... wrote RAR frame %d\n',i);
        end
      end
            
      fname = fullfile(outdir,'matlabres.mat');
      save(fname,'ISR','IRAR','matinfo');
      fprintf(1,'Wrote matinfo ''%s''\n',fname);
      
      fname = fullfile(outdir,'info.json');
      fh = fopen(fname,'w');
      fprintf(fh,'%s\n',jsonencode(txtinfo));
      fclose(fh);
      fprintf(1,'Wrote jsoninfo ''%s''\n',fname);
    end
    
    function test1compare(testdir,varargin)
      mov = myparse(varargin,...
        'mov',''); % read from testdir, but can also be supplied if eg on diff platform or movie has moved
      
      res = load(fullfile(testdir,'matlabres.mat'));
      fh = fopen(fullfile(testdir,'info.json'));
      json = jsondecode(fgetl(fh));
      fclose(fh);
      
      if isempty(mov)
        mov = json.mov;
      end
      
      if json.nmaxsupplied
        nmax = json.nmaxused;
      else
        nmax = [];
      end
      VideoTest.test1(mov,...
        'nmax',nmax,...
        'frms',json.frms,...
        'ISR3p',res.ISR,...
        'ISR3pname',testdir);
    end
    
    function test1core(ISR,IRAR,frms,varargin)
      % ISR: seq-read frames, 1..n.
      % IRAR: random-read frames. Labeled by frms.
      
      plusminus = myparse(varargin,...
        'plusminus',3);
      
      ICOMB = [ISR;IRAR];
      
      cls = cellfun(@class,ICOMB,'uni',0);
      cls = unique(cls);
      if isscalar(cls)
        fprintf(1,'img class: %s\n',cls{1});
      else
        fprintf(2,'Multiple img classes: %s\n',...
          String.cellstr2CommaSepList(cls));
      end
      
      chan = cellfun(@(x)size(x,3),ICOMB);
      chanUn = unique(chan);
      if isscalar(chanUn)
        fprintf(1,'num chans: %d\n',chanUn);
      else
        fprintf(2,'Multiple chans: %s\n',mat2str(chanUn));
      end
      
      if isequal(chanUn,3)
        tfgray = cellfun(@(x)isequal(x(:,:,1),x(:,:,2),x(:,:,3)),ICOMB);
        if all(tfgray)
          fprintf(1,'All ims grayscale.\n');
        elseif ~any(tfgray)
          fprintf(1,'No ims grayscale.\n');
        else
          fprintf(2,'Grayscale/nongrayscale mix.\n');
        end
      end
      
      nfrms = numel(frms);
      assert(nfrms==numel(IRAR));
      frmsUn = unique(frms);
      nfrmsUn = numel(frmsUn);
      fprintf(1,'%d frames sampled.\n',nfrmsUn);
      for i=1:nfrmsUn
        f = frmsUn(i);
        irar = frms==f;
        imrar = IRAR(irar);
        if ~isequal(imrar{:})
          fprintf(2,'Frame %d, inconsistent within-RAR.\n',f);
        elseif ~isequal(ISR{f},imrar{1})
          fprintf(2,'Frame %d, inconsistent RAR vs SR.\n',f);
          frmsplusminus = max(1,f-plusminus):min(numel(ISR),f+plusminus);
          for ff=frmsplusminus
            if isequal(ISR{ff},imrar{1})
              fprintf(2,'... found it at delta=%d.\n',ff-f);
              break;
            end
          end
        else
          fprintf(1,'Frame %d, OK %d samps.\n',f,numel(imrar));
        end
      end
    end
    
    function matvspng(testdir)
      res = load(fullfile(testdir,'matlabres.mat'));
      fh = fopen(fullfile(testdir,'info.json'));
      json = jsondecode(fgetl(fh));
      fclose(fh);
      for i=1:numel(res.ISR)
        fname = fullfile(testdir,sprintf('sr_%06d.png',i));
        im = imread(fname);
        if isequal(im,res.ISR{i})
          %fprintf('SR %d ok|',i);
        else
          fprintf(2,'SR %d NOT OK!\n\n',i);
        end
      end
      fprintf(1,'Read/checked %d ISR frames.\n',numel(res.ISR));
      
      for i=1:numel(res.IRAR)
        fname = sprintf('rar_%06d_%06d.png',i,json.frms(i));
        fname = fullfile(testdir,fname);
        im = imread(fname);
        if isequal(im,res.IRAR{i})
          %fprintf('IRAR %d ok|',i);
        else
          fprintf(2,'IRAR %d NOT OK!\n\n',i);
        end
      end
      fprintf(1,'Read/checked %d IRAR frames.\n',numel(res.IRAR));
      
      fprintf('ok.\n');
    end
  end
end

function name = getHostName
[ret,name] = system('hostname');
if ret~=0
  if ispc
    name = getenv('COMPUTERNAME');
  else
    name = getenv('HOSTNAME');
  end
end
end

% 
% function imgRead2(mov,frms)
% 
% if nargin==0 || isempty(mov)
%   mov = fullfile(fileparts(mfilename('fullpath')),'C001H001S0007_c.avi');
% end
% 
% fprintf('Test mov is %s\n',mov);
% 
% ims0 = readAllFrames(mov,max(frms)+50);
% 
% nfrm = numel(frms);
% vr = VideoReader(mov);
% for i=1:nfrm
%   im = vr.read(frms(i));
%   if isequal(im(:,:,1),im(:,:,2),im(:,:,3))
%     im = im(:,:,1);
%   else
%     warning('image not grayscale.');
%   end
%   
%   fprintf('Read frm %d, matches hopefully-true frm %d.\n',...
%     frms(i),find(cellfun(@(x)isequal(im,x),ims0)));
% end
% 
% 
% % * cluster 16b: sometimes ftrue==fread, mostly ftrue==fread+1
% % * cluster 18a: all over, sampled 10 frames saw as high as 50 frames off
% % * verman-ws1 18a. assert on readAllFrames/line12 errors. images are not 
% % read as perfect grayscale. comparing rgb (3-chan) ims: sometimes 
% % ftrue==fread, ftrue==fread+1
% % * verman-ws1 16b. either ftrue==fread or ftrue==fread-1
% % * allen win 17b: perfect match
% % * allen win 16b: perfect match
