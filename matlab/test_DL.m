%% script to test APT_interface on 3 different projects

user = char(java.lang.System.getProperty('user.name'));

% setup the LD_LIBRARY_PATH so that we can run python from within matlab.
[a,b] = system('echo $LD_LIBRARY_PATH');
X = strsplit(b,':');
Y = {};
for ndx = 1:numel(X)
  if isempty(strfind(X{ndx},matlabroot));
    Y{end+1} = X{ndx};
  end
end

pp = Y{1};
for ndx = 2:numel(Y);
  pp = sprintf('%s:%s',pp,Y{ndx});
end

train = false;
track = true;

% cmd = sprintf('export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/local/MATLAB/R2017a/sys/opengl/lib/glnxa64/;');
cmd = sprintf('export LD_LIBRARY_PATH=%s;',pp(1:end-1));
cmd = sprintf('%s source /groups/branson/home/kabram/bransonlab/venv/bin/activate;',cmd);
cmd = sprintf('%s cd /groups/branson/home/kabram/bransonlab/PoseTF;',cmd);
cmd = sprintf('%s export CUDA_VISIBLE_DEVICES=0;',cmd);
basecmd = cmd;

APT.setpath
cd /groups/branson/home/kabram/PycharmProjects/poseTF
cachedir = '/groups/branson/bransonlab/mayank/PoseTF/cache/apt_interface';
% if strcmp(user,'kabram'),
%   lbldir = '/groups/branson/home/kabram/bransonlab/PoseTF/data/apt_interface/';
% else
%   error('Allen, set your cache directory here');
%   cachedir = '';
%   lbldir = '';
% end

%% Double pendulum
% I save the modified file as that way if we ever update tracker parameters
% we can just update it to the modified lbl file instead of original lbl
% file. This is just to protect the labels in original lbl file.

lbl_file = '/groups/branson/home/kabram/bransonlab/PoseTF/data/apt_interface/pend/Doub_pend_stripped.lbl';
ref_lbl_file = '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/stephen_ref.lbl'; % this has the default params.
B = load(ref_lbl_file,'-mat');
A = load(lbl_file,'-mat');
A.trackerDeepData = B.trackerDeepData;
A.trackerDeepData.sPrm.dl_steps = 2000;
A.trackerDeepData.sPrm.sizex = 1280;
A.trackerDeepData.sPrm.sizey = 720;
A.trackerDeepData.sPrm.CropX_view1 = 0;
A.trackerDeepData.sPrm.CropY_view1 = 0;
A.trackerDeepData.sPrm.scale = 4;
A.trackerDeepData.sPrm.CacheDir = cachedir;


lbl_file = sprintf('%s_modified.lbl',splitext(lbl_file));
save(lbl_file,'-struct','A','-v7.3');

mov_file = '/groups/branson/home/leea30/apt/deeptrackIntegrate20180427/doubpend.mp4';
out_file = [splitext(mov_file)  '_interface_test.trk'];
name = 'pend_test_apt';
cmd = sprintf('%s python APT_interface.py %s -name %s train',basecmd,lbl_file,name);
if train,
  system(cmd)
end
cmd = sprintf('%s python APT_interface.py  %s -name %s track -mov %s -out %s -end_frame 1000',basecmd, lbl_file, name, mov_file, out_file);
if track,
  system(cmd);
end

%% Alice: for projects with trx files.
lbl_file = '/groups/branson/bransonlab/mayank/PoseTF/data/alice/multitarget_bubble_expandedbehavior_20180425.lbl';
ref_lbl_file = '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/stephen_ref.lbl';
B = load(ref_lbl_file,'-mat');
A = load(lbl_file,'-mat');

A.trackerDeepData = B.trackerDeepData;
A.trackerDeepData.sPrm.unet_steps = 2000;
A.trackerDeepData.sPrm.sizex = 180;
A.trackerDeepData.sPrm.sizey = 180;
A.trackerDeepData.sPrm.CropX_view1 = 0;
A.trackerDeepData.sPrm.CropY_view1 = 0;
A.trackerDeepData.sPrm.scale = 1;
A.trackerDeepData.sPrm.normalize = 0;

lbl_file = sprintf('%s_modified.lbl',splitext(lbl_file));
save(lbl_file,'-struct','A','-v7.3');

mov_file = 'source ~/bransonlab/venv/bin/activate; /groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/alice/cx_GMR_SS00168_CsChr_RigD_20150909T111218/movie.ufmf';
out_file = [splitext(mov_file)  '_interface_test.trk'];
name = 'alice_test_apt';
cmd = sprintf('%s python APT_interface.py %s -name %s train',basecmd, lbl_file,name);
if train
  system(cmd);
end
cmd = sprintf('%s python APT_interface.py %s -name %s track -mov %s -out %s -end_frame 1000',basecmd, lbl_file, name, mov_file, out_file);
if track,  
  system(cmd);
end

%% stephen for multi view
lbl_file = '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/stephen_ref.lbl';

A = load(lbl_file,'-mat');
A.trackerDeepData.sPrm.unet_steps = 2000;
A.trackerDeepData.sPrm.sizex = 512;
A.trackerDeepData.sPrm.sizey = 512;
A.trackerDeepData.sPrm.CropX_view1 = 0;
A.trackerDeepData.sPrm.CropY_view1 = 0;
A.trackerDeepData.sPrm.CropX_view2 = 128;
A.trackerDeepData.sPrm.CropY_view2 = 0;
A.trackerDeepData.sPrm.scale = 2;
A.trackerDeepData.sPrm.NChannels = 3;

lbl_file = sprintf('%s_modified.lbl',splitext(lbl_file));
save(lbl_file,'-struct','A','-v7.3');

mov_files = {'/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/fly516/C001H001S0001/C001H001S0001_c.avi',...
            '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/fly516/C002H001S0001/C002H001S0001_c.avi'};
out_files = {};
for ndx = 1:numel( mov_files),
    out_files{end+1} = [splitext(mov_files{ndx}) '_interface_test.trk'];
end

name = 'stephen_test_apt';
cmd = sprintf('%s python APT_interface.py %s -name %s train',basecmd, lbl_file,name);
if train
  system(cmd);
end
cmd = sprintf('%s python APT_interface.py %s -name %s track -mov %s %s -out %s %s', basecmd, ...
  lbl_file, name, mov_files{1}, mov_files{2}, out_files{1}, out_files{2});
if track
  system(cmd);
end

%% stephen for open pose
lbl_file = '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/stephen_ref.lbl';

A = load(lbl_file,'-mat');
A.trackerDeepData.sPrm.unet_steps = 2000;
A.trackerDeepData.sPrm.sizex = 512;
A.trackerDeepData.sPrm.sizey = 512;
A.trackerDeepData.sPrm.CropX_view1 = 0;
A.trackerDeepData.sPrm.CropY_view1 = 0;
A.trackerDeepData.sPrm.CropX_view2 = 128;
A.trackerDeepData.sPrm.CropY_view2 = 0;
A.trackerDeepData.sPrm.scale = 2;

lbl_file = sprintf('%s_modified.lbl',splitext(lbl_file));
save(lbl_file,'-struct','A','-v7.3');

mov_files = {'/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/fly516/C001H001S0001/C001H001S0001_c.avi',...
            '/groups/branson/bransonlab/mayank/PoseTF/data/apt_interface/stephen/fly516/C002H001S0001/C002H001S0001_c.avi'};
out_files = {};
for ndx = 1:numel( mov_files),
    out_files{end+1} = [splitext(mov_files{ndx}) '_interface_test.trk'];
end

name = 'stephen_test_apt';
cmd = sprintf('%s python APT_interface.py %s -name %s train',basecmd, lbl_file,name);
if train
  system(cmd);
end
cmd = sprintf('%s python APT_interface.py %s -name %s track -mov %s %s -out %s %s', basecmd, ...
  lbl_file, name, mov_files{1}, mov_files{2}, out_files{1}, out_files{2});
if track
  system(cmd);
end


