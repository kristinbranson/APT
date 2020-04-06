hfig = figure(1234);
%% make a video that just shows the frame number

assert(exist('videotest','dir')>0);

outmovfile = 'videotest/FrameNumber.avi';
writerobj = VideoWriter(outmovfile,'Uncompressed AVI');
writerobj.FrameRate = 30;
open(writerobj);

nframeswrite = 1000;

hfig = figure(1234);
set(hfig,'Position',[10,10,200,100]);
clf;
hax = axes('Position',[0,0,1,1]);
set(hax,'XTick',[],'YTick',[]);
htext = text(.5,.5,'Test','HorizontalAlignment','center','VerticalAlignment','middle','FontSize',48);

for i = 1:nframeswrite,
  set(htext,'String',num2str(i));
  fr = getframe(hax);
  drawnow;
  writeVideo(writerobj,fr);
end

close(writerobj);

%% matlab mjpeg

[readframe,nframes] =  get_readframe_fcn('videotest/FrameNumber.avi');
outmovfile = 'videotest/FrameNumber_linuxmatlab17a_mjpeg.avi';
writerobj = VideoWriter(outmovfile);
writerobj.FrameRate = 30;
open(writerobj);

for i = 1:nframes,
  writeVideo(writerobj,readframe(i));
end
close(writerobj);

%% test a bunch of movies created in a variety of ways

% mencoder FrameNumber.avi -o FrameNumber_xvid.avi -ovc xvid -xvidencopts fixed_quant=4
% mencoder FrameNumber.avi -o FrameNumber_x264.avi -ovc x264
% mencoder FrameNumber.avi -o FrameNumber_lavc.avi -ovc lavc (this is by default mpeg4)
% mencoder FrameNumber.avi -o FrameNumber_lavc_mjpeg.avi -ovc lavc -lavcopts vcodec=mjpeg
% avconv -i FrameNumber.avi -c:v mpeg4 -b:v 600k -mbd rd -flags +mv4+aic -trellis 2 -cmp 2 -subcmp 2 -g 300 FrameNumber_mpeg4.avi
% windows virtualdub, xvid default settings -> FrameNumber_vdub_xvid.avi => PROBLEM
% windows virtualdub, xvid default settings + force key frame every 1 frames -> FrameNumber_vdub_xvid_keyframes1.avi => PROBLEM
% windows virtualdub, h264 default settings -> FrameNumber_vdub_h264.avi
% windows virtualdub, ffdshow, mjpeg -> FrameNumber_vdub_ffdshow_mjpg.avi
% windows handbrake, h264 (x264) default settings -> FrameNumber_handbrake_x264.avi => PROBLEM with the first frame only
% windows handbrake, h264 (intel qsv) default settings -> FrameNumber_handbrake_h264_intelqsv.avi => PROBLEM with first two frames
% windows handbrake, x265 default settings -> FrameNumber_handbrake_h265.avi => COULD NOT OPEN
% windows handbrake, mpeg4 default settings -> FrameNumber_handbrake_mpeg4.avi
% windows handbrake, mpeg2 default settings -> FrameNumber_handbrake_mpeg2.avi => PROBLEM with first two frames
% linux matlab 2017a, mjpeg -> videotest/FrameNumber_linuxmatlab17a_mjpeg.avi
% from stephen: vdub, windows, h264 -> FrameNumber_H264_singlePass_zeroLatency_NoHack.avi
% from alice: mac, don't know what else:
% FrameNumberImageJjpeg.avi 
% FrameNumberImageJpng.avi 
% FrameNumberiSkySoftavi.avi
% FrameNumberiSkySoftmp4.mp4
% FrameNumberiSkysoftmov.mov
% from alice, windows -> FrameNumberffmpeg.mp4 => PROBLEM with first frame. frames are also stretched weirdly

%inmovfile = 'videotest/FrameNumber_vdub_h264.avi';
%inmovfile = 'videotest/FrameNumber_vdub_xvid.avi';
%inmovfile = 'videotest/FrameNumber_xvidFFmpegWindows.avi';

allinmovfiles = mydir('videotest');

nframesread = 200;
naxr = 20;
naxc = 10;
order = randperm(nframesread);

for movi = i:numel(allinmovfiles),
  
  inmovfile = allinmovfiles{movi};
  [p,n,ext] = fileparts(inmovfile);
  %try
    readframe = get_readframe_fcn(inmovfile,'preload',true);
  %catch ME,
  %  disp(getReport(ME));
  %  continue;
  %end

  hfig = figure;
  set(gcf,'Position',[10,10,920,1300]);
  clf;
  hax = createsubplots(naxr,naxc,0);
  set(hax,'XTick',[],'YTick',[]);
  
  for ii = 1:nframesread,
    i = order(ii);
    im = readframe(i);
    image(hax(i),im);
    axis(hax(i),'image');
    xlabel(hax(i),num2str(i));
    set(hax(i),'XTick',[],'YTick',[]);
  end
  
  set(hfig,'Name',n);
  drawnow;
  
  SaveFigLotsOfWays(hfig,fullfile(p,[n,'_VideoTest']),{'png'});
  
end
