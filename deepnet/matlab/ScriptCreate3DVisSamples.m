%% Need to be stopped in debugger at line 361 in compute3Dfrom2D.

figure;
hax = subplot(1,1,1);
hold on;
scale = 5000;
nframes = 10;
st = 3;
cc = jet(nframes);
for tt = st:(st+nframes-1);
  ns = nsamples(tt,4);
  scatter3(Psample(1,1:ns,4,tt),Psample(2,1:ns,4,tt),Psample(3,1:ns,4,tt),w(1:ns,4,tt)*scale,cc(tt-st+1,:),'.');
end


%%
OptionZ.FrameRate=30;
OptionZ.Duration=5.5;
OptionZ.Periodic=true;
CaptureFigVid([0,0;360,0],'Test3DTracking',OptionZ)
