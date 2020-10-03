%%
% BSIZES = [1 2 4 8 16 32 64 128];
% 
% for n=nets(:)',n=n{1};
%   s.(n).t_overall = cell2mat(cat(1,s.(n).t_overall{:}));
%   s.(n).t_pred = cellfun(@cell2mat,s.(n).t_pred,'uni',0);
%   s.(n).t_read = cellfun(@cell2mat,s.(n).t_read,'uni',0);
% end

s = load('res20200210.mat');
nets = setdiff(fieldnames(s),'bsizes');
s.bsizes = double(s.bsizes);
nnets = numel(nets);
bszlbls = arrayfun(@num2str,s.bsizes,'uni',0);

%% time breakdown
times = nan(nnets,numel(s.bsizes),4);
for inet=1:nnets
  n = nets{inet};
  snet = s.(n).vw0;
  tnet = TrackFPS.getTimesTbl(s.bsizes,snet);
  times(inet,:,:) = tnet{:,2:end};
end
%times = permute(times,[2 1 3]);
%pnet = [3 1 2];
pnet = 1:nnets;
times = times(pnet,:,:);
netsplot = nets(pnet);
FIGNUM = 12;
hfig = figure(FIGNUM);
clf;
h = plotBarStackGroups(times, netsplot, FIGNUM);
clrs = lines(4);
%clrs = clrs([1 2 4 3],:);
for iclr=1:4
  [h(:,iclr).FaceColor] = deal(clrs(iclr,:));
end
TIMEFLDS = {'t_predinnr' 't_predothr' 't_read_tot' 't_overothr'};
TIMEFLDSPRTY = {'pred/net' 'pred/othr' 'read' 'othr/unk'};
assert(isequal(tnet.Properties.VariableNames(2:end),TIMEFLDS));             
legend(TIMEFLDSPRTY,'interpreter','none');
grid on;
ax = gca;
ax.FontSize = 16;
args = {'fontweight','bold','interpreter','none'};
xlabel('net',args{:});
ylabel('time (s)',args{:})
title('Inference compute time, FlyBub Ngt=1800',args{:});



%% Inf FPS
TIME_FLD = 't_pred';
TSTR = sprintf('FlyBub GT inference speed, all pred (%s)',TIME_FLD);
FIGNUM = 15;

PTILES = [25 50 75];
VWS = [0];
nbsizes = numel(s.bsizes);
nvw = numel(VWS);
nnets = numel(nets);
nptls = numel(PTILES);
mfps = nan(nbsizes,nnets,nptls);
for inet=1:nnets
  n = nets{inet};
  for ivw=1:nvw
    vw = VWS(ivw);    
    snetvw = s.(n).(['vw' num2str(vw)]);
    assert(numel(snetvw.(TIME_FLD))==nbsizes);
    for ib=1:nbsizes
      texec = snetvw.(TIME_FLD){ib};  % batch pred exec times in s. these are raw but we are taking ptiles anyway
      fps = s.bsizes(ib)./texec;
      p = prctile(fps,PTILES);
      mfps(ib,inet,:) = p;
    end
  end
end

hfig = figure(FIGNUM);
clf
plot(log2(s.bsizes),mfps(:,:,2),'.','markersize',30);
hold on
ax = gca;
ax.ColorOrderIndex = 1;
plot(log2(s.bsizes),mfps(:,:,1),'--');
ax.ColorOrderIndex = 1;
plot(log2(s.bsizes),mfps(:,:,3),'--');
plot([nan nan],[nan nan],'k--');
legend([nets(:); '+/- 25%ile'],'location','northwest')
grid on;
xlabel('batch size','fontweight','bold');
ax.XTick = log2(s.bsizes);
ax.XTickLabel = arrayfun(@num2str,s.bsizes,'uni',0);
ax.FontSize = 16;
ylabel('Inference FPS','fontweight','bold');
title(TSTR,'fontweight','bold','interpreter','none');
    