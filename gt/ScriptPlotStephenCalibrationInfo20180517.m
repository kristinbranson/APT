%load('Z:\apt\experiments\data\trnDataSH_20180503.mat');
%shinfo = importdata('Z:\apt\experiments\data\shflies.csv');

load('/groups/branson/bransonlab/apt/experiments/data/trnDataSH_20180503.mat');
shinfo = readtable('/groups/branson/bransonlab/apt/experiments/data/old/shflies.csv');
excludeflyids = importdata('/groups/branson/bransonlab/apt/experiments/data/exclude_flyids_20180601.txt');

%% locs of videos

sfid = fopen('/groups/branson/bransonlab/apt/experiments/data/stephen_side_list.txt','r');
tline = fgetl(sfid);
slist = {};
while ischar(tline),
  slist{end+1} = tline;
  tline = fgetl(sfid);
end
fclose(sfid);
mflyid = nan(numel(slist),1);
remove = [];
for ndx = 1:numel(slist)
  splits = strsplit(slist{ndx},'/');
  [~,~,~,~,flystr] = regexpi(splits{end-2},'fly( *)(\d+)');
  if isempty(flystr),
%     fprintf('%s %s\n',splits{end-2},slist{ndx});
    continue;
  end
  mflyid(ndx) = str2double(flystr{1}{2});
end

[ism,idx] = ismember(flyids,mflyid);
movfiles = cell(1,numel(flyids));
movfiles(ism) = slist(idx(ism));


%%

flyids = shinfo.fly;
%flyids = shinfo.data(:,1);
[ism,flyidx] = ismember(tMain.flyID,flyids);
assert(all(ism));
[nlabelsperfly] = hist(flyidx,1:numel(flyids));
[trainingflies,sampletrainingframe] = unique(tMain.flyID);

assert(all(ismember(excludeflyids,flyids)));

%%

hfig = 1;
figure(hfig);
clf;
isbadcalib = ismember(flyids,excludeflyids);
oldisbadcalib = shinfo.isBadCalib==1;

hold on;
legends = {};
for isoldbad = [false,true],
  for isnewbad = [false,true],
    plot(flyids( (oldisbadcalib==isoldbad)&(isbadcalib==isnewbad) ),...
      nlabelsperfly((oldisbadcalib==isoldbad)&(isbadcalib==isnewbad)),...
      '.','LineWidth',3,'MarkerSize',12);
    if isoldbad && isnewbad,
      legends{end+1} = 'Both bad';
    elseif isoldbad && ~isnewbad,
      legends{end+1} = 'Only old bad';
    elseif ~isoldbad && isnewbad,
      legends{end+1} = 'Only new bad';
    elseif ~isoldbad && ~isnewbad,
      legends{end+1} = 'OK';
    else
      error('!');
    end
  end
end

legend(legends);
xlabel('Fly ID');
ylabel('Number of training labels');

%% plot one frame from each training fly

%naxc = ceil(sqrt(numel(trainingflies)));
%naxr = ceil(numel(trainingflies)/naxc);
naxr = 9;
naxc = 11;

hfig = 2;
figure(hfig);
clf;
hax = createsubplots(naxr,naxc,0);
hax = reshape(hax,[naxr,naxc]);

colors = lines(4);

for i = 1:numel(trainingflies),

  image(hax(i),repmat(IMain_crop3{sampletrainingframe(i),2},[1,1,3]));
  hold(hax(i),'on');
  axis(hax(i),'image','off');
  j = find(flyids == trainingflies(i));
  isnewbadcurr = isbadcalib(j);
  isoldbadcurr = oldisbadcalib(j);
  colori = sub2ind([2,2],isnewbadcurr+1,isoldbadcurr+1);
  colorcurr = colors(colori,:);
%   if isbadcurr,
%     colorcurr = 'r';
%   else
%     colorcurr = 'g';
%   end
  text(5,5,sprintf('%d, %d',trainingflies(i),nlabelsperfly(j)),'HorizontalAlignment','left','VerticalAlignment','top','Color',colorcurr,'Parent',hax(i));
  drawnow;
  
end

subplot(hax(1,end));
hold on;
h = [];
for isoldbad = [false,true],
  for isnewbad = [false,true],
    h(end+1) = plot(nan,nan,'.','LineWidth',3,'MarkerSize',12);
  end
end
legend(h,legends)
delete(hax(numel(trainingflies)+1:end));

%% plot one frame for each test fly

idxplot = find(~cellfun(@isempty,movfiles));

naxr = 20;
naxc = ceil(numel(idxplot)/naxr);


hfig = 3;
figure(hfig);
clf;
hax = createsubplots(naxr,naxc,0);
hax = reshape(hax,[naxr,naxc]);

ims = cell(1,numel(idxplot));
parfor ii = 1:numel(idxplot),
  fprintf('%d / %d\n',ii,numel(idxplot));

  frontmovfile = strrep(movfiles{idxplot(ii)},'C001','C002');
  readframe = get_readframe_fcn(frontmovfile);
  ims{ii} = readframe(1);
end

for ii = 1:numel(idxplot),
  fprintf('%d / %d\n',ii,numel(idxplot));
    
  i = idxplot(ii);
%   htext = findobj(hax(ii),'type','text');
%   delete(htext);
  
  
  im = ims{ii};
  image(hax(ii),im);
  
  hold(hax(ii),'on');
  axis(hax(ii),'image','off');
  j = find(flyids(i) == trainingflies);
  if isempty(j),
    nlabelscurr = 0;
  else
    nlabelscurr = nlabelsperfly(i);
  end
  isnewbadcurr = isbadcalib(i);
  isoldbadcurr = oldisbadcalib(i);
  colori = sub2ind([2,2],isnewbadcurr+1,isoldbadcurr+1);
  colorcurr = colors(colori,:);
  
%   if isbadcurr,
%     colorcurr = 'r';
%   else
%     colorcurr = 'g';
%   end
  text(5,5,sprintf('%d, %d',flyids(i),nlabelscurr),'HorizontalAlignment','left','VerticalAlignment','top','Color',colorcurr,'Parent',hax(ii));
  %drawnow;
  
end


subplot(hax(numel(idxplot)));
hold on;
h = [];
for isoldbad = [false,true],
  for isnewbad = [false,true],
    h(end+1) = plot(nan,nan,'.','LineWidth',3,'MarkerSize',12);
  end
end
legend(h,legends)

delete(hax(numel(idxplot)+1:end));