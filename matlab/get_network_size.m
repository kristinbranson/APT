function nsz = get_network_size(net_type,im_sz,batch_size)
% Estimates GPU memory requirement.
%  net_type is the network type. Eg: 'mdn', 'deeplabcut' etc.
% imsz is the image size in pixels. eg [1024,1024].
% batch_size is the batch size. eg. 8
% return the memory size in MB.

dat_file = fullfile('deepnet','data','network_size.mat');
nsz = -1;
try
  A = load(dat_file);
catch ME
  msg = sprintf('Could not load network size data file. Cannot estimate network size.\n%',...
    ME.message);
  warning(msg);
  return;
end

if ~any(strcmp(net_type,fieldnames(A.mem_use)))
  warning('No data for network type:%s\n',net_type);
  return;
end

net_dat = A.mem_use.(net_type);
% columns are different image size.
% rows are different batch sizes.
n_px_in = prod(im_sz(1:2));
bszs = A.batch_size;
n_px = [];
n_px(1) = A.im_sz_1(1)*A.im_sz_2(1);
n_px(2) = A.im_sz_1(end)*A.im_sz_2(end);

% can't use interp2 because it doesn't work outside the input ranges.
% We interpolate using the highest and lowest values for batch and image
% sizes.

mem_batch = zeros(2,1);
% first interpolate along batch sizes.
mem_per_batch = (net_dat(1,end)-net_dat(1,1))/(bszs(end)-bszs(1));
mem_batch(1) = net_dat(1,1) + mem_per_batch*(batch_size-bszs(1));

mem_per_batch = (net_dat(end,end)-net_dat(end,1))/(bszs(end)-bszs(1));
mem_batch(2) = net_dat(end,1) + mem_per_batch*(batch_size-bszs(1));

% mem_batch now has memory requred for existing image sizes but for input
% batch size.

% now interpolate along number of pixels
mem_per_px = (mem_batch(2)-mem_batch(1))/(n_px(end)-n_px(1));
nsz = mem_batch(1) + mem_per_px*(n_px_in-n_px(1));
