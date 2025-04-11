function nsz = get_network_size(net_type,im_sz,batch_size,is_ma)
% Estimates GPU memory requirement.
%  net_type is the network type. Eg: 'mdn', 'deeplabcut' etc.
% imsz is the image size in pixels. eg [1024,1024].
% batch_size is the batch size. eg. 8
% return the memory size in MB.

if is_ma
dat_file = fullfile(APT.Root,'deepnet','data','network_size_ma.mat');
else
dat_file = fullfile(APT.Root,'deepnet','data','network_size.mat');
end
nsz = -1;
try
  A = load(dat_file);
catch ME
  msg = sprintf('Could not load network size data file. Cannot estimate network size.\n%',...
    ME.message);
  warning(msg);
  return;
end

% if strcmp(net_type,'leap')
%   nsz = 2000;
%   return;
% end

if ~any(strcmp(net_type,fieldnames(A.mem_use)))
  warningNoTrace('No data on GPU memory requirements for network type %s\n',net_type);
  return
end

net_dat = double(A.mem_use.(net_type)(:,:,1));
valid_dat = A.mem_use.(net_type)(:,:,2)>0.5;
bszs = double(A.batch_size);
i_sz1 = double(A.im_sz);
i_sz = (i_sz1./A.scales).^2;

%% This isn't that useful.
% [bb,ss] = meshgrid(bszs,i_sz);
% F = scatteredInterpolant(bb(valid_dat),...
%       ss(valid_dat),...
%       net_dat(valid_dat));
% 
% nsz = F(batch_size,im_sz(1)*im_sz(2));
% nsz = round(nsz/1024/1024/1024*10)/10;


%% Find bilinear coefficients by using differences

% First take difference along batch-size then along image size
d_b = diff(bszs);
d_s = diff(i_sz);
[bb,ss] = meshgrid(bszs,i_sz);
valid_d_b = valid_dat(:,1:end-1)&valid_dat(:,2:end);
valid_d_s = valid_dat(1:end-1,:)&valid_dat(2:end,:);

dat_d_b = diff(net_dat,1,2);
dat_d_1 = dat_d_b./d_b;

dat_d_s = diff(dat_d_1,1,1);
dat_d_2 = dat_d_s./d_s';
valid_d_2 = valid_d_b(1:end-1,:)&valid_d_b(2:end,:);
bi_coeff = mean(dat_d_2(valid_d_2));

% now that we have the bilinear coeff, find the individual coeff.
dat1 = net_dat-bi_coeff.*bb.*ss;

dat1_d_b = diff(dat1,1,2)./d_b;
b_coeff = mean(dat1_d_b(valid_d_b));

dat1_d_s = diff(dat1,1,1)./d_s';
s_coeff = mean(dat1_d_s(valid_d_s));

% now the constant
dat2 = dat1-bb*b_coeff-ss*s_coeff;
const = mean(dat2(valid_dat));

re_dat = bi_coeff.*bb.*ss + s_coeff*ss + b_coeff*bb + const;

%%
nsz = const + s_coeff*im_sz(1)*im_sz(2) + b_coeff*batch_size + ...
  bi_coeff*im_sz(1)*im_sz(2)*batch_size;
nsz = nsz/1024/1024;