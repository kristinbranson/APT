function [f,nf,fid,hifo] = bag_get_readframe_fcn(bagfname,varargin)

topicpat = myparse(varargin,...
  'topicpat',[] ... % topic pattern to match in Bag
  );

% if ~isStringScalar(cam_num) || ~ischar(cam_num)
%   try
%     cam_num = num2str(cam_num);
%   catch
%     error('Invalid camera number input.')
%   end
% end

[path, fn, bag_ext] = fileparts(bagfname);
fprintf('Getting file info for %s%s ... \n', fn, bag_ext)
% load bag info and get topics
bagInfo = rosbag('info',bagfname);
% find topics related to video (in Kinefly case, these contain the string
% 'camera'
% camera_topic_idx = arrayfun(...
%   @(x) contains(x.Topic, 'camera') && contains(x.Topic,cam_num), bagInfo.Topics);

image_msg_idx = arrayfun(@(x) contains(x.MessageType, 'Image'), bagInfo.Topics);
if nnz(image_msg_idx)==0
  error('No Topic with *Image MessageType found in %s.',bagfname);
end
if isempty(topicpat)
  % No topic supplied, but maybe there is only one topic with *Image MessageType
  tfcam = image_msg_idx; 
  if nnz(tfcam)>1
    errstr = sprintf('Available topics are: \n');
    errstr = [errstr sprintf('%s \n', bagInfo.Topics(tfcam).Topic)];
    error('Multiple Topics with *Image MessageType found in %s. %s',bagfname,errstr);
  end
else
  camera_topic_idx = arrayfun(@(x) contains(x.Topic,topicpat), bagInfo.Topics);
  tfcam = camera_topic_idx & image_msg_idx;
  
  if nnz(tfcam)==0
    error('No Topic matching pattern ''%s'' with *Image MessageType found in %s.',...
      topicpat,bagfname);
  elseif nnz(tfcam)>1 
    errstr = sprintf('Matching topics are: \n');
    errstr = [errstr sprintf('%s \n', bagInfo.Topics(tfcam).Topic)];
    error('Multiple Topics match pattern ''%s'' in %s. %s',...
      topicpat,bagfname,errstr);
  end
end
  
assert(nnz(tfcam)==1);
%else
%   topic_name = bagInfo.Topics(tfcam).Topic;
%   topic_name_split = strsplit(topic_name,'/') ;
%   topic_name_short = topic_name_split{2} ;
%end

% -------------------------------------------------
%% load bag file and get camera messages
% the code below is pretty slow, probably a better way to do this...
fprintf('Loading %s%s ... \n', fn, bag_ext)
bag = rosbag(bagfname);
cam_bag = select(bag,'Topic',bagInfo.Topics(tfcam).Topic);
clear bag % free up some memory?
% get info on general frame rate (we'll also save specific time stamps)
cam_time = [cam_bag.MessageList.Time]; % time stamps for messages
nf = length(cam_time);             % number of messages for this topic

hifo = struct;
FLDS = {'FilePath' 'StartTime' 'EndTime' 'NumMessages' 'AvailableTopics'};
for fld=FLDS,fld=fld{1};
  hifo.(fld) = cam_bag.(fld);
end
hifo.mean_fps = 1.0/mean(diff(cam_time)) ;   % frame rate estimated as average difference between time stamps
hifo.timeStamps = datenum(datetime(cam_time,'ConvertFrom','posixtime')) ; % convert to datenum

f = @lclread;
fid = [];

  function im = lclread(msgnum)
    cam_msgs = readMessages(cam_bag,msgnum);
    assert(isscalar(cam_msgs) && iscell(cam_msgs));
    im = readImage(cam_msgs{1});    
  end
end