function [stimulusOnOff,stimulusCase] = flyNum2stimFrames_SJH(flynum)
% Takes Stephen's fly number and outputs the frames the stimulus turned on and off
%
% input:
%       
%       flynum = number of fly in Stephen's optogenetic experiments.  Same
%       number that comes after "fly" in the name of the folder containing
%       the videos.
%
% Output:
%
%       stimulusOnOff = 2 columns.  1st column = frame at which the
%       stimulus turned on, 2nd column = last frame at which the stimulus
%       was on.  Each row = a new stimulus epoch.  Indexing for video frames starts
%       at 1.  Note that the outputted frame numbers are only accurate to
%       roughly +/- 1 frames.  This is just a rough number for bransonlab to chop 
%       up stimulus vs. non-stimulus epochs.  If you want exact numbers for a particular 
%       video run Stephens_matlab_root/optogeneticsBehvior/PhotronAnalysis/getVidTimingInfo.m
%       on the DAQ files for that particular video.


%% turning fly number into stimulus case

% My notes from going through my old files:
%
% 1-18 stimulusCase = 6%LEDbursts_1sTrial_0.3sPulse_1pulsesInBursts_0.3sIBI
% 19-39 stimulusCase =5%LEDbursts_2.4sTrial_1sPulse_1pulsesInBursts_0.3sIBI
% 40-87 stimulusCase =4 %LEDbursts_8sTrial_0.5sPulse_1pulsesInBursts_0.7sIBI_
% 80-88 error - double named flies
% 89-99 stimulusCase=2;  %LEDbursts_10sTrial_0.3sPulse_1pulsesInBursts_0.3sIBI
% 100-132 stimulusCase =0; %LEDbursts_10sTrial_0.3sPulse_1pulsesInBursts_1sIBI
% 133-157 stimulusCase= 1; %LEDbursts_10sTrial_0.001sPulse_150pulsesInBursts_1sIBI
% 158-628 stimulusCase =0; %LEDbursts_10sTrial_0.3sPulse_1pulsesInBursts_1sIBI
% 629-1631   stimulusCase =3; photron_LEDbursts(12,0.3,0.3,1,1.2)
% 1632+ stimulusCase =7;  DAQ altered for panels photron_LEDbursts(12,0.3,0.3,1,1.2,0/1)

if flynum <=18
    stimulusCase = 6;
elseif (flynum>=19) && (flynum<=39)
    stimulusCase =5;
elseif (flynum>=40) && (flynum<=87)
    stimulusCase =4;
elseif (flynum>=80) && (flynum<=88)
    error('Fly numbers 80-88 are used twice.  Specify stimulus times manually')
elseif (flynum>=89) && (flynum<=99)
    stimulusCase =2;
elseif (flynum>=100) && (flynum<=132)
    stimulusCase =0;
elseif (flynum>=133) && (flynum<=157)
    stimulusCase =1;
elseif (flynum>=158) && (flynum<=628)
    stimulusCase =0;
elseif (flynum>=629) && (flynum<=1631)
    stimulusCase =3;
elseif flynum>=1632
    stimulusCase =7;
else
    error('Fly number not recognized or not in my list')
end





%% Getting stimulus on/off times for all stimulus cases.

%stimulusOnOff = frames in which the stimulus turned on and off
if stimulusCase == 0 %0 = most common stimulus times between fly # 119-132 & 158-628:
    % 234 frames stimmed
    
    stimulusOnOff(1,:) = [124 162];
    stimulusOnOff(2,:) = [324 362];
    stimulusOnOff(3,:) = [524 562];
    stimulusOnOff(4,:) = [724 762];
    stimulusOnOff(5,:) = [924 962];
    stimulusOnOff(6,:) = [1124 1162];

elseif stimulusCase ==1 % each pulse consists of a burst.  Filenames: LEDbursts_10sTrial_0.001sPulse_150pulsesInBursts_1sIBI
    % 266 frames stimmed
    % %stimulus times for old fly 157 142 etc:
    stimulusOnOff(1,:) = [124 161];
    stimulusOnOff(2,:) = [286 323];
    stimulusOnOff(3,:) = [449 486];
    stimulusOnOff(4,:) = [611 648];
    stimulusOnOff(5,:) = [773 810];
    stimulusOnOff(6,:) = [936 973];
    stimulusOnOff(7,:) = [1099 1136];

elseif  stimulusCase ==2 %fly 89-90: photron_LEDbursts(1,0.1,1,0.5) - old version
    %418 frms stimmed
    %stimulus times for fly 90:
    stimulusOnOff(1,:) = [37 74];
    stimulusOnOff(2,:) = [150 187];
    stimulusOnOff(3,:) = [262 299];
    stimulusOnOff(4,:) = [375 412];
    stimulusOnOff(5,:) = [487 524];
    stimulusOnOff(6,:) = [600 637];
    stimulusOnOff(7,:) = [712 749];
    stimulusOnOff(8,:) = [825 862];
    stimulusOnOff(9,:) = [937 974];
    stimulusOnOff(10,:) = [1050 1087];
    stimulusOnOff(11,:) = [1162 1199];

elseif  stimulusCase ==3 %fly numbers 629 and up using photron_LEDbursts(12,0.3,0.3,1,1.2)
    % 234 frms stimmed
    stimulusOnOff(1,:) = [150 188];
    stimulusOnOff(2,:) = [376 414];
    stimulusOnOff(3,:) = [601 639];
    stimulusOnOff(4,:) = [827 865];
    stimulusOnOff(5,:) = [1052 1090];
    stimulusOnOff(6,:) = [1278 1316];
    
elseif stimulusCase == 4 %LEDbursts_8sTrial_0.5sPulse_1pulsesInBursts_0.7sIBI_
    % 252 frms stimmed
    stimulusOnOff(1,:) = [86,148];
    stimulusOnOff(2,:) = [299,361];
    stimulusOnOff(3,:) = [511,573];
    stimulusOnOff(4,:) = [724,786];
    
elseif stimulusCase == 5 %LEDbursts_2.4sTrial_1sPulse_1pulsesInBursts_0.3sIBI
    % 125 frms stimmed
    stimulusOnOff=[37,161];
    
elseif stimulusCase == 6%LEDbursts_1sTrial_0.3sPulse_1pulsesInBursts_0.3sIBI
    % 38 frms stimmed
    stimulusOnOff=[36,73];
    
        
elseif stimulusCase ==7 %Stephen's stimulusCase = 4.  flynum>=1632 DAQ altered for panels photron_LEDbursts(12,0.3,0.3,1,1.2,0/1)
    
    stimulusOnOff(1,:) =     [139     177];
    stimulusOnOff(2,:) =     [364     402];
    stimulusOnOff(3,:) =     [589     627];
    stimulusOnOff(4,:) =     [814     852];
    stimulusOnOff(5,:) =     [1039	1077];
    stimulusOnOff(6,:) =     [1264	1302];
    
else
error('Stimulus timing not properly specified')
end
