%mmf writer
function [nkeyframes,nframes]=mmfwriter(inputfile,outputfile,firstframe,lastframe)

KEYFRAMEINTERVAL = 180;

obj=VideoReader(inputfile);
foutput=fopen(outputfile,'wb');
%load text
thisPath = mfilename('fullpath');
[parentDir] = fileparts(thisPath);
textfilename = fullfile(parentDir,'text.text');
ftexto=fopen(textfilename,'r');

texto=[];
for i=1:15
linea=fgets(ftexto);
texto=[texto linea];
end

%write header text
fwrite(foutput,texto,'char')
% end of the header
fwrite(foutput,0,'char')
%id_code
d=hex2dec('a3d2d45d');
fwrite(foutput,d,'ulong')
%header size
fwrite(foutput,10240,'int')
%calculate keyframes
nkeyframes=ceil((lastframe-firstframe+1)/KEYFRAMEINTERVAL);
%keyframe
fwrite(foutput,KEYFRAMEINTERVAL,'int')
%other recording parameters TODO
% zero padded to reach header size
for i=ftell(foutput)+1:10240
fwrite(foutput,0,'uint8');
end
%calculate nframes
%
nframes=zeros(1,nkeyframes);
nframes(1:end-1)=KEYFRAMEINTERVAL;
nframes(end)=(lastframe-firstframe+1)-(nkeyframes-1)*KEYFRAMEINTERVAL;

isfirst=1;
% for the stacks
for stack=1:nkeyframes
    
%Image Stacks
stackstart_loc = ftell(foutput);
begining_header=ftell(foutput);
%idcode
d=hex2dec('bb67ca20');
fwrite(foutput,d,'ulong')
%header size
fwrite(foutput,512,'int')
stacksize_loc = ftell(foutput);
%total size of stack in disk TODO
fwrite(foutput,0,'int')
%nframes in this stack default=180 but the last one can be smaller
fwrite(foutput,nframes(stack),'int');
% zero padded to reach header size
for i=ftell(foutput)-begining_header+1:512
fwrite(foutput,0,'uint8');
end

%Backgorund image
begining_header=ftell(foutput);
%header size
fwrite(foutput,112,'int')
%imageid
fwrite(foutput,0,'int')
%numchannels
fwrite(foutput,1,'int')
%alphachannel
fwrite(foutput,0,'int')
%depth 
fwrite(foutput,8,'int')
%colormodel
fwrite(foutput,'GRAY','char')
%channel seq
fwrite(foutput,'GRAY','char')
%data order
fwrite(foutput,0,'int')
% origin
fwrite(foutput,0,'int')
%align should 4 or 8 TODO
fwrite(foutput,4,'int')
%width 
fwrite(foutput,obj.width,'int')
%height
fwrite(foutput,obj.height,'int')
%roi 0=> full image
fwrite(foutput,0,'int')
%mask roi
fwrite(foutput,0,'int')
%im id
fwrite(foutput,0,'int')
%tile info
fwrite(foutput,0,'int')
%images size (heightxwidth
fwrite(foutput,obj.width*obj.height,'int')
%imagedata
fwrite(foutput,0,'int')
%widthstep
fwrite(foutput,obj.width,'int')
%bordermode
fwrite(foutput,zeros(1,4),'int')
%borderconst
fwrite(foutput,zeros(1,4),'int')
%imagedataorigin
fwrite(foutput,0,'int')
% zero padded to reach header size
for i=ftell(foutput)-begining_header+1:112
fwrite(foutput,0,'uint8');
end
s=0;
im=zeros(obj.height,obj.width,nframes(stack));
if isfirst==1
    for i=firstframe:firstframe+nframes(stack);
        s=s+1;
        imrgbtemp=read(obj,i);
        imrgbtemp=double(imrgbtemp);
        im(:,:,s)=sum(imrgbtemp,3)/3;
    end
    isfirst=0;
else 
    for i=firstframe+sum(nframes(1:stack-1)):firstframe+sum(nframes(1:stack))
        s=s+1;
        imrgbtemp=read(obj,i);
        imrgbtemp=double(imrgbtemp);
        im(:,:,s)=sum(imrgbtemp,3)/3;
    end
end
    
%%creting mean backgfix
bkgim=median(im,3);
%writing background image
fwrite(foutput,reshape(bkgim',1,obj.width*obj.height))


%Extracting larvae out of the images

 frames=zeros(size(im));
 SE=strel('disk',8);
 bounding=cell(1,nframes(stack));
 
for i=1:nframes(stack)
   begining_header=ftell(foutput);
   %id code
   d=hex2dec('f80921af');
   fwrite(foutput,d,'ulong')
   %header size
   fwrite(foutput,1024,'int')
   %depth
   fwrite(foutput,8,'int')
   %nchannels
   fwrite(foutput,1,'int')
   
   %calculating numims
    frames(:,:,i)=abs(im(:,:,i)-bkgim);
    %threshold automatically set
    %level(i)=graythresh(frames(:,:,i));
     bw=zeros(size(frames(:,:,1)));
     bw(frames(:,:,i)*10>max(max(frames(:,:,i))))=1;
     bw2=bwareaopen(bw,7);
     dilated=imdilate(bw2,SE);
     cc=bwconncomp(dilated);
     
     %numims
     fwrite(foutput,cc.NumObjects,'int')
     % metadata idcode
     fwrite(foutput,hex2dec('c15ac674'),'ulong');
     % 6 key values
     fwrite(foutput,6,'int');
     % ROIX = 360
     fwrite(foutput,'ROIX','char');
     fwrite(foutput,0,'char');
     fwrite(foutput,360,'double');
      % ROIY = 30
     fwrite(foutput,'ROIY','char');
     fwrite(foutput,0,'char');
     fwrite(foutput,30,'double');
     % bufnum = 23226335
     fwrite(foutput,'bufnum','char');
     fwrite(foutput,0,'char');
     fwrite(foutput,23226335,'double');
     % bufnum_time = 41
     fwrite(foutput,'bufnum_time','char');
     fwrite(foutput,0,'char');
     fwrite(foutput,41,'double');
     % camtime = 2224245094
     fwrite(foutput,'camtime','char');
     fwrite(foutput,0,'char');
     fwrite(foutput,2224245094,'double');
     % frameAddedTimeStamp = 9.454610
     fwrite(foutput,'frameAddedTimeStamp','char');
     fwrite(foutput,0,'char');
     fwrite(foutput,9.454610,'double');
     
     % zero padded to reach header size
        for k=ftell(foutput)-begining_header+1:1024
        fwrite(foutput,0,'uint8');
        end
%     %Creating bounding box
     STATS=regionprops(cc,'BoundingBox');
      bounding{i}=zeros(cc.NumObjects,4);
     for j=1:cc.NumObjects
         bounding{i}(j,:)=STATS(j,1).BoundingBox;
         temp=im(bounding{i}(j,2)+.5:bounding{i}(j,2)+bounding{i}(j,4)-.5,...
           bounding{i}(j,1)+.5:bounding{i}(j,1)+bounding{i}(j,3)-.5,i);
%          %temp=zeros(fix(bounding{i}(j,4)),fix(bounding{i}(j,3)));
%          if fix(bounding{i}(j,2))==0 && fix(bounding{i}(j,1))==0
%             temp=im(1:fix(bounding{i}(j,4)),1:fix(bounding{i}(j,3)),i); 
%         elseif fix(bounding{i}(j,2))==0
%             temp=im(1:fix(bounding{i}(j,4)),fix(bounding{i}(j,1)):fix(bounding{i}(j,1))+fix(bounding{i}(j,3))-1,i);
%         elseif fix(bounding{i}(j,1))==0
%             temp=im(fix(bounding{i}(j,2)):fix(bounding{i}(j,2))+fix(bounding{i}(j,4))-1,1:fix(bounding{i}(j,3)),i);     
%         else
%             temp=im(fix(bounding{i}(j,2)):fix(bounding{i}(j,2))+fix(bounding{i}(j,4))-1,fix(bounding{i}(j,1)):fix(bounding{i}(j,1))+fix(bounding{i}(j,3))-1,i);
%          end
        fwrite(foutput,bounding{i}(j,:)-[.5,.5,0,0],'int')
        fwrite(foutput,reshape(temp',1,bounding{i}(j,3)*bounding{i}(j,4)))
      end
   
end

stackend_loc = ftell(foutput);
stacksize = stackend_loc - stackstart_loc;
fseek(foutput,stacksize_loc,'bof');
fwrite(foutput,stacksize,'int');
fseek(foutput,stackend_loc,'bof');

end
fclose(foutput)

