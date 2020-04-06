%mmf writer
function [keyframes,nframes]=mmfwritertrans(inputfile,outputfile,obj,firstframe,lastframe)

%obj=VideoReader(inputfile);
foutput=fopen(outputfile,'wb');
%load text
ftexto=fopen('/groups/visitors/home/albam/Desktop/mmfreader/text.text');

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
keyframes=fix((lastframe-firstframe+1)/180)+1;
%keyframe
fwrite(foutput,keyframes,'int')
%other recording parameters TODO
% zero padded to reach header size
for i=ftell(foutput)+1:10240
fwrite(foutput,0,'uint8');
end
%calculate nframes
%
nframes=zeros(1,keyframes);
nframes(1:end-1)=180;
nframes(end)=(lastframe-firstframe+1)-(keyframes-1)*180;

isfirst=1;
% for the stacks
for stack=1:keyframes
    
%Image Stacks
begining_header=ftell(foutput);
%idcode
d=hex2dec('bb67ca20');
fwrite(foutput,d,'ulong')
%header size
fwrite(foutput,512,'int')
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
fwrite(foutput,[0 0 0 0],'char')
%channel seq
fwrite(foutput,[0 0 0 0],'char')
%data order
fwrite(foutput,0,'int')
% origin
fwrite(foutput,0,'int')
%align should 4 or 8 TODO
fwrite(foutput,0,'int')
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
fwrite(foutput,0,'int')
%borderconst
fwrite(foutput,0,'int')
%imagedataorigin
fwrite(foutput,0,'int')
% zero padded to reach header size
for i=ftell(foutput)-begining_header+1:112
fwrite(foutput,0,'uint8');
end
s=0;
im=zeros(obj.width,obj.height,nframes(stack));
if isfirst==1
    for i=firstframe:firstframe+nframes(stack);
        s=s+1;
        imrgbtemp=read(obj,i);
        imrgbtemp=double(imrgbtemp);
        imtemp=sum(imrgbtemp,3)/3;
        im(:,:,s)=imtemp';
    end
    isfirst=0;
else 
    for i=firstframe+sum(nframes(1:stack-1)):firstframe+sum(nframes(1:stack))
        s=s+1;
        imrgbtemp=read(obj,i);
        imrgbtemp=double(imrgbtemp);
        imtemp=sum(imrgbtemp,3)/3;
        im(:,:,s)=imtemp';
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
     % zero padded to reach header size
        for k=ftell(foutput)-begining_header+1:1024
        fwrite(foutput,0,'uint8');
        end
%     %Creating bounding box
     STATS=regionprops(cc,'BoundingBox');
      bounding{i}=zeros(cc.NumObjects,4);
     for j=1:cc.NumObjects
         bounding{i}(j,:)=STATS(j,1).BoundingBox;
         %temp=zeros(fix(bounding{i}(j,4)),fix(bounding{i}(j,3)));
         if fix(bounding{i}(j,2))==0 && fix(bounding{i}(j,1))==0
            temp=im(1:fix(bounding{i}(j,4)),1:fix(bounding{i}(j,3)),i); 
        elseif fix(bounding{i}(j,2))==0
            temp=im(1:fix(bounding{i}(j,4)),fix(bounding{i}(j,1)):fix(bounding{i}(j,1))+fix(bounding{i}(j,3))-1,i);
        elseif fix(bounding{i}(j,1))==0
            temp=im(fix(bounding{i}(j,2)):fix(bounding{i}(j,2))+fix(bounding{i}(j,4))-1,1:fix(bounding{i}(j,3)),i);     
        else
            temp=im(fix(bounding{i}(j,2)):fix(bounding{i}(j,2))+fix(bounding{i}(j,4))-1,fix(bounding{i}(j,1)):fix(bounding{i}(j,1))+fix(bounding{i}(j,3))-1,i);
         end
        fwrite(foutput,fix(bounding{i}(j,[2 1 4 3])),'int')
        fwrite(foutput,reshape(temp',1,fix(bounding{i}(j,3))*fix(bounding{i}(j,4))))
      end
   
end

end
fclose(foutput)

