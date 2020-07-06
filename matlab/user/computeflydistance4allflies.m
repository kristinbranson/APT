function [flydist,flyID] = computeflydistance4allflies(fly,frame,trxstruct)

% find all the fly alive at current frame
isalive = zeros(1,numel(trxstruct));
for i = 1:numel(trxstruct)
    if frame >= trxstruct(i).firstframe && frame <= trxstruct(i).endframe
        isalive(i) = 1;
    end
end

% find dcenter from fly to all other flies in frame
flyIDs = 1:1:numel(trxstruct);
liveIDs = flyIDs(logical(isalive));

if numel(liveIDs) <= 1
    flydist = [];
    flyID = [];
else  
    flyx = trxstruct(fly).x_mm(frame+trxstruct(fly).off);
    flyy = trxstruct(fly).y_mm(frame+trxstruct(fly).off);    
    for i = 1:numel(liveIDs)
        flyi_x = trxstruct(liveIDs(i)).x_mm(frame+trxstruct(liveIDs(i)).off);
        flyi_y = trxstruct(liveIDs(i)).y_mm(frame+trxstruct(liveIDs(i)).off);
        dx = flyx-flyi_x;
        dy = flyy - flyi_y;
        fdist(i) = sqrt(dx.^2 + dy.^2);
    end   
    [a,b] = sort(fdist);
    flyID = liveIDs(b(2:end));
    flydist = a(2:end);
end