clc;close;clear
c=webcam;
load myNet2;
teethDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',128);
while true
    e=c.snapshot;
    bboxes =step(teethDetect,e);
    if(sum(sum(bboxes))~=0)
     es=imcrop(e,bboxes(1,:));
    es=imresize(es,[227 227]);
    label=classify(myNet2,es);
    image(e);
    title(char(label));
    drawnow;
    else
        image(e);
        title('No Face Detected');
    end
end