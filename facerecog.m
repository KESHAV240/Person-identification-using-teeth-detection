%training the model
clc
clear all
close all
warning off;
cao=webcam;%turning on webcam
teethDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',64); %detection 
c=150;%number of images taken
temp=0;
while true
    e=cao.snapshot;
    bboxes =step(teethDetect,e);%detecting teeth
    if(sum(sum(bboxes))~=0)
    if(temp>=c)
        break;
    else
    es=imcrop(e,bboxes(1,:));%cropping the image 
    es=imresize(es,[277 277]);%resizing the image for training 
    filename=strcat(num2str(temp),'.bmp');
    imwrite(es,filename);
    temp=temp+1;
    imshow(es);
    drawnow;
    end
    else
        imshow(e);
        drawnow;
    end
end