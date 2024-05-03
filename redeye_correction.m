%red eye correction

clear
clc

%tuning parameters
t_eye = .05;

%image files location and suffixes
filepath = "redeyeimages/";
filename_flash = "_flash.jpg";
filename_noflash = "_noflash.jpg";

%read 1st image pair and convert to ycbcr luminance/chrominance space
imID = "01";
fn = strcat(filepath,imID,filename_noflash);
A_YCC = rgb2ycbcr(double(imread(fn))/256.0);
fn = strcat(filepath,imID,filename_flash);
F_YCC = rgb2ycbcr(double(imread(fn))/256.0);

%calculate relative redness measure
R = F_YCC(:,:,3) - A_YCC(:,:,3);

%segment based on t_eye
Rseg = R.*(R>t_eye);
figure(1);
imshow(Rseg, []);
title("relative redness measure, image " + imID + ", τ_{eye} = " + t_eye);

%find seed pixels using k-means clustering isntead of t_dark
k = 4;
[Rseed,centers] = imsegkmeans(int8(Rseg*256),k);
figure(2);
[~, i] = max(centers);

%sample Rseg on most intense clusters; blur for connectivity
Rseed = imgaussfilt(Rseg.*(Rseed==i), 2);
imshow(Rseed, []);
title("seed pixels, image " + imID + ", k = " + k);

%determine elliptical constraints
R = R - min(min(R));
Rmask = zeros(size(R));
stats = regionprops("table", Rseed>0, "Centroid", "MajorAxisLength", "MinorAxisLength");
centers = stats.Centroid;
majors = stats.MajorAxisLength;
for i = 1:size(centers)
    for row = floor(centers(i,1)-majors(i)):ceil(centers(i,1)+majors(i))
        for col = floor(centers(i,2)-majors(i)):ceil(centers(i,2)+majors(i))
            if (row-centers(i,1))^2 + (col-centers(i,2))^2 < (majors(i)/2)^2
                Rmask(col,row) = Rseg(col,row);
            else %feathering (failed)
                %Rmask(col,row) = max(0, R(col,row) / (((row-centers(i,1))^2 + (col-centers(i,2))^2) / (majors(i)/2)^2)^4);
            end
        end
    end
end

%remove flash specularities; final red eye mask
Rmask = (Rmask / max(max(Rmask))).^.25;
Rmask = imgaussfilt(Rmask, 1);
Rmask(F_YCC(:,:,1) > .80*max(max(F_YCC(:,:,1)))) = 0;
Rmask = imgaussfilt(Rmask, .5);
figure(3);
imshow(Rmask, []);
title("final red eye mask");

%correct red eye artifact
FinalImage = double(imread(fn))/256.0;
figure(4)
imshow(FinalImage, []);
title("original image");
FinalImage(:,:,1) = (1-Rmask).*FinalImage(:,:,1) + .8*Rmask.*(A_YCC(:,:,1));
FinalImage(:,:,2) = (1-Rmask).*FinalImage(:,:,2) + .8*Rmask.*(A_YCC(:,:,1));
FinalImage(:,:,3) = (1-Rmask).*FinalImage(:,:,3) + .8*Rmask.*(A_YCC(:,:,1));

figure(5);
imshow(FinalImage, []);
title("final corrected image");

figure(6);
subplot(1,3,1);
imshow(Rmask, []);
title("final red eye mask");
subplot(1,3,2);
imshow(imread(fn), []);
title("original image");
subplot(1,3,3);
imshow(FinalImage, []);
title("final corrected image");

figure(7);

fn = strcat(filepath,imID,filename_noflash);
A_YCC = rgb2ycbcr(double(imread(fn))/256.0);

subplot(1,3,1);
imshow(imread(fn), []);
title("original no flash image");

fn = strcat(filepath,imID,filename_flash);
F_YCC = rgb2ycbcr(double(imread(fn))/256.0);

subplot(1,3,2);
imshow(imread(fn), []);
title("original flash image");
subplot(1,3,3);
imshow(FinalImage, []);
title("final corrected image");