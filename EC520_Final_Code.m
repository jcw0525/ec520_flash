%Carpet Image
filepath = '/Users/abbyskerker/Documents/Spring24/EC520/Project/flash_data_JBF_Detail_transfer/flash_data_JBF_Detail_transfer/';
filename_flash = 'carpet_00_flash.tif';
filename_noflash = 'carpet_01_noflash.tif';
filename_bilat = 'carpet_02_bilateral.tif';
filename_result = 'carpet_03_our_result.tif';

%Load in Files and Settings
fn = strcat(filepath,filename_flash);
flash_img = (imread(fn)); 
magnificationFactor = 0.25; 
flash_img = imresize(flash_img,magnificationFactor);
flash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
flash_exp_t = 1/40; 
fn = strcat(filepath,filename_noflash);
noflash_img = (imread(fn)); 
noflash_img = imresize(noflash_img,magnificationFactor);
noflash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
noflash_exp_t = 1/15; 
fn = strcat(filepath,filename_bilat);
bilat_img = (imread(fn)); 
bilat_img = imresize(bilat_img,magnificationFactor);
fn = strcat(filepath,filename_result);
result_img = (imread(fn)); 
result_img = imresize(result_img,magnificationFactor);

%% Sigma D Test

f_width = 0.1; 
sigma_d = 48; 
sigma_r_bilat = 20; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final1,F_base1,F_detail1,A_base1,A_NR1] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

f_width = 0.1; 
sigma_d = 36; 
sigma_r_bilat = 20; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final2,F_base2,F_detail2,A_base2,A_NR2] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

f_width = 0.1; 
sigma_d = 24; 
sigma_r_bilat = 20; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final3,F_base3,F_detail3,A_base3,A_NR3] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

%%
figure; 
subplot(2,2,1);
imshow(uint8(noflash_img));
title("No Flash Image")
xlim([250,400])
ylim([100,250])
subplot(2,2,2);
imshow(uint8(A_base1));
title("Sigma_d = 48")
xlim([250,400])
ylim([100,250])

subplot(2,2,3);
imshow(uint8(A_base2));
title("Sigma_d = 36")
xlim([250,400])
ylim([100,250])

subplot(2,2,4);
imshow(uint8(A_base3));
title("Sigma_d = 24")
xlim([250,400])
ylim([100,250])


set(gcf,'color','w');

%%
figure; 
subplot(2,2,1);
imshow(uint8(flash_img));
title("Flash Image")
xlim([250,400])
ylim([100,250])
subplot(2,2,2);
imshow(rescale(rgb2gray(F_detail1)))
title("Sigma_d = 48")
xlim([250,400])
ylim([100,250])

subplot(2,2,3);
imshow(rescale(rgb2gray(F_detail2)))
title("Sigma_d = 36")
xlim([250,400])
ylim([100,250])

subplot(2,2,4);
imshow(rescale(rgb2gray(F_detail3)))
title("Sigma_d = 24")
xlim([250,400])
ylim([100,250])

set(gcf,'color','w');

%%
figure; 
subplot(2,2,1);
imshow(uint8(noflash_img));
title("No Flash Image")
xlim([250,400])
ylim([100,250])
subplot(2,2,2);
imshow(uint8(A_final1));
title("Sigma_d = 48")
xlim([250,400])
ylim([100,250])

subplot(2,2,3);
imshow(uint8(A_final2));
title("Sigma_d = 36")
xlim([250,400])
ylim([100,250])

subplot(2,2,4);
imshow(uint8(A_final3));
title("Sigma_d = 24")
xlim([250,400])
ylim([100,250])


set(gcf,'color','w');

%% Sigma R Test

f_width = 0.1; 
sigma_d = 24; 
sigma_r_bilat = 20; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final1,F_base1,F_detail1,A_base1,A_NR1] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

f_width = 0.1; 
sigma_d = 24; 
sigma_r_bilat = 16; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final2,F_base2,F_detail2,A_base2,A_NR2] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

f_width = 0.1; 
sigma_d = 24; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final3,F_base3,F_detail3,A_base3,A_NR3] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

%%
figure; 
subplot(2,2,1);
imshow(uint8(noflash_img));
title("No Flash Image")
xlim([250,400])
ylim([100,250])
subplot(2,2,2);
imshow(uint8(A_base1));
title("Sigma_r = 20")
xlim([250,400])
ylim([100,250])

subplot(2,2,3);
imshow(uint8(A_base2));
title("Sigma_r = 16")
xlim([250,400])
ylim([100,250])

subplot(2,2,4);
imshow(uint8(A_base3));
title("Sigma_r = 12")
xlim([250,400])
ylim([100,250])

set(gcf,'color','w');

%%
figure; 
subplot(2,2,1);
imshow(uint8(flash_img));
title("Flash Image")
xlim([250,400])
ylim([100,250])
subplot(2,2,2);
imshow(rescale(rgb2gray(F_detail1)))
title("Sigma_r = 20")
xlim([250,400])
ylim([100,250])

subplot(2,2,3);
imshow(rescale(rgb2gray(F_detail2)))
title("Sigma_r = 16")
xlim([250,400])
ylim([100,250])

subplot(2,2,4);
imshow(rescale(rgb2gray(F_detail3)))
title("Sigma_r = 12")
xlim([250,400])
ylim([100,250])

set(gcf,'color','w');

%%
figure; 
subplot(2,2,1);
imshow(uint8(noflash_img));
title("No Flash Image")
xlim([250,400])
ylim([100,250])
subplot(2,2,2);
imshow(uint8(A_final1));
title("Sigma_r = 20")
xlim([250,400])
ylim([100,250])

subplot(2,2,3);
imshow(uint8(A_final2));
title("Sigma_r = 16")
xlim([250,400])
ylim([100,250])

subplot(2,2,4);
imshow(uint8(A_final3));
title("Sigma_r = 12")
xlim([250,400])
ylim([100,250])


set(gcf,'color','w');

%% Filter Width test

f_width = 0.1; 
sigma_d = 24; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final1,F_base1,F_detail1,A_base1,A_NR1] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

f_width = 0.3; 
sigma_d = 24; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final2,F_base2,F_detail2,A_base2,A_NR2] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

f_width = 0.5; 
sigma_d = 24; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final3,F_base3,F_detail3,A_base3,A_NR3] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

f_width = 0.9; 
sigma_d = 24; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final4,F_base4,F_detail4,A_base4,A_NR4] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

set(gcf,'color','w');


%%
figure; 
subplot(2,2,1);
imshow(uint8(noflash_img));
title("No Flash Image")
xlim([250,400])
ylim([100,250])

subplot(2,2,2);
imshow(uint8(A_base1));
title("0.1")
xlim([250,400])
ylim([100,250])

subplot(2,2,3);
imshow(uint8(A_base2));
title("0.3")
xlim([250,400])
ylim([100,250])

subplot(2,2,4);
imshow(uint8(A_base3));
title("0.5")
xlim([250,400])
ylim([100,250])


set(gcf,'color','w');

%%
figure; 
subplot(2,2,1);
imshow(uint8(flash_img));
title("Flash Image")
xlim([250,400])
ylim([100,250])

subplot(2,2,2);
imshow(rescale(rgb2gray(F_detail1)))
title("0.1")
xlim([250,400])
ylim([100,250])

subplot(2,2,3);
imshow(rescale(rgb2gray(F_detail2)))
title("0.3")
xlim([250,400])
ylim([100,250])

subplot(2,2,4);
imshow(rescale(rgb2gray(F_detail3)))
title("0.5")
xlim([250,400])
ylim([100,250])

set(gcf,'color','w');


%%
figure; 
subplot(2,2,1);
imshow(uint8(noflash_img));
title("No Flash Image")
xlim([250,400])
ylim([100,250])

subplot(2,2,2);
imshow(uint8(A_final1));
title("0.1")
xlim([250,400])
ylim([100,250])

subplot(2,2,3);
imshow(uint8(A_final2));
title("0.3")
xlim([250,400])
ylim([100,250])

subplot(2,2,4);
imshow(uint8(A_final3));
title("0.5")
xlim([250,400])
ylim([100,250])

set(gcf,'color','w');


%% Calculate Error Images from Above
err1 = double(rgb2gray(uint8(A_final1)))-double(rgb2gray((result_img)));
err2 = double(rgb2gray(uint8(A_final2)))-double(rgb2gray((result_img)));
err3 = double(rgb2gray(uint8(A_final3)))-double(rgb2gray((result_img)));
err4 = double(rgb2gray(uint8(A_final3)))-double(rgb2gray((result_img)));

p1_errb = sum(sum(abs(err1)))/size(err1,1)/size(err1,2)
p3_errb = sum(sum(abs(err2)))/size(err1,1)/size(err1,2)
p5_errb = sum(sum(abs(err3)))/size(err1,1)/size(err1,2)
p9_errb = sum(sum(abs(err4)))/size(err1,1)/size(err1,2)

berr = [err1, err2, err3];%, %err4, err5, err6];


figure;
subplot(2,2,1);
imshow(err1,[-128,128]);
title('Final 0.1')

subplot(2,2,2);
imshow(err2,[-128,128]);
title('Final 0.3')

subplot(2,2,3);
imshow(err3,[-128,128]);
title('Final 0.5')

subplot(2,2,4);
imshow(err4,[-128,128]);
title('Final 0.9')

set(gcf,'color','w');

%% Run other images

%Cave  Image
filepath = '/Users/abbyskerker/Documents/Spring24/EC520/Project/flash_data_JBF_Detail_transfer/flash_data_JBF_Detail_transfer/';
filename_flash = 'cave01_00_flash.tif';
filename_noflash = 'cave01_01_noflash.tif';


%Load in Files and Settings
fn = strcat(filepath,filename_flash);
flash_img = (imread(fn)); 
magnificationFactor = 0.25; 
flash_img = imresize(flash_img,magnificationFactor);
flash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
flash_exp_t = 1/40; 
fn = strcat(filepath,filename_noflash);
noflash_img = (imread(fn)); 
noflash_img = imresize(noflash_img,magnificationFactor);
noflash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
noflash_exp_t = 1/15; 

f_width = 0.1; 
sigma_d = 24; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final1,F_base1,F_detail1,A_base1,A_NR1] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

figure;
subplot(1,3,1);
imshow(uint8(flash_img));
title("Flash Image")

subplot(1,3,2);
imshow(uint8(noflash_img));
title("No Flash Image")

subplot(1,3,3);
imshow(uint8(A_final1));
title('Merged Image')

set(gcf,'color','w');

%% Run other images

%Lamp  Image
filepath = '/Users/abbyskerker/Documents/Spring24/EC520/Project/flash_data_JBF_Detail_transfer/flash_data_JBF_Detail_transfer/';
filename_flash = 'lamp_00_flash.tif';
filename_noflash = 'lamp_01_noflash.tif';


%Load in Files and Settings
fn = strcat(filepath,filename_flash);
flash_img = (imread(fn)); 
magnificationFactor = 0.25; 
flash_img = imresize(flash_img,magnificationFactor);
flash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
flash_exp_t = 1/40; 
fn = strcat(filepath,filename_noflash);
noflash_img = (imread(fn)); 
noflash_img = imresize(noflash_img,magnificationFactor);
noflash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
noflash_exp_t = 1/15; 

f_width = 0.1; 
sigma_d = 24; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final1,F_base1,F_detail1,A_base1,A_NR1] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

figure;
subplot(1,3,1);
imshow(uint8(flash_img));
title("Flash Image")

subplot(1,3,2);
imshow(uint8(noflash_img));
title("No Flash Image")

subplot(1,3,3);
imshow(uint8(A_final1));
title('Merged Image')

set(gcf,'color','w');

%% Run other images

%Pots  Image
filepath = '/Users/abbyskerker/Documents/Spring24/EC520/Project/flash_data_JBF_Detail_transfer/flash_data_JBF_Detail_transfer/';
filename_flash = 'potsdetail_00_flash.tif';
filename_noflash = 'potsdetail_01_noflash.tif';


%Load in Files and Settings
fn = strcat(filepath,filename_flash);
flash_img = (imread(fn)); 
magnificationFactor = 0.25; 
flash_img = imresize(flash_img,magnificationFactor);
flash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
flash_exp_t = 1/40; 
fn = strcat(filepath,filename_noflash);
noflash_img = (imread(fn)); 
noflash_img = imresize(noflash_img,magnificationFactor);
noflash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
noflash_exp_t = 1/15; 

f_width = 0.1; 
sigma_d = 24; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final1,F_base1,F_detail1,A_base1,A_NR1] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

figure;
subplot(1,3,1);
imshow(uint8(flash_img));
title("Flash Image")

subplot(1,3,2);
imshow(uint8(noflash_img));
title("No Flash Image")

subplot(1,3,3);
imshow(uint8(A_final1));
title('Merged Image')

set(gcf,'color','w');

%% Run other images

%Puppets  Image
filepath = '/Users/abbyskerker/Documents/Spring24/EC520/Project/flash_data_JBF_Detail_transfer/flash_data_JBF_Detail_transfer/';
filename_flash = 'puppets_00_flash.tif';
filename_noflash = 'puppets_01_noflash.tif';


%Load in Files and Settings
fn = strcat(filepath,filename_flash);
flash_img = (imread(fn)); 
magnificationFactor = 0.25; 
flash_img = imresize(flash_img,magnificationFactor);
flash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
flash_exp_t = 1/40; 
fn = strcat(filepath,filename_noflash);
noflash_img = (imread(fn)); 
noflash_img = imresize(noflash_img,magnificationFactor);
noflash_iso = imfinfo(fn).DigitalCamera.ISOSpeedRatings; 
noflash_exp_t = 1/15; 

f_width = 0.1; 
sigma_d = 24; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
linear_scaling_factor = flash_iso*flash_exp_t/(noflash_iso*noflash_exp_t);
[A_final1,F_base1,F_detail1,A_base1,A_NR1] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);

figure;
subplot(1,3,1);
imshow(uint8(flash_img));
title("Flash Image")

subplot(1,3,2);
imshow(uint8(noflash_img));
title("No Flash Image")

subplot(1,3,3);
imshow(uint8(A_final1));
title('Final Merged Image')

set(gcf,'color','w');
%% Functions

function [A_final,F_base,F_detail,A_base,A_NR] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint, linear_scaling_factor)
    %Bilateral Filter on Flash Image
    F1 = flash_img(:,:,1);
    F2 = flash_img(:,:,2);
    F3 = flash_img(:,:,3);
    F = flash_img;
    F_base = zeros(size(flash_img));
    F_base(:,:,1) = bilat_filt(double(F1),f_width,sigma_d,sigma_r_bilat);
    F_base(:,:,2) = bilat_filt(double(F2),f_width,sigma_d,sigma_r_bilat);
    F_base(:,:,3) = bilat_filt(double(F3),f_width,sigma_d,sigma_r_bilat);
   
    
    %Detail Transfer on Flash Image
    epsilon = 0.02; 
    F_rescale = rescale(F); 
    F_base_rescale = rescale(F_base);
    F_detail = (F_rescale+epsilon)./(F_base_rescale + epsilon); 
    
    % Detail Transfer
    threshold = 2;
    F_Lin = double(rgb2lin(flash_img)); %flash lumanance
    A_Lin = double(rgb2lin(noflash_img))*linear_scaling_factor; % no flash luminance
    M = zeros(size(F,1),size(F,2),size(F,3)); % initialize final mask
    M_shadow = M; %zeros(size(F,1), size(F,2)); % initialize shadow mask
    M_shadow((abs(F_Lin-A_Lin)<=threshold)) = 1; % get shadow mask
    M_specularities = M; %zeros(size(F,1), size(F,2)); % initialize specular
    M_specularities((F_Lin/max(F_Lin(:))) > 0.95 ) = 1; % get specular mask
    %for i = 1:3 % get final mask
        M(:,:,:) = M_shadow | M_specularities; % merge two masks
    %end
    M = imdilate(M, strel('disk',2)); % blur the mask

    %Bilateral Filter on No Flash Image
    A1 = noflash_img(:,:,1);
    A2 = noflash_img(:,:,2);
    A3 = noflash_img(:,:,3);
    A_base = zeros(size(noflash_img));
    A_base(:,:,1) = bilat_filt(double(A1),f_width,sigma_d,sigma_r_bilat);
    A_base(:,:,2) = bilat_filt(double(A2),f_width,sigma_d,sigma_r_bilat);
    A_base(:,:,3) = bilat_filt(double(A3),f_width,sigma_d,sigma_r_bilat);
    
    
    %Joint Bilateral Filter on No Flash Image
    A1 = noflash_img(:,:,1);
    A2 = noflash_img(:,:,2);
    A3 = noflash_img(:,:,3);
    A_NR = zeros(size(noflash_img));
    A_NR(:,:,1) = joint_bilat_filt(double(F1),double(A1),f_width,sigma_d,sigma_r_joint);
    A_NR(:,:,2) = joint_bilat_filt(double(F2),double(A2),f_width,sigma_d,sigma_r_joint);
    A_NR(:,:,3) = joint_bilat_filt(double(F3),double(A3),f_width,sigma_d,sigma_r_joint);
    
    %Final Image
    A_final = (1-M).*A_NR.*F_detail + M.*A_base;
    
    figure; 
    subplot(2,3,1)
    imshow(flash_img);
    title("F: Flash Image")
    subplot(2,3,4);
    imshow(noflash_img);
    title("A: No Flash Image")
    subplot(2,3,2);
    imshow(uint8(F_base));
    title("F Base: Flash Bilateral Image")
    subplot(2,3,5)
    imshow(rescale(rgb2gray(F_detail)))
    title("F Detail: Detail Transfer Image")
    subplot(2,3,3)
    imshow(uint8(A_NR));
    title("A NR: Joint Bilateral Filter Image")
    subplot(2,3,6)
    imshow(uint8(A_final));
    title("A Final: Final Image")
end 

function h = bilat_filt(f, mag_cutoff, sigma_d, sigma_r)
    
    %Note: variable notation used is from the Tomasi Paper

    % Compute Closeness Function (Gaussian)
    width = round(sqrt(-sigma_d^2*log(mag_cutoff)));
    [X,Y] = meshgrid(-width:width,-width:width);
    c = exp(-1/2*(X.^2+Y.^2)/sigma_d^2); %g_d in the Petshnigg Paper
    
    %Setup Variables
    dim = size(f);
    h = zeros(dim);
    im_height = dim(1); 
    im_width = dim(2);
    width = round(sqrt(-sigma_r^2*log(mag_cutoff)));

    %Loop through each location in image
    for i = 1:im_height
        for j = 1:im_width
            % Get local region
            iMin = max(i-width,1);
            iMax = min(i+width,im_height);
            jMin = max(j-width,1);
            jMax = min(j+width,im_width);
            I = f(iMin:iMax, jMin:jMax);
            
            % Compute similarity function (Gaussian)
            d1 = I(:,:)-f(i,j);
            s = exp(-1/2*(d1.^2)/sigma_r^2); %g_r in the Petshnigg Paper
            
            % Compute Normalization Constant
            k = s.*c((iMin:iMax)-i+width+1, (jMin:jMax)-j+width+1);
            k_inv = 1/sum(k(:));

            % Compute Output Image
            h(i,j) = sum(sum(k.*I(:,:)))*k_inv;           
        end
    end
    
end

function h = joint_bilat_filt(flash_img,noflash_img, mag_cutoff, sigma_d, sigma_r)
    
    %Note: variable notation used is from the Tomasi Paper

    % Compute Closeness Function (Gaussian)
    width = round(sqrt(-sigma_d^2*log(mag_cutoff)));
    [X,Y] = meshgrid(-width:width,-width:width);
    c = exp(-1/2*(X.^2+Y.^2)/sigma_d^2); %g_d in the Petshnigg Paper
    
    %Setup Variables
    dim = size(flash_img);
    h = zeros(dim);
    im_height = dim(1); 
    im_width = dim(2);
    width = round(sqrt(-sigma_r^2*log(mag_cutoff)));

    %Loop through each location in image
    for i = 1:im_height
        for j = 1:im_width
            % Get local region
            iMin = max(i-width,1);
            iMax = min(i+width,im_height);
            jMin = max(j-width,1);
            jMax = min(j+width,im_width);
            I_flash = flash_img(iMin:iMax, jMin:jMax);
            I_noflash = noflash_img(iMin:iMax, jMin:jMax);
            
            % Compute similarity function (Gaussian)
            d1 = I_flash(:,:)-flash_img(i,j);
            s_flash = exp(-1/2*(d1.^2)/sigma_r^2); %g_r in the Petshnigg Paper
            d1 = I_noflash(:,:)-flash_img(i,j);
            s_noflash = exp(-1/2*(d1.^2)/sigma_r^2); %g_r in the Petshnigg Paper

            % Compute Normalization Constant
            k = s_noflash.*c((iMin:iMax)-i+width+1, (jMin:jMax)-j+width+1);
            k_inv = 1/sum(k(:));
            k_flash = s_flash.*c((iMin:iMax)-i+width+1, (jMin:jMax)-j+width+1);
            k_inv = 1/sum(k_flash(:));
            % Compute Output Image
            h(i,j) = sum(sum(k_flash.*I_noflash(:,:)))*k_inv;           
        end
    end

end
