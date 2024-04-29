% filepath = '/Users/abbyskerker/Documents/GitHub/ec520_flash/';
% filename_flash = 'flash1.jpg';
% filename_noflash = 'noflash1.jpg';
% 
% fn = strcat(filepath,filename_flash);
% flash_img = (imread(fn)); 
% magnificationFactor = 0.2; 
% flash_img = imresize(flash_img,magnificationFactor);
% flash_iso = 500;
% flash_exposuret = 1/40; 
% 
% fn = strcat(filepath,filename_noflash);
% noflash_img = (imread(fn)); 
% noflash_img = imresize(noflash_img,magnificationFactor);
% noflash_iso = 2000;
% noflash_exposuret = 1/15; 

filepath = '/Users/abbyskerker/Documents/GitHub/ec520_flash/';
filename_flash = 'flash2.jpg';
filename_noflash = 'noflash2.jpg';

fn = strcat(filepath,filename_flash);
flash_img = (imread(fn)); 
magnificationFactor = 0.2; 
flash_img = imresize(flash_img,magnificationFactor);
flash_iso = 32;
flash_exposuret = 1/32; 

fn = strcat(filepath,filename_noflash);
noflash_img = (imread(fn)); 
noflash_img = imresize(noflash_img,magnificationFactor);
noflash_iso = 2000;
noflash_exposuret = 1/15; 
linear_scaling_factor = flash_iso*flash_exposuret/(noflash_iso*noflash_exposuret);



%Test Different f_widths: 
f_width = 0.5; 
sigma_d = 48; 
sigma_r_bilat = 20; 
sigma_r_joint = 0.001*256; 
[A_final,F_base,F_detail,A_base,A_NR] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);
A_final_p5 = {A_final,F_base,F_detail,A_base,A_NR};
sgtitle("Mag Cutoff = 0.5, Sigma_d = 48, Sigma_r = 20")
f_width = 0.9; 
[A_final,F_base,F_detail,A_base,A_NR] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);
A_final_p9 = {A_final,F_base,F_detail,A_base,A_NR};
sgtitle("Mag Cutoff = 0.9, Sigma_d = 48, Sigma_r = 20")
f_width = 0.3; 
[A_final,F_base,F_detail,A_base,A_NR] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);
A_final_p3 = {A_final,F_base,F_detail,A_base,A_NR};
sgtitle("Mag Cutoff = 0.3, Sigma_d = 48, Sigma_r = 20")
f_width = 0.1; 
[A_final,F_base,F_detail,A_base,A_NR] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);
A_final_p1 = {A_final,F_base,F_detail,A_base,A_NR};
sgtitle("Mag Cutoff = 0.1, Sigma_d = 48, Sigma_r = 20")

%Test Different Sigma_d
f_width = 0.9; 
sigma_d = 24; 
sigma_r_bilat = 20; 
sigma_r_joint = 0.001*256; 
M = 0; 
[A_final,F_base,F_detail,A_base,A_NR] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);
A_final_sigmad_24 = {A_final,F_base,F_detail,A_base,A_NR};
sgtitle("Mag Cutoff = 0.9, Sigma_d = 24, Sigma_r = 20")

%Test Different Sigma_r_bilat
f_width = 0.9; 
sigma_d = 48; 
sigma_r_bilat = 12; 
sigma_r_joint = 0.001*256; 
M = 0; 
[A_final,F_base,F_detail,A_base,A_NR] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint,linear_scaling_factor);
A_final_sigmar_bilat_12 = {A_final,F_base,F_detail,A_base,A_NR};
sgtitle("Mag Cutoff = 0.9, Sigma_d = 48, Sigma_r = 12")

%% Functions

function [A_final,F_base,F_detail,A_base,A_NR] = run_algorithm(noflash_img, flash_img, f_width, sigma_d, sigma_r_bilat, sigma_r_joint, linear_scaling_factor)
    %Bilateral Filter on Flash Image
    F1 = flash_img(:,:,1);
    F2 = flash_img(:,:,2);
    F3 = flash_img(:,:,3);
    F = flash_img;
    F_base = zeros(size(flash_img));
    F_base(:,:,1) = bilat_filt_1d(double(F1),f_width,sigma_d,sigma_r_bilat);
    F_base(:,:,2) = bilat_filt_1d(double(F2),f_width,sigma_d,sigma_r_bilat);
    F_base(:,:,3) = bilat_filt_1d(double(F3),f_width,sigma_d,sigma_r_bilat);
   
    
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
    A_base(:,:,1) = bilat_filt_1d(double(A1),f_width,sigma_d,sigma_r_bilat);
    A_base(:,:,2) = bilat_filt_1d(double(A2),f_width,sigma_d,sigma_r_bilat);
    A_base(:,:,3) = bilat_filt_1d(double(A3),f_width,sigma_d,sigma_r_bilat);
    
    
    %Joint Bilateral Filter on No Flash Image
    A1 = noflash_img(:,:,1);
    A2 = noflash_img(:,:,2);
    A3 = noflash_img(:,:,3);
    A_NR = zeros(size(noflash_img));
    A_NR(:,:,1) = joint_bilat_filt_1d(double(F1),double(A1),f_width,sigma_d,sigma_r_joint);
    A_NR(:,:,2) = joint_bilat_filt_1d(double(F2),double(A2),f_width,sigma_d,sigma_r_joint);
    A_NR(:,:,3) = joint_bilat_filt_1d(double(F3),double(A3),f_width,sigma_d,sigma_r_joint);
    
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
    imshow(rescale(F_detail),[0, max(max(max(F_detail)))+1]);
    title("F Detail: Detail Transfer Image")
    subplot(2,3,3)
    imshow(uint8(A_NR));
    title("A NR: Joint Bilateral Filter Image")
    subplot(2,3,6)
    imshow(uint8(A_final));
    title("A Final: Final Image")
end 

function h = bilat_filt_1d(f, mag_cutoff, sigma_d, sigma_r)
    
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

function h = joint_bilat_filt_1d(flash_img,noflash_img, mag_cutoff, sigma_d, sigma_r)
    
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