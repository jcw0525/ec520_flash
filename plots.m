load lamp.mat; 

%Carpet Image
filepath = '/Users/abbyskerker/Documents/Spring24/EC520/Project/flash_data_JBF_Detail_transfer/flash_data_JBF_Detail_transfer/';
filename_flash = 'lamp_00_flash.tif';
filename_noflash = 'lamp_01_noflash.tif';
filename_bilat = 'lamp_02_bilateral.tif';
filename_result = 'lamp_03_our_result.tif';

fn = strcat(filepath,filename_flash);
flash_img = (imread(fn)); 
magnificationFactor = .2; 
flash_img = imresize(flash_img,magnificationFactor);
fn = strcat(filepath,filename_noflash);
noflash_img = (imread(fn)); 
noflash_img = imresize(noflash_img,magnificationFactor);

A_final = A_final_sigmar_bilat_12{1,1} ;
F_base = A_final_sigmar_bilat_12{1,2} ;
F_detail = A_final_sigmar_bilat_12{1,3} ;
A_base = A_final_sigmar_bilat_12{1,4};
A_NR = A_final_sigmar_bilat_12{1,5} ;

figure; 
subplot(1,3,1)
imshow(noflash_img);
% xlim([200, 350])
% ylim([100, 250]); 
title("A")
subplot(1,3,2)
imshow(uint8(A_base));
% xlim([200, 350])
% ylim([100, 250]); 
title("A Base")
subplot(1,3,3)
imshow(uint8(A_NR));
% xlim([200, 350])
% ylim([100, 250]); 
title("A NR")

% subplot(2,3,5)
% imshow(double(noflash_img) - (A_base));
% xlim([200, 350])
% ylim([100, 250]); 
% title("A Base")
% subplot(2,3,6)
% imshow(noflash_img - uint8(A_NR));
% xlim([200, 350])
% ylim([100, 250]); 
% title("A NR")


%%
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
imshow(rescale(F_detail));
title("F Detail: Detail Transfer Image")
subplot(2,3,3)
imshow(uint8(A_NR));
title("A NR: Joint Bilateral Filter Image")
subplot(2,3,6)
imshow(uint8(A_final));
title("A Final: Final Image")