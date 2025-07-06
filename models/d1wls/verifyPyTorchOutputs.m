% verify pytorch outputs
clear all, close all, clc;

addpath('original')

load('test/sample.mat', 'irn','result');

for im = 1:size(irn, 2)
    im_noisy = double(squeeze(irn(:, im, :, :, :))); 
    expected = -double(squeeze(result(:, im, 2, :, :))); 

    % directly taken from the reference paper
    im_denoised = d1_WLS_Destriping(im_noisy, 40, 3);
    noise = im_noisy - im_denoised;
    fprintf("diff: %.5f\n", sum(abs(noise(:) - expected(:))));
    figure,imshow(noise-expected, []);
end