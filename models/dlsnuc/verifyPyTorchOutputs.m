% verify pytorch outputs
clear all, close all, clc;


addpath(genpath('C:\matconvnet-1.0-beta24\matlab'));
vl_setupnn();

addpath('original')
load('model1.mat');

load('test/sample.mat', 'irn','result');

for im = 1:size(irn, 2)
    im_noise = squeeze(irn(:, im, :, :, :)); 
    expected = -squeeze(result(:, im, 2, :, :)); 

    % directly taken from the reference paper
    im_denoised = des_ds_Matconvnet(im_noise, model, 4);
    noise = im_noise - im_denoised;
    fprintf("diff: %.5f\n", sum(abs(noise(:) - expected(:))));
    figure,imshow(noise-expected, []);
end