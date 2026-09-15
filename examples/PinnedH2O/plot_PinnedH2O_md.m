clc; clear all; close all;

%%
plot_fom = 0;
rdims = [77, 39];

%% 
load md_fom.mat
if plot_fom
    plotAngleHistogram(H1_fom, H2_fom, 'fom');
end

%%

all_H1_rom = zeros(length(rdims), size(H1_fom, 1), 3);
all_H2_rom = zeros(length(rdims), size(H2_fom, 1), 3);
k = 0;

for rdim = rdims
    k = k + 1;
    load(['md_rom' int2str(rdim) '.mat'])
    plotAngleHistogram(H1_rom, H2_rom, ['rom' int2str(rdim)]);
    all_H1_rom(k, :, :) = H1_rom;
    all_H2_rom(k, :, :) = H2_rom;
end

plotAtomTrajectory(H1_fom(:,1), all_H1_rom(:,:,1), rdims, 'x', 1)
plotAtomTrajectory(H1_fom(:,2), all_H1_rom(:,:,2), rdims, 'y', 1)
plotAtomTrajectory(H1_fom(:,3), all_H1_rom(:,:,3), rdims, 'z', 1)

plotAtomTrajectory(H2_fom(:,1), all_H2_rom(:,:,1), rdims, 'x', 2)
plotAtomTrajectory(H2_fom(:,2), all_H2_rom(:,:,2), rdims, 'y', 2)
plotAtomTrajectory(H2_fom(:,3), all_H2_rom(:,:,3), rdims, 'z', 2)

%% 
function plotAtomTrajectory(X_fom, all_X_rom, rdims, var, idx)
    figure;
    hold on;
    plot(X_fom, 'Linewidth', 2, 'DisplayName', 'FOM');
    k = 0;
    for rdim = rdims
        k = k + 1;
        X_rom = all_X_rom(k, :);
        plot(X_rom, 'Linewidth', 2, 'DisplayName', ['ROM dim = ' int2str(rdim)]);
    end
    title([var '-coordinate of H' int2str(idx)])
    legend;
    saveas(gcf, [var '_H' int2str(idx)], 'jpg');    
end

function plotAngleHistogram(X1, X2, suffix)
    figure;
    A = acosd(sum(X1.*X2,2) ./ sqrt(sum(X1.^2,2)) ./ sqrt(sum(X2.^2,2)));
    histogram(A, 20);
    xlabel('Angle');
    ylabel('Frequency');
    title('Histogram of angle');
    saveas(gcf, ['angle_' suffix], 'jpg');
end