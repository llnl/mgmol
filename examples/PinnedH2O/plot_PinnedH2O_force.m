clc; clear all; close all;

%%
plot_fom = 0;
plot_rom = 0;
rdim = 77;

%%
load F_fom.mat
fprintf(1, 'Force statistics using FOM orbitals\n');
fprintf(1, 'Mean of force on H1: %6.4e, %6.4e, %6.4e\n', mean(F1_fom));
fprintf(1, 'Variance of force on H1: %6.4e, %6.4e, %6.4e\n', var(F1_fom));
fprintf(1, 'Mean of force on H2: %6.4e, %6.4e, %6.4e\n', mean(F2_fom));
fprintf(1, 'Variance of force on H2: %6.4e, %6.4e, %6.4e\n', var(F2_fom));

if plot_fom
    plotForce(F1_fom, 'F_H1_fom');
    plotForce(F2_fom, 'F_H2_fom');
    plotForceHistograms(F1_fom, 'H1_fom');
    plotForceHistograms(F2_fom, 'H2_fom');
end

%%
load(['F_rom' int2str(rdim) '.mat'])
fprintf(1, 'Force statistics using projected orbitals\n');
fprintf(1, 'Mean of force on H1: %6.4e, %6.4e, %6.4e\n', mean(F1_rom));
fprintf(1, 'Variance of force on H1: %6.4e, %6.4e, %6.4e\n', var(F1_rom));
%H1_correlation = sum(F1_fom(:) .* F1_rom(:)) / (norm(F1_fom(:)) * norm(F1_rom(:)))
fprintf(1, 'Mean of force on H2: %6.4e, %6.4e, %6.4e\n', mean(F2_rom));
fprintf(1, 'Variance of force on H2: %6.4e, %6.4e, %6.4e\n', var(F2_rom));
%H2_correlation = sum(F2_fom(:) .* F2_rom(:)) / (norm(F2_fom(:)) * norm(F2_rom(:)))

if plot_rom
    plotForce(F1_rom, ['F_H1_rom' int2str(rdim)]);
    plotForce(F2_rom, ['F_H2_rom' int2str(rdim)]);
    plotForceHistograms(F1_rom, ['H1_rom' int2str(rdim)]);
    plotForceHistograms(F2_rom, ['H2_rom' int2str(rdim)]);
    plotForceHistogram(abs(F1_fom - F1_rom), ['H1_rom' int2str(rdim)], 'Fdiff');
    plotForceHistogram(abs(F2_fom - F2_rom), ['H2_rom' int2str(rdim)], 'Fdiff');
end

%% 
function plotForce(F, suffix)
    figure;
    imagesc(F');
    axis tight;
    axis equal;
    colorbar;
    saveas(gcf, suffix, 'jpg');
end

function plotForceHistogram(F, suffix, var)
    figure;
    if strcmp(var,'Fx')
        X = F(:,1);
        var_name = 'x-directional Force';
    elseif strcmp(var,'Fy')
        X = F(:,2);
        var_name = 'y-directional Force';
    elseif strcmp(var,'Fz')
        X = F(:,3);
        var_name = 'z-directional Force';
    elseif strcmp(var,'Fmag')
        X = sqrt(sum(F.^2, 2));
        var_name = 'Force Magitude';
    elseif strcmp(var,'Fdiff')
        X = sqrt(sum(F.^2, 2));
        var_name = 'Magitude of Difference in Force';
    else
        error('Invalid type');
    end
    histogram(X, 20);
    xlabel(var_name);
    ylabel('Frequency');
    title(['Histogram of ' var_name]);
    saveas(gcf, [var '_' suffix], 'jpg');
end

function plotForceHistograms(F, suffix)
    plotForceHistogram(F, suffix, 'Fx');
    plotForceHistogram(F, suffix, 'Fy');
    plotForceHistogram(F, suffix, 'Fz');
    plotForceHistogram(F, suffix, 'Fmag');
end
