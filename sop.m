%% SOP - Epileptic seizure detection on Bonn EEG (MATLAB port of sop.py)
% EMD / VMD decomposition -> 72 features -> SVM / RF / KNN, 10-fold CV
% Requires: Signal Processing Toolbox, Statistics and Machine Learning Toolbox
% (Parallel Computing Toolbox is optional; parfor just runs serially without it)
clear; clc; close all; rng(0);

%% 0. Settings  -------------------------------------------------------------
DATA_DIR   = 'DATASET';          % folder with the Bonn .txt files (searched recursively)
FS         = 173.61;
K          = 6;                  % number of modes
PY_FEAT_DIR = 'features';        % folder with features_emd.csv / features_vmd.csv exported from Python
PY_RESULTS  = 'classification_results.csv';   % Python results table (optional)

%% 1. Load Bonn EEG  --------------------------------------------------------
files = dir(fullfile(DATA_DIR, '**', '*.txt'));
[~, ord] = sort({files.name}); files = files(ord);
setMap = containers.Map({'Z','O','N','F','S'}, {'A','B','C','D','E'});

X = zeros(0, 4096); setsC = {};
for i = 1:numel(files)
    c = upper(files(i).name(1));
    if ~isKey(setMap, c), continue; end
    x = readmatrix(fullfile(files(i).folder, files(i).name));
    x = x(:)';
    X(end+1, :) = x(1:4096); %#ok<SAGROW>
    setsC{end+1, 1} = setMap(c); %#ok<SAGROW>
end
sets = categorical(setsC);
grp = containers.Map({'A','B','C','D','E'}, {'Normal','Normal','Interictal','Interictal','Seizure'});
y = categorical(cellfun(@(s) grp(s), setsC, 'UniformOutput', false));

fprintf('X size: %d x %d (expect 500 x 4096)\n', size(X));
disp(countcats(sets)');

T = size(X, 2);  t = (0:T-1) / FS;

%% 2. Raw EEG, one recording per set  --------------------------------------
letters = 'ABCDE';
figure('Name','Raw EEG');
for k = 1:5
    subplot(5,1,k);
    xi = X(find(sets == letters(k), 1), :);
    plot(t, xi, 'LineWidth', 0.6); ylabel(['Set ' letters(k)]);
end
xlabel('Time (s)');

%% 3. 40 Hz low-pass filter (4th-order Butterworth, zero-phase)  -----------
[b, a] = butter(4, 40/(FS/2), 'low');
Xf = filtfilt(b, a, X')';        % filtfilt works on columns, so transpose

figure('Name','Raw vs filtered');
for k = 1:5
    subplot(5,1,k);

    i = find(sets == letters(k), 1);
    plot(t(1:500), X(i,1:500)); hold on;
    plot(t(1:500), Xf(i,1:500)); hold off;
    ylabel(['Set ' letters(k)]);
end
legend('Raw','Filtered (40 Hz LP)'); xlabel('Time (s)');

%% 4. EMD and VMD decomposition (cached in modes.mat)  ---------------------
% Mode 1 = highest frequency for both methods (same convention as Python).
if isfile('modes.mat')
    load('modes.mat', 'modes_emd', 'modes_vmd');
    fprintf('Modes loaded from cache.\n');
else
    N = size(Xf, 1);
    modes_emd = zeros(N, K, T);
    modes_vmd = zeros(N, K, T);
    tic;
    parfor i = 1:N
        x = Xf(i, :)';
        % ---- EMD ----
        imf = emd(x, 'MaxNumIMF', K, 'Display', 0);          % T x nIMF
        modes_emd(i, :, :) = padModes(imf', K, T);
        % ---- VMD ----
        % alpha = 2000, tau = 0, uniform ("grid") initialisation as in vmdpy
        imfv = vmd(x, 'NumIMF', K, 'PenaltyFactor', 2000, 'LMUpdateRate', 0);
        [~, o] = sort(meanfreq(imfv, FS), 'descend');         % high -> low frequency
        modes_vmd(i, :, :) = padModes(imfv(:, o)', K, T);
    end
    fprintf('Decomposition done in %.0f s\n', toc);
    save('modes.mat', 'modes_emd', 'modes_vmd', '-v7.3');
end

%% 5. EMD vs VMD comparison (table + plot)  --------------------------------
fprintf('\n%-8s | %-12s | %-17s | %-10s\n', 'Set', 'Mean MSE', 'Mean Correlation', 'Energy ratio (VMD/EMD)');
for s = 1:5
    ix = find(sets == letters(s));
    E = modes_emd(ix, :, :);  V = modes_vmd(ix, :, :);
    mse = mean((E(:) - V(:)).^2);
    cs = [];
    for ii = 1:numel(ix)
        for k = 1:K
            c = corrcoef(squeeze(E(ii,k,:)), squeeze(V(ii,k,:)));
            if ~isnan(c(1,2)), cs(end+1) = c(1,2); end %#ok<SAGROW>
        end
    end
    fprintf('Set %-4s | %-12.5f | %-17.4f | %-10.4f\n', letters(s), mse, mean(cs), sum(V(:).^2)/(sum(E(:).^2)+1e-12));
end

figure('Name','EMD vs VMD modes','Position',[50 50 1500 800]);
for s = 1:5
    i = find(sets == letters(s), 1);
    for k = 1:K
        subplot(K, 5, (k-1)*5 + s);
        plot(t, squeeze(modes_emd(i,k,:)), 'LineWidth', 0.6); hold on;
        plot(t, squeeze(modes_vmd(i,k,:)), 'LineWidth', 0.6); hold off;
        if s == 1, ylabel(sprintf('Mode %d', k)); end
        if k == 1, title(['Set ' letters(s)]); end
    end
end
saveas(gcf, 'bonn_all5_sets_modes.png');

%% 6. Feature extraction: 12 features x 6 modes = 72 per recording  --------
featNames = {'mean','std','skew','kurt','energy','ptp','zcr','hj_mob','hj_comp','perm_en','spec_en','samp_en'};
colNames = cell(1, K * numel(featNames));
c = 0;
for k = 1:K
    for f = 1:numel(featNames)
        c = c + 1; colNames{c} = sprintf('m%d_%s', k, featNames{f});
    end
end

methods = {'emd', 'vmd'};
FEATS = struct();
if ~exist('features_matlab', 'dir'), mkdir('features_matlab'); end
for mi = 1:2
    M = modes_emd; if mi == 2, M = modes_vmd; end
    N = size(M, 1);
    F = zeros(N, K * numel(featNames));
    tic;
    parfor i = 1:N
        row = zeros(1, K * numel(featNames));
        for k = 1:K
            row((k-1)*12 + (1:12)) = modeFeatures(squeeze(M(i, k, :))', FS);
        end
        F(i, :) = row;
    end
    F(~isfinite(F)) = 0;
    FEATS.(methods{mi}) = F;
    Tout = array2table(F, 'VariableNames', colNames);
    Tout.set = cellstr(sets); Tout.label3 = cellstr(y);
    writetable(Tout, fullfile('features_matlab', ['features_' methods{mi} '.csv']));
    fprintf('%s features: %d x %d (%.0f s)\n', upper(methods{mi}), size(F,1), size(F,2), toc);
end

%% 7. Classification with MATLAB classifiers (10-fold stratified CV)  ------
tasks = {'5-class', sets; '3-class', y};
res = table();
for ti = 1:size(tasks, 1)
    for mi = 1:2
        r = evalAll(FEATS.(methods{mi}), tasks{ti,2}, tasks{ti,1}, upper(methods{mi}));
        res = [res; r]; %#ok<AGROW>
    end
end
disp(res);
writetable(res, 'classification_results_matlab.csv');

% Best combination per task: confusion matrix + per-class report
for ti = 1:size(tasks, 1)
    sub = res(strcmp(res.task, tasks{ti,1}), :);
    [~, bi] = max(sub.accuracy);
    best = sub(bi, :);
    F = FEATS.(lower(best.decomposition{1}));
    [~, pred] = evalOne(F, tasks{ti,2}, best.classifier{1});
    figure; confusionchart(tasks{ti,2}, pred);
    title(sprintf('%s + %s (%s) - %.1f%%', best.decomposition{1}, best.classifier{1}, best.task{1}, best.accuracy));
    saveas(gcf, sprintf('confusion_%s_matlab.png', tasks{ti,1}));
    fprintf('\nBest %s: %s + %s (%.1f%%)\n', best.task{1}, best.decomposition{1}, best.classifier{1}, best.accuracy);
    cm = confusionmat(tasks{ti,2}, pred);
    prec = diag(cm) ./ sum(cm, 1)';  rec = diag(cm) ./ sum(cm, 2);
    f1 = 2*prec.*rec ./ (prec + rec);
    disp(table(categories(tasks{ti,2}), prec, rec, f1, 'VariableNames', {'class','precision','recall','F1'}));
end

%% 8. VALIDATION A: run MATLAB classifiers on the PYTHON features  ---------
% If these accuracies match Python's table, the classification step is validated.
resPy = table();
if isfile(fullfile(PY_FEAT_DIR, 'features_emd.csv'))
    for mi = 1:2
        Tp = readtable(fullfile(PY_FEAT_DIR, ['features_' methods{mi} '.csv']));
        vn = Tp.Properties.VariableNames;
        Fp = table2array(Tp(:, startsWith(vn, 'm')));
        yp5 = categorical(Tp.set);  yp3 = categorical(Tp.label3);
        resPy = [resPy; evalAll(Fp, yp5, '5-class', upper(methods{mi}))]; %#ok<AGROW>
        resPy = [resPy; evalAll(Fp, yp3, '3-class', upper(methods{mi}))]; %#ok<AGROW>
    end
    resPy.Properties.VariableNames{'accuracy'} = 'acc_MATLAB_on_PyFeatures';
    resPy.macro_F1 = [];
    writetable(resPy, 'validation_matlab_on_python_features.csv');
end

%% 9. VALIDATION B: side-by-side comparison with the Python results  -------
if isfile(PY_RESULTS)
    P = readtable(PY_RESULTS);
    P = P(:, {'task','decomposition','classifier','accuracy'});
    P.Properties.VariableNames{'accuracy'} = 'acc_Python';
    cmp = innerjoin(P, res(:, {'task','decomposition','classifier','accuracy'}), ...
                    'Keys', {'task','decomposition','classifier'});
    cmp.Properties.VariableNames{'accuracy'} = 'acc_MATLAB';
    if ~isempty(resPy)
        cmp = innerjoin(cmp, resPy(:, {'task','decomposition','classifier','acc_MATLAB_on_PyFeatures'}), ...
                        'Keys', {'task','decomposition','classifier'});
    end
    cmp.diff_MATLAB_vs_Python = cmp.acc_MATLAB - cmp.acc_Python;
    disp(cmp);
    writetable(cmp, 'python_vs_matlab_comparison.csv');
end

%% 10. VALIDATION C: feature-by-feature agreement (EMD/VMD implementations)
if isfile(fullfile(PY_FEAT_DIR, 'features_vmd.csv'))
    for mi = 1:2
        Tp = readtable(fullfile(PY_FEAT_DIR, ['features_' methods{mi} '.csv']));
        Fp = table2array(Tp(:, startsWith(Tp.Properties.VariableNames, 'm')));
        Fm = FEATS.(methods{mi});
        r = zeros(1, size(Fm, 2));
        for j = 1:size(Fm, 2)
            cc = corrcoef(Fp(:, j), Fm(:, j));
            r(j) = cc(1, 2);
        end
        fprintf('%s: median per-feature correlation Python vs MATLAB = %.3f\n', upper(methods{mi}), median(r, 'omitnan'));
    end
end

%% ===================== LOCAL FUNCTIONS ===================================

function out = padModes(modes, K, T)
% modes: nModes x T  ->  K x T (truncate or zero-pad)
    modes = modes(1:min(K, size(modes, 1)), 1:T);
    if size(modes, 1) < K
        modes = [modes; zeros(K - size(modes, 1), T)];
    end
    out = modes;
end

function v = modeFeatures(m, fs)
% 12 features, same order as Python: mean,std,skew,kurt,energy,ptp,zcr,hj_mob,hj_comp,perm_en,spec_en,samp_en
    flat = std(m, 1) == 0;
    [mob, comp] = hjorth(m);
    v = zeros(1, 12);
    v(1)  = mean(m);
    v(2)  = std(m, 1);                         % population std (numpy default)
    if ~flat
        v(3) = skewness(m);                    % biased, matches scipy default
        v(4) = kurtosis(m) - 3;                % excess kurtosis, matches scipy default
        v(10) = permEntropy(m, 3);
        v(11) = spectralEntropy(m, fs);
    end
    v(5)  = sum(m.^2);
    v(6)  = max(m) - min(m);
    v(7)  = sum(diff(sign(m)) ~= 0);           % zero crossings
    v(8)  = mob;
    v(9)  = comp;
    v(12) = sampleEntropy(m, 2, 0.2);
end

function [mob, comp] = hjorth(x)
    dx = diff(x); ddx = diff(dx);
    vx = var(x, 1); vdx = var(dx, 1); vddx = var(ddx, 1);
    if vx == 0 || vdx == 0, mob = 0; comp = 0; return; end
    mob = sqrt(vdx / vx);
    comp = sqrt(vddx / vdx) / mob;
end

function h = permEntropy(x, order)
    n = numel(x) - order + 1;
    W = zeros(n, order);
    for j = 1:order, W(:, j) = x(j:j+n-1); end
    [~, idx] = sort(W, 2);
    code = idx * (order .^ (0:order-1))';
    [~, ~, j] = unique(code);
    p = accumarray(j, 1); p = p / sum(p);
    h = -sum(p .* log2(p)) / log2(factorial(order));
end

function h = spectralEntropy(x, fs)
    nseg = min(512, numel(x));
    [p, f] = pwelch(x, hann(nseg, 'periodic'), floor(nseg/2), [], fs);
    p = p(p > 0) / sum(p(p > 0));
    h = -sum(p .* log2(p)) / log2(numel(f));
end

function h = sampleEntropy(x, m, rf)
    s = std(x, 1);
    if s == 0, h = 0; return; end
    x = (x(:) - mean(x)) / s;
    n = numel(x) - m;
    B = countMatches(x, m, n, rf);
    A = countMatches(x, m + 1, n, rf);
    if A > 0 && B > 0, h = -log(A / B); else, h = 0; end
end

function cnt = countMatches(x, mm, n, r)
    emb = zeros(n, mm);
    for j = 1:mm, emb(:, j) = x(j:j+n-1); end
    idx = rangesearch(emb, emb, r, 'Distance', 'chebychev');
    cnt = sum(cellfun(@numel, idx)) - n;       % remove self-matches
end

function res = evalAll(F, target, taskName, decName)
% Run SVM / RF / KNN with 10-fold stratified CV, return a results table
    clfs = {'SVM', 'RF', 'KNN'};
    rows = cell(numel(clfs), 5);
    for c = 1:numel(clfs)
        [acc, pred] = evalOne(F, target, clfs{c});
        cm = confusionmat(target, pred);
        prec = diag(cm) ./ max(sum(cm, 1)', eps);
        rec  = diag(cm) ./ max(sum(cm, 2), eps);
        f1 = 2*prec.*rec ./ max(prec + rec, eps);
        rows(c, :) = {taskName, decName, clfs{c}, round(acc, 2), round(mean(f1) * 100, 2)};
    end
    res = cell2table(rows, 'VariableNames', {'task','decomposition','classifier','accuracy','macro_F1'});
end

function [acc, pred] = evalOne(F, target, clfName)
    rng(0);
    cv = cvpartition(target, 'KFold', 10);     % stratified
    nf = size(F, 2);
    switch clfName
        case 'SVM'   % RBF, C = 10, gamma = 'scale' ~ 1/nfeat on standardized data
            tpl = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 10, ...
                              'KernelScale', sqrt(nf), 'Standardize', true);
            mdl = fitcecoc(F, target, 'Learners', tpl, 'Coding', 'onevsone', 'CVPartition', cv);
        case 'RF'    % 300 trees
            tpl = templateTree('NumVariablesToSample', round(sqrt(nf)));
            mdl = fitcensemble(F, target, 'Method', 'Bag', 'NumLearningCycles', 300, ...
                               'Learners', tpl, 'CVPartition', cv);
        case 'KNN'   % k = 5
            mdl = fitcknn(F, target, 'NumNeighbors', 5, 'Standardize', true, 'CVPartition', cv);
    end
    pred = kfoldPredict(mdl);
    acc = mean(pred == target) * 100;
end

