function sweepResults = main_eht_receiver_sweep_8k_16k_32k(userOpts)
%MAIN_EHT_RECEIVER_SWEEP_8K_16K_32K
% Runs 8K, 16K, and 32K CPU/GPU receiver benchmarks and creates:
%   1. CPU vs GPU packet throughput
%   2. CPU vs GPU payload throughput in Mbps
%   3. Average receiver latency per packet
%   4. Receiver-side speedup
%   5. GPU timing composition
%   6. CPU timing composition
%   7. Median component time versus workload
%
% CPU and GPU composition figures use the same y-axis scale.
% CPU is blue. GPU is green.

    arguments
        userOpts struct = struct()
    end


% LOAD PROJECT CONFIGURATION

srcDir = fileparts(mfilename("fullpath"));
projectRoot = fileparts(srcDir);

addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

cfg = project_config(projectRoot);


% BUILD MAIN OPTIONS FROM CONFIG

opts = struct;

% Dynamic packet sweep
opts.PacketCounts = build_packet_sweep(cfg);

% CPU worker configuration
opts.NumWorkers = cfg.CPU.NumWorkers;

% Automatically choose sequential or parallel CPU execution.
if opts.NumWorkers == 1
    opts.CPUExecutionMode = "sequential";
else
    opts.CPUExecutionMode = "parallel";
end

opts.PoolType = cfg.CPU.PoolType;
opts.MaximumSafeWorkers = cfg.CPU.MaximumSafeWorkers;
opts.GPUBatchSize = cfg.GPU.BatchSize;

% Portable project directory
opts.ProjectDir = projectRoot;

% Keep existing plotting behavior for now.
opts.SaveFigures = cfg.Plot.Enable;
opts.CloseExistingFigures = cfg.Plot.CloseExistingFigures;

    names = fieldnames(userOpts);
    for k = 1:numel(names)
        name = names{k};
        if ~isfield(opts,name)
            error('wifi7cuda:UnknownOption', ...
                'Unknown sweep option: %s',name);
        end
        opts.(name) = userOpts.(name);
    end

    if opts.CloseExistingFigures
        close all;
    end
    clc;

    packetCounts = double(opts.PacketCounts(:).');
    numRuns = numel(packetCounts);
    runResults = cell(numRuns,1);

    timestamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
    outputDir = fullfile(opts.ProjectDir,'results', ...
        sprintf('EHT_Sweep_%s_%s', ...
        lower(string(opts.CPUExecutionMode)),timestamp));
    plotDir = fullfile(outputDir,'Plots');

    if ~isfolder(outputDir)
        mkdir(outputDir);
    end
    if ~isfolder(plotDir)
        mkdir(plotDir);
    end

    fprintf('\n============================================================\n');
    fprintf('EHT RECEIVER WORKLOAD SWEEP\n');
    fprintf('============================================================\n');
    fprintf('CPU mode    : %s\n',opts.CPUExecutionMode);
    fprintf('CPU workers : %d\n',opts.NumWorkers);
    fprintf('GPU batch   : %d packets\n',opts.GPUBatchSize);
    fprintf('Output      : %s\n',outputDir);
    fprintf('============================================================\n');

    for runIndex = 1:numRuns
        benchmarkOpts = struct;
        benchmarkOpts.ProjectDir = opts.ProjectDir;
        benchmarkOpts.NumPackets = packetCounts(runIndex);
        benchmarkOpts.CPUExecutionMode = opts.CPUExecutionMode;
        benchmarkOpts.NumWorkers = opts.NumWorkers;
        if lower(string(opts.CPUExecutionMode)) == "sequential"
            benchmarkOpts.NumWorkers = 0;
        end
        benchmarkOpts.PoolType = opts.PoolType;
        benchmarkOpts.MaximumSafeWorkers = opts.MaximumSafeWorkers;
        benchmarkOpts.GPUBatchSize = opts.GPUBatchSize;
        benchmarkOpts.ValidationPackets = ...
    cfg.Execution.ValidationPackets;

benchmarkOpts.WarmupPackets = ...
    cfg.Execution.WarmupPackets;

benchmarkOpts.LLRScale = ...
    cfg.Execution.LLRScale;
        benchmarkOpts.SaveIndividualResult = true;
        benchmarkOpts.PrintHeader = true;

        runResults{runIndex} = ...
            benchmark_eht_integrated_cpu_gpu_batched(benchmarkOpts);
    end

    cpuThroughput = cellfun(@(r) r.CPUPacketsPerSecond,runResults);
    gpuThroughput = cellfun(@(r) r.GPUPacketsPerSecond,runResults);
    cpuPayloadMbps = cellfun(@(r) r.CPUPayloadMbps,runResults);
    gpuPayloadMbps = cellfun(@(r) r.GPUPayloadMbps,runResults);
    cpuLatency = cellfun(@(r) r.CPULatencyMsPerPacket,runResults);
    gpuLatency = cellfun(@(r) r.GPULatencyMsPerPacket,runResults);
    speedup = cellfun(@(r) r.Speedup,runResults);

        
    sweepTable = table( ...
        packetCounts(:), ...
        repmat(string(opts.CPUExecutionMode),numRuns,1), ...
        repmat(opts.NumWorkers,numRuns,1), ...
        cpuThroughput(:),gpuThroughput(:), ...
        cpuPayloadMbps(:),gpuPayloadMbps(:), ...
        cpuLatency(:),gpuLatency(:),speedup(:), ...
        'VariableNames',{ ...
        'Packets','CPUExecutionMode','NumWorkers', ...
        'CPU_Packets_per_s','GPU_Packets_per_s', ...
        'CPU_Payload_Mbps','GPU_Payload_Mbps', ...
        'CPU_Latency_ms_per_packet','GPU_Latency_ms_per_packet', ...
        'ReceiverSpeedup_x'});

    % Combine all 12 CPU/GPU stage timings from every workload into one CSV.
    % This does not alter any existing plots or summary metrics.
    stageTimingTables = cellfun(@(r) r.StageTimingTable,runResults, ...
        'UniformOutput',false);
    receiverStageTimingTable = vertcat(stageTimingTables{:});

    save(fullfile(outputDir,'EHT_Sweep_Results.mat'), ...
        'runResults','sweepTable','receiverStageTimingTable','opts','-v7.3');
    writetable(sweepTable, ...
        fullfile(outputDir,'EHT_Sweep_Summary.csv'));
    stageTimingCsvPath = fullfile(outputDir,'Receiver_Stage_Timings.csv');
    writetable(receiverStageTimingTable,stageTimingCsvPath);
    fprintf('Stage timing CSV saved to:\n%s\n',stageTimingCsvPath);

    cpuColor = [0 0.4470 0.7410];
    gpuColor = [0.20 0.65 0.25];
    xLabels = strings(1,numRuns);

    for runIndex = 1:numRuns
        if packetCounts(runIndex) >= 1024 && ...
                mod(packetCounts(runIndex),1024) == 0
            xLabels(runIndex) = sprintf('%dK', ...
                packetCounts(runIndex)/1024);
        else
            xLabels(runIndex) = sprintf('%d', ...
                packetCounts(runIndex));
        end
    end

    xPositions = 1:numRuns;

   %% 1. Throughput

if cfg.Plot.Enable && cfg.Plot.PacketThroughput

    fig = figure( ...
        'Name','CPU vs GPU Throughput', ...
        'Color','w');

    b = bar(xPositions, ...
        [cpuThroughput(:),gpuThroughput(:)], ...
        'grouped');

    xticks(xPositions);
    xticklabels(xLabels);

    b(1).FaceColor = cpuColor;
    b(2).FaceColor = gpuColor;

    ylabel('Throughput (packets/s)');
    xlabel('Receiver workload');
    title('CPU vs GPU Receiver Throughput');

    legend({'CPU','GPU'},'Location','best');

    grid on;

    addGroupedBarLabelsLocal(b,'%.1f');

    addPeakAnnotationLocal( ...
        xPositions, ...
        gpuThroughput, ...
        sprintf('Peak GPU: %.1f packets/s', ...
        max(gpuThroughput)));

saveFigureLocal( ...
    fig,plotDir,'01_CPU_vs_GPU_Throughput',opts,cfg);

end
    %% 2. Payload throughput in Mbps
    if cfg.Plot.Enable && cfg.Plot.PayloadMbps
        fig = figure('Name','CPU vs GPU Payload Throughput','Color','w');
    b = bar(xPositions, ...
        [cpuPayloadMbps(:),gpuPayloadMbps(:)],'grouped');
    xticks(xPositions);
    xticklabels(xLabels);
    b(1).FaceColor = cpuColor;
    b(2).FaceColor = gpuColor;
    ylabel('End-to-end payload throughput (Mbps)');
    xlabel('Receiver workload');
    title('CPU vs GPU End-to-End Payload Throughput');
    legend({'CPU','GPU'},'Location','best');
    grid on;
    addGroupedBarLabelsLocal(b,'%.2f');
    saveFigureLocal( ...
    fig,plotDir,'02_CPU_vs_GPU_Payload_Mbps',opts,cfg);

  
end
    

    %% 3. Average latency
    if cfg.Plot.Enable && cfg.Plot.Latency
        fig = figure('Name','Average Receiver Latency','Color','w');
    b = bar(xPositions, ...
        [cpuLatency(:),gpuLatency(:)],'grouped');
    xticks(xPositions);
    xticklabels(xLabels);
    b(1).FaceColor = cpuColor;
    b(2).FaceColor = gpuColor;
    ylabel('Average latency (ms/packet)');
    xlabel('Receiver workload');
    title('GPU/CPU Equivalent ms/packet');
    legend({'CPU','GPU'},'Location','best');
    grid on;
    addGroupedBarLabelsLocal(b,'%.3f');
    saveFigureLocal( ...
    fig,plotDir,'03_Average_Receiver_Latency',opts,cfg);
end
    

    %% 4. Speedup
    if cfg.Plot.Enable && cfg.Plot.Speedup
    fig = figure('Name','Receiver Speedup','Color','w');
    b = bar(xPositions,speedup(:));
    xticks(xPositions);
    xticklabels(xLabels);
    b.FaceColor = gpuColor;
    ylabel('Speedup (CPU time / GPU time)');
    xlabel('Receiver workload');
    title('Receiver-Side GPU Speedup');
    grid on;
    addSingleBarLabelsLocal(b,'%.2fx');
    saveFigureLocal( ...
    fig,plotDir,'04_Receiver_Speedup',opts,cfg);
end


   
%%
    sweepResults = struct;
    sweepResults.Options = opts;
    sweepResults.RunResults = runResults;
    sweepResults.SummaryTable = sweepTable;
    sweepResults.StageTimingTable = receiverStageTimingTable;
    sweepResults.StageTimingCSVPath = stageTimingCsvPath;
    sweepResults.OutputDirectory = outputDir;
    sweepResults.PlotDirectory = plotDir;
    
    fprintf('\n============================================================\n');
    fprintf('SWEEP COMPLETE\n');
    fprintf('============================================================\n');
    disp(sweepTable);
    fprintf('Plots and results saved to:\n%s\n',outputDir);
end



function addGroupedBarLabelsLocal(barHandles,numberFormat)
    for seriesIndex = 1:numel(barHandles)
        x = barHandles(seriesIndex).XEndPoints;
        y = barHandles(seriesIndex).YEndPoints;
        labels = compose(numberFormat,barHandles(seriesIndex).YData);

        text(x,y,labels, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', ...
            'FontWeight','bold', ...
            'FontSize',9);
    end

    currentLimits = ylim;
    ylim([currentLimits(1),currentLimits(2)*1.12]);
end


function addSingleBarLabelsLocal(barHandle,numberFormat)
    x = barHandle.XEndPoints;
    y = barHandle.YEndPoints;
    labels = compose(numberFormat,barHandle.YData);

    text(x,y,labels, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom', ...
        'FontWeight','bold', ...
        'FontSize',9);

    currentLimits = ylim;
    ylim([currentLimits(1),currentLimits(2)*1.12]);
end


function addPeakAnnotationLocal(xPositions,values,labelText)
    [peakValue,peakIndex] = max(values);

    text(xPositions(peakIndex),peakValue,labelText, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom', ...
        'FontWeight','bold', ...
        'FontSize',9);
end


function addCompositionTotalLabelsLocal(composition)
    totals = sum(composition,2);

    for barIndex = 1:numel(totals)
        text(barIndex,totals(barIndex), ...
            sprintf('%.3f ms',totals(barIndex)), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', ...
            'FontWeight','bold', ...
            'FontSize',9);
    end
end


function saveFigureLocal(fig,plotDir,stem,opts,cfg)

    if ~opts.SaveFigures
        return;
    end

    % Convert stem to char so filename construction is consistent.
    stem = char(stem);

    % Hide axes toolbar before exporting.
    axesHandles = findall(fig,'Type','axes');

    for k = 1:numel(axesHandles)
        try
            axtoolbar(axesHandles(k),'Visible','off');
        catch
        end
    end

    if cfg.Plot.SavePNG
        exportgraphics( ...
            fig, ...
            fullfile(plotDir,[stem '.png']), ...
            'Resolution',300);
    end

    if cfg.Plot.SavePDF
        exportgraphics( ...
            fig, ...
            fullfile(plotDir,[stem '.pdf']), ...
            'ContentType','vector');
    end

    if cfg.Plot.SaveFIG
        savefig( ...
            fig, ...
            fullfile(plotDir,[stem '.fig']));
    end

end
