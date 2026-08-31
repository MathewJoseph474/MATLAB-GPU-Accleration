function packetCounts = build_packet_sweep(cfg)
%BUILD_PACKET_SWEEP
% Generates the packet counts used by the benchmark based
% on cfg.Packets.Mode.


mode = lower(string(cfg.Packets.Mode));


switch mode

    %% ========================================================
    % MANUAL MODE
    % =========================================================

    case "manual"

        packetCounts = cfg.Packets.Values(:).';


    %% ========================================================
    % POWERS OF TWO
    % =========================================================

    case "power2"

        minPackets = cfg.Packets.Min;
        maxPackets = cfg.Packets.Max;

        if minPackets <= 0 || maxPackets <= 0
            error("Packet counts must be positive.");
        end

        if minPackets > maxPackets
            error( ...
                "cfg.Packets.Min cannot exceed cfg.Packets.Max.");
        end

        minExponent = ceil(log2(minPackets));
        maxExponent = floor(log2(maxPackets));

        packetCounts = 2.^(minExponent:maxExponent);


    %% ========================================================
    % MULTIPLES OF BASE DATASET
    % =========================================================

    case "basemultiples"

        basePackets = cfg.Packets.Base;

        multipliers = cfg.Packets.Multipliers(:).';

        packetCounts = basePackets .* multipliers;


    %% ========================================================
    % UNKNOWN MODE
    % =========================================================

    otherwise

        error( ...
            "Unknown packet mode: %s", ...
            cfg.Packets.Mode);

end


%% ============================================================
% SANITY CHECKS
% =============================================================

packetCounts = unique(packetCounts, "stable");

if any(packetCounts <= 0)
    error("Packet counts must all be positive.");
end

if any(mod(packetCounts,1) ~= 0)
    error("Packet counts must all be integers.");
end

end