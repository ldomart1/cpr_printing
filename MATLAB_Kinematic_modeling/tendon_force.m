function tendon_notch_slider_demo()
%% Tendon-pull notch demo (FORCE + TUBE STIFFNESS + ELASTIC LIMIT) with live geometry sliders + animation + GIF export
%
% Key additions vs your original:
% 1) Force-driven bending with tube stiffness:
%       theta_i = (F * r_eff(i)) / k_theta(i), capped by geometric theta0(i)
%    where k_theta(i) = E * I_y(i) / L_i and I_y(i) is computed numerically for a notched ring section.
%
% 2) Elastic-limit detection (linear elastic proxy) using max bending strain / stress:
%       kappa_i = theta_i / L_i
%       eps_max(i)   = |kappa_i| * c_i
%       sigma_max(i) = E * eps_max(i)
%    If eps_max > epsElasticLimit OR sigma_max > sigmaElasticLimit => "out of elastic"
%    Those notches are colored RED and their pin markers become RED.
%
% Tube dimensions:
%   ID = 0.0414"  (1.05156 mm)
%   OD = 0.0570"  (1.44780 mm)

clc; close all;

%% ------------------- DEFAULT PARAMS + RANGES -------------------
P = struct();
P.height          = 30.0;  R.height          = [0 70];

% IMPORTANT: "depth" here is interpreted as notch cut depth across diameter along +X,
% removing region x > (ro - depth). Range 0..~OD(mm) makes sense.
P.depth           = 1.30;  R.depth           = [0 1.45];

P.offset          = 0.5;   R.offset          = [-2 2];

P.startNotchWidth = 1.5;   R.startNotchWidth = [0 4];
P.endNotchWidth   = 1.5;   R.endNotchWidth   = [0 4];

P.startSpacing    = 0.0;   R.startSpacing    = [-2 2];
P.endSpacing      = 0.0;   R.endSpacing      = [-2 2];

P.startOffsetApex = -1.0;  R.startOffsetApex = [-3 3];
P.endOffsetApex   = -2.0;  R.endOffsetApex   = [-3 3];

% ------------------- PHYSICS -------------------
P.tubeID_in   = 0.0414;
P.tubeOD_in   = 0.0570;
P.E_GPa       = 60;        % Nitinol effective E (GPa)
P.tendon_x_mm = 0.92456;       % NaN -> assume tendon rides at +ri (inner wall at +x)

% Elastic limit proxy (linear elastic region):
% Use either strain limit, stress limit, or both. If either is exceeded => "out of elastic".
P.epsElasticLimit       = 0.010;  % 1% (change to match your definition)
P.sigmaElasticLimit_MPa = 800;    % MPa (change to match your definition)

% Force slider cap (user preference); actual max is also limited by theta0 caps.
P.Fmax_user_N = 20;

distribution = "force_based"; % metadata only
bend_dir = +1;

%% ------------------- ANIMATION / GIF SETTINGS -------------------
gifFilename           = "full_curl_force_elastic.gif";
gifFrames             = 140;
gifDelay              = 0.03;
gifLoopCount          = inf;
gifTransparentBg      = true;

animFPS               = 40;
animBaseFramesPerCurl = 140;

%% ------------------- COLORS -------------------
colLinkUndeformed     = [0.80 0.80 0.80];
colTendonPins         = [1.00 1.00 1.00];
colOverElastic        = [1.00 0.00 0.00]; % RED highlight
undeformedDarkenFactor = 0.80;

%% ------------------- FIGURE / AXES (transparent) -------------------
fig = figure('Name','Tendon Pull (Force + Elastic limit): Notches + Pins', ...
    'Position',[100 140 1200 820], ...
    'Color','none', ...
    'InvertHardcopy','off');

ax = axes('Parent',fig, 'Position',[0.07 0.34 0.90 0.62]);
grid(ax,'on'); axis(ax,'equal'); hold(ax,'on');
xlabel(ax,'X (mm)'); ylabel(ax,'Y (mm)'); zlabel(ax,'Z (mm)');
ht = title(ax,'Undeformed vs Force-Driven Deformation (red = out of elastic)');
view(ax, 35, 20);
set(ax,'Color','none');

try
    set(ht,'Units','normalized');
    p = get(ht,'Position'); p(2) = p(2) + 0.04; set(ht,'Position',p);
catch
end

%% ------------------- FORCE SLIDER + LABEL -------------------
pullSld = uicontrol(fig,'Style','slider', ...
    'Units','pixels', ...
    'Position',[110 215 980 18], ...
    'Min',0,'Max',P.Fmax_user_N,'Value',0);

pullLbl = uicontrol(fig,'Style','text', ...
    'Units','pixels', ...
    'Position',[110 235 980 18], ...
    'BackgroundColor','none', ...
    'HorizontalAlignment','left', ...
    'String','Tendon force F = 0.000 N | predicted ΔL ≈ 0.0000 mm | over-elastic: 0');

%% ------------------- BUTTONS -------------------
btnY = 150;

btnPlay = uicontrol(fig,'Style','pushbutton','String','Play', ...
    'Units','pixels','Position',[110 btnY 90 30]);

btnStop = uicontrol(fig,'Style','pushbutton','String','Stop', ...
    'Units','pixels','Position',[210 btnY 90 30], 'Enable','off');

btnGif = uicontrol(fig,'Style','pushbutton','String','Save GIF...', ...
    'Units','pixels','Position',[330 btnY 110 30]);

btnClose = uicontrol(fig,'Style','pushbutton','String','Close', ...
    'Units','pixels','Position',[460 btnY 90 30]);

uicontrol(fig,'Style','text','String','Speed:', ...
    'Units','pixels','Position',[580 btnY+6 50 18], ...
    'BackgroundColor','none','HorizontalAlignment','right');

popSpeed = uicontrol(fig,'Style','popupmenu', ...
    'Units','pixels','Position',[640 btnY 90 28], ...
    'String',{'0.25x','0.5x','1x','2x','4x'}, ...
    'Value',3);

chkLockView = uicontrol(fig,'Style','checkbox', ...
    'String','Lock side view (X-Z)', ...
    'Units','pixels','Position',[740 btnY+5 170 22], ...
    'BackgroundColor','none', ...
    'Value',0);

tipTxt = uicontrol(fig,'Style','text', ...
    'Units','pixels', ...
    'Position',[920 btnY-2 270 44], ...
    'BackgroundColor','none', ...
    'HorizontalAlignment','left', ...
    'String','Tip wrt base: (x,y,z) = (—, —, —) mm | Notches: — | Attack = —°');

%% ------------------- PARAMETER SLIDERS (small) -------------------
panelY = 10;
panelH = 125;
uicontrol(fig,'Style','text','String','Geometry sliders (live):', ...
    'Units','pixels','Position',[110 panelY+panelH-18 250 16], ...
    'BackgroundColor','none','HorizontalAlignment','left','FontWeight','bold');

col1x = 110; col2x = 660;
rowYs = panelY + [90 70 50 30 10];

UI = struct();
[UI.h_height, UI.t_height, UI.v_height] = makeParamSlider('height', P.height, R.height, col1x, rowYs(1));
[UI.h_depth,  UI.t_depth,  UI.v_depth ] = makeParamSlider('depth (cut across dia)',  P.depth,  R.depth,  col1x, rowYs(2));
[UI.h_offset, UI.t_offset, UI.v_offset] = makeParamSlider('offset', P.offset, R.offset, col1x, rowYs(3));
[UI.h_sNW,    UI.t_sNW,    UI.v_sNW   ] = makeParamSlider('startNotchWidth', P.startNotchWidth, R.startNotchWidth, col1x, rowYs(4));
[UI.h_eNW,    UI.t_eNW,    UI.v_eNW   ] = makeParamSlider('endNotchWidth',   P.endNotchWidth,   R.endNotchWidth,   col1x, rowYs(5));

[UI.h_sSp,    UI.t_sSp,    UI.v_sSp   ] = makeParamSlider('startSpacing',    P.startSpacing,    R.startSpacing,    col2x, rowYs(1));
[UI.h_eSp,    UI.t_eSp,    UI.v_eSp   ] = makeParamSlider('endSpacing',      P.endSpacing,      R.endSpacing,      col2x, rowYs(2));
[UI.h_sOA,    UI.t_sOA,    UI.v_sOA   ] = makeParamSlider('startOffsetApex', P.startOffsetApex, R.startOffsetApex, col2x, rowYs(3));
[UI.h_eOA,    UI.t_eOA,    UI.v_eOA   ] = makeParamSlider('endOffsetApex',   P.endOffsetApex,   R.endOffsetApex,   col2x, rowYs(4));

%% ------------------- STATE -------------------
S = struct();
S.fig = fig; S.ax = ax;
S.pullSld = pullSld; S.pullLbl = pullLbl;
S.tipTxt = tipTxt;

S.btnPlay = btnPlay; S.btnStop = btnStop;
S.popSpeed = popSpeed;

S.chkLockView = chkLockView;
S.lockView = false;
S.viewFreeAzEl = get(ax,'View');
S.viewSideDir  = [0 -1 0];
S.viewSideUp   = [0 0 1];

S.P = P; S.R = R;
S.distribution = distribution;
S.bend_dir = bend_dir;

S.animTimer = [];
S.animFPS = animFPS;
S.animBaseFramesPerCurl = animBaseFramesPerCurl;
S.speedFactor = 1.0;

S.colLinkUndeformed     = colLinkUndeformed;
S.colTendonPins         = colTendonPins;
S.colOverElastic        = colOverElastic;
S.undeformedDarkenFactor = undeformedDarkenFactor;

S.hOrig = gobjects(0);
S.hDef  = gobjects(0);
S.hInterLinkUndeformed = gobjects(0);
S.hPins = gobjects(1);
S.hTendon = gobjects(1);
S.hLegend = gobjects(1);
S.hLegendOverElastic = gobjects(1);
S.notchColors = zeros(0,3);

S.notches0 = {};
S.pins0 = [];
S.p0_list = [];
S.p2_list = [];
S.L_notch = [];
S.theta0 = [];

% physics caches
S.k_theta = [];
S.r_eff   = [];
S.c_fiber = [];
S.E_MPa   = 45000;
S.F_max_model = P.Fmax_user_N;

set(fig,'CloseRequestFcn',@(~,~) closeAndCleanup());

%% ------------------- CALLBACKS -------------------
set(pullSld,'Callback',@(~,~) onPull());
addlistener(pullSld,'Value','PostSet',@(~,~) onPull());

set(btnPlay,'Callback',@(~,~) startAnimation());
set(btnStop,'Callback',@(~,~) stopAnimation());
set(btnGif,'Callback',@(~,~) saveGifFromUI());
set(btnClose,'Callback',@(~,~) closeAndCleanup());
set(popSpeed,'Callback',@(~,~) onSpeedChanged());
set(chkLockView,'Callback',@(~,~) onLockViewToggled());

paramNames = {'height','depth','offset','startNotchWidth','endNotchWidth', ...
              'startSpacing','endSpacing','startOffsetApex','endOffsetApex'};
for k = 1:numel(paramNames)
    name = paramNames{k};
    h = UI.(['h_' shortName(name)]);
    set(h,'Callback',@(src,~) onParamChanged(name, get(src,'Value')));
    addlistener(h,'Value','PostSet',@(~,~) onParamChanged(name, get(h,'Value')));
end

%% ------------------- INITIAL BUILD -------------------
rebuildGeometryAndPlots(true);

%% ============================ NESTED FUNCTIONS ============================
    function s = shortName(full)
        switch full
            case 'height', s='height';
            case 'depth', s='depth';
            case 'offset', s='offset';
            case 'startNotchWidth', s='sNW';
            case 'endNotchWidth', s='eNW';
            case 'startSpacing', s='sSp';
            case 'endSpacing', s='eSp';
            case 'startOffsetApex', s='sOA';
            case 'endOffsetApex', s='eOA';
            otherwise, s=full;
        end
    end

    function applyViewLock()
        if ~ishandle(S.fig) || ~ishandle(S.ax), return; end
        if S.lockView
            view(S.ax, S.viewSideDir);
            camup(S.ax, S.viewSideUp);
            camproj(S.ax,'orthographic');
            axis(S.ax,'vis3d');
            rotate3d(S.fig,'off');
        else
            view(S.ax, S.viewFreeAzEl(1), S.viewFreeAzEl(2));
            rotate3d(S.fig,'on');
        end
    end

    function onLockViewToggled()
        if ~ishandle(S.fig), return; end
        newLock = logical(get(S.chkLockView,'Value'));
        if newLock && ~S.lockView
            S.viewFreeAzEl = get(S.ax,'View');
        end
        S.lockView = newLock;
        applyViewLock();
    end

    function onPull()
        if ~ishandle(S.fig), return; end
        F = get(S.pullSld,'Value');
        updatePlot(S, F);
    end

    function onSpeedChanged()
        strs = get(S.popSpeed,'String');
        val  = get(S.popSpeed,'Value');
        s = strs{val};
        S.speedFactor = str2double(strrep(s,'x',''));
        if isnan(S.speedFactor) || S.speedFactor <= 0, S.speedFactor = 1; end
    end

    function onParamChanged(paramName, newVal)
        if ~ishandle(S.fig), return; end
        S.P.(paramName) = newVal;
        updateParamValueText(paramName, newVal);
        rebuildGeometryAndPlots(false);
    end

    function updateParamValueText(paramName, newVal)
        switch paramName
            case 'height',          hval = UI.v_height;
            case 'depth',           hval = UI.v_depth;
            case 'offset',          hval = UI.v_offset;
            case 'startNotchWidth', hval = UI.v_sNW;
            case 'endNotchWidth',   hval = UI.v_eNW;
            case 'startSpacing',    hval = UI.v_sSp;
            case 'endSpacing',      hval = UI.v_eSp;
            case 'startOffsetApex', hval = UI.v_sOA;
            case 'endOffsetApex',   hval = UI.v_eOA;
            otherwise, return;
        end
        set(hval,'String',sprintf('%.3f', newVal));
    end

    function rebuildGeometryAndPlots(isFirst)
        wasRunning = ~isempty(S.animTimer) && isvalid(S.animTimer) && strcmp(S.animTimer.Running,'on');
        if wasRunning, stopAnimation(); end

        prevMax = S.F_max_model;
        prevPull = get(S.pullSld,'Value');
        pullRatio = 0;
        if prevMax > 1e-12
            pullRatio = min(max(prevPull / prevMax, 0), 1);
        end

        [notches0, info] = generateNotchesOriginal( ...
            S.P.height, S.P.depth, S.P.offset, ...
            S.P.startNotchWidth, S.P.endNotchWidth, ...
            S.P.startSpacing, S.P.endSpacing, ...
            S.P.startOffsetApex, S.P.endOffsetApex);

        N = numel(notches0);
        if N == 0
            cla(S.ax);
            title(S.ax,'No notches generated with current parameters');
            return;
        end

        pins0   = info.p1;
        p0_list = info.p0;
        p2_list = info.p2;
        L_notch = info.L_notch;

        theta0 = zeros(N,1);
        for i = 1:N
            v0 = p0_list(i,:) - pins0(i,:);
            v2 = p2_list(i,:) - pins0(i,:);
            theta0(i) = angleBetween(v0, v2);
        end

        % ----------- PHYSICS PRECOMPUTE -----------
        in2mm = 25.4;
        ro = (S.P.tubeOD_in * in2mm)/2;
        ri = (S.P.tubeID_in * in2mm)/2;

        E_MPa = S.P.E_GPa * 1000; % GPa -> MPa (= N/mm^2)
        S.E_MPa = E_MPa;

        if isnan(S.P.tendon_x_mm)
            xT = ri; % tendon rides on +x inner wall
        else
            xT = S.P.tendon_x_mm;
        end

        depthCut = max(S.P.depth, 0);
        depthCut = min(depthCut, 2*ro);

        k_theta = zeros(N,1);
        r_eff   = zeros(N,1);
        c_fiber = zeros(N,1);

        for i = 1:N
            [~, xbar, Iy] = ringNotchSectionProps(ro, ri, depthCut, 140, 360);

            % Remaining material max-x boundary after cut is xMax = ro - depthCut.
            xMax = ro - depthCut;
            xMax = min(max(xMax, -ro), ro);

            % Distance from NA to extreme fiber in bending direction (x):
            cpos = abs(xMax - xbar);
            cneg = abs((-ro) - xbar);
            c    = max(cpos, cneg);
            c    = max(c, 1e-6);

            r_eff(i)   = max(xT - xbar, 1e-6);             % mm
            k_theta(i) = (E_MPa * Iy) / max(L_notch(i),1e-6); % (MPa*mm^4)/mm = N*mm/rad
            k_theta(i) = max(k_theta(i), 1e-9);

            c_fiber(i) = c;
        end

        % Model-based force cap from theta0 caps (avoids unreachable beyond geometric limit)
        Fcap_each = (k_theta .* max(theta0,0)) ./ max(r_eff,1e-6);
        Fcap_model = min(Fcap_each);
        if ~isfinite(Fcap_model) || Fcap_model <= 0
            Fcap_model = S.P.Fmax_user_N;
        end

        Fmax_use = min(S.P.Fmax_user_N, 1.05*Fcap_model);
        if ~isfinite(Fmax_use) || Fmax_use <= 1e-9
            Fmax_use = S.P.Fmax_user_N;
        end

        % ----------- Store -----------
        S.notches0 = notches0;
        S.pins0   = pins0;
        S.p0_list = p0_list;
        S.p2_list = p2_list;
        S.L_notch = L_notch;
        S.theta0  = theta0;

        S.k_theta = k_theta;
        S.r_eff   = r_eff;
        S.c_fiber = c_fiber;
        S.F_max_model = Fmax_use;

        S.notchColors = lines(max(N,7));
        S.notchColors = S.notchColors(1:N,:);

        % ----------- Plot rebuild -----------
        cla(S.ax); hold(S.ax,'on'); grid(S.ax,'on'); axis(S.ax,'equal');
        xlabel(S.ax,'X (mm)'); ylabel(S.ax,'Y (mm)'); zlabel(S.ax,'Z (mm)');
        ht2 = title(S.ax,'Undeformed (thin) vs Force-Driven Deformed (thicker); red = out of elastic');
        set(S.ax,'Color','none');
        try
            set(ht2,'Units','normalized');
            p = get(ht2,'Position'); p(2) = p(2) + 0.04; set(ht2,'Position',p);
        catch
        end

        applyViewLock();

        % Undeformed notches
        S.hOrig = gobjects(N,1);
        for i = 1:N
            P0 = notches0{i};
            c = S.notchColors(i,:) * S.undeformedDarkenFactor;
            S.hOrig(i) = plot3(S.ax, P0(:,1), P0(:,2), P0(:,3), '-o', ...
                'LineWidth', 1.0, 'MarkerSize', 3.5, 'Color', c);
        end

        % Undeformed inter-notch links: p2(i) -> p0(i+1)
        nLinks = max(N-1, 0);
        S.hInterLinkUndeformed = gobjects(nLinks,1);
        for i = 1:nLinks
            a = p2_list(i,:);
            b = p0_list(i+1,:);
            S.hInterLinkUndeformed(i) = plot3(S.ax, [a(1) b(1)], [a(2) b(2)], [a(3) b(3)], '-', ...
                'LineWidth', 0.8, 'Color', S.colLinkUndeformed);
        end

        % Deformed notches (created once; color updated each frame based on elastic status)
        S.hDef = gobjects(N,1);
        for i = 1:N
            P0 = notches0{i};
            S.hDef(i) = plot3(S.ax, P0(:,1), P0(:,2), P0(:,3), '-o', ...
                'LineWidth', 1.8, 'MarkerSize', 3.8, 'Color', S.notchColors(i,:));
        end

        % Pins -> scatter3 so we can per-point color
        S.hPins = scatter3(S.ax, pins0(:,1), pins0(:,2), pins0(:,3), 34, ...
            'Marker','s','LineWidth',0.9,'MarkerEdgeColor','flat','MarkerFaceColor','none');
        set(S.hPins, 'CData', repmat(S.colTendonPins, N, 1));

        % Tendon (through p2)
        S.hTendon = plot3(S.ax, p2_list(:,1), p2_list(:,2), p2_list(:,3), '-', ...
            'LineWidth', 1.0, 'Color', S.colTendonPins);

        % Dummy red handle for legend (over-elastic)
        S.hLegendOverElastic = plot3(S.ax, nan, nan, nan, '-o', ...
            'LineWidth', 1.8, 'MarkerSize', 3.8, 'Color', S.colOverElastic);

        % LEGEND
        try
            if nLinks >= 1
                S.hLegend = legend(S.ax, ...
                    [S.hOrig(1) S.hDef(1) S.hLegendOverElastic S.hInterLinkUndeformed(1) S.hPins S.hTendon], ...
                    {'Undeformed notches','Deformed (elastic)','Deformed (over-elastic)','Undeformed p2→p0 link','Pins (color-coded)','Tendon (through p2)'}, ...
                    'Location','best');
            else
                S.hLegend = legend(S.ax, ...
                    [S.hOrig(1) S.hDef(1) S.hLegendOverElastic S.hPins S.hTendon], ...
                    {'Undeformed notches','Deformed (elastic)','Deformed (over-elastic)','Pins (color-coded)','Tendon (through p2)'}, ...
                    'Location','best');
            end
        catch
        end

        % Force slider scaling
        set(S.pullSld,'Min',0,'Max',S.F_max_model);
        newPull = pullRatio * S.F_max_model;
        set(S.pullSld,'Value',newPull);

        updatePlot(S, newPull);

        if wasRunning
            startAnimation();
        elseif isFirst
            % nothing
        end
    end

    function startAnimation()
        if ~isempty(S.animTimer) && isvalid(S.animTimer) && strcmp(S.animTimer.Running,'on')
            return;
        end
        set(S.btnPlay,'Enable','off');
        set(S.btnStop,'Enable','on');

        dt = 1/max(S.animFPS,1);
        S.animTimer = timer( ...
            'ExecutionMode','fixedRate', ...
            'Period',dt, ...
            'BusyMode','drop', ...
            'TimerFcn',@onTick);

        start(S.animTimer);

        function onTick(~,~)
            if ~ishandle(S.fig)
                stopAnimation();
                return;
            end
            baseStep = S.F_max_model / max(S.animBaseFramesPerCurl, 10);
            step = baseStep * S.speedFactor;

            v = get(S.pullSld,'Value') + step;
            if v > S.F_max_model, v = 0; end
            set(S.pullSld,'Value',v);
            updatePlot(S, v);
        end
    end

    function stopAnimation()
        set(S.btnPlay,'Enable','on');
        set(S.btnStop,'Enable','off');
        if ~isempty(S.animTimer) && isvalid(S.animTimer)
            try, stop(S.animTimer); catch, end
            try, delete(S.animTimer); catch, end
        end
        S.animTimer = [];
    end

    function closeAndCleanup()
        stopAnimation();
        if ishandle(S.fig)
            try, set(S.fig,'CloseRequestFcn',[]); catch, end
            delete(S.fig);
        end
    end

    function saveGifFromUI()
        wasRunning = ~isempty(S.animTimer) && isvalid(S.animTimer) && strcmp(S.animTimer.Running,'on');
        stopAnimation();

        [file, path] = uiputfile('*.gif','Save full curl GIF as...', char(gifFilename));
        if isequal(file,0)
            if wasRunning, startAnimation(); end
            return;
        end
        out = fullfile(path,file);

        viewAzEl = get(S.ax,'View');

        saveFullCurlGifForceElastic( ...
            out, gifFrames, gifDelay, gifLoopCount, gifTransparentBg, ...
            S.P, S.distribution, S.bend_dir, ...
            S.lockView, S.viewSideDir, S.viewSideUp, viewAzEl, ...
            S.notches0, S.pins0, S.p0_list, S.p2_list, S.L_notch, S.theta0, ...
            S.k_theta, S.r_eff, S.c_fiber, S.E_MPa, S.F_max_model, ...
            S.notchColors, S.undeformedDarkenFactor, S.colLinkUndeformed, S.colTendonPins, S.colOverElastic);

        msgbox(sprintf('Saved GIF:\n%s', out), 'GIF Saved');

        if wasRunning, startAnimation(); end
    end

    function [hSld, hTxt, hVal] = makeParamSlider(label, defaultVal, range, x, y)
        wLabel = 160; wSlider = 250; wVal = 60; h = 16;

        hTxt = uicontrol(fig,'Style','text', ...
            'Units','pixels','Position',[x y+2 wLabel h], ...
            'BackgroundColor','none', ...
            'HorizontalAlignment','left', ...
            'String',label);

        hSld = uicontrol(fig,'Style','slider', ...
            'Units','pixels','Position',[x+wLabel+10 y wSlider h], ...
            'Min',range(1),'Max',range(2),'Value',defaultVal);

        hVal = uicontrol(fig,'Style','text', ...
            'Units','pixels','Position',[x+wLabel+10+wSlider+10 y+2 wVal h], ...
            'BackgroundColor','none', ...
            'HorizontalAlignment','left', ...
            'String',sprintf('%.3f',defaultVal));
    end
end

%% ========================= UPDATE (FORCE + ELASTIC CHECK) =========================
function updatePlot(S, F)

[theta, DeltaL_geom] = solveAnglesFromForce(F, S.k_theta, S.r_eff, S.theta0);

% Elastic check
[overElastic, epsMax, sigmaMax] = elasticStatus(theta, S.L_notch, S.c_fiber, S.E_MPa, ...
    S.P.epsElasticLimit, S.P.sigmaElasticLimit_MPa);

nOver = nnz(overElastic);
set(S.pullLbl, 'String', sprintf('Tendon force F = %.3f N | predicted ΔL ≈ %.4f mm (Fmax ≈ %.3f N) | over-elastic: %d', ...
    F, DeltaL_geom, S.F_max_model, nOver));

[pinsDef, yawBefore, yawAfter] = deformPinsFromThetas(S.pins0, theta, S.bend_dir);

N = numel(S.notches0);
p2_def = zeros(N,3);

for i = 1:N
    p0 = S.p0_list(i,:);
    p1 = S.pins0(i,:);
    p2 = S.p2_list(i,:);

    v0 = p0 - p1;
    v2 = p2 - p1;

    Rb = Ry(yawBefore(i));
    Ra = Ry(yawAfter(i));

    p1d = pinsDef(i,:);
    p0d = p1d + (Rb * v0(:)).';
    p2d = p1d + (Ra * v2(:)).';

    Pseg = [p0d; p1d; p2d];
    set(S.hDef(i), 'XData', Pseg(:,1), 'YData', Pseg(:,2), 'ZData', Pseg(:,3));

    % Color the notch red if out-of-elastic
    if overElastic(i)
        set(S.hDef(i), 'Color', S.colOverElastic);
    else
        set(S.hDef(i), 'Color', S.notchColors(i,:));
    end

    p2_def(i,:) = p2d;
end

% Pins: per-point color
pinColors = repmat(S.colTendonPins, N, 1);
pinColors(overElastic,:) = repmat(S.colOverElastic, nOver, 1);

set(S.hPins,  'XData', pinsDef(:,1), 'YData', pinsDef(:,2), 'ZData', pinsDef(:,3), 'CData', pinColors);
set(S.hTendon,'XData', p2_def(:,1), 'YData', p2_def(:,2), 'ZData', p2_def(:,3));

% Tip relative to tendon base:
tipAbs  = p2_def(end,:);
baseAbs = p2_def(1,:) + [0 0 -S.P.startSpacing];
tipRel  = tipAbs - baseAbs;

attackDeg = rad2deg(abs(yawAfter(end)));
degSym = char(176);

% Show worst-case elastic metric at current force
epsWorst   = max(epsMax);
sigmaWorst = max(sigmaMax);

set(S.tipTxt, 'String', sprintf(['Tip wrt base: (x,y,z) = (%.3f, %.3f, %.3f) mm\n' ...
                                 'Notches: %d | Attack = %.2f%s | eps(max)=%.3f | sig(max)=%.0f MPa'], ...
                                 tipRel(1), tipRel(2), tipRel(3), N, attackDeg, degSym, epsWorst, sigmaWorst));

drawnow limitrate;
end

%% ========================= GIF EXPORT (FORCE + ELASTIC) =========================
function saveFullCurlGifForceElastic( ...
    filename, nFrames, delay, loopCount, makeTransparent, ...
    P, distribution, bend_dir, ...
    lockView, viewSideDir, viewSideUp, viewAzEl, ...
    notches0, pins0, p0_list, p2_list, L_notch, theta0, ...
    k_theta, r_eff, c_fiber, E_MPa, F_max, ...
    notchColors, undeformedDarkenFactor, colLinkUndeformed, colTendonPins, colOverElastic)

N = numel(notches0);

forceBlackOpaque = false;
if forceBlackOpaque
    makeTransparent = false;
end

keyRGB = [0 0 0];

f = figure('Visible','off', 'Position',[100 100 950 640], 'InvertHardcopy','off');
set(f,'Renderer','opengl');
try, set(f,'GraphicsSmoothing','off'); catch, end

if makeTransparent
    set(f,'Color', keyRGB);
else
    set(f,'Color','k');
end

ax = axes('Parent',f);
hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');
xlabel(ax,'X (mm)'); ylabel(ax,'Y (mm)'); zlabel(ax,'Z (mm)');
ht = title(ax,'CPR Curl Deformation (Force + Elastic limit)');
try
    set(ht,'Units','normalized');
    p = get(ht,'Position'); p(2) = p(2) + 0.06; set(ht,'Position',p);
catch
end

set(ax,'XColor',[1 1 1],'YColor',[1 1 1],'ZColor',[1 1 1]);
set(ax,'GridColor',[0.6 0.6 0.6],'GridAlpha',0.35);

if lockView
    view(ax, viewSideDir);
    camup(ax, viewSideUp);
    camproj(ax,'orthographic');
    axis(ax,'vis3d');
else
    view(ax, viewAzEl(1), viewAzEl(2));
end

if makeTransparent
    set(ax,'Color', keyRGB);
else
    set(ax,'Color','k');
end

% --- Overlays ---
hTip = annotation(f,'textbox',[0.0 0.90 0.92 0.08], ...
    'String','', ...
    'FitBoxToText','on', ...
    'Interpreter','none', ...
    'EdgeColor','none', ...
    'BackgroundColor','none', ...
    'Color',[1 1 1], ...
    'FontSize',11);

paramStr = formatParamStringForceElastic(P, distribution, bend_dir, F_max);
annotation(f,'textbox',[0.00 0.01 0.75 0.27], ...
    'String',paramStr, ...
    'FitBoxToText','on', ...
    'Interpreter','none', ...
    'EdgeColor','none', ...
    'BackgroundColor','none', ...
    'Color',[1 1 1], ...
    'FontSize',8);

% --- Plot objects (create once; update per frame) ---
hOrig = gobjects(N,1);
for i = 1:N
    P0 = notches0{i};
    c = notchColors(i,:) * undeformedDarkenFactor;
    hOrig(i) = plot3(ax, P0(:,1), P0(:,2), P0(:,3), '-o', ...
        'LineWidth', 1.0, 'MarkerSize', 3.5, 'Color', c);
end

nLinks = max(N-1,0);
hLink = gobjects(nLinks,1);
for i = 1:nLinks
    a = p2_list(i,:);
    b = p0_list(i+1,:);
    hLink(i) = plot3(ax, [a(1) b(1)], [a(2) b(2)], [a(3) b(3)], '-', ...
        'LineWidth', 0.8, 'Color', colLinkUndeformed);
end

hDef = gobjects(N,1);
for i = 1:N
    P0 = notches0{i};
    hDef(i) = plot3(ax, P0(:,1), P0(:,2), P0(:,3), '-o', ...
        'LineWidth', 1.8, 'MarkerSize', 3.8, 'Color', notchColors(i,:));
end

hPins = scatter3(ax, pins0(:,1), pins0(:,2), pins0(:,3), 34, ...
    'Marker','s','LineWidth',0.9,'MarkerEdgeColor','flat','MarkerFaceColor','none');
set(hPins, 'CData', repmat(colTendonPins, N, 1));

hTendon = plot3(ax, p2_list(:,1), p2_list(:,2), p2_list(:,3), '-', ...
    'LineWidth', 1.0, 'Color', colTendonPins);

hOver = plot3(ax, nan, nan, nan, '-o', 'LineWidth',1.8,'MarkerSize',3.8,'Color',colOverElastic);

try
    if nLinks >= 1
        lgd = legend(ax, [hOrig(1) hDef(1) hOver hLink(1) hPins hTendon], ...
            {'Undeformed notches','Deformed (elastic)','Deformed (over-elastic)','Undeformed p2→p0 link','Pins (color-coded)','Tendon (through p2)'}, ...
            'Location','northeast');
    else
        lgd = legend(ax, [hOrig(1) hDef(1) hOver hPins hTendon], ...
            {'Undeformed notches','Deformed (elastic)','Deformed (over-elastic)','Pins (color-coded)','Tendon (through p2)'}, ...
            'Location','northeast');
    end
    set(lgd,'TextColor',[1 1 1],'Color','none','Box','off');
catch
end

degSym = char(176);

baseMap = [];
keyIdx0 = 0; % 0-based

for k = 1:nFrames
    t = (k-1) / max(nFrames-1,1);
    F = t * F_max;

    [theta, DeltaL_geom] = solveAnglesFromForce(F, k_theta, r_eff, theta0);
    [overElastic, epsMax, sigmaMax] = elasticStatus(theta, L_notch, c_fiber, E_MPa, ...
        P.epsElasticLimit, P.sigmaElasticLimit_MPa);

    [pinsDef, yawBefore, yawAfter] = deformPinsFromThetas(pins0, theta, bend_dir);

    p2_def = zeros(N,3);
    for i = 1:N
        p0 = p0_list(i,:);
        p1 = pins0(i,:);
        p2 = p2_list(i,:);

        v0 = p0 - p1;
        v2 = p2 - p1;

        p1d = pinsDef(i,:);
        p0d = p1d + (Ry(yawBefore(i)) * v0(:)).';
        p2d = p1d + (Ry(yawAfter(i))  * v2(:)).';

        Pseg = [p0d; p1d; p2d];
        set(hDef(i), 'XData', Pseg(:,1), 'YData', Pseg(:,2), 'ZData', Pseg(:,3));

        if overElastic(i)
            set(hDef(i), 'Color', colOverElastic);
        else
            set(hDef(i), 'Color', notchColors(i,:));
        end

        p2_def(i,:) = p2d;
    end

    pinColors = repmat(colTendonPins, N, 1);
    pinColors(overElastic,:) = repmat(colOverElastic, nnz(overElastic), 1);
    set(hPins,'XData',pinsDef(:,1),'YData',pinsDef(:,2),'ZData',pinsDef(:,3),'CData',pinColors);
    set(hTendon,'XData',p2_def(:,1),'YData',p2_def(:,2),'ZData',p2_def(:,3));

    tipAbs  = p2_def(end,:);
    baseAbs = p2_def(1,:) + [0 0 -P.startSpacing];
    tipRel  = tipAbs - baseAbs;

    attackDeg = rad2deg(abs(yawAfter(end)));

    set(hTip,'String',sprintf(['Tip: (%.3f, %.3f, %.3f) mm | F: %.3f N | ΔL≈%.3f mm | ' ...
                              'Over-elastic: %d | eps(max)=%.3f | sig(max)=%.0f MPa | Attack: %.2f%s'], ...
        tipRel(1), tipRel(2), tipRel(3), F, DeltaL_geom, nnz(overElastic), max(epsMax), max(sigmaMax), attackDeg, degSym));

    drawnow;
    fr = getframe(f);

    if k == 1
        [im, map] = rgb2ind(fr.cdata, 256);
        baseMap = map;

        if makeTransparent
            [~, idx] = min(sum((baseMap - keyRGB).^2, 2));
            keyIdx0 = idx - 1;
            imwrite(im, baseMap, filename, 'gif', ...
                'LoopCount', loopCount, ...
                'DelayTime', delay, ...
                'DisposalMethod','restorebg', ...
                'TransparentColor', keyIdx0);
        else
            imwrite(im, baseMap, filename, 'gif', ...
                'LoopCount', loopCount, ...
                'DelayTime', delay, ...
                'DisposalMethod','restorebg');
        end
    else
        im = rgb2ind(fr.cdata, baseMap, 'nodither');
        if makeTransparent
            imwrite(im, baseMap, filename, 'gif', ...
                'WriteMode','append', ...
                'DelayTime', delay, ...
                'DisposalMethod','restorebg', ...
                'TransparentColor', keyIdx0);
        else
            imwrite(im, baseMap, filename, 'gif', ...
                'WriteMode','append', ...
                'DelayTime', delay, ...
                'DisposalMethod','restorebg');
        end
    end
end

delete(f);
end

function s = formatParamStringForceElastic(P, distribution, bend_dir, Fmax)
in2mm = 25.4;
IDmm = P.tubeID_in * in2mm;
ODmm = P.tubeOD_in * in2mm;

if isnan(P.tendon_x_mm)
    tendonStr = 'tendon_x_mm = NaN (assume +ri)';
else
    tendonStr = sprintf('tendon_x_mm = %.4f', P.tendon_x_mm);
end

s = sprintf([ ...
    'Settings used:\n' ...
    'height = %.3f\n' ...
    'depthCut = %.3f\n' ...
    'offset = %.3f\n' ...
    'startNotchWidth = %.3f\n' ...
    'endNotchWidth   = %.3f\n' ...
    'startSpacing    = %.3f\n' ...
    'endSpacing      = %.3f\n' ...
    'startOffsetApex = %.3f\n' ...
    'endOffsetApex   = %.3f\n' ...
    'tube ID/OD = %.4f / %.4f mm\n' ...
    'E = %.2f GPa\n' ...
    '%s\n' ...
    'elastic limits: eps<=%.4f, sigma<=%.0f MPa\n' ...
    'model = %s\n' ...
    'bend_dir = %+d\n' ...
    'Fmax(model/slider) = %.3f N' ], ...
    P.height, P.depth, P.offset, ...
    P.startNotchWidth, P.endNotchWidth, ...
    P.startSpacing, P.endSpacing, ...
    P.startOffsetApex, P.endOffsetApex, ...
    IDmm, ODmm, ...
    P.E_GPa, ...
    tendonStr, ...
    P.epsElasticLimit, P.sigmaElasticLimit_MPa, ...
    char(distribution), bend_dir, Fmax);
end

%% ========================= GEOMETRY BUILD =========================
function [notches, info] = generateNotchesOriginal( ...
    height, depth, offset, ...
    startNotchWidth, endNotchWidth, ...
    startSpacing, endSpacing, ...
    startOffsetApex, endOffsetApex)

notches = {};
p0_list = [];
p1_list = [];
p2_list = [];
L_list  = [];
S_list  = [];

start_z  = 0.0;
min_step = 1e-9;

while start_z < height
    t = 0.0;
    if height ~= 0, t = start_z / height; end

    L = lerp(startNotchWidth, endNotchWidth, t);
    L = max(L, 0.0);

    end_z = start_z + L;
    if end_z > height, break; end

    apex_z_offset = lerp(startOffsetApex, endOffsetApex, t);

    % NOTE: These are still your demo kinematic points. The stiffness model
    % uses tube ID/OD and "depth" as cut depth across diameter for section props.
    p0 = [ depth, 0.0, start_z ];
    p1 = [ -offset, 0.0, start_z + 0.5*L + apex_z_offset ];
    p2 = [ depth, 0.0, end_z ];

    notches{end+1} = [p0; p1; p2]; %#ok<AGROW>

    spacing_now = lerp(startSpacing, endSpacing, t);

    p0_list = [p0_list; p0]; %#ok<AGROW>
    p1_list = [p1_list; p1]; %#ok<AGROW>
    p2_list = [p2_list; p2]; %#ok<AGROW>
    L_list  = [L_list;  L];  %#ok<AGROW>
    S_list  = [S_list;  max(spacing_now,0.0)]; %#ok<AGROW>

    if L <= min_step && spacing_now <= min_step, break; end
    start_z = end_z + max(spacing_now,0.0);
end

info.p0 = p0_list;
info.p1 = p1_list;
info.p2 = p2_list;
info.L_notch = L_list;
info.S_after = S_list;
end

%% ========================= PIN DEFORMATION =========================
function [pinsDef, yawBefore, yawAfter] = deformPinsFromThetas(pins0, theta, bend_dir)
N = size(pins0,1);
pinsDef = zeros(N,3);
yawBefore = zeros(N,1);
yawAfter  = zeros(N,1);

xNA = pins0(1,1);
p = [xNA, 0, 0];
yaw = 0;

L0 = pins0(1,3) - 0;
p = p + stepXZ(L0, yaw);
pinsDef(1,:) = p;

for i = 1:N
    yawBefore(i) = yaw;
    yaw = yaw + bend_dir * theta(i);
    yawAfter(i) = yaw;

    if i < N
        seg = pins0(i+1,3) - pins0(i,3);
        p = p + stepXZ(seg, yaw);
        pinsDef(i+1,:) = p;
    end
end
end

%% ========================= FORCE-BASED ANGLES =========================
function [theta, DeltaL_geom] = solveAnglesFromForce(F, k_theta, r_eff, thetaCap)
if isempty(k_theta) || isempty(r_eff) || F <= 0
    theta = zeros(size(thetaCap));
    DeltaL_geom = 0;
    return;
end

theta = (F .* r_eff) ./ max(k_theta,1e-12);
theta = min(max(theta,0), max(thetaCap,0));

DeltaL_geom = sum(r_eff .* theta);  % mm (geometric tendon shortening estimate)
end

%% ========================= ELASTIC STATUS (NEW) =========================
function [overElastic, epsMax, sigmaMax] = elasticStatus(theta, L_notch, c_fiber, E_MPa, epsLim, sigmaLim_MPa)
Lsafe = max(L_notch, 1e-9);
kappa = theta ./ Lsafe;              % 1/mm
epsMax = abs(kappa) .* c_fiber;      % unitless
sigmaMax = E_MPa .* epsMax;          % MPa (since E in MPa)

% if either limit exceeded -> mark as out-of-elastic
overElastic = (epsMax > epsLim + 1e-12) | (sigmaMax > sigmaLim_MPa + 1e-9);
end

%% ========================= SECTION PROPS (NOTCHED RING) =========================
function [A, xbar, Iy_centroid] = ringNotchSectionProps(ro, ri, depthCut, nR, nTh)
% Remaining section = ring (ri..ro) MINUS region with x > (ro - depthCut)
% depthCut in mm, measured from +x outer surface inward.

if nargin < 4, nR = 140; end
if nargin < 5, nTh = 360; end

depthCut = max(depthCut, 0);
depthCut = min(depthCut, 2*ro);
xCut = ro - depthCut;   % remove x > xCut

dr  = (ro - ri) / nR;
dth = 2*pi / nTh;

r  = ri + (0.5:1:nR-0.5)*dr;
th = (0.5:1:nTh-0.5)*dth;
[R,TH] = ndgrid(r, th);

X = R .* cos(TH);
dA = R * dr * dth;

keep = (X <= xCut);

A = sum(dA(keep));
if A < 1e-12
    xbar = 0;
    Iy_centroid = 1e-12;
    return;
end

xbar = sum(X(keep).*dA(keep)) / A;
Iy_centroid = sum( (X(keep) - xbar).^2 .* dA(keep) );
end

%% ========================= MATH HELPERS =========================
function y = lerp(a,b,t), y = a + t*(b-a); end

function ang = angleBetween(a,b)
na = norm(a); nb = norm(b);
if na < 1e-12 || nb < 1e-12
    ang = 0; return;
end
c = dot(a,b) / (na*nb);
c = max(min(c,1),-1);
ang = acos(c);
end

function v = stepXZ(s, yaw)
v = [s*sin(yaw), 0, s*cos(yaw)];
end

function R = Ry(yaw)
c = cos(yaw); s = sin(yaw);
R = [ c  0  s;
      0  1  0;
     -s  0  c ];
end
