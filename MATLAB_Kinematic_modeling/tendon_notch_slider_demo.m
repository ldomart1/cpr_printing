function tendon_notch_slider_demo()
%% Tendon-pull notch demo with live geometry sliders + animation + GIF export
% Final requested updates:
% - Legend restored
% - Attack angle is UNWRAPPED (0 -> 370 -> ...), not folding after 180
% - Title placed slightly higher
% - Tip display includes total number of notches (before attack angle)
% - Side view lock is X-Z plane (camera along +Y)
% - Tip reference base is offset by -startSpacing in Z
% - GIF export: black-key transparency (or black opaque fallback), no frame accumulation,
%   metadata bottom-left (smaller), tip+notch+attack top-left, legend included.

clc; close all;

%% ------------------- DEFAULT PARAMS + RANGES -------------------
P = struct();
P.height          = 30.0;  R.height          = [0 70];
P.depth           = 1.3;   R.depth           = [0 1.45];
P.offset          = 0.5;   R.offset          = [-2 2];

P.startNotchWidth = 1.5;   R.startNotchWidth = [0 4];
P.endNotchWidth   = 1.5;   R.endNotchWidth   = [0 4];

P.startSpacing    = 0.0;   R.startSpacing    = [-2 2];
P.endSpacing      = 0.0;   R.endSpacing      = [-2 2];

P.startOffsetApex = -1.0;  R.startOffsetApex = [-3 3];
P.endOffsetApex   = -2.0;  R.endOffsetApex   = [-3 3];

distribution = "proportional_to_length"; % "uniform" or "proportional_to_length"
bend_dir = +1;

%% ------------------- ANIMATION / GIF SETTINGS -------------------
gifFilename           = "full_curl.gif";
gifFrames             = 140;
gifDelay              = 0.03;
gifLoopCount          = inf;
gifTransparentBg      = true;

animFPS               = 40;
animBaseFramesPerCurl = 140;

%% ------------------- COLORS -------------------
colLinkUndeformed = [0.80 0.80 0.80];   % undeformed inter-notch links
colTendonPins     = [1.00 1.00 1.00];   % tendon + deformed pins (WHITE)
undeformedDarkenFactor = 0.80;          % darken undeformed notch colors

%% ------------------- FIGURE / AXES (transparent) -------------------
fig = figure('Name','Tendon Pull: Deformed Pins + Notches', ...
    'Position',[100 140 1200 820], ...
    'Color','none', ...
    'InvertHardcopy','off');

% Move axes up a bit (more room for UI)
ax = axes('Parent',fig, 'Position',[0.07 0.34 0.90 0.62]);
grid(ax,'on'); axis(ax,'equal'); hold(ax,'on');
xlabel(ax,'X (mm)'); ylabel(ax,'Y (mm)'); zlabel(ax,'Z (mm)');
ht = title(ax,'Undeformed vs Deformed continuum robot');
view(ax, 35, 20);
set(ax,'Color','none');

% Title slightly higher
try
    set(ht,'Units','normalized');
    p = get(ht,'Position'); p(2) = p(2) + 0.04; set(ht,'Position',p);
catch
end

%% ------------------- PULL SLIDER + LABEL -------------------
pullSld = uicontrol(fig,'Style','slider', ...
    'Units','pixels', ...
    'Position',[110 215 980 18], ...
    'Min',0,'Max',1,'Value',0);

pullLbl = uicontrol(fig,'Style','text', ...
    'Units','pixels', ...
    'Position',[110 235 980 18], ...
    'BackgroundColor','none', ...
    'HorizontalAlignment','left', ...
    'String','Pulled tendon ΔL = 0.0000 mm');

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

% Lock view checkbox (side view = X-Z plane)
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
[UI.h_depth,  UI.t_depth,  UI.v_depth ] = makeParamSlider('depth',  P.depth,  R.depth,  col1x, rowYs(2));
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
S.viewSideDir  = [0 -1 0];   % camera along +Y => screen plane is X-Z
S.viewSideUp   = [0 0 1];   % Z up

S.P = P; S.R = R;
S.distribution = distribution;
S.bend_dir = bend_dir;

S.animTimer = [];
S.animFPS = animFPS;
S.animBaseFramesPerCurl = animBaseFramesPerCurl;
S.speedFactor = 1.0;

S.colLinkUndeformed = colLinkUndeformed;
S.colTendonPins     = colTendonPins;
S.undeformedDarkenFactor = undeformedDarkenFactor;

S.hOrig = gobjects(0);
S.hDef  = gobjects(0);
S.hInterLinkUndeformed = gobjects(0);
S.hPins = gobjects(1);
S.hTendon = gobjects(1);
S.hLegend = gobjects(1);
S.notchColors = zeros(0,3);

S.notches0 = {};
S.pins0 = [];
S.p0_list = [];
S.p2_list = [];
S.L_notch = [];
S.theta0 = [];
S.d = [];
S.DeltaL_max = 1;

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
            S.viewFreeAzEl = get(S.ax,'View'); % store current view
        end
        S.lockView = newLock;
        applyViewLock();
    end

    function onPull()
        if ~ishandle(S.fig), return; end
        DeltaL = get(S.pullSld,'Value');
        updatePlot(S, DeltaL);
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

        prevMax = S.DeltaL_max;
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

        d = abs(p2_list(:,1) - pins0(:,1));
        DeltaL_max = sum(d .* theta0);
        if DeltaL_max < 1e-9, DeltaL_max = 1.0; end

        S.notches0 = notches0;
        S.pins0 = pins0;
        S.p0_list = p0_list;
        S.p2_list = p2_list;
        S.L_notch = L_notch;
        S.theta0 = theta0;
        S.d = d;
        S.DeltaL_max = DeltaL_max;

        S.notchColors = lines(max(N,7));
        S.notchColors = S.notchColors(1:N,:);

        cla(S.ax); hold(S.ax,'on'); grid(S.ax,'on'); axis(S.ax,'equal');
        xlabel(S.ax,'X (mm)'); ylabel(S.ax,'Y (mm)'); zlabel(S.ax,'Z (mm)');
        ht2 = title(S.ax,'Undeformed (thin) vs Deformed (thicker)');
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

        % Deformed notches
        S.hDef = gobjects(N,1);
        for i = 1:N
            P0 = notches0{i};
            S.hDef(i) = plot3(S.ax, P0(:,1), P0(:,2), P0(:,3), '-o', ...
                'LineWidth', 1.8, 'MarkerSize', 3.8, 'Color', S.notchColors(i,:));
        end

        % Deformed pins (thin white, smaller)
        S.hPins = plot3(S.ax, pins0(:,1), pins0(:,2), pins0(:,3), 's', ...
            'MarkerSize', 4.6, 'LineWidth', 0.9, ...
            'MarkerEdgeColor', S.colTendonPins, ...
            'MarkerFaceColor', 'none');

        % Tendon (thin white)
        S.hTendon = plot3(S.ax, p2_list(:,1), p2_list(:,2), p2_list(:,3), '-', ...
            'LineWidth', 1.0, 'Color', S.colTendonPins);

        % LEGEND (restored)
        try
            if nLinks >= 1
                S.hLegend = legend(S.ax, ...
                    [S.hOrig(1) S.hDef(1) S.hInterLinkUndeformed(1) S.hPins S.hTendon], ...
                    {'Undeformed notches','Deformed notches','Undeformed p2→p0 link','Deformed pins','Tendon (through p2)'}, ...
                    'Location','best');
            else
                S.hLegend = legend(S.ax, ...
                    [S.hOrig(1) S.hDef(1) S.hPins S.hTendon], ...
                    {'Undeformed notches','Deformed notches','Deformed pins','Tendon (through p2)'}, ...
                    'Location','best');
            end
        catch
        end

        set(S.pullSld,'Min',0,'Max',DeltaL_max);
        newPull = pullRatio * DeltaL_max;
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
            baseStep = S.DeltaL_max / max(S.animBaseFramesPerCurl, 10);
            step = baseStep * S.speedFactor;

            v = get(S.pullSld,'Value') + step;
            if v > S.DeltaL_max, v = 0; end
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
            delete(S.fig);  % use delete to avoid recursive close warnings
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

        saveFullCurlGif( ...
            out, gifFrames, gifDelay, gifLoopCount, gifTransparentBg, ...
            S.P, S.distribution, S.bend_dir, ...
            S.lockView, S.viewSideDir, S.viewSideUp, viewAzEl, ...
            S.notches0, S.pins0, S.p0_list, S.p2_list, S.L_notch, S.theta0, S.d, S.DeltaL_max, ...
            S.notchColors, S.undeformedDarkenFactor, S.colLinkUndeformed, S.colTendonPins);

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

%% ========================= UPDATE =========================
function updatePlot(S, DeltaL)
set(S.pullLbl, 'String', sprintf('Pulled tendon ΔL = %.4f mm (max ≈ %.4f mm)', DeltaL, S.DeltaL_max));

theta = solveAnglesWithLimits(DeltaL, S.d, S.L_notch, S.theta0, S.distribution);
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
    p2_def(i,:) = p2d;
end

set(S.hPins,  'XData', pinsDef(:,1), 'YData', pinsDef(:,2), 'ZData', pinsDef(:,3));
set(S.hTendon,'XData', p2_def(:,1), 'YData', p2_def(:,2), 'ZData', p2_def(:,3));

% Tip relative to tendon base:
% base = first tendon point shifted by -startSpacing in Z
tipAbs  = p2_def(end,:);
baseAbs = p2_def(1,:) + [0 0 -S.P.startSpacing];
tipRel  = tipAbs - baseAbs;

% Unwrapped attack angle: use cumulative yaw at the tip (can exceed 360°)
attackDeg = rad2deg(abs(yawAfter(end)));

degSym = char(176);
set(S.tipTxt, 'String', sprintf(['Tip wrt base: (x,y,z) = (%.3f, %.3f, %.3f) mm\n' ...
                                 'Notches: %d | Attack = %.2f%s'], ...
                                 tipRel(1), tipRel(2), tipRel(3), N, attackDeg, degSym));

drawnow limitrate;
end

%% ========================= GIF EXPORT =========================
function saveFullCurlGif( ...
    filename, nFrames, delay, loopCount, makeTransparent, ...
    P, distribution, bend_dir, ...
    lockView, viewSideDir, viewSideUp, viewAzEl, ...
    notches0, pins0, p0_list, p2_list, L_notch, theta0, d, DeltaL_max, ...
    notchColors, undeformedDarkenFactor, colLinkUndeformed, colTendonPins)

N = numel(notches0);

% If transparency edges still bother you, set this true (forces black opaque background).
forceBlackOpaque = false;
if forceBlackOpaque
    makeTransparent = false;
end

% Use BLACK as the transparency key -> no pink halos.
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
ht = title(ax,'CPR Curl Deformation');
% title slightly higher
try
    set(ht,'Units','normalized');
    p = get(ht,'Position'); p(2) = p(2) + 0.06; set(ht,'Position',p);
catch
end

% Make axis readable on black
set(ax,'XColor',[1 1 1],'YColor',[1 1 1],'ZColor',[1 1 1]);
set(ax,'GridColor',[0.6 0.6 0.6],'GridAlpha',0.35);

% View
if lockView
    view(ax, viewSideDir);   % should be [0 1 0]
    camup(ax, viewSideUp);   % [0 0 1]
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
% Tip text (top-left)
hTip = annotation(f,'textbox',[0.0 0.90 0.86 0.08], ...
    'String','', ...
    'FitBoxToText','on', ...
    'Interpreter','none', ...
    'EdgeColor','none', ...
    'BackgroundColor','none', ...
    'Color',[1 1 1], ...
    'FontSize',11);

% Metadata (bottom-left), smaller
paramStr = formatParamString(P, distribution, bend_dir);
annotation(f,'textbox',[0.00 0.01 0.6 0.20], ...
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

hPins = plot3(ax, pins0(:,1), pins0(:,2), pins0(:,3), 's', ...
    'MarkerSize', 4.6, 'LineWidth', 0.9, ...
    'MarkerEdgeColor', colTendonPins, 'MarkerFaceColor', 'none');

hTendon = plot3(ax, p2_list(:,1), p2_list(:,2), p2_list(:,3), '-', ...
    'LineWidth', 1.0, 'Color', colTendonPins);

% Legend (restored) — place it away from top-left overlays
try
    if nLinks >= 1
        lgd = legend(ax, [hOrig(1) hDef(1) hLink(1) hPins hTendon], ...
            {'Undeformed notches','Deformed notches','Undeformed p2→p0 link','Deformed pins','Tendon (through p2)'}, ...
            'Location','northeast');
    else
        lgd = legend(ax, [hOrig(1) hDef(1) hPins hTendon], ...
            {'Undeformed notches','Deformed notches','Deformed pins','Tendon (through p2)'}, ...
            'Location','northeast');
    end
    set(lgd,'TextColor',[1 1 1],'Color','none','Box','off');
catch
end

degSym = char(176);

% --- GIF anti-trail strategy ---
% - Fixed colormap from frame 1
% - DisposalMethod='restorebg' for every frame
% - Stable TransparentColor index
baseMap = [];
keyIdx0 = 0; % 0-based

for k = 1:nFrames
    t = (k-1) / max(nFrames-1,1);
    DeltaL = t * DeltaL_max;

    theta = solveAnglesWithLimits(DeltaL, d, L_notch, theta0, distribution);
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
        p2_def(i,:) = p2d;
    end

    set(hPins,'XData',pinsDef(:,1),'YData',pinsDef(:,2),'ZData',pinsDef(:,3));
    set(hTendon,'XData',p2_def(:,1),'YData',p2_def(:,2),'ZData',p2_def(:,3));

    % Tip relative to base with -startSpacing in Z
    tipAbs  = p2_def(end,:);
    baseAbs = p2_def(1,:) + [0 0 -P.startSpacing];
    tipRel  = tipAbs - baseAbs;

    % Unwrapped attack angle (can exceed 360°)
    attackDeg = rad2deg(abs(yawAfter(end)));

    set(hTip,'String',sprintf(['Tip wrt base: (%.3f, %.3f, %.3f) mm   |   ' ...
                              'ΔL: %.3f mm   |   Notches: %d   |   Attack: %.2f%s'], ...
        tipRel(1), tipRel(2), tipRel(3), DeltaL, N, attackDeg, degSym));

    drawnow;
    fr = getframe(f);

    if k == 1
        [im, map] = rgb2ind(fr.cdata, 256);
        baseMap = map;

        if makeTransparent
            [~, idx] = min(sum((baseMap - keyRGB).^2, 2));
            keyIdx0 = idx - 1; % 0-based
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

function s = formatParamString(P, distribution, bend_dir)
s = sprintf([ ...
    'Settings used:\n' ...
    'height = %.3f\n' ...
    'depth = %.3f\n' ...
    'offset = %.3f\n' ...
    'startNotchWidth = %.3f\n' ...
    'endNotchWidth   = %.3f\n' ...
    'startSpacing    = %.3f\n' ...
    'endSpacing      = %.3f\n' ...
    'startOffsetApex = %.3f\n' ...
    'endOffsetApex   = %.3f\n' ...
    'distribution    = %s\n' ...
    'bend_dir        = %+d' ], ...
    P.height, P.depth, P.offset, ...
    P.startNotchWidth, P.endNotchWidth, ...
    P.startSpacing, P.endSpacing, ...
    P.startOffsetApex, P.endOffsetApex, ...
    char(distribution), bend_dir);
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

%% ========================= ANGLES FROM PULL (WITH LIMITS) =========================
function theta = solveAnglesWithLimits(DeltaL, d, L_notch, theta0, distribution)
N = numel(d);
theta = zeros(N,1);
if N == 0 || DeltaL <= 0
    return;
end

d_safe = max(d, 1e-12);
thetaCap = max(theta0, 0);

switch string(distribution)
    case "uniform"
        w = ones(N,1);
    otherwise
        w = max(L_notch, 1e-12);
end

free = true(N,1);
remaining = DeltaL;

while remaining > 1e-12 && any(free)
    wf = w .* free;
    wf_sum = sum(wf);
    if wf_sum < 1e-12
        break;
    end

    DeltaL_i = remaining * (wf / wf_sum);
    theta_proposed = DeltaL_i ./ d_safe;

    saturate = free & (theta + theta_proposed > thetaCap + 1e-12);

    if ~any(saturate)
        theta(free) = theta(free) + theta_proposed(free);
        remaining = 0;
        break;
    end

    for i = find(saturate).'
        theta_to_cap = max(thetaCap(i) - theta(i), 0);
        used = d_safe(i) * theta_to_cap;

        theta(i) = thetaCap(i);
        remaining = max(remaining - used, 0);
        free(i) = false;
    end
end

theta = min(max(theta,0), thetaCap);
end

%% ========================= MATH HELPERS =========================
function y = lerp(a,b,t), y = a + t*(b-a); end

function ang = angleBetween(a,b)
na = norm(a); nb = norm(b);
if na < 1e-12 || nb < 1e-12
    ang = 0;
    return;
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
