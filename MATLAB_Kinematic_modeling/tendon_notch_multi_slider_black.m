function tendon_notch_ui_side_sliders_black()
%% Interactive tendon-pull deformation (side sliders + bottom pull slider)
% + Tip position display (Cartesian)
% + "Lock view" checkbox to keep the camera fixed while sliders update
%
% Visual rules:
% - Black background, grid on
% - Undeformed = light grey solid (no markers)
% - Deformed notches = white solid (no markers)
% - Pins = smaller white circles
% - Tendon (through p2) = grey line
% - No legend

clc; close all;

%% ------------------- DEFAULTS + SLIDER LIMITS -------------------
D.height          = 30.0;   L.height          = [0 50];
D.depth           = 1.3;    L.depth           = [0 5];    % reasonable range
D.offset          = 0.5;    L.offset          = [-2 2];

D.startNotchWidth = 1.5;    L.startNotchWidth = [0 4];
D.endNotchWidth   = 1.5;    L.endNotchWidth   = [0 4];

D.startSpacing    = 0.0;    L.startSpacing    = [-2 2];
D.endSpacing      = 0.0;    L.endSpacing      = [-2 2];

D.startOffsetApex = -1.0;   L.startOffsetApex = [-3 3];
D.endOffsetApex   = -2.0;   L.endOffsetApex   = [-3 3];

distribution = "proportional_to_length"; % or "uniform"
bend_dir     = +1;

% Capstan / tendon friction model
% T_out = T_in * exp(-mu * beta), where beta is the local wrap angle.
% This simulation uses the resulting relative tendon tension as a weighting
% on the local tendon shortening assigned to each notch.
useCapstan = true;
pullSide   = "base";   % "base" if tendon is pulled from notch 1; use "tip" otherwise
D.muCapstan = 0.15;    L.muCapstan = [0 1.5];

% Colors / style
colUndef  = [0.65 0.65 0.65];  % light grey
colDef    = [1 1 1];           % white
colTendon = [0.70 0.70 0.70];  % grey
gridColor = [0.25 0.25 0.25];
pinMarkerSize = 5;             % smaller pins

defaultView = [35, 20];

%% ------------------- UI LAYOUT -------------------
fig = uifigure('Name','Tendon Pull Arm', 'Position',[80 60 1250 820]);
fig.Color = 'k';

% 2 rows (main + bottom pull), 2 cols (axes + side controls)
gl = uigridlayout(fig, [2 2]);
gl.RowHeight = {'1x', 70};
gl.ColumnWidth = {'1x', 360};
gl.Padding = [10 10 10 10];
gl.BackgroundColor = 'k';

% Axes (row 1, col 1)
ax = uiaxes(gl);
ax.Layout.Row = 1; ax.Layout.Column = 1;
styleAxes(ax, gridColor);
xlabel(ax,'X (mm)'); ylabel(ax,'Y (mm)'); zlabel(ax,'Z (mm)');
title(ax,'Original (dashed) vs Deformed (solid) Arm by Tendon','Color','w');
view(ax, defaultView(1), defaultView(2));
hold(ax,'on');

% Side controls panel (row 1, col 2)
sidePanel = uipanel(gl, 'BackgroundColor','k', 'BorderType','none');
sidePanel.Layout.Row = 1; sidePanel.Layout.Column = 2;

% 2 header rows (lock view + tip label) + 9 geometry sliders
% + capstan checkbox + pull-side dropdown + mu slider = 14 rows
sideGL = uigridlayout(sidePanel, [14 3]);
sideGL.RowHeight = [24, 24, repmat(24,1,12)];
sideGL.ColumnWidth = {190, '1x', 110};
sideGL.Padding = [10 10 10 10];
sideGL.BackgroundColor = 'k';

% Row 1: Lock view checkbox (span columns 1..3)
lockViewCB = uicheckbox(sideGL, 'Text','Lock view (keep camera fixed)', 'FontColor','w', 'Value', true);
lockViewCB.Layout.Row = 1;
lockViewCB.Layout.Column = [1 3];

% Row 2: Tip position label (span columns 1..3)
tipLbl = uilabel(sideGL, 'Text','Tip (mm): X=—  Y=—  Z=—', 'FontColor','w');
tipLbl.Layout.Row = 2;
tipLbl.Layout.Column = [1 3];

% Helper to create a slider row in a given grid
    function [sld, valLbl] = addSliderRow(parentGL, row, name, lim, val)
        nameLbl = uilabel(parentGL, 'Text', name, 'FontColor','w');
        nameLbl.Layout.Row = row; nameLbl.Layout.Column = 1;

        sld = uislider(parentGL, 'Limits', lim, 'Value', val);
        sld.Layout.Row = row; sld.Layout.Column = 2;

        valLbl = uilabel(parentGL, 'Text', sprintf('%.3f', val), ...
            'FontColor','w', 'HorizontalAlignment','right');
        valLbl.Layout.Row = row; valLbl.Layout.Column = 3;
    end

% Geometry sliders (side) start at row 3
[sHeight, lHeight] = addSliderRow(sideGL, 3,  'height',          L.height,          D.height);
[sDepth,  lDepth]  = addSliderRow(sideGL, 4,  'depth',           L.depth,           D.depth);
[sOffset, lOffset] = addSliderRow(sideGL, 5,  'offset',          L.offset,          D.offset);

[sSNW, lSNW] = addSliderRow(sideGL, 6,  'startNotchWidth', L.startNotchWidth, D.startNotchWidth);
[sENW, lENW] = addSliderRow(sideGL, 7,  'endNotchWidth',   L.endNotchWidth,   D.endNotchWidth);

[sSS,  lSS]  = addSliderRow(sideGL, 8,  'startSpacing',    L.startSpacing,    D.startSpacing);
[sES,  lES]  = addSliderRow(sideGL, 9,  'endSpacing',      L.endSpacing,      D.endSpacing);

[sSOA, lSOA] = addSliderRow(sideGL, 10, 'startOffsetApex', L.startOffsetApex, D.startOffsetApex);
[sEOA, lEOA] = addSliderRow(sideGL, 11, 'endOffsetApex',   L.endOffsetApex,   D.endOffsetApex);

% Capstan/friction controls
capstanCB = uicheckbox(sideGL, 'Text','Use capstan friction', ...
    'FontColor','w', 'Value', useCapstan);
capstanCB.Layout.Row = 12;
capstanCB.Layout.Column = [1 3];

pullSideLbl = uilabel(sideGL, 'Text','pull side', 'FontColor','w');
pullSideLbl.Layout.Row = 13;
pullSideLbl.Layout.Column = 1;

pullSideDD = uidropdown(sideGL, 'Items', {'base','tip'}, 'Value', char(pullSide));
pullSideDD.Layout.Row = 13;
pullSideDD.Layout.Column = 2;

pullSideValueLbl = uilabel(sideGL, 'Text','', 'FontColor','w'); %#ok<NASGU>
pullSideValueLbl.Layout.Row = 13;
pullSideValueLbl.Layout.Column = 3;

[sMu, lMu] = addSliderRow(sideGL, 14, '\mu capstan', L.muCapstan, D.muCapstan);

% Bottom pull slider panel (row 2, spanning both cols)
pullPanel = uipanel(gl, 'BackgroundColor','k', 'BorderType','none');
pullPanel.Layout.Row = 2; pullPanel.Layout.Column = [1 2];

pullGL = uigridlayout(pullPanel, [1 3]);
pullGL.RowHeight = {24};
pullGL.ColumnWidth = {190, '1x', 110};
pullGL.Padding = [10 10 10 10];
pullGL.BackgroundColor = 'k';

[pullSld, pullValLbl] = addSliderRow(pullGL, 1, 'Pull \DeltaL', [0 1], 0);

%% ------------------- STATE -------------------
S.ax = ax;
S.defaultView = defaultView;

S.colUndef  = colUndef;
S.colDef    = colDef;
S.colTendon = colTendon;
S.gridColor = gridColor;
S.pinMarkerSize = pinMarkerSize;

S.distribution = distribution;
S.bend_dir = bend_dir;
S.useCapstan = useCapstan;
S.pullSide = pullSide;

S.lockViewCB = lockViewCB;
S.tipLbl = tipLbl;

S.pullSld = pullSld; S.pullValLbl = pullValLbl;

S.sHeight=sHeight; S.lHeight=lHeight;
S.sDepth=sDepth;   S.lDepth=lDepth;
S.sOffset=sOffset; S.lOffset=lOffset;

S.sSNW=sSNW; S.lSNW=lSNW;
S.sENW=sENW; S.lENW=lENW;

S.sSS=sSS;   S.lSS=lSS;
S.sES=sES;   S.lES=lES;

S.sSOA=sSOA; S.lSOA=lSOA;
S.sEOA=sEOA; S.lEOA=lEOA;

S.capstanCB = capstanCB;
S.pullSideDD = pullSideDD;
S.sMu = sMu; S.lMu = lMu;

%% ------------------- CALLBACKS -------------------
geomSliders = [sHeight,sDepth,sOffset,sSNW,sENW,sSS,sES,sSOA,sEOA,sMu];
for k = 1:numel(geomSliders)
    geomSliders(k).ValueChangingFcn = @(src,evt) updateAll(S, src, evt.Value);
    geomSliders(k).ValueChangedFcn  = @(src,evt) updateAll(S, src, src.Value);
end

pullSld.ValueChangingFcn = @(src,evt) updateAll(S, src, evt.Value);
pullSld.ValueChangedFcn  = @(src,evt) updateAll(S, src, src.Value);

% Redraw if lock-view toggled or capstan settings change
lockViewCB.ValueChangedFcn = @(src,evt) updateAll(S, src, S.pullSld.Value);
capstanCB.ValueChangedFcn = @(src,evt) updateAll(S, src, S.pullSld.Value);
pullSideDD.ValueChangedFcn = @(src,evt) updateAll(S, src, S.pullSld.Value);

%% ------------------- INITIAL DRAW -------------------
updateAll(S, pullSld, pullSld.Value);

end

%% ========================= UPDATE =========================
function updateAll(S, changedSlider, pullValueMaybe)
% Preserve current view if locked (or if user has adjusted it)
lockView = logical(S.lockViewCB.Value);
if lockView
    oldView = view(S.ax);
else
    oldView = S.defaultView;
end

% Read geometry
height          = S.sHeight.Value;
depth           = S.sDepth.Value;
offset          = S.sOffset.Value;

startNotchWidth = S.sSNW.Value;
endNotchWidth   = S.sENW.Value;

startSpacing    = S.sSS.Value;
endSpacing      = S.sES.Value;

startOffsetApex = S.sSOA.Value;
endOffsetApex   = S.sEOA.Value;

useCapstan = logical(S.capstanCB.Value);
pullSide   = string(S.pullSideDD.Value);
muCapstan  = S.sMu.Value;

% Update numeric labels
S.lHeight.Text = sprintf('%.3f', height);
S.lDepth.Text  = sprintf('%.3f', depth);
S.lOffset.Text = sprintf('%.3f', offset);
S.lSNW.Text    = sprintf('%.3f', startNotchWidth);
S.lENW.Text    = sprintf('%.3f', endNotchWidth);
S.lSS.Text     = sprintf('%.3f', startSpacing);
S.lES.Text     = sprintf('%.3f', endSpacing);
S.lSOA.Text    = sprintf('%.3f', startOffsetApex);
S.lEOA.Text    = sprintf('%.3f', endOffsetApex);
S.lMu.Text     = sprintf('%.3f', muCapstan);

% Build original geometry
[notches0, info] = generateNotchesOriginal( ...
    height, depth, offset, ...
    startNotchWidth, endNotchWidth, ...
    startSpacing, endSpacing, ...
    startOffsetApex, endOffsetApex);

N = numel(notches0);

% Clear axes and restyle
cla(S.ax); hold(S.ax,'on');
styleAxes(S.ax, S.gridColor);
xlabel(S.ax,'X (mm)'); ylabel(S.ax,'Y (mm)'); zlabel(S.ax,'Z (mm)');
title(S.ax,'Original (dashed) vs Deformed (solid) Arm by Tendon','Color','w');

% Restore view: locked => keep camera; unlocked => default
view(S.ax, oldView(1), oldView(2));

if N == 0
    S.tipLbl.Text = 'Tip (mm): X=—  Y=—  Z=—';
    S.pullSld.Limits = [0 1];
    S.pullSld.Value = 0;
    S.pullValLbl.Text = sprintf('%.3f', 0);
    drawnow limitrate;
    return;
end

pins0   = info.p1;      % [N x 3]
p0_list = info.p0;      % [N x 3]
p2_list = info.p2;      % [N x 3]
L_notch = info.L_notch; % [N x 1]

% Kinematic caps: initial opening angles at pins
theta0 = zeros(N,1);
for i = 1:N
    v0 = p0_list(i,:) - pins0(i,:);
    v2 = p2_list(i,:) - pins0(i,:);
    theta0(i) = angleBetween(v0, v2);
end

% Tendon moment arm (tendon through p2 guides)
d = abs(p2_list(:,1) - pins0(:,1));
DeltaL_max = sum(max(d,1e-12) .* max(theta0,0));
if DeltaL_max < 1e-9, DeltaL_max = 1.0; end

% Update pull slider limits
isPullSlider = isequal(changedSlider, S.pullSld);
if ~isPullSlider
    oldMax = S.pullSld.Limits(2);
    frac = 0;
    if oldMax > 0, frac = S.pullSld.Value / oldMax; end
    S.pullSld.Limits = [0 DeltaL_max];
    S.pullSld.Value  = min(max(frac * DeltaL_max, 0), DeltaL_max);
else
    S.pullSld.Limits = [0 DeltaL_max];
    S.pullSld.Value  = min(max(pullValueMaybe, 0), DeltaL_max);
end

DeltaL = S.pullSld.Value;
S.pullValLbl.Text = sprintf('%.3f', DeltaL);

% Solve angles with saturation.
% If capstan friction is enabled, distal/proximal notches receive less
% effective tendon pull depending on pullSide and local wrap angle.
if useCapstan
    theta = solveAnglesWithCapstanLimits( ...
        DeltaL, d, L_notch, theta0, S.distribution, ...
        muCapstan, pullSide);
else
    theta = solveAnglesWithLimits(DeltaL, d, L_notch, theta0, S.distribution);
end

% Deform pins (neutral axis)
[pinsDef, yawBefore, yawAfter] = deformPinsFromThetas(pins0, theta, S.bend_dir);

% Deform each notch: rotate original vectors about pin around +Y
p2_def = zeros(N,3);

for i = 1:N
    p0 = p0_list(i,:); p1 = pins0(i,:); p2 = p2_list(i,:);
    v0 = p0 - p1;  % pin->p0
    v2 = p2 - p1;  % pin->p2

    Rb = Ry(yawBefore(i));
    Ra = Ry(yawAfter(i));

    p1d = pinsDef(i,:);
    p0d = p1d + (Rb * v0(:)).';
    p2d = p1d + (Ra * v2(:)).';

    % Undeformed (light grey, solid, no markers)
    P0 = [p0; p1; p2];
    plot3(S.ax, P0(:,1), P0(:,2), P0(:,3), '-', 'LineWidth', 1.2, 'Color', S.colUndef);

    % Deformed (white, solid, no markers)
    Pd = [p0d; p1d; p2d];
    plot3(S.ax, Pd(:,1), Pd(:,2), Pd(:,3), '-', 'LineWidth', 2.2, 'Color', S.colDef);

    p2_def(i,:) = p2d;
end

% Pins only (smaller circles)
plot3(S.ax, pinsDef(:,1), pinsDef(:,2), pinsDef(:,3), 'o', ...
    'MarkerSize', S.pinMarkerSize, 'LineWidth', 1.2, ...
    'MarkerEdgeColor', S.colDef, 'MarkerFaceColor', 'none', 'Color', S.colDef);

% Tendon polyline (grey) through p2
plot3(S.ax, p2_def(:,1), p2_def(:,2), p2_def(:,3), '-', ...
    'LineWidth', 2.6, 'Color', S.colTendon);

% Tip marker + tip label (Cartesian)
tip = p2_def(end,:);
plot3(S.ax, tip(1), tip(2), tip(3), '^', 'MarkerSize', 7, ...
    'LineWidth', 1.2, 'Color', S.colDef, 'MarkerFaceColor', 'none');

S.tipLbl.Text = sprintf('Tip (mm): X=%.3f  Y=%.3f  Z=%.3f', tip(1), tip(2), tip(3));

% Keep the camera fixed if locked (user may have rotated after last draw)
if lockView
    view(S.ax, oldView(1), oldView(2));
else
    view(S.ax, S.defaultView(1), S.defaultView(2));
end

drawnow limitrate;
end

%% ========================= STYLING =========================
function styleAxes(ax, gridColor)
ax.Color = 'k';
ax.XColor = 'w'; ax.YColor = 'w'; ax.ZColor = 'w';
ax.GridColor = gridColor;
ax.MinorGridColor = gridColor;
grid(ax,'on');
axis(ax,'equal');
hold(ax,'on');
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
    p1 = [ -offset, 0.0, start_z + 0.5*L + apex_z_offset ]; % pin / neutral axis
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

%% ========================= SOLVER / KINEMATICS =========================
function theta = solveAnglesWithLimits(DeltaL, d, L_notch, theta0, distribution)
N = numel(d);
theta = zeros(N,1);
if N == 0 || DeltaL <= 0, return; end

switch string(distribution)
    case "uniform"
        w = ones(N,1);
    otherwise
        w = max(L_notch(:), 1e-12);
end

theta = solveAnglesWeightedWithLimits(DeltaL, d, theta0, w);
end

function theta = solveAnglesWithCapstanLimits( ...
    DeltaL, d, L_notch, theta0, distribution, mu, pullSide)
% Kinematic capstan approximation for a tendon routed through the notch tips.
%
% Capstan law:
%   T_out = T_in * exp(-mu * beta)
%
% Here beta is approximated as the local notch closing angle. The resulting
% relative tension distribution is used as an effective weighting for how
% much of the commanded tendon shortening is assigned to each notch.

N = numel(d);
theta = zeros(N,1);

if N == 0 || DeltaL <= 0
    return;
end

switch string(distribution)
    case "uniform"
        baseW = ones(N,1);
    otherwise
        baseW = max(L_notch(:), 1e-12);
end

if mu <= 0
    theta = solveAnglesWeightedWithLimits(DeltaL, d, theta0, baseW);
    return;
end

% Start from the frictionless solution, then iterate because the frictional
% tension decay depends on wrap angle, which depends on theta.
theta = solveAnglesWeightedWithLimits(DeltaL, d, theta0, baseW);

maxIter = 40;
tol     = 1e-6;
relax   = 0.5;

for it = 1:maxIter %#ok<NASGU>
    thetaOld = theta;

    % Local tendon wrap angle approximation at each notch.
    beta = abs(thetaOld);

    % Relative local tendon tension according to pull direction.
    Trel = capstanRelativeTension(beta, mu, pullSide);

    % Less tension means that notch gets a smaller share of the global pull.
    wEff = baseW .* Trel;

    thetaNew = solveAnglesWeightedWithLimits(DeltaL, d, theta0, wEff);

    % Under-relaxation improves numerical stability when mu is high.
    theta = (1 - relax)*thetaOld + relax*thetaNew;

    if norm(theta - thetaOld, inf) < tol
        break;
    end
end

thetaCap = max(theta0(:), 0);
theta = min(theta, thetaCap);
theta = max(theta, 0);
end

function Trel = capstanRelativeTension(beta, mu, pullSide)
beta = beta(:);
N = numel(beta);
Trel = ones(N,1);

switch string(pullSide)
    case "base"
        % Tendon is pulled from notch 1 toward notch N.
        % Notch 1 sees the highest tension; distal notches see reduced tension.
        accumulatedWrap = 0;
        for i = 1:N
            Trel(i) = exp(-mu * accumulatedWrap);
            accumulatedWrap = accumulatedWrap + beta(i);
        end

    case "tip"
        % Tendon is pulled from notch N toward notch 1.
        % Tip notch sees the highest tension; proximal notches see reduced tension.
        accumulatedWrap = 0;
        for i = N:-1:1
            Trel(i) = exp(-mu * accumulatedWrap);
            accumulatedWrap = accumulatedWrap + beta(i);
        end

    otherwise
        error('pullSide must be "base" or "tip".');
end

% Avoid a completely zero weight in the downstream allocation solver.
Trel = max(Trel, 1e-6);
end

function theta = solveAnglesWeightedWithLimits(DeltaL, d, theta0, w)
N = numel(d);
theta = zeros(N,1);

if N == 0 || DeltaL <= 0
    return;
end

d_safe = max(d(:), 1e-12);
thetaCap = max(theta0(:), 0);

w = max(w(:), 0);
if sum(w) < 1e-12
    return;
end

free = true(N,1);
remaining = DeltaL;

while remaining > 1e-12 && any(free)
    wf = w .* free;
    wf_sum = sum(wf);
    if wf_sum < 1e-12, break; end

    DeltaL_i = remaining * (wf / wf_sum);
    theta_prop = DeltaL_i ./ d_safe;

    saturate = free & (theta + theta_prop > thetaCap + 1e-12);
    if ~any(saturate)
        theta(free) = theta(free) + theta_prop(free);
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

theta = min(theta, thetaCap);
theta = max(theta, 0);
end

function [pinsDef, yawBefore, yawAfter] = deformPinsFromThetas(pins0, theta, bend_dir)
N = size(pins0,1);
pinsDef = zeros(N,3);
yawBefore = zeros(N,1);
yawAfter  = zeros(N,1);

xNA = pins0(1,1);
p = [xNA, 0, 0];
yaw = 0;

% base -> first pin (originally along Z)
L0 = pins0(1,3);
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

%% ========================= MATH HELPERS =========================
function v = stepXZ(s, yaw), v = [s*sin(yaw), 0, s*cos(yaw)]; end

function R = Ry(yaw)
c = cos(yaw); s = sin(yaw);
R = [ c  0  s;
      0  1  0;
     -s  0  c ];
end

function ang = angleBetween(a,b)
na = norm(a); nb = norm(b);
if na < 1e-12 || nb < 1e-12, ang = 0; return; end
c = dot(a,b)/(na*nb);
c = max(min(c,1),-1);
ang = acos(c);
end

function y = lerp(a,b,t), y = a + t*(b-a); end
