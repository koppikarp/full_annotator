// Client Application JavaScript for POPCROP Base Caller & Signal Validator

let state = {
    mode: 'base_caller', // 'base_caller' or 'blob_validator'
    showFishProbe: false,
    config: null,
    cells: [],
    currentIndex: 0,
    currentCycle: 1,
    currentTile: 14,
    isStarted: false,
    nuclearDiameter: 55.0,
    hysteresisHigh: 4.0,
    imageWidth: 2048,
    imageHeight: 2048,

    // LUT State (min/max ADU)
    greenMin: 0,
    greenMax: 65535,
    redMin: 0,
    redMax: 65535,
    dapiMin: 0,
    dapiMax: 65535,
    c0FarredMin: 0,
    c0FarredMax: 65535,
    c0DapiMin: 0,
    c0DapiMax: 65535,

    greenLut: 'green',
    redLut: 'red',
    dapiLut: 'dapi',
    greenMode: 'pseudocolor',
    redMode: 'pseudocolor',
    greenBlue: false,
    redBlue: false,
    overlayBlue: true,
    cycle0OverlayBlue: true,

    // Contours toggle (Independent Per-Channel and Nuclear Outline)
    showNucContour: true,
    showGreenContour: true,
    showRedContour: true,
    showFovDots: true,

    // Request synchronization
    cropRequestId: 0,
    fovRequestId: 0,
    cropAbortController: null,
    fovAbortController: null,
    isAnnotating: false,

    // Progression Mode ('cell' for cell A -> cell B, 'cycle' for cycle 1..12 of same cell)
    progressionMode: 'cell',

    // Cached preloaded images & crop data
    fovImagesLoaded: false,
    fovImages: {
        green: new Image(),
        red: new Image(),
        overlay: new Image(),
        cycle0: new Image()
    },
    activeCropData: null,
    lutProfiles: {}
};

// Global Debounce Timers for Smooth Slider Interactions
let refreshImagesTimer = null;

function debouncedRefreshImages(delay = 130, isLutOnly = true) {
    if (refreshImagesTimer) {
        clearTimeout(refreshImagesTimer);
    }
    refreshImagesTimer = setTimeout(() => {
        refreshImagesTimer = null;
        refreshImages(isLutOnly);
    }, delay);
}

function flushRefreshImages(isLutOnly = true) {
    if (refreshImagesTimer) {
        clearTimeout(refreshImagesTimer);
        refreshImagesTimer = null;
    }
    refreshImages(isLutOnly);
}

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    setupWorkflowTabs();
    setupFishProbeToggle();
    setupLUTDrawer();
    setupDragAndDrop();
    setupControls();
    setupHotkeys();

    const initTimer = setTimeout(() => {
        showDatasetLoading('Loading POPCROP Workspace', 'Initializing microscopy dataset & nuclei...', false);
        setLoadingProgress(60, 'Loading dataset & rendering full-FOV projections...');
    }, 350);

    try {
        await fetchConfig();
        await loadLutProfiles();
        await loadDataset();
    } finally {
        clearTimeout(initTimer);
        hideDatasetLoading(300);
    }
}

function setupWorkflowTabs() {
    const tabBase = document.getElementById('tab-base-caller');
    const tabBlob = document.getElementById('tab-blob-validator');
    const secBase = document.getElementById('base-caller-workflow-section');
    const secBlob = document.getElementById('blob-validator-workflow-section');
    const csvBadge = document.getElementById('output-csv-badge');

    const setMode = (mode) => {
        state.mode = mode;
        if (mode === 'base_caller') {
            tabBase.classList.add('active');
            tabBlob.classList.remove('active');
            secBase.style.display = 'block';
            secBlob.style.display = 'none';
            if (csvBadge) csvBadge.innerText = 'annotated_base_ground_truth.csv';
            document.getElementById('table-title').innerHTML = `Annotated Base Ground Truth (<span id="gt-count">0</span> records)`;
        } else {
            tabBlob.classList.add('active');
            tabBase.classList.remove('active');
            secBlob.style.display = 'block';
            secBase.style.display = 'none';
            if (csvBadge) csvBadge.innerText = 'annotated_blob_ground_truth.csv';
            document.getElementById('table-title').innerHTML = `Annotated Blob Signal Ground Truth (<span id="gt-count">0</span> records)`;
        }
        refreshProgressUI();
        updateGroundTruthTable();
    };

    tabBase.addEventListener('click', () => setMode('base_caller'));
    tabBlob.addEventListener('click', () => setMode('blob_validator'));

    const initialMode = document.body.getAttribute('data-mode') || 'base_caller';
    setMode(initialMode);
}

function setupFishProbeToggle() {
    const btn = document.getElementById('fish-probe-toggle-btn');
    const statusText = document.getElementById('fish-status-text');
    const fovGrid = document.getElementById('fov-grid');
    const cropGrid = document.getElementById('crop-grid');
    const fovC0Panel = document.getElementById('fov-cycle0-panel');
    const cropC0Panel = document.getElementById('crop-cycle0-panel');
    const lutC0Card = document.getElementById('c0-lut-card');

    btn.addEventListener('click', () => {
        state.showFishProbe = !state.showFishProbe;
        if (state.showFishProbe) {
            btn.classList.add('active');
            statusText.innerText = 'ON';
            fovC0Panel.style.display = 'flex';
            cropC0Panel.style.display = 'flex';
            if (lutC0Card) lutC0Card.style.display = 'flex';

            fovGrid.classList.remove('grid-3col');
            fovGrid.classList.add('grid-4col');
            cropGrid.classList.remove('grid-3col');
            cropGrid.classList.add('grid-4col');
        } else {
            btn.classList.remove('active');
            statusText.innerText = 'OFF';
            fovC0Panel.style.display = 'none';
            cropC0Panel.style.display = 'none';
            if (lutC0Card) lutC0Card.style.display = 'none';

            fovGrid.classList.remove('grid-4col');
            fovGrid.classList.add('grid-3col');
            cropGrid.classList.remove('grid-4col');
            cropGrid.classList.add('grid-3col');
        }
        renderAllFovCanvases();
        renderHighPowerCrop(state.cropRequestId);
    });
}

function showCropLoading(isLoading) {
    ['green', 'red', 'composite', 'cycle0'].forEach(ch => {
        const overlay = document.getElementById(`${ch}-loading-overlay`);
        if (overlay) {
            overlay.style.display = isLoading ? 'flex' : 'none';
        }
        if (isLoading) {
            const canvas = document.getElementById(`crop-${ch}-canvas`);
            if (canvas) {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }
    });
}

function showFovLoading(isLoading) {
    ['green', 'red', 'overlay', 'cycle0'].forEach(ch => {
        const overlay = document.getElementById(`fov-${ch}-loading-overlay`);
        if (overlay) {
            overlay.style.display = isLoading ? 'flex' : 'none';
        }
        if (isLoading) {
            const canvas = document.getElementById(`fov-${ch}-canvas`);
            if (canvas) {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }
    });
}

function setProgressionMode(mode) {
    if (mode !== 'cell' && mode !== 'cycle') return;
    state.progressionMode = mode;
    document.querySelectorAll('.segmented-control button[data-prog]').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-prog') === mode);
    });
    const isCycle = (mode === 'cycle');
    const baseHelp = document.getElementById('base-advance-help-text');
    if (baseHelp) {
        baseHelp.innerHTML = isCycle
            ? 'Press keyboard key (<kbd>A</kbd>, <kbd>T</kbd>, <kbd>C</kbd>, <kbd>G</kbd>, <kbd>N</kbd>) to annotate and advance to next cycle'
            : 'Press keyboard key (<kbd>A</kbd>, <kbd>T</kbd>, <kbd>C</kbd>, <kbd>G</kbd>, <kbd>N</kbd>) to annotate and advance to next cell';
    }
    const blobHelp = document.getElementById('blob-advance-help-text');
    if (blobHelp) {
        blobHelp.innerHTML = isCycle
            ? 'Verify channel accuracy: (<kbd>B</kbd>, <kbd>G</kbd>, <kbd>R</kbd>, <kbd>N</kbd>) to annotate and advance to next cycle'
            : 'Verify channel accuracy: (<kbd>B</kbd>, <kbd>G</kbd>, <kbd>R</kbd>, <kbd>N</kbd>) to annotate and advance to next cell';
    }
}

function toggleProgressionMode() {
    setProgressionMode(state.progressionMode === 'cell' ? 'cycle' : 'cell');
}

function setupControls() {
    // Start button
    document.getElementById('start-btn').addEventListener('click', () => {
        state.isStarted = true;
        document.getElementById('start-btn').classList.remove('btn-glow');
        document.getElementById('start-btn').innerText = '✓ ANNOTATING';
        if (state.cells.length > 0) {
            displayCell(state.currentIndex);
        }
    });

    // Cycle selector
    document.getElementById('cycle-select').addEventListener('change', (e) => {
        changeCycle(parseInt(e.target.value));
    });

    // Nuclear diameter
    document.getElementById('nuclear-dia-input').addEventListener('change', (e) => {
        state.nuclearDiameter = parseFloat(e.target.value) || 55.0;
        if (state.cells.length > 0) {
            fetchCellCrop(state.cells[state.currentIndex].cell_id);
        }
    });

    // Hysteresis threshold
    const hSlider = document.getElementById('hysteresis-high-slider');
    const hBadge = document.getElementById('hysteresis-val-badge');
    hSlider.addEventListener('input', (e) => {
        state.hysteresisHigh = parseFloat(e.target.value);
        hBadge.innerText = `${state.hysteresisHigh.toFixed(1)} σ`;
    });
    hSlider.addEventListener('change', () => {
        if (state.cells.length > 0) {
            fetchCellCrop(state.cells[state.currentIndex].cell_id);
        }
    });

    // Contours toggle (Separate Nuclear, Green Spot, Red Spot)
    const nucCb = document.getElementById('toggle-nuc-contour-cb');
    if (nucCb) {
        nucCb.addEventListener('change', (e) => {
            state.showNucContour = e.target.checked;
            renderHighPowerCrop(state.cropRequestId);
        });
    }
    const greenCb = document.getElementById('toggle-green-contour-cb');
    if (greenCb) {
        greenCb.addEventListener('change', (e) => {
            state.showGreenContour = e.target.checked;
            renderHighPowerCrop(state.cropRequestId);
        });
    }
    const redCb = document.getElementById('toggle-red-contour-cb');
    if (redCb) {
        redCb.addEventListener('change', (e) => {
            state.showRedContour = e.target.checked;
            renderHighPowerCrop(state.cropRequestId);
        });
    }

    // Low Power FOV dots toggle
    const fovDotsCb = document.getElementById('toggle-fov-dots-cb');
    if (fovDotsCb) {
        fovDotsCb.addEventListener('change', (e) => {
            state.showFovDots = e.target.checked;
            renderAllFovCanvases();
        });
    }

    // Pseudocolor / Grayscale mode toggles
    const setupChannelModeToggle = (ch) => {
        const toggle = () => {
            const prop = `${ch}Mode`;
            state[prop] = state[prop] === 'pseudocolor' ? 'greyscale' : 'pseudocolor';
            ['fov', 'crop'].forEach(prefix => {
                const btn = document.getElementById(`${prefix}-${ch}-mode-btn`);
                if (btn) {
                    btn.innerText = state[prop] === 'pseudocolor' ? 'Pseudocolor' : 'Greyscale';
                    btn.classList.toggle('active', state[prop] === 'pseudocolor');
                }
            });
            flushRefreshImages(true);
        };
        const fovBtn = document.getElementById(`fov-${ch}-mode-btn`);
        if (fovBtn) fovBtn.addEventListener('click', toggle);
        const cropBtn = document.getElementById(`crop-${ch}-mode-btn`);
        if (cropBtn) cropBtn.addEventListener('click', toggle);
    };
    setupChannelModeToggle('green');
    setupChannelModeToggle('red');

    // Blue DAPI in Overlay toggles
    const fovBlueCb = document.getElementById('fov-overlay-blue-cb');
    const cropBlueCb = document.getElementById('crop-overlay-blue-cb');
    if (fovBlueCb) {
        fovBlueCb.addEventListener('change', (e) => {
            state.overlayBlue = e.target.checked;
            if (cropBlueCb) cropBlueCb.checked = state.overlayBlue;
            flushRefreshImages(true);
        });
    }
    if (cropBlueCb) {
        cropBlueCb.addEventListener('change', (e) => {
            state.overlayBlue = e.target.checked;
            if (fovBlueCb) fovBlueCb.checked = state.overlayBlue;
            flushRefreshImages(true);
        });
    }

    // Navigation
    document.getElementById('prev-btn').addEventListener('click', prevCell);
    document.getElementById('next-btn').addEventListener('click', nextCell);
    document.getElementById('skip-btn').addEventListener('click', advanceAfterAnnotation);

    // Progression segmented controls
    document.querySelectorAll('.segmented-control button[data-prog]').forEach(btn => {
        btn.addEventListener('click', () => {
            setProgressionMode(btn.getAttribute('data-prog'));
        });
    });

    // Base call buttons (A, T, C, G, N)
    document.querySelectorAll('.base-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const base = btn.getAttribute('data-base');
            annotateBaseCall(base);
        });
    });

    // Blob validator buttons (Both, Only Green, Only Red, Neither)
    const btnBoth = document.getElementById('btn-call-both');
    const btnOnlyG = document.getElementById('btn-call-only-green');
    const btnOnlyR = document.getElementById('btn-call-only-red');
    const btnNeither = document.getElementById('btn-call-neither');

    if (btnBoth) btnBoth.addEventListener('click', () => annotateBlobSignal('both'));
    if (btnOnlyG) btnOnlyG.addEventListener('click', () => annotateBlobSignal('only_green'));
    if (btnOnlyR) btnOnlyR.addEventListener('click', () => annotateBlobSignal('only_red'));
    if (btnNeither) btnNeither.addEventListener('click', () => annotateBlobSignal('neither'));

    // Refresh and clear table
    document.getElementById('refresh-gt-btn').addEventListener('click', updateGroundTruthTable);
    document.getElementById('clear-gt-btn').addEventListener('click', clearCurrentGroundTruth);

    // FOV canvas clicks
    ['green', 'red', 'overlay', 'cycle0'].forEach(ch => {
        const canvas = document.getElementById(`fov-${ch}-canvas`);
        if (canvas) {
            canvas.addEventListener('click', (e) => handleFovClick(e, canvas));
        }
    });
}

async function changeCycle(newCycle, forceReload = false) {
    const available = (state.config && state.config.available_cycles && state.config.available_cycles.length > 0)
        ? state.config.available_cycles
        : [1];
    const minCycle = Math.min(...available);
    const maxCycle = Math.max(...available);
    if (newCycle < minCycle) newCycle = minCycle;
    if (newCycle > maxCycle) newCycle = maxCycle;
    if (newCycle === state.currentCycle && !forceReload) return;
    
    state.currentCycle = newCycle;
    const cycleSel = document.getElementById('cycle-select');
    if (cycleSel) cycleSel.value = newCycle;
    const saveCycleLbl = document.getElementById('save-cycle-lbl');
    if (saveCycleLbl) saveCycleLbl.innerText = newCycle;
    
    // Apply LUT controls for the new cycle without triggering duplicate refreshImages()
    applyLutSettingsForCycle(state.currentCycle, false);
    
    // Show immediate loading feedback
    showCropLoading(true);
    await loadDataset();
}

function setupHotkeys() {
    window.addEventListener('keydown', (e) => {
        // Avoid hotkeys when typing in text inputs
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

        const key = e.key.toUpperCase();

        if (state.mode === 'base_caller') {
            if (['A', 'T', 'C', 'G', 'N'].includes(key)) {
                e.preventDefault();
                annotateBaseCall(key);
            }
        } else if (state.mode === 'blob_validator') {
            if (key === 'B' || key === '1' || key === 'Y') {
                e.preventDefault();
                annotateBlobSignal('both');
            } else if (key === 'G' || key === '2') {
                e.preventDefault();
                annotateBlobSignal('only_green');
            } else if (key === 'R' || key === '3') {
                e.preventDefault();
                annotateBlobSignal('only_red');
            } else if (key === 'N' || key === '0' || key === 'X') {
                e.preventDefault();
                annotateBlobSignal('neither');
            }
        }

        // Toggle progression mode with 'P'
        if (key === 'P') {
            e.preventDefault();
            toggleProgressionMode();
        }

        // Toggle FOV dots with 'D'
        if (key === 'D') {
            e.preventDefault();
            state.showFovDots = !state.showFovDots;
            const fovDotsCb = document.getElementById('toggle-fov-dots-cb');
            if (fovDotsCb) fovDotsCb.checked = state.showFovDots;
            renderAllFovCanvases();
        }

        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            prevCell();
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            nextCell();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            changeCycle(state.currentCycle + 1);
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            changeCycle(state.currentCycle - 1);
        }
    });
}

function setupLUTDrawer() {
    const sidebar = document.getElementById('lut-sidebar');
    const handle = document.getElementById('lut-sidebar-handle');
    if (!handle || !sidebar) return;
    handle.addEventListener('click', () => {
        const isCollapsed = sidebar.classList.toggle('collapsed');
        document.body.classList.toggle('sidebar-collapsed', isCollapsed);
        handle.querySelector('.tab-arrow').innerText = isCollapsed ? '◀' : '▶';
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
            renderAllFovCanvases();
        }, 260);
    });

    const bindSlider = (minSliderId, maxSliderId, minInputId, maxInputId, propMin, propMax) => {
        const minS = document.getElementById(minSliderId);
        const maxS = document.getElementById(maxSliderId);
        const minI = document.getElementById(minInputId);
        const maxI = document.getElementById(maxInputId);

        const sync = (vMin, vMax, immediate = false) => {
            state[propMin] = vMin;
            state[propMax] = vMax;
            if (minS) minS.value = vMin;
            if (maxS) maxS.value = vMax;
            if (minI) minI.value = vMin;
            if (maxI) maxI.value = vMax;
            if (immediate) {
                flushRefreshImages(true);
            } else {
                debouncedRefreshImages(130, true);
            }
        };

        if (minS) {
            minS.addEventListener('input', (e) => sync(Math.min(parseInt(e.target.value), state[propMax] - 1), state[propMax], false));
            minS.addEventListener('change', (e) => sync(Math.min(parseInt(e.target.value), state[propMax] - 1), state[propMax], true));
        }
        if (maxS) {
            maxS.addEventListener('input', (e) => sync(state[propMin], Math.max(parseInt(e.target.value), state[propMin] + 1), false));
            maxS.addEventListener('change', (e) => sync(state[propMin], Math.max(parseInt(e.target.value), state[propMin] + 1), true));
        }
        if (minI) minI.addEventListener('change', (e) => sync(Math.min(parseInt(e.target.value) || 0, state[propMax] - 1), state[propMax], true));
        if (maxI) maxI.addEventListener('change', (e) => sync(state[propMin], Math.max(parseInt(e.target.value) || 65535, state[propMin] + 1), true));
    };

    bindSlider('green-min-slider', 'green-max-slider', 'green-min-input', 'green-max-input', 'greenMin', 'greenMax');
    bindSlider('red-min-slider', 'red-max-slider', 'red-min-input', 'red-max-input', 'redMin', 'redMax');
    bindSlider('dapi-min-slider', 'dapi-max-slider', 'dapi-min-input', 'dapi-max-input', 'dapiMin', 'dapiMax');
    bindSlider('c0-farred-min-slider', 'c0-farred-max-slider', 'c0-farred-min-input', 'c0-farred-max-input', 'c0FarredMin', 'c0FarredMax');
    bindSlider('c0-dapi-min-slider', 'c0-dapi-max-slider', 'c0-dapi-min-input', 'c0-dapi-max-input', 'c0DapiMin', 'c0DapiMax');

    // Colormap dropdowns
    const gLutSel = document.getElementById('green-lut-select');
    if (gLutSel) {
        gLutSel.addEventListener('change', (e) => {
            state.greenLut = e.target.value;
            flushRefreshImages(true);
        });
    }
    const rLutSel = document.getElementById('red-lut-select');
    if (rLutSel) {
        rLutSel.addEventListener('change', (e) => {
            state.redLut = e.target.value;
            flushRefreshImages(true);
        });
    }
    const dLutSel = document.getElementById('dapi-lut-select');
    if (dLutSel) {
        dLutSel.addEventListener('change', (e) => {
            state.dapiLut = e.target.value;
            flushRefreshImages(true);
        });
    }

    // Save LUT settings for current cycle
    document.getElementById('save-luts-btn').addEventListener('click', saveLutSettingsForCycle);
}

function refreshImages(isLutOnly = false) {
    fetchFullFovImages();
    if (state.cells.length > 0) {
        fetchCellCrop(state.cells[state.currentIndex].cell_id, isLutOnly);
    }
}

// --- Per-Cycle, Per-Channel LUT Persistence ---

async function loadLutProfiles() {
    try {
        const res = await fetch(`/api/lut_profile?_t=${Date.now()}`);
        const data = await res.json();
        if (data.success && data.profile) {
            state.lutProfiles = data.profile;
            applyLutSettingsForCycle(state.currentCycle, false);
        }
    } catch (e) {
        console.warn('Could not load LUT profiles:', e);
    }
}

function applyLutSettingsForCycle(cycle, triggerRefresh = true) {
    const cycleKey = String(cycle);
    const cycles = state.lutProfiles.cycles || {};
    const settings = cycles[cycleKey];
    if (settings) {
        // Green channel (per cycle)
        if (settings.greenMin !== undefined) state.greenMin = settings.greenMin;
        if (settings.greenMax !== undefined) state.greenMax = settings.greenMax;
        if (settings.greenLut) state.greenLut = settings.greenLut;
        if (settings.greenMode) state.greenMode = settings.greenMode;
        if (settings.greenBlue !== undefined) state.greenBlue = settings.greenBlue;

        // Red channel (per cycle)
        if (settings.redMin !== undefined) state.redMin = settings.redMin;
        if (settings.redMax !== undefined) state.redMax = settings.redMax;
        if (settings.redLut) state.redLut = settings.redLut;
        if (settings.redMode) state.redMode = settings.redMode;
        if (settings.redBlue !== undefined) state.redBlue = settings.redBlue;

        // DAPI channel (per cycle)
        if (settings.dapiMin !== undefined) state.dapiMin = settings.dapiMin;
        if (settings.dapiMax !== undefined) state.dapiMax = settings.dapiMax;
        if (settings.dapiLut) state.dapiLut = settings.dapiLut;

        if (settings.overlayBlue !== undefined) state.overlayBlue = settings.overlayBlue;

        // Update UI inputs and sliders
        ['green', 'red', 'dapi'].forEach(ch => {
            const propMin = `${ch}Min`;
            const propMax = `${ch}Max`;
            const sMin = document.getElementById(`${ch}-min-slider`);
            const sMax = document.getElementById(`${ch}-max-slider`);
            const iMin = document.getElementById(`${ch}-min-input`);
            const iMax = document.getElementById(`${ch}-max-input`);
            if (sMin) sMin.value = state[propMin];
            if (sMax) sMax.value = state[propMax];
            if (iMin) iMin.value = state[propMin];
            if (iMax) iMax.value = state[propMax];
        });

        // Update colormap dropdowns
        const gLutSel = document.getElementById('green-lut-select');
        if (gLutSel && state.greenLut) gLutSel.value = state.greenLut;
        const rLutSel = document.getElementById('red-lut-select');
        if (rLutSel && state.redLut) rLutSel.value = state.redLut;
        const dLutSel = document.getElementById('dapi-lut-select');
        if (dLutSel && state.dapiLut) dLutSel.value = state.dapiLut;

        // Sync mode buttons
        const setModeBtn = (btnId, mode) => {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.innerText = mode === 'pseudocolor' ? 'Pseudocolor' : 'Greyscale';
                btn.classList.toggle('active', mode === 'pseudocolor');
            }
        };
        setModeBtn('crop-green-mode-btn', state.greenMode);
        setModeBtn('fov-green-mode-btn', state.greenMode);
        setModeBtn('crop-red-mode-btn', state.redMode);
        setModeBtn('fov-red-mode-btn', state.redMode);

        const fovBlueCb = document.getElementById('fov-overlay-blue-cb');
        const cropBlueCb = document.getElementById('crop-overlay-blue-cb');
        if (fovBlueCb) fovBlueCb.checked = state.overlayBlue;
        if (cropBlueCb) cropBlueCb.checked = state.overlayBlue;
    }

    // Cycle 0 settings (anchor channel)
    const c0 = state.lutProfiles.cycle0 || {};
    if (c0.cycle0FarredMin !== undefined) state.c0FarredMin = c0.cycle0FarredMin;
    else if (c0.c0FarredMin !== undefined) state.c0FarredMin = c0.c0FarredMin;
    if (c0.cycle0FarredMax !== undefined) state.c0FarredMax = c0.cycle0FarredMax;
    else if (c0.c0FarredMax !== undefined) state.c0FarredMax = c0.c0FarredMax;

    if (c0.cycle0DapiMin !== undefined) state.c0DapiMin = c0.cycle0DapiMin;
    else if (c0.c0DapiMin !== undefined) state.c0DapiMin = c0.c0DapiMin;
    if (c0.cycle0DapiMax !== undefined) state.c0DapiMax = c0.cycle0DapiMax;
    else if (c0.c0DapiMax !== undefined) state.c0DapiMax = c0.c0DapiMax;

    if (c0.cycle0OverlayBlue !== undefined) state.cycle0OverlayBlue = c0.cycle0OverlayBlue;

    const c0FrMinS = document.getElementById('c0-farred-min-slider');
    const c0FrMaxS = document.getElementById('c0-farred-max-slider');
    const c0FrMinI = document.getElementById('c0-farred-min-input');
    const c0FrMaxI = document.getElementById('c0-farred-max-input');
    if (c0FrMinS) c0FrMinS.value = state.c0FarredMin;
    if (c0FrMaxS) c0FrMaxS.value = state.c0FarredMax;
    if (c0FrMinI) c0FrMinI.value = state.c0FarredMin;
    if (c0FrMaxI) c0FrMaxI.value = state.c0FarredMax;

    const c0DapiMinS = document.getElementById('c0-dapi-min-slider');
    const c0DapiMaxS = document.getElementById('c0-dapi-max-slider');
    const c0DapiMinI = document.getElementById('c0-dapi-min-input');
    const c0DapiMaxI = document.getElementById('c0-dapi-max-input');
    if (c0DapiMinS) c0DapiMinS.value = state.c0DapiMin;
    if (c0DapiMaxS) c0DapiMaxS.value = state.c0DapiMax;
    if (c0DapiMinI) c0DapiMinI.value = state.c0DapiMin;
    if (c0DapiMaxI) c0DapiMaxI.value = state.c0DapiMax;

    const subtitle = document.querySelector('.sidebar-subtitle');
    if (subtitle) {
        subtitle.innerText = `Cycle ${cycle} LUT (Per-channel per-cycle)`;
    }
    const saveCycleLbl = document.getElementById('save-cycle-lbl');
    if (saveCycleLbl) {
        saveCycleLbl.innerText = cycle;
    }

    if (triggerRefresh) {
        refreshImages();
    }
}

async function saveLutSettingsForCycle() {
    const cycleKey = String(state.currentCycle);
    const payload = {
        cycle: state.currentCycle,
        settings: {
            greenMin: state.greenMin,
            greenMax: state.greenMax,
            greenLut: state.greenLut,
            greenMode: state.greenMode,
            greenBlue: state.greenBlue,
            redMin: state.redMin,
            redMax: state.redMax,
            redLut: state.redLut,
            redMode: state.redMode,
            redBlue: state.redBlue,
            dapiMin: state.dapiMin,
            dapiMax: state.dapiMax,
            dapiLut: state.dapiLut,
            overlayBlue: state.overlayBlue
        },
        cycle0_settings: {
            cycle0FarredMin: state.c0FarredMin,
            cycle0FarredMax: state.c0FarredMax,
            cycle0DapiMin: state.c0DapiMin,
            cycle0DapiMax: state.c0DapiMax,
            cycle0OverlayBlue: state.cycle0OverlayBlue
        }
    };

    try {
        const res = await fetch('/api/lut_profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            state.lutProfiles.cycles = state.lutProfiles.cycles || {};
            state.lutProfiles.cycles[cycleKey] = payload.settings;
            state.lutProfiles.cycle0 = payload.cycle0_settings;
            const saveBtn = document.getElementById('save-luts-btn');
            const origText = saveBtn.innerText;
            saveBtn.innerText = `✓ SAVED CYCLE ${state.currentCycle}!`;
            setTimeout(() => { saveBtn.innerText = origText; }, 1400);
        }
    } catch (err) {
        console.error('Error saving LUT settings:', err);
    }
}

// --- Dataset & FOV Loading ---

async function fetchConfig() {
    try {
        const res = await fetch(`/api/config?_t=${Date.now()}`);
        const data = await res.json();
        if (!state.config && Number.isFinite(data.nuclear_diameter)) {
            state.nuclearDiameter = data.nuclear_diameter;
            document.getElementById('nuclear-dia-input').value = state.nuclearDiameter;
        }
        state.config = data;
        if (data.image_width) state.imageWidth = data.image_width;
        if (data.image_height) state.imageHeight = data.image_height;
        document.getElementById('tiff-filename-badge').innerText = data.tiff_filename;

        const cycleSelect = document.getElementById('cycle-select');
        cycleSelect.innerHTML = '';
        (data.available_cycles || [1]).forEach(c => {
            const opt = document.createElement('option');
            opt.value = c;
            opt.innerText = `Cycle ${c}`;
            cycleSelect.appendChild(opt);
        });
    } catch (e) {
        console.error('Config fetch failed:', e);
    }
}

async function loadDataset() {
    try {
        const res = await fetch(`/api/cells?cycle=${state.currentCycle}&_t=${Date.now()}`);
        const data = await res.json();
        state.cells = data.cells || [];
        updateProgressUI(data.annotated_base_count, data.annotated_blob_count, data.total_cells);
        fetchFullFovImages();
        if (state.cells.length > 0) {
            displayCell(state.currentIndex);
        }
        updateGroundTruthTable();
    } catch (e) {
        console.error('Cells fetch failed:', e);
    }
}

async function fetchFullFovImages() {
    state.fovRequestId++;
    const reqId = state.fovRequestId;

    state.fovImagesLoaded = false;
    showFovLoading(true);

    if (state.fovAbortController) {
        try {
            state.fovAbortController.abort();
        } catch (_) {}
    }
    state.fovAbortController = new AbortController();

    const params = new URLSearchParams({
        cycle: state.currentCycle,
        green_min: state.greenMin, green_max: state.greenMax,
        red_min: state.redMin, red_max: state.redMax,
        dapi_min: state.dapiMin, dapi_max: state.dapiMax,
        c0_farred_min: state.c0FarredMin, c0_farred_max: state.c0FarredMax,
        c0_dapi_min: state.c0DapiMin, c0_dapi_max: state.c0DapiMax,
        green_lut: state.greenLut, red_lut: state.redLut, dapi_lut: state.dapiLut,
        green_mode: state.greenMode, red_mode: state.redMode,
        overlay_blue: state.overlayBlue,
        c0_overlay_blue: state.cycle0OverlayBlue,
        _t: Date.now()
    });

    try {
        const res = await fetch(`/api/full_fov_images?${params.toString()}`, {
            signal: state.fovAbortController.signal
        });
        const data = await res.json();
        if (reqId !== state.fovRequestId) return;

        const loadImg = (src) => new Promise((resolve) => {
            if (!src) return resolve(null);
            const img = new Image();
            img.onload = async () => {
                try {
                    await img.decode();
                } catch (_) {}
                resolve(img);
            };
            img.onerror = () => resolve(null);
            img.src = src;
        });

        const [gImg, rImg, oImg, c0Img] = await Promise.all([
            loadImg(data.green_fov),
            loadImg(data.red_fov),
            loadImg(data.overlay_fov),
            loadImg(data.cycle0_fov)
        ]);

        if (reqId !== state.fovRequestId) return;
        state.fovImages.green = gImg;
        state.fovImages.red = rImg;
        state.fovImages.overlay = oImg;
        state.fovImages.cycle0 = c0Img;
        state.fovImagesLoaded = true;

        renderAllFovCanvases();
    } catch (e) {
        if (e.name === 'AbortError') return;
        console.error('Error loading full FOV:', e);
    } finally {
        if (reqId === state.fovRequestId) {
            showFovLoading(false);
        }
    }
}

function renderAllFovCanvases() {
    if (!state.fovImagesLoaded) return;
    ['green', 'red', 'overlay', 'cycle0'].forEach(ch => renderFovCanvas(ch));
}

function renderFovCanvas(ch) {
    const canvas = document.getElementById(`fov-${ch}-canvas`);
    const img = state.fovImages[ch];
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);

    const fovW = state.imageWidth || 2048.0;
    const fovH = state.imageHeight || 2048.0;
    const scaleX = width / fovW;
    const scaleY = height / fovH;

    // Draw centroids (toggleable via state.showFovDots)
    if (state.showFovDots) {
        state.cells.forEach((cell, idx) => {
            const cx = cell.x * scaleX;
            const cy = cell.y * scaleY;
            ctx.beginPath();
            ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);

            if (state.mode === 'base_caller') {
                if (cell.annotated_base === 'A') ctx.fillStyle = '#10b981';
                else if (cell.annotated_base === 'T') ctx.fillStyle = '#ef4444';
                else if (cell.annotated_base === 'C') ctx.fillStyle = '#f59e0b';
                else if (cell.annotated_base === 'G') ctx.fillStyle = '#6b7280';
                else if (cell.annotated_base === 'N') ctx.fillStyle = '#ffffff';
                else ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
            } else {
                if (cell.is_real_blob === true) ctx.fillStyle = '#10b981';
                else if (cell.is_real_blob === false) ctx.fillStyle = '#ef4444';
                else ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
            }
            ctx.fill();
        });
    }

    // Reticle for active cell
    if (state.cells.length > 0 && state.currentIndex < state.cells.length) {
        const activeCell = state.cells[state.currentIndex];
        const ax = activeCell.x * scaleX;
        const ay = activeCell.y * scaleY;
        const diaCanvas = (state.nuclearDiameter * scaleX);

        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.strokeRect(ax - diaCanvas / 2.0, ay - diaCanvas / 2.0, diaCanvas, diaCanvas);

        ctx.beginPath();
        ctx.moveTo(ax - 8, ay); ctx.lineTo(ax + 8, ay);
        ctx.moveTo(ax, ay - 8); ctx.lineTo(ax, ay + 8);
        ctx.stroke();
    }
}

function handleFovClick(e, canvas) {
    if (!state.cells || state.cells.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const clickY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const fovW = state.imageWidth || 2048.0;
    const fovH = state.imageHeight || 2048.0;
    const scaleX = canvas.width / fovW;
    const scaleY = canvas.height / fovH;

    let closestIdx = -1;
    let minDist = Infinity;
    state.cells.forEach((cell, idx) => {
        const cx = cell.x * scaleX;
        const cy = cell.y * scaleY;
        const d = Math.hypot(clickX - cx, clickY - cy);
        if (d < minDist) {
            minDist = d;
            closestIdx = idx;
        }
    });

    if (closestIdx !== -1 && minDist < 35) {
        displayCell(closestIdx);
    }
}

// --- Cell Crop & Distribution Featurization ---

function displayCell(index) {
    if (index < 0) index = 0;
    if (index >= state.cells.length) index = state.cells.length - 1;
    state.currentIndex = index;

    const cell = state.cells[index];
    document.getElementById('active-cell-id').innerText = `Cell #${cell.cell_id} (${index + 1}/${state.cells.length})`;
    document.getElementById('active-coords').innerText = `X: ${cell.x.toFixed(1)}, Y: ${cell.y.toFixed(1)}`;
    document.getElementById('active-crop-cell-label').innerText = cell.cell_id;

    // Current call badge
    const badge = document.getElementById('active-base-badge');
    if (state.mode === 'blob_validator') {
        if (cell.blob_call_status) {
            const labels = {
                'both': 'BOTH CORRECT',
                'only_green': 'ONLY GREEN',
                'only_red': 'ONLY RED',
                'neither': 'NEITHER'
            };
            badge.innerText = labels[cell.blob_call_status] || String(cell.blob_call_status).toUpperCase();
            badge.className = `meta-val base-badge tag-call-${String(cell.blob_call_status).replace('_', '-')}`;
        } else {
            badge.innerText = 'UNANNOTATED';
            badge.className = 'meta-val base-badge';
        }
    } else {
        if (cell.annotated_base) {
            badge.innerText = cell.annotated_base;
            badge.className = `meta-val base-badge tag-${cell.annotated_base.toLowerCase()}`;
        } else {
            badge.innerText = 'UNANNOTATED';
            badge.className = 'meta-val base-badge';
        }
    }

    renderAllFovCanvases();
    fetchCellCrop(cell.cell_id);
}

async function fetchCellCrop(cellId, isLutOnly = false) {
    state.cropRequestId++;
    const reqId = state.cropRequestId;
    if (!isLutOnly) {
        showCropLoading(true);
    }

    if (state.cropAbortController) {
        try {
            state.cropAbortController.abort();
        } catch (_) {}
    }
    state.cropAbortController = new AbortController();

    const params = new URLSearchParams({
        cycle: state.currentCycle,
        nuclear_diameter: state.nuclearDiameter,
        hysteresis_high: state.hysteresisHigh,
        hysteresis_low: Math.max(1.0, state.hysteresisHigh * 0.5),
        green_min: state.greenMin, green_max: state.greenMax,
        red_min: state.redMin, red_max: state.redMax,
        dapi_min: state.dapiMin, dapi_max: state.dapiMax,
        c0_farred_min: state.c0FarredMin, c0_farred_max: state.c0FarredMax,
        c0_dapi_min: state.c0DapiMin, c0_dapi_max: state.c0DapiMax,
        green_lut: state.greenLut, red_lut: state.redLut, dapi_lut: state.dapiLut,
        green_mode: state.greenMode, red_mode: state.redMode,
        overlay_blue: state.overlayBlue,
        c0_overlay_blue: state.cycle0OverlayBlue,
        lut_only: isLutOnly ? '1' : '0',
        _t: Date.now()
    });

    try {
        const res = await fetch(`/api/crop/${cellId}?${params.toString()}`, {
            signal: state.cropAbortController.signal
        });
        const data = await res.json();
        if (reqId !== state.cropRequestId) return;

        state.activeCropData = data;
        await renderHighPowerCrop(reqId);
        if (!isLutOnly) {
            updateDistributionCards(data.base_features);
            updateBlobValidationPanel(data.blob_features);
            updateSpotFooters(data);
        }
    } catch (e) {
        if (e.name === 'AbortError') return;
        console.error('Error fetching crop:', e);
    } finally {
        if (reqId === state.cropRequestId) {
            showCropLoading(false);
        }
    }
}

async function renderHighPowerCrop(reqId) {
    if (!state.activeCropData) return;
    const data = state.activeCropData;
    const crops = data.crops || {};
    const contours = data.contours || {};

    const loadAndDraw = async (ch, imgSrc) => {
        const canvas = document.getElementById(`crop-${ch}-canvas`);
        if (!canvas || !imgSrc) return;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.src = imgSrc;
        try {
            await img.decode();
        } catch (err) {
            await new Promise((resolve) => {
                img.onload = resolve;
                img.onerror = resolve;
            });
        }
        if (reqId && reqId !== state.cropRequestId) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        const scaleX = canvas.width / (data.crop_box ? data.crop_box.width : 280.0);
        const scaleY = canvas.height / (data.crop_box ? data.crop_box.height : 280.0);

        const drawPolygon = (poly, strokeStyle, lineWidth) => {
            ctx.strokeStyle = strokeStyle;
            ctx.lineWidth = lineWidth;
            ctx.beginPath();
            poly.forEach((pt, i) => {
                const px = pt[0] * scaleX;
                const py = pt[1] * scaleY;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            ctx.closePath();
            ctx.stroke();
        };

        // Draw Outlines: strictly channel-specific
        if (ch === 'green') {
            // ONLY green blob outline for green channel (no blue nuclear outline)
            if (state.showGreenContour && contours.green_blob) {
                contours.green_blob.forEach(poly => drawPolygon(poly, '#10b981', 2.5));
            }
        } else if (ch === 'red') {
            // ONLY red blob outline for red channel (no blue nuclear outline)
            if (state.showRedContour && contours.red_blob) {
                contours.red_blob.forEach(poly => drawPolygon(poly, '#ef4444', 2.5));
            }
        } else if (ch === 'composite') {
            // Blue nuclear outline + whichever blobs detected on overlay
            if (state.showNucContour && contours.nuc) {
                contours.nuc.forEach(poly => drawPolygon(poly, '#38bdf8', 2.0));
            }
            if (state.showGreenContour && contours.green_blob) {
                contours.green_blob.forEach(poly => drawPolygon(poly, '#10b981', 2.2));
            }
            if (state.showRedContour && contours.red_blob) {
                contours.red_blob.forEach(poly => drawPolygon(poly, '#ef4444', 2.2));
            }
        } else if (ch === 'cycle0') {
            // Blue nuclear outline on Cycle 0 reference
            if (state.showNucContour && contours.nuc) {
                contours.nuc.forEach(poly => drawPolygon(poly, '#38bdf8', 2.0));
            }
        }
    };

    const drawPromises = [
        loadAndDraw('green', crops.green),
        loadAndDraw('red', crops.red),
        loadAndDraw('composite', crops.composite)
    ];
    if (state.showFishProbe && crops.cycle0) {
        drawPromises.push(loadAndDraw('cycle0', crops.cycle0));
    }
    await Promise.all(drawPromises);
}

function updateSpotFooters(data) {
    const bf = data.base_features || {};
    const det = data.detection || {};

    // GREEN FOOTER
    const gStatus = document.getElementById('green-crop-status');
    const gArea = document.getElementById('green-crop-area');
    const gMed = document.getElementById('green-crop-med-val');
    if (gStatus) {
        if (det.green_detected) {
            gStatus.innerText = 'DETECTED';
            gStatus.className = 'spot-status-badge tag-detected-green';
        } else {
            gStatus.innerText = 'NO BLOB';
            gStatus.className = 'spot-status-badge tag-no-blob';
        }
    }
    if (gArea) gArea.innerText = det.green_area !== undefined ? det.green_area : (bf.green ? bf.green.area || 0 : '--');
    if (gMed) gMed.innerText = bf.green ? bf.green.median : '--';

    // RED FOOTER
    const rStatus = document.getElementById('red-crop-status');
    const rArea = document.getElementById('red-crop-area');
    const rMed = document.getElementById('red-crop-med-val');
    if (rStatus) {
        if (det.red_detected) {
            rStatus.innerText = 'DETECTED';
            rStatus.className = 'spot-status-badge tag-detected-red';
        } else {
            rStatus.innerText = 'NO BLOB';
            rStatus.className = 'spot-status-badge tag-no-blob';
        }
    }
    if (rArea) rArea.innerText = det.red_area !== undefined ? det.red_area : (bf.red ? bf.red.area || 0 : '--');
    if (rMed) rMed.innerText = bf.red ? bf.red.median : '--';

    // COMPOSITE FOOTER
    const cStatus = document.getElementById('composite-crop-status');
    const cArea = document.getElementById('blob-area-val');
    if (cStatus) {
        if (det.composite_detected) {
            cStatus.innerText = 'DETECTED';
            cStatus.className = 'spot-status-badge tag-detected-green';
        } else {
            cStatus.innerText = 'NO BLOB';
            cStatus.className = 'spot-status-badge tag-no-blob';
        }
    }
    if (cArea) cArea.innerText = det.composite_area !== undefined ? det.composite_area : (bf.blob_area !== undefined ? bf.blob_area : '--');
}

function updateDistributionCards(bf) {
    if (!bf || !bf.green || !bf.red) return;

    // GREEN CARD
    const g = bf.green;
    document.getElementById('g-dist-median').innerText = `${g.median} ADU`;
    document.getElementById('g-dist-trimmed').innerText = `${g.trimmed_mean} ADU`;
    document.getElementById('g-dist-iqr').innerText = `${g.p25} – ${g.p75}`;
    document.getElementById('g-dist-p90').innerText = `${g.p90}`;
    document.getElementById('g-dist-max').innerText = `${g.max}`;
    document.getElementById('g-dist-delta').innerText = `+${g.excess_median} ADU`;

    // Green Range Bar positioning
    const maxVal = Math.max(4000, g.max, bf.red.max);
    const gP25Pct = Math.min(100, (g.p25 / maxVal) * 100);
    const gP75Pct = Math.min(100, (g.p75 / maxVal) * 100);
    const gMedPct = Math.min(100, (g.median / maxVal) * 100);
    const gP95Pct = Math.min(100, (g.p95 / maxVal) * 100);

    const gIqr = document.getElementById('g-range-iqr');
    gIqr.style.left = `${gP25Pct}%`;
    gIqr.style.width = `${Math.max(4, gP75Pct - gP25Pct)}%`;
    document.getElementById('g-range-pin').style.left = `${gMedPct}%`;
    const gWhisker = document.getElementById('g-range-whisker');
    gWhisker.style.left = `${gP75Pct}%`;
    gWhisker.style.width = `${Math.max(0, gP95Pct - gP75Pct)}%`;

    // RED CARD
    const r = bf.red;
    document.getElementById('r-dist-median').innerText = `${r.median} ADU`;
    document.getElementById('r-dist-trimmed').innerText = `${r.trimmed_mean} ADU`;
    document.getElementById('r-dist-iqr').innerText = `${r.p25} – ${r.p75}`;
    document.getElementById('r-dist-p90').innerText = `${r.p90}`;
    document.getElementById('r-dist-max').innerText = `${r.max}`;
    document.getElementById('r-dist-delta').innerText = `+${r.excess_median} ADU`;

    // Red Range Bar positioning
    const rP25Pct = Math.min(100, (r.p25 / maxVal) * 100);
    const rP75Pct = Math.min(100, (r.p75 / maxVal) * 100);
    const rMedPct = Math.min(100, (r.median / maxVal) * 100);
    const rP95Pct = Math.min(100, (r.p95 / maxVal) * 100);

    const rIqr = document.getElementById('r-range-iqr');
    rIqr.style.left = `${rP25Pct}%`;
    rIqr.style.width = `${Math.max(4, rP75Pct - rP25Pct)}%`;
    document.getElementById('r-range-pin').style.left = `${rMedPct}%`;
    const rWhisker = document.getElementById('r-range-whisker');
    rWhisker.style.left = `${rP75Pct}%`;
    rWhisker.style.width = `${Math.max(0, rP95Pct - rP75Pct)}%`;
}

function updateBlobValidationPanel(feat) {
    if (!feat) return;

    document.getElementById('feat-area').innerText = feat.area || 0;
    document.getElementById('feat-frac-nuc').innerText = feat.fraction_nucleus_occupied ? `${(feat.fraction_nucleus_occupied * 100).toFixed(1)}%` : '0%';
    document.getElementById('feat-med-contrast').innerText = feat.median_contrast ? `${feat.median_contrast.toFixed(1)} σ` : '--';
    document.getElementById('feat-peak-contrast').innerText = feat.peak_contrast ? `${feat.peak_contrast.toFixed(1)} σ` : '--';
    document.getElementById('feat-smoothness').innerText = feat.spatial_smoothness !== undefined ? feat.spatial_smoothness.toFixed(3) : '--';
    document.getElementById('feat-solidity').innerText = feat.solidity !== undefined ? feat.solidity.toFixed(2) : '--';
    document.getElementById('feat-eccentricity').innerText = feat.eccentricity !== undefined ? feat.eccentricity.toFixed(2) : '--';
    document.getElementById('feat-perimeter-area').innerText = feat.perimeter_to_area_ratio !== undefined ? feat.perimeter_to_area_ratio.toFixed(2) : '--';
    document.getElementById('feat-dist-boundary').innerText = feat.distance_from_nuclear_boundary !== undefined ? `${feat.distance_from_nuclear_boundary.toFixed(1)} px` : '--';
    document.getElementById('feat-sat-frac').innerText = feat.saturated_pixel_fraction !== undefined ? `${(feat.saturated_pixel_fraction * 100).toFixed(1)}%` : '0%';

    const pBadge = document.getElementById('val-prob-badge');
    const p = feat.p_real_blob !== undefined ? feat.p_real_blob : 0.0;
    pBadge.innerText = `P(real) = ${(p * 100).toFixed(0)}%`;
    if (p >= 0.75) {
        pBadge.style.color = '#065f46';
        pBadge.style.backgroundColor = '#d1fae5';
        pBadge.style.borderColor = '#10b981';
    } else if (p <= 0.35) {
        pBadge.style.color = '#991b1b';
        pBadge.style.backgroundColor = '#fee2e2';
        pBadge.style.borderColor = '#ef4444';
    } else {
        pBadge.style.color = '#92400e';
        pBadge.style.backgroundColor = '#fef3c7';
        pBadge.style.borderColor = '#f59e0b';
    }
}

// --- Annotation Handlers (Two Distinct Workflows) ---

async function advanceAfterAnnotation() {
    if (state.progressionMode === 'cycle') {
        const available = (state.config && state.config.available_cycles && state.config.available_cycles.length > 0)
            ? state.config.available_cycles
            : [1];
        const sortedCycles = [...available].sort((a, b) => a - b);
        const nextCycles = sortedCycles.filter(c => c > state.currentCycle);

        if (nextCycles.length > 0) {
            // Advance to next cycle for this cell
            await changeCycle(nextCycles[0]);
        } else {
            // Reached the last cycle! Advance to next cell, and reset to first cycle
            if (state.currentIndex < state.cells.length - 1) {
                state.currentIndex += 1;
                await changeCycle(sortedCycles[0], true);
            } else {
                displayCell(state.currentIndex);
                console.log('Finished annotating all cycles for all cells!');
            }
        }
    } else {
        nextCell();
    }
}

async function annotateBaseCall(base) {
    if (state.cells.length === 0 || state.isAnnotating) return;
    state.isAnnotating = true;
    try {
        const cell = state.cells[state.currentIndex];
        const cropData = state.activeCropData || {};
        if (!cropData.analysis_token || cropData.cell_id !== cell.cell_id || cropData.cycle !== state.currentCycle) return;
        const bf = cropData.base_features || {};
        const g = bf.green || {};
        const r = bf.red || {};

        const payload = {
            cell_id: cell.cell_id,
            analysis_token: cropData.analysis_token,
            base: base,
            cycle: state.currentCycle,
            tile: cell.tile || state.currentTile,
            well: cell.well || 'B2',
            x: cell.x,
            y: cell.y,
            blob_area: bf.blob_area || 0,
            // Green
            green_median: g.median || 0,
            green_trimmed_mean: g.trimmed_mean || 0,
            green_p25: g.p25 || 0,
            green_p75: g.p75 || 0,
            green_p90: g.p90 || 0,
            green_p95: g.p95 || 0,
            green_mean: g.mean || 0,
            green_std: g.std || 0,
            green_excess_median: g.excess_median || 0,
            green_integrated_excess: g.integrated_excess || 0,
            // Red
            red_median: r.median || 0,
            red_trimmed_mean: r.trimmed_mean || 0,
            red_p25: r.p25 || 0,
            red_p75: r.p75 || 0,
            red_p90: r.p90 || 0,
            red_p95: r.p95 || 0,
            red_mean: r.mean || 0,
            red_std: r.std || 0,
            red_excess_median: r.excess_median || 0,
            red_integrated_excess: r.integrated_excess || 0,
            // Ratios
            rg_ratio_median: bf.rg_ratio_median || 0,
            rg_ratio_trimmed: bf.rg_ratio_trimmed || 0,
            // Tier 2 distributions
            green_distribution: g.histogram || {},
            red_distribution: r.histogram || {}
        };

        const res = await fetch('/api/annotate_base', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Annotation save failed');
        if (data.success) {
            cell.annotated_base = base;
            refreshProgressUI();
            await advanceAfterAnnotation();
            updateGroundTruthTable();
        }
    } catch (e) {
        console.error('Error saving base annotation:', e);
    } finally {
        state.isAnnotating = false;
    }
}

async function annotateBlobSignal(callStatus) {
    if (state.cells.length === 0 || state.isAnnotating) return;
    state.isAnnotating = true;
    try {
        const cell = state.cells[state.currentIndex];
        const cropData = state.activeCropData || {};
        if (!cropData.analysis_token || cropData.cell_id !== cell.cell_id || cropData.cycle !== state.currentCycle) return;
        const feat = cropData.blob_features || {};
        const det = cropData.detection || {};

        const greenCorrect = (callStatus === 'both' || callStatus === 'only_green');
        const redCorrect = (callStatus === 'both' || callStatus === 'only_red');
        const isReal = (Boolean(det.green_detected) === greenCorrect) || (Boolean(det.red_detected) === redCorrect);

        const payload = {
            cell_id: cell.cell_id,
            analysis_token: cropData.analysis_token,
            call_status: callStatus,
            is_real_blob: isReal,
            green_called_correctly: greenCorrect,
            red_called_correctly: redCorrect,
            green_detected: Boolean(det.green_detected),
            red_detected: Boolean(det.red_detected),
            green_area: det.green_area || 0,
            red_area: det.red_area || 0,
            cycle: state.currentCycle,
            tile: cell.tile || state.currentTile,
            well: cell.well || 'B2',
            x: cell.x,
            y: cell.y,
            area: feat.area || 0,
            fraction_nucleus_occupied: feat.fraction_nucleus_occupied || 0,
            median_contrast: feat.median_contrast || 0,
            peak_contrast: feat.peak_contrast || 0,
            upper_quantile_contrast: feat.upper_quantile_contrast || 0,
            integrated_excess_intensity: feat.integrated_excess_intensity || 0,
            spatial_smoothness: feat.spatial_smoothness || 0,
            solidity: feat.solidity || 0,
            eccentricity: feat.eccentricity || 0,
            perimeter_to_area_ratio: feat.perimeter_to_area_ratio || 0,
            persistence_cycles: feat.persistence_cycles || 1.0,
            distance_from_nuclear_boundary: feat.distance_from_nuclear_boundary || 0,
            saturated_pixel_fraction: feat.saturated_pixel_fraction || 0,
            p_real_blob: feat.p_real_blob || 0
        };

        const res = await fetch('/api/annotate_blob', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Annotation save failed');
        if (data.success) {
            cell.blob_call_status = callStatus;
            cell.is_real_blob = data.is_real_blob;
            refreshProgressUI();
            await advanceAfterAnnotation();
            updateGroundTruthTable();
        }
    } catch (e) {
        console.error('Error saving blob validation:', e);
    } finally {
        state.isAnnotating = false;
    }
}

function prevCell() {
    if (state.currentIndex > 0) {
        displayCell(state.currentIndex - 1);
    }
}

function nextCell() {
    if (state.currentIndex < state.cells.length - 1) {
        displayCell(state.currentIndex + 1);
    }
}

function updateProgressUI(baseCount, blobCount, total) {
    const isBase = state.mode === 'base_caller';
    const count = isBase ? (baseCount || 0) : (blobCount || 0);
    const tot = total || (state.cells ? state.cells.length : 1) || 1;
    const pct = Math.min(100, Math.round((count / tot) * 100));

    const modeLabel = isBase ? 'Base' : 'Blob';
    const progText = document.getElementById('progress-text');
    if (progText) progText.innerText = `Cycle ${state.currentCycle} ${modeLabel} Annotated: ${count} / ${tot}`;
    const progPct = document.getElementById('progress-percent');
    if (progPct) progPct.innerText = `${pct}%`;
    const progFill = document.getElementById('progress-bar-fill');
    if (progFill) progFill.style.width = `${pct}%`;
    const sessBadge = document.getElementById('session-count-badge');
    if (sessBadge) sessBadge.innerText = `Total Ground Truth: ${count}`;
}

function refreshProgressUI() {
    if (!state.cells || state.cells.length === 0) return;
    const baseCount = state.cells.filter(c => c.annotated_base !== null && c.annotated_base !== undefined && String(c.annotated_base).trim() !== '').length;
    const blobCount = state.cells.filter(c => c.blob_call_status !== null && c.blob_call_status !== undefined && String(c.blob_call_status).trim() !== '').length;
    updateProgressUI(baseCount, blobCount, state.cells.length);
}

async function updateGroundTruthTable() {
    const isBase = state.mode === 'base_caller';
    const endpoint = isBase ? '/api/ground_truth_base' : '/api/ground_truth_blob';
    const thead = document.getElementById('gt-table-head');
    const tbody = document.getElementById('gt-table-body');
    const countBadge = document.getElementById('gt-count');

    try {
        const res = await fetch(`${endpoint}?_t=${Date.now()}`);
        const data = await res.json();
        const records = data.annotations || [];
        countBadge.innerText = records.length;

        if (isBase) {
            thead.innerHTML = `
                <tr>
                    <th>Cell</th><th>Cycle</th><th>Base</th><th>Blob Area</th>
                    <th>Green Med</th><th>Green Trimmed</th><th>Red Med</th><th>Red Trimmed</th>
                    <th>R/G Ratio</th><th>Timestamp</th>
                </tr>
            `;
            if (records.length === 0) {
                tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: var(--text-muted);">No base annotations recorded yet.</td></tr>`;
            } else {
                tbody.innerHTML = records.slice(-100).reverse().map(r => `
                    <tr>
                        <td>${r.cell}</td><td>${r.cycle}</td>
                        <td><span class="base-badge tag-${String(r.annotated_base).toLowerCase()}">${r.annotated_base}</span></td>
                        <td>${r.blob_area} px</td>
                        <td>${r.green_median}</td><td>${r.green_trimmed_mean}</td>
                        <td>${r.red_median}</td><td>${r.red_trimmed_mean}</td>
                        <td>${r.rg_ratio_trimmed}</td><td>${r.timestamp}</td>
                    </tr>
                `).join('');
            }
        } else {
            thead.innerHTML = `
                <tr>
                    <th>Cell</th><th>Cycle</th><th>Validation Call</th>
                    <th>Green Det</th><th>Red Det</th><th>Area</th><th>P(real)</th>
                    <th>Contrast</th><th>Smoothness</th><th>Timestamp</th>
                </tr>
            `;
            if (records.length === 0) {
                tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: var(--text-muted);">No blob validation annotations recorded yet.</td></tr>`;
            } else {
                tbody.innerHTML = records.slice(-100).reverse().map(r => {
                    const status = r.call_status || (r.is_real_blob ? 'both' : 'neither');
                    const badges = {
                        'both': '<span class="tag-call-both">Both Correct</span>',
                        'only_green': '<span class="tag-call-green">Only Green</span>',
                        'only_red': '<span class="tag-call-red">Only Red</span>',
                        'neither': '<span class="tag-call-neither">Neither</span>'
                    };
                    const badgeHtml = badges[status] || `<span class="tag-call-neither">${status}</span>`;
                    const gDet = r.green_detected ? `<span style="color:#059669;font-weight:700;">YES (${r.green_area || 0}px)</span>` : '<span style="color:#94a3b8;">NO</span>';
                    const rDet = r.red_detected ? `<span style="color:#dc2626;font-weight:700;">YES (${r.red_area || 0}px)</span>` : '<span style="color:#94a3b8;">NO</span>';
                    return `
                        <tr>
                            <td>${r.cell}</td><td>${r.cycle}</td>
                            <td>${badgeHtml}</td>
                            <td>${gDet}</td><td>${rDet}</td>
                            <td>${r.area} px</td>
                            <td>${(Number(r.p_real_blob) * 100).toFixed(0)}%</td>
                            <td>${Number(r.median_contrast).toFixed(1)} σ</td>
                            <td>${Number(r.spatial_smoothness).toFixed(3)}</td>
                            <td>${r.timestamp}</td>
                        </tr>
                    `;
                }).join('');
            }
        }
    } catch (e) {
        console.error('Failed to update ground truth table:', e);
    }
}

async function clearCurrentGroundTruth() {
    const isBase = state.mode === 'base_caller';
    const target = isBase ? 'base' : 'blob';
    if (!confirm(`Are you sure you want to clear the ${isBase ? 'Base Call' : 'Blob Signal'} ground truth table?`)) return;

    try {
        await fetch('/api/clear_ground_truth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target })
        });
        await loadDataset();
    } catch (e) {
        console.error('Error clearing ground truth:', e);
    }
}

function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    ['dragenter', 'dragover'].forEach(name => {
        dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.style.borderColor = '#2563eb'; });
    });
    ['dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.style.borderColor = ''; });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) uploadFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) uploadFile(fileInput.files[0]);
    });
}

// --- Fullscreen Dataset Loading Screen & Pipeline Step Cycling ---

const PIPELINE_STEPS = [
    { id: 'upload', title: 'Uploading image stack to server...', detail: 'Streaming raw bytes' },
    { id: 'metadata', title: 'Inspecting TIFF container & metadata...', detail: 'Validating 16-bit multi-page layout and dimensions' },
    { id: 'memory', title: 'Loading 13-cycle image stack into memory...', detail: 'Buffering ~2.5 GB imaging array in RAM' },
    { id: 'dapi', title: 'Filtering DAPI & distance transform...', detail: 'Gaussian smoothing and Euclidean distance map on 32M px' },
    { id: 'nuclei', title: 'Segmenting nuclei & calculating centroids...', detail: 'Topological watershed & Cython regionprops' },
    { id: 'overview', title: 'Generating downsampled FOV overview...', detail: 'Computing 4-channel composite overview projections' },
    { id: 'warmup', title: 'Initializing cell crops & spot engine...', detail: 'Calibrating hysteresis segmenter & UI viewer' }
];

let loadingStepTimer = null;

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
}

function showDatasetLoading(title, subtitle, includeUpload = true) {
    const overlay = document.getElementById('dataset-loading-overlay');
    if (!overlay) return;

    const titleEl = document.getElementById('loading-card-title');
    const subEl = document.getElementById('loading-card-subtitle');
    const container = document.getElementById('pipeline-steps-container');

    if (titleEl) titleEl.textContent = title || 'Processing Image Stack';
    if (subEl) subEl.textContent = subtitle || 'Analyzing raw microscopy stack...';

    if (container) {
        container.innerHTML = '';
        const stepsToUse = includeUpload ? PIPELINE_STEPS : PIPELINE_STEPS.slice(1);
        stepsToUse.forEach((step, idx) => {
            const item = document.createElement('div');
            item.className = 'pipeline-step-item pending';
            item.id = `pipe-step-${step.id}`;
            item.innerHTML = `
                <span class="step-status-icon">${idx + 1}</span>
                <span class="step-name">${step.title}</span>
            `;
            container.appendChild(item);
        });
    }

    setLoadingProgress(0, 'Initializing pipeline...');
    overlay.style.display = 'flex';
    requestAnimationFrame(() => overlay.classList.add('active'));
}

function updatePipelineStep(stepId, stepState = 'active', text = null) {
    const item = document.getElementById(`pipe-step-${stepId}`);
    if (!item) return;
    item.classList.remove('pending', 'active', 'done');
    item.classList.add(stepState);

    const icon = item.querySelector('.step-status-icon');
    if (icon) {
        if (stepState === 'done') {
            icon.innerHTML = '✓';
        } else if (stepState === 'active') {
            icon.innerHTML = '●';
        }
    }
    if (text) {
        const nameEl = item.querySelector('.step-name');
        if (nameEl) nameEl.textContent = text;
    }
}

function setLoadingProgress(percent, stageText) {
    const fill = document.getElementById('loading-bar-fill');
    const pLabel = document.getElementById('loading-percent-label');
    const sLabel = document.getElementById('loading-stage-label');
    const pct = Math.max(0, Math.min(100, Math.round(percent)));
    if (fill) fill.style.width = `${pct}%`;
    if (pLabel) pLabel.textContent = `${pct}%`;
    if (sLabel && stageText) sLabel.textContent = stageText;
}

function hideDatasetLoading(delay = 600) {
    const overlay = document.getElementById('dataset-loading-overlay');
    if (!overlay) return;
    if (loadingStepTimer) {
        clearTimeout(loadingStepTimer);
        loadingStepTimer = null;
    }
    setTimeout(() => {
        overlay.classList.remove('active');
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 300);
    }, delay);
}

async function uploadFile(file) {
    const sizeStr = formatBytes(file.size);
    showDatasetLoading('Uploading & Analyzing Image Stack', `${file.name} (${sizeStr})`, true);
    updatePipelineStep('upload', 'active', `Uploading ${file.name} (0% of ${sizeStr})...`);

    const formData = new FormData();
    formData.append('file', file);

    const uploadPromise = new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload', true);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                const overallPct = Math.round(pct * 0.45); // 0% to 45% for file transfer
                setLoadingProgress(overallPct, `Uploading image stack (${pct}% of ${sizeStr})...`);
                updatePipelineStep('upload', 'active', `Uploading ${file.name} (${pct}% of ${sizeStr})...`);
            }
        };

        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const data = JSON.parse(xhr.responseText);
                    resolve(data);
                } catch (err) {
                    reject(err);
                }
            } else {
                reject(new Error(`Server returned status ${xhr.status}`));
            }
        };

        xhr.onerror = () => reject(new Error('Network error during file upload'));
        xhr.send(formData);
    });

    try {
        const uploadResult = await uploadPromise;
        if (!uploadResult.success) {
            throw new Error(uploadResult.error || 'Upload failed');
        }

        // Upload complete! Mark Step 1 Done
        updatePipelineStep('upload', 'done', `Upload complete (${sizeStr} stored)`);
        setLoadingProgress(48, 'Verifying TIFF container & metadata...');

        // Start step animation simulation for the backend processing
        updatePipelineStep('metadata', 'active');
        await new Promise(r => setTimeout(r, 450));
        updatePipelineStep('metadata', 'done');
        setLoadingProgress(55, 'Loading 13-cycle image stack into memory...');

        updatePipelineStep('memory', 'active');

        // Asynchronously execute backend dataset loading
        const processingPromise = (async () => {
            await fetchConfig();
            await loadDataset();
        })();

        // Progressively step through pipeline stages while waiting for backend
        const stepSchedule = [
            { delay: 1800, step: 'memory', next: 'dapi', pct: 68, label: 'Filtering DAPI channel & distance transform...' },
            { delay: 4200, step: 'dapi', next: 'nuclei', pct: 82, label: 'Segmenting nuclei across 32M pixels...' },
            { delay: 7000, step: 'nuclei', next: 'overview', pct: 90, label: 'Generating downsampled FOV overview...' },
            { delay: 9200, step: 'overview', next: 'warmup', pct: 96, label: 'Extracting initial cell crops & spot engine...' },
        ];

        const scheduledTimeouts = stepSchedule.map(item => {
            return setTimeout(() => {
                const overlay = document.getElementById('dataset-loading-overlay');
                if (overlay && overlay.classList.contains('active')) {
                    updatePipelineStep(item.step, 'done');
                    updatePipelineStep(item.next, 'active');
                    setLoadingProgress(item.pct, item.label);
                }
            }, item.delay);
        });

        await processingPromise;
        scheduledTimeouts.forEach(t => clearTimeout(t));

        // All steps successfully completed!
        PIPELINE_STEPS.forEach(s => updatePipelineStep(s.id, 'done'));
        const totalLoaded = state.cells.length;
        setLoadingProgress(100, `Complete! ${totalLoaded} nuclei analyzed.`);
        const cardTitle = document.getElementById('loading-card-title');
        if (cardTitle) cardTitle.textContent = `Ready! (${totalLoaded} Cells Analyzed)`;

        hideDatasetLoading(700);
    } catch (e) {
        console.error('Upload or processing error:', e);
        setLoadingProgress(100, `Failed: ${e.message}`);
        updatePipelineStep('upload', 'pending', `Error: ${e.message}`);
        alert(`File upload or processing failed: ${e.message}`);
        hideDatasetLoading(1200);
    }
}
