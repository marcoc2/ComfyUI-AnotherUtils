import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DEBUG = true;
function log(...args) {
    if (DEBUG) console.log("[AudioWaveformSlicer]", ...args);
}

const WAVEFORM_TOP_PADDING = 80;
const WAVEFORM_BOTTOM_PADDING = 50;
const WAVEFORM_H_PADDING = 10;
const CUT_LINE_HIT_RADIUS = 6;
const PLAYHEAD_COLOR = "#FFFFFF";
const CUT_LINE_COLOR = "#FF3333";
const SEGMENT_COLORS = [
    "rgba(60, 130, 80, 0.12)",
    "rgba(60, 80, 160, 0.12)",
    "rgba(160, 120, 50, 0.12)",
    "rgba(130, 60, 130, 0.12)",
];

function formatTime(seconds) {
    if (!seconds || !isFinite(seconds)) return "0:00.0";
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return `${m}:${s.padStart(4, "0")}`;
}

app.registerExtension({
    name: "AnotherUtils.AudioWaveformSlicer",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "AudioWaveformSlicer") return;

        log("Registering AudioWaveformSlicer extension");

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            log("onNodeCreated called");
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            try {
                this._aws = {
                    peaks: null,
                    duration: 0,
                    audioBuffer: null,
                    audioContext: null,
                    sourceNode: null,
                    isPlaying: false,
                    playStartTime: 0,
                    playOffset: 0,
                    playheadPos: 0,
                    cutLines: [],
                    draggingCut: -1,
                    animFrameId: null,
                    waveformRect: null,
                    lastAudioFile: null,
                };

                // Find widgets
                this._awsWidgets = {};
                if (this.widgets) {
                    for (const w of this.widgets) {
                        if (w.name === "audio" || w.name === "cut_positions") {
                            this._awsWidgets[w.name] = w;
                        }
                    }
                }

                // Hide cut_positions widget
                const cpWidget = this._awsWidgets.cut_positions;
                if (cpWidget) {
                    cpWidget.type = "hidden";
                    cpWidget.computeSize = function () { return [0, -4]; };
                }

                // Set audio widget callback
                const audioWidget = this._awsWidgets.audio;
                if (audioWidget) {
                    const self = this;
                    audioWidget.callback = function () {
                        self._loadAudioFile();
                    };
                }

                // Set minimum size
                if (this.size) {
                    this.size[0] = Math.max(this.size[0] || 400, 400);
                    this.size[1] = Math.max(this.size[1] || 300, 300);
                }

                // Restore cut_positions from widget
                this._restoreCutPositions();

                // Load initial audio if set
                if (audioWidget && audioWidget.value) {
                    this._loadAudioFile();
                }

                log("onNodeCreated completed successfully");
            } catch (e) {
                console.error("[AudioWaveformSlicer] Error in onNodeCreated:", e);
            }

            return r;
        };

        // --- Load audio file ---
        nodeType.prototype._loadAudioFile = async function () {
            var filename = null;
            try {
                filename = this._awsWidgets && this._awsWidgets.audio ? this._awsWidgets.audio.value : null;
            } catch (e) {
                return;
            }
            if (!filename) return;
            if (!this._aws) return;
            if (filename === this._aws.lastAudioFile) return;

            this._stopPlayback();
            this._aws.lastAudioFile = filename;
            this._aws.peaks = null;
            this._aws.audioBuffer = null;
            this._aws.duration = 0;
            this._aws.playheadPos = 0;

            try {
                var url = api.apiURL("/view?filename=" + encodeURIComponent(filename) + "&type=input");
                log("Loading audio from:", url);
                var response = await fetch(url);
                var arrayBuffer = await response.arrayBuffer();

                if (!this._aws.audioContext) {
                    this._aws.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }

                var audioBuffer = await this._aws.audioContext.decodeAudioData(arrayBuffer);
                this._aws.audioBuffer = audioBuffer;
                this._aws.duration = audioBuffer.duration;
                log("Audio loaded, duration:", audioBuffer.duration);

                this._calculatePeaks();
                this.setDirtyCanvas(true, true);
            } catch (e) {
                console.error("[AudioWaveformSlicer] Error loading audio:", e);
            }
        };

        // --- Calculate peaks ---
        nodeType.prototype._calculatePeaks = function () {
            if (!this._aws || !this._aws.audioBuffer) return;
            var buf = this._aws.audioBuffer;

            var rect = this._getWaveformRect();
            var numBars = Math.max(1, Math.floor(rect.w));
            var channelData = buf.getChannelData(0);
            var samplesPerBar = Math.max(1, Math.floor(channelData.length / numBars));

            var peaks = new Float32Array(numBars);
            for (var i = 0; i < numBars; i++) {
                var max = 0;
                var start = i * samplesPerBar;
                var end = Math.min(start + samplesPerBar, channelData.length);
                for (var j = start; j < end; j++) {
                    var abs = Math.abs(channelData[j]);
                    if (abs > max) max = abs;
                }
                peaks[i] = max;
            }

            this._aws.peaks = peaks;
        };

        // --- Get waveform drawing rect ---
        nodeType.prototype._getWaveformRect = function () {
            var widgetCount = (this.widgets && this.widgets.length) ? this.widgets.length : 2;
            var topY = Math.max(WAVEFORM_TOP_PADDING, widgetCount * 28 + 10);
            var nodeW = (this.size && this.size[0]) ? this.size[0] : 400;
            var nodeH = (this.size && this.size[1]) ? this.size[1] : 300;
            var w = nodeW - WAVEFORM_H_PADDING * 2;
            var h = nodeH - topY - WAVEFORM_BOTTOM_PADDING;
            return {
                x: WAVEFORM_H_PADDING,
                y: topY,
                w: Math.max(w, 10),
                h: Math.max(h, 20),
            };
        };

        // --- Sync cut_positions widget ---
        nodeType.prototype._syncCutPositions = function () {
            if (!this._aws || !this._awsWidgets) return;
            var cpWidget = this._awsWidgets.cut_positions;
            if (cpWidget) {
                var sorted = this._aws.cutLines.slice().sort(function (a, b) { return a - b; });
                this._aws.cutLines = sorted;
                cpWidget.value = JSON.stringify(sorted);
            }
        };

        // --- Restore cut positions from widget ---
        nodeType.prototype._restoreCutPositions = function () {
            if (!this._aws) return;
            try {
                var val = (this._awsWidgets && this._awsWidgets.cut_positions) ? this._awsWidgets.cut_positions.value : "[]";
                val = val || "[]";
                var arr = JSON.parse(val);
                if (Array.isArray(arr)) {
                    this._aws.cutLines = arr.map(Number).filter(function (n) { return !isNaN(n); });
                }
            } catch (e) {
                this._aws.cutLines = [];
            }
        };

        // --- Playback ---
        nodeType.prototype._togglePlayback = function () {
            if (!this._aws) return;
            if (this._aws.isPlaying) {
                this._stopPlayback();
            } else {
                this._startPlayback();
            }
        };

        nodeType.prototype._startPlayback = function () {
            if (!this._aws) return;
            var buf = this._aws.audioBuffer;
            var ctx = this._aws.audioContext;
            if (!buf || !ctx) return;

            if (ctx.state === "suspended") ctx.resume();

            var source = ctx.createBufferSource();
            source.buffer = buf;
            source.connect(ctx.destination);

            var offset = this._aws.playOffset;
            source.start(0, offset);

            this._aws.sourceNode = source;
            this._aws.playStartTime = ctx.currentTime - offset;
            this._aws.isPlaying = true;

            var self = this;
            source.onended = function () {
                if (self._aws && self._aws.isPlaying) {
                    self._aws.isPlaying = false;
                    self._aws.playOffset = 0;
                    self._aws.playheadPos = 0;
                    if (self._aws.animFrameId) {
                        cancelAnimationFrame(self._aws.animFrameId);
                        self._aws.animFrameId = null;
                    }
                    self.setDirtyCanvas(true, true);
                }
            };

            this._animatePlayhead();
        };

        nodeType.prototype._stopPlayback = function () {
            if (!this._aws) return;
            if (this._aws.sourceNode) {
                try {
                    this._aws.sourceNode.onended = null;
                    this._aws.sourceNode.stop();
                } catch (e) { /* ignore */ }
                this._aws.sourceNode = null;
            }
            if (this._aws.isPlaying && this._aws.audioContext) {
                this._aws.playOffset = this._aws.audioContext.currentTime - this._aws.playStartTime;
            }
            this._aws.isPlaying = false;
            if (this._aws.animFrameId) {
                cancelAnimationFrame(this._aws.animFrameId);
                this._aws.animFrameId = null;
            }
            this.setDirtyCanvas(true, true);
        };

        nodeType.prototype._animatePlayhead = function () {
            if (!this._aws || !this._aws.isPlaying) return;

            var ctx = this._aws.audioContext;
            if (!ctx) return;
            var elapsed = ctx.currentTime - this._aws.playStartTime;
            var duration = this._aws.duration || 1;
            this._aws.playheadPos = Math.min(elapsed / duration, 1);

            if (this._aws.playheadPos >= 1) {
                this._aws.isPlaying = false;
                this._aws.playOffset = 0;
                this._aws.playheadPos = 0;
                this._aws.sourceNode = null;
                this.setDirtyCanvas(true, true);
                return;
            }

            this.setDirtyCanvas(true, true);
            var self = this;
            this._aws.animFrameId = requestAnimationFrame(function () {
                self._animatePlayhead();
            });
        };

        // --- Time <-> X coordinate conversion ---
        nodeType.prototype._timeToX = function (seconds) {
            var rect = this._getWaveformRect();
            var duration = (this._aws && this._aws.duration) ? this._aws.duration : 1;
            return rect.x + (seconds / duration) * rect.w;
        };

        nodeType.prototype._xToTime = function (x) {
            var rect = this._getWaveformRect();
            var duration = (this._aws && this._aws.duration) ? this._aws.duration : 1;
            var ratio = (x - rect.x) / rect.w;
            return Math.max(0, Math.min(ratio * duration, duration));
        };

        // --- Draw ---
        var origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);

            if (!this._aws) return;

            try {
                var rect = this._getWaveformRect();
                this._aws.waveformRect = rect;

                // Background
                ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
                ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

                if (!this._aws.peaks || this._aws.peaks.length === 0) {
                    ctx.fillStyle = "#888";
                    ctx.font = "12px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Select an audio file", rect.x + rect.w / 2, rect.y + rect.h / 2);
                    ctx.textAlign = "left";
                    // Still draw transport bar
                    this._drawTransportBar(ctx, rect);
                    return;
                }

                // Recalculate peaks if width changed
                if (this._aws.peaks.length !== Math.floor(rect.w)) {
                    this._calculatePeaks();
                }

                var peaks = this._aws.peaks;
                if (!peaks) return;

                // Draw segment backgrounds
                var cuts = this._aws.cutLines.slice().sort(function (a, b) { return a - b; });
                var segments = [0].concat(cuts, [this._aws.duration]);
                for (var i = 0; i < segments.length - 1; i++) {
                    var sx = this._timeToX(segments[i]);
                    var ex = this._timeToX(segments[i + 1]);
                    ctx.fillStyle = SEGMENT_COLORS[i % SEGMENT_COLORS.length];
                    ctx.fillRect(sx, rect.y, ex - sx, rect.h);
                }

                // Draw waveform bars
                var centerY = rect.y + rect.h / 2;
                var maxBarH = rect.h / 2 - 2;
                for (var i = 0; i < peaks.length; i++) {
                    var amp = peaks[i];
                    var barH = amp * maxBarH;
                    var x = rect.x + i;

                    var r = Math.floor(60 + amp * 195);
                    var g = Math.floor(180 - amp * 100);
                    var b = 60;
                    ctx.fillStyle = "rgb(" + r + "," + g + "," + b + ")";
                    ctx.fillRect(x, centerY - barH, 1, barH * 2);
                }

                // Draw cut lines
                for (var i = 0; i < this._aws.cutLines.length; i++) {
                    var t = this._aws.cutLines[i];
                    var cx = this._timeToX(t);

                    ctx.strokeStyle = CUT_LINE_COLOR;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(cx, rect.y);
                    ctx.lineTo(cx, rect.y + rect.h);
                    ctx.stroke();

                    // Handle triangle
                    ctx.fillStyle = CUT_LINE_COLOR;
                    ctx.beginPath();
                    ctx.moveTo(cx - 5, rect.y);
                    ctx.lineTo(cx + 5, rect.y);
                    ctx.lineTo(cx, rect.y + 10);
                    ctx.closePath();
                    ctx.fill();

                    // Timestamp label
                    ctx.fillStyle = "#FFF";
                    ctx.font = "10px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText(formatTime(t), cx, rect.y - 3);
                }

                // Draw playhead
                if (this._aws.duration > 0) {
                    var phX = rect.x + this._aws.playheadPos * rect.w;
                    ctx.strokeStyle = PLAYHEAD_COLOR;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(phX, rect.y);
                    ctx.lineTo(phX, rect.y + rect.h);
                    ctx.stroke();
                }

                // Transport bar
                this._drawTransportBar(ctx, rect);

            } catch (e) {
                console.error("[AudioWaveformSlicer] Draw error:", e);
            }

            // Reset text alignment
            ctx.textAlign = "left";
        };

        // --- Draw transport bar ---
        nodeType.prototype._drawTransportBar = function (ctx, rect) {
            if (!this._aws) return;
            var barY = rect.y + rect.h + 5;

            // Play/pause button
            ctx.fillStyle = "#CCC";
            ctx.font = "16px sans-serif";
            ctx.textAlign = "left";
            var btnText = this._aws.isPlaying ? "\u23F8" : "\u25B6";
            ctx.fillText(btnText, rect.x + 5, barY + 16);

            // Time display
            var currentTime = this._aws.playheadPos * (this._aws.duration || 0);
            ctx.fillStyle = "#AAA";
            ctx.font = "12px sans-serif";
            ctx.fillText(
                formatTime(currentTime) + " / " + formatTime(this._aws.duration),
                rect.x + 28,
                barY + 14
            );

            // Slice count
            var numCuts = this._aws.cutLines ? this._aws.cutLines.length : 0;
            ctx.textAlign = "right";
            ctx.fillText(
                numCuts + " cuts \u2192 " + (numCuts + 1) + " slices",
                rect.x + rect.w,
                barY + 14
            );
            ctx.textAlign = "left";
        };

        // --- Mouse events ---
        var origOnMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (event, pos, graphCanvas) {
            if (!this._aws || !this._aws.waveformRect) {
                return origOnMouseDown ? origOnMouseDown.apply(this, arguments) : false;
            }

            var rect = this._aws.waveformRect;
            var x = pos[0];
            var y = pos[1];

            // Check play button click
            var barY = rect.y + rect.h + 5;
            if (x >= rect.x && x <= rect.x + 25 && y >= barY && y <= barY + 22) {
                this._togglePlayback();
                return true;
            }

            // Check if inside waveform area
            if (x < rect.x || x > rect.x + rect.w || y < rect.y || y > rect.y + rect.h) {
                return origOnMouseDown ? origOnMouseDown.apply(this, arguments) : false;
            }

            // Check if near a cut line (for dragging)
            if (this._aws.cutLines) {
                for (var i = 0; i < this._aws.cutLines.length; i++) {
                    var cx = this._timeToX(this._aws.cutLines[i]);
                    if (Math.abs(x - cx) <= CUT_LINE_HIT_RADIUS) {
                        this._aws.draggingCut = i;
                        return true;
                    }
                }
            }

            // Single click: seek playhead
            if (this._aws.duration > 0) {
                var time = this._xToTime(x);
                this._aws.playheadPos = time / this._aws.duration;
                this._aws.playOffset = time;

                if (this._aws.isPlaying) {
                    this._stopPlayback();
                    this._startPlayback();
                }

                this.setDirtyCanvas(true, true);
            }
            return true;
        };

        var origOnMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (event, pos, graphCanvas) {
            if (this._aws && this._aws.draggingCut >= 0) {
                var x = pos[0];
                var time = this._xToTime(x);
                this._aws.cutLines[this._aws.draggingCut] = time;
                this._syncCutPositions();
                this.setDirtyCanvas(true, true);
                return true;
            }
            return origOnMouseMove ? origOnMouseMove.apply(this, arguments) : false;
        };

        var origOnMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function (event, pos, graphCanvas) {
            if (this._aws && this._aws.draggingCut >= 0) {
                this._aws.draggingCut = -1;
                this._syncCutPositions();
                this.setDirtyCanvas(true, true);
                return true;
            }
            return origOnMouseUp ? origOnMouseUp.apply(this, arguments) : false;
        };

        var origOnDblClick = nodeType.prototype.onDblClick;
        nodeType.prototype.onDblClick = function (event, pos, graphCanvas) {
            if (!this._aws || !this._aws.waveformRect || !this._aws.peaks) {
                return origOnDblClick ? origOnDblClick.apply(this, arguments) : undefined;
            }

            var rect = this._aws.waveformRect;
            var x = pos[0];
            var y = pos[1];

            // Check if inside waveform
            if (x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h) {
                // Check if near existing cut line -> remove it
                for (var i = 0; i < this._aws.cutLines.length; i++) {
                    var cx = this._timeToX(this._aws.cutLines[i]);
                    if (Math.abs(x - cx) <= CUT_LINE_HIT_RADIUS) {
                        this._aws.cutLines.splice(i, 1);
                        this._syncCutPositions();
                        this.setDirtyCanvas(true, true);
                        return true;
                    }
                }

                // Add new cut line
                var time = this._xToTime(x);
                this._aws.cutLines.push(time);
                this._syncCutPositions();
                this.setDirtyCanvas(true, true);
                return true;
            }

            return origOnDblClick ? origOnDblClick.apply(this, arguments) : undefined;
        };

        // --- Configure (workflow load) ---
        var origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);

            try {
                // Re-find widgets after configure
                this._awsWidgets = {};
                if (this.widgets) {
                    for (var i = 0; i < this.widgets.length; i++) {
                        var w = this.widgets[i];
                        if (w.name === "audio" || w.name === "cut_positions") {
                            this._awsWidgets[w.name] = w;
                        }
                    }
                }

                // Hide cut_positions
                var cpWidget = this._awsWidgets.cut_positions;
                if (cpWidget) {
                    cpWidget.type = "hidden";
                    cpWidget.computeSize = function () { return [0, -4]; };
                }

                if (this._aws) {
                    this._restoreCutPositions();
                }

                if (this._awsWidgets.audio && this._awsWidgets.audio.value) {
                    this._loadAudioFile();
                }
            } catch (e) {
                console.error("[AudioWaveformSlicer] Error in onConfigure:", e);
            }
        };

        // --- Compute size ---
        var origComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            var size = origComputeSize ? origComputeSize.apply(this, arguments) : [400, 300];
            if (!size) size = [400, 300];
            return [Math.max(size[0], 400), Math.max(size[1], 300)];
        };

        log("Extension registered successfully");
    },
});
