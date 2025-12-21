import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "AnotherUtils.CaptionImageLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CaptionImageLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.captionData = [];
                this.selectedBasename = "";
                this.statusMessage = "No images loaded. Set specific path.";

                // Scrolling state
                this.scrollY = 0;
                this.itemHeight = 110;
                this.listStartY = 140;

                // Helper to fetch images
                this.fetchImages = async () => {
                    const dirWidget = this.widgets.find(w => w.name === "directory");
                    if (!dirWidget || !dirWidget.value) {
                        this.statusMessage = "Please set a directory path.";
                        this.setDirtyCanvas(true);
                        return;
                    }

                    this.statusMessage = "Loading...";
                    this.setDirtyCanvas(true);

                    try {
                        const response = await api.fetchApi(`/another_utils/list_images?directory=${encodeURIComponent(dirWidget.value)}`);
                        if (response.status !== 200) {
                            throw new Error(response.statusText);
                        }
                        const data = await response.json();
                        if (data.error) {
                            this.statusMessage = "Error: " + data.error;
                            this.setDirtyCanvas(true);
                            return;
                        }

                        this.captionData = data.files || [];
                        if (this.captionData.length === 0) {
                            this.statusMessage = "No images found in directory.";
                        } else {
                            this.statusMessage = "";
                        }

                        // Reset scroll
                        this.scrollY = 0;

                        // We don't auto-resize infinitely anymore, we fix height to something reasonable or let user resize
                        // But to be nice, let's set a min height if small
                        this.setSize([this.size[0], Math.max(this.size[1], 500)]);
                        this.setDirtyCanvas(true);

                    } catch (e) {
                        this.statusMessage = "Error loading: " + e.message;
                        this.setDirtyCanvas(true);
                    }
                };

                // Add "Load List" button
                this.addWidget("button", "Load Images / Refresh", null, () => {
                    this.fetchImages();
                });

                // Add callback to directory widget for auto-load
                const dirWidget = this.widgets.find(w => w.name === "directory");
                if (dirWidget) {
                    const originalCallback = dirWidget.callback;
                    dirWidget.callback = (value) => {
                        if (originalCallback) originalCallback(value);
                        this.fetchImages();
                    };
                }

                const selWidget = this.widgets.find(w => w.name === "selected_basename");
                if (selWidget) {
                    selWidget.computeSize = () => [0, -4];
                }

                return r;
            };

            // Scrollbar state
            this.isDraggingScroll = false;
            this.lastMouseY = 0;

            // Input Handling
            const onMouseDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function (event, pos, graphCanvas) {
                const r = onMouseDown ? onMouseDown.apply(this, arguments) : undefined;

                if (!this.captionData || this.captionData.length === 0) return r;

                const x = pos[0];
                const y = pos[1];

                // Helper to get scrollbar rect
                const w = this.size[0];
                const viewHeight = this.size[1] - this.listStartY;
                const totalHeight = this.captionData.length * this.itemHeight;
                const sbWidth = 10; // Wider for easier grab
                const sbX = w - sbWidth - 2;

                // Check Scrollbar click
                if (totalHeight > viewHeight && x >= sbX && x <= sbX + sbWidth && y >= this.listStartY) {
                    this.isDraggingScroll = true;
                    this.lastMouseY = y;
                    console.log("[CaptionImageLoader] Scrollbar drag start");
                    // Capture input
                    if (graphCanvas.canvas) {
                        // This ensures we get mouse moves even outside node
                        // LiteGraph doesn't always expose setCapture easily, but we can rely on graphCanvas.dragging_canvas = false
                    }
                    return true;
                }

                if (y < this.listStartY) return r;

                // Item Click
                // Adjust index calculation correctly
                const index = Math.floor((y - this.listStartY - this.scrollY) / this.itemHeight);

                if (index >= 0 && index < this.captionData.length) {
                    this.selectedBasename = this.captionData[index].basename;

                    const widget = this.widgets.find(w => w.name === "selected_basename");
                    if (widget) {
                        widget.value = this.selectedBasename;
                    }
                    this.setDirtyCanvas(true);
                    return true;
                }

                return r;
            };

            const onMouseMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function (event, pos, graphCanvas) {
                if (this.isDraggingScroll) {
                    const y = pos[1];
                    const dy = y - this.lastMouseY;
                    this.lastMouseY = y;

                    const totalHeight = this.captionData.length * this.itemHeight;
                    const viewHeight = this.size[1] - this.listStartY;

                    // Map mouse delta to scroll delta
                    // Scrollbar moves by (viewHeight / totalHeight) * scrollDelta? 
                    // No, we are moving the thumb. 
                    // Thumb range = sbHeight - thumbHeight. Content range = totalHeight - viewHeight.
                    // ratio = contentRange / thumbRange.

                    const sbHeight = viewHeight;
                    const thumbHeight = Math.max(20, (viewHeight / totalHeight) * viewHeight);
                    const scrollRange = totalHeight - viewHeight;
                    const thumbRange = sbHeight - thumbHeight;

                    if (thumbRange > 0) {
                        const scrollDelta = -(dy * (scrollRange / thumbRange));
                        this.scrollY += scrollDelta;
                        this.scrollY = Math.max(-scrollRange, Math.min(0, this.scrollY));
                    }

                    this.setDirtyCanvas(true);
                    return true;
                }

                return onMouseMove ? onMouseMove.apply(this, arguments) : undefined;
            };

            const onMouseUp = nodeType.prototype.onMouseUp;
            nodeType.prototype.onMouseUp = function (event, pos, graphCanvas) {
                if (this.isDraggingScroll) {
                    this.isDraggingScroll = false;
                    console.log("[CaptionImageLoader] Scrollbar drag end");
                    this.setDirtyCanvas(true);
                    return true;
                }
                return onMouseUp ? onMouseUp.apply(this, arguments) : undefined;
            };

            // Wheel Handling
            // Note: LiteGraph nodes should contain onMouseWheel or getExtraMenuOptions
            // If checking "pos" is problematic, trust the event.
            nodeType.prototype.onWheel = function (event) {
                // Older LiteGraph support
                console.log("onWheel");
            };

            // This is the correct method name for ComfyUI's graph
            nodeType.prototype.onMouseWheel = function (event, pos, graphCanvas) {
                // pos is relative to node top-left
                if (!this.captionData || this.captionData.length === 0) return false;

                const x = pos[0];
                const y = pos[1];

                // Only scroll if over the list area
                if (y > this.listStartY && x < this.size[0]) {
                    // Standard deltaY is like 100 or -100.
                    const delta = event.deltaY;
                    // Invert logic: Scroll down (positive delta) -> Move content up (negative scrollY)
                    this.scrollY -= delta * 0.5; // Scale speed if needed

                    const totalHeight = this.captionData.length * this.itemHeight;
                    const viewHeight = this.size[1] - this.listStartY;
                    const minScroll = Math.min(0, -(totalHeight - viewHeight) - 20);

                    this.scrollY = Math.max(minScroll, Math.min(0, this.scrollY));

                    this.setDirtyCanvas(true);
                    return true; // Consume event
                }

                return false;
            }


            // Custom Drawing
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;

                const margin = 10;
                let y = this.listStartY;
                const w = this.size[0];

                ctx.save();

                // Clip content area
                ctx.beginPath();
                ctx.rect(0, y, w, this.size[1] - y);
                ctx.clip();

                // Status
                if (this.statusMessage) {
                    ctx.fillStyle = "#ffaaaa";
                    if (this.statusMessage.startsWith("Loading")) ctx.fillStyle = "#aaaaff";
                    if (this.statusMessage.startsWith("No images")) ctx.fillStyle = "#ffffaa";
                    ctx.font = "italic 14px Arial";
                    ctx.fillText(this.statusMessage, margin, y + 20 + this.scrollY);
                }

                if (!this.captionData || this.captionData.length === 0) {
                    ctx.restore();
                    return;
                }

                // Optimization: Only draw visible items
                // Item Top = listStartY + i*h + scrollY
                // Visible range: [listStartY, size[1]]
                const totalHeight = this.captionData.length * this.itemHeight;

                // Which index starts at top?
                // i*h + scrollY >= 0 (relative to list start) -> i*h >= -scrollY -> i >= -scrollY/h
                const startIndex = Math.max(0, Math.floor(-this.scrollY / this.itemHeight));
                // End index
                // i*h + scrollY <= viewHeight
                const viewHeight = this.size[1] - this.listStartY;
                const endIndex = Math.min(this.captionData.length, Math.ceil((-this.scrollY + viewHeight) / this.itemHeight));

                for (let i = startIndex; i < endIndex; i++) {
                    const item = this.captionData[i];
                    const itemY = y + (i * this.itemHeight) + this.scrollY;
                    const isSelected = item.basename === this.selectedBasename;

                    // Background
                    ctx.fillStyle = isSelected ? "#445" : (i % 2 === 0 ? "#222" : "#2a2a2a");
                    ctx.fillRect(margin, itemY, w - margin * 2, this.itemHeight);

                    if (isSelected) {
                        ctx.strokeStyle = "#4a9";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(margin, itemY, w - margin * 2, this.itemHeight);
                    }

                    // Thumbnail
                    if (!item.imgObj) {
                        item.imgObj = new Image();
                        item.imgObj.src = item.thumbnail;
                    }

                    if (item.imgObj.complete) {
                        const scale = Math.min(100 / item.imgObj.width, 100 / item.imgObj.height);
                        const dw = item.imgObj.width * scale;
                        const dh = item.imgObj.height * scale;
                        ctx.drawImage(item.imgObj, margin + 5, itemY + 5, dw, dh);
                    }

                    // Utils for Text 
                    const textX = margin + 115;
                    const textMaxWidth = w - margin * 3 - 115;

                    // Filename
                    ctx.fillStyle = "#ddd";
                    ctx.font = "bold 12px Arial";
                    ctx.fillText(item.filename, textX, itemY + 20);

                    // Caption
                    ctx.font = "12px Arial";
                    const caption = item.caption || "";
                    const words = caption.split(" ");
                    let line = "";
                    let ly = itemY + 40;

                    for (let n = 0; n < words.length; n++) {
                        const testLine = line + words[n] + " ";
                        const metrics = ctx.measureText(testLine);
                        if (metrics.width > textMaxWidth && n > 0) {
                            ctx.fillText(line, textX, ly);
                            line = words[n] + " ";
                            ly += 15;
                            if (ly > itemY + this.itemHeight - 5) break;
                        } else {
                            line = testLine;
                        }
                    }
                    ctx.fillText(line, textX, ly);
                }

                ctx.restore();

                // Scrollbar
                if (totalHeight > viewHeight) {
                    const sbWidth = 6;
                    const sbX = w - sbWidth - 2;
                    const sbY = y;
                    const sbHeight = viewHeight;

                    ctx.fillStyle = "#111";
                    ctx.fillRect(sbX, sbY, sbWidth, sbHeight);

                    const thumbHeight = Math.max(20, (viewHeight / totalHeight) * viewHeight);
                    const scrollRatio = -this.scrollY / (totalHeight - viewHeight); // 0 to 1
                    const thumbY = sbY + scrollRatio * (sbHeight - thumbHeight);

                    ctx.fillStyle = "#666";
                    ctx.fillRect(sbX, thumbY, sbWidth, thumbHeight);
                }
            };
        }
    },
});
