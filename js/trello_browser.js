import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "AnotherUtils.TrelloBrowser",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "TrelloBrowser") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.trelloData = [];
                this.statusMessage = "Load JSON to browse prompts.";
                
                // UI State
                this.scrollY = 0;
                this.itemHeight = 120; // Increased
                this.listStartY = 140; 
                this.selectedId = "";

                this.setSize([400, 600]); // Set default size

                // Helper to fetch Trello data
                this.fetchTrelloData = async () => {
                    const jsonPathWidget = this.widgets.find(w => w.name === "json_path");
                    if (!jsonPathWidget || !jsonPathWidget.value) {
                        this.statusMessage = "Missing JSON path.";
                        this.setDirtyCanvas(true);
                        return;
                    }

                    this.statusMessage = "Loading Trello data...";
                    this.setDirtyCanvas(true);

                    try {
                        const response = await api.fetchApi(`/another_utils/trello_prompts?json_path=${encodeURIComponent(jsonPathWidget.value)}`);
                        if (!response.ok) throw new Error("Failed to fetch");
                        
                        const data = await response.json();
                        this.trelloData = data.cards || [];
                        this.statusMessage = this.trelloData.length === 0 ? "No cards found." : "";
                        this.scrollY = 0;
                        
                        // Sync selectedId with widget
                        const idWidget = this.widgets.find(w => w.name === "selected_id");
                        if (idWidget) this.selectedId = idWidget.value;

                        this.setDirtyCanvas(true);
                    } catch (e) {
                        this.statusMessage = "Error: " + e.message;
                        this.setDirtyCanvas(true);
                    }
                };

                // Add Refresh Button
                this.addWidget("button", "Refresh Trello Browser", null, () => {
                   this.fetchTrelloData();
                });

                // Auto-load on path change
                const pathWidget = this.widgets.find(w => w.name === "json_path");
                if (pathWidget) {
                    const cb = pathWidget.callback;
                    pathWidget.callback = (v) => {
                        if (cb) cb(v);
                        this.fetchTrelloData();
                    };
                }

                // Hide the selected_id widget - we handle it visually
                const idWidget = this.widgets.find(w => w.name === "selected_id");
                if (idWidget) {
                    idWidget.computeSize = () => [0, -4];
                }

                // Initial fetch
                setTimeout(() => this.fetchTrelloData(), 500);

                return r;
            };

            // Input Handling (simplified from caption_image_loader)
            nodeType.prototype.onMouseDown = function (event, pos, graphCanvas) {
                const x = pos[0];
                const y = pos[1];
                if (y < this.listStartY) return;

                const filterWidget = this.widgets.find(w => w.name === "list_filter");
                const filter = filterWidget ? filterWidget.value : "All";
                const filtered = filter === "All" ? this.trelloData : this.trelloData.filter(c => c.list === filter);
                
                const w = this.size[0];
                const totalHeight = filtered.length * this.itemHeight;
                const viewHeight = this.size[1] - this.listStartY;
                const sbWidth = 10;
                const sbX = w - sbWidth - 2;

                // Scrollbar interaction
                if (totalHeight > viewHeight && x >= sbX) {
                    this.isDraggingScroll = true;
                    this.lastMouseY = y;
                    return true;
                }

                const index = Math.floor((y - this.listStartY - this.scrollY) / this.itemHeight);
                if (index >= 0 && index < filtered.length) {
                    const card = filtered[index];
                    this.selectedId = card.id;
                    const idWidget = this.widgets.find(w => w.name === "selected_id");
                    if (idWidget) idWidget.value = card.id;
                    this.setDirtyCanvas(true);
                    return true;
                }
            };

            nodeType.prototype.onMouseMove = function (event, pos, graphCanvas) {
                if (this.isDraggingScroll) {
                    const y = pos[1];
                    const dy = y - this.lastMouseY;
                    this.lastMouseY = y;

                    const filterWidget = this.widgets.find(w => w.name === "list_filter");
                    const filter = filterWidget ? filterWidget.value : "All";
                    const filtered = filter === "All" ? this.trelloData : this.trelloData.filter(c => c.list === filter);

                    const totalHeight = filtered.length * this.itemHeight;
                    const viewHeight = this.size[1] - this.listStartY;
                    const scrollRange = totalHeight - viewHeight;
                    const thumbHeight = Math.max(20, (viewHeight / totalHeight) * viewHeight);
                    const thumbRange = viewHeight - thumbHeight;

                    if (thumbRange > 0) {
                        this.scrollY -= dy * (scrollRange / thumbRange);
                        this.scrollY = Math.max(-scrollRange, Math.min(0, this.scrollY));
                        this.setDirtyCanvas(true);
                    }
                    return true;
                }
            };

            nodeType.prototype.onMouseUp = function (event, pos, graphCanvas) {
                if (this.isDraggingScroll) {
                    this.isDraggingScroll = false;
                    return true;
                }
            };

            nodeType.prototype.onMouseWheel = function (event, pos, graphCanvas) {
                if (pos[1] > this.listStartY) {
                    this.scrollY -= event.deltaY * 0.8; // Faster
                    const viewHeight = this.size[1] - this.listStartY;
                    
                    const filterWidget = this.widgets.find(w => w.name === "list_filter");
                    const filter = filterWidget ? filterWidget.value : "All";
                    const filtered = filter === "All" ? this.trelloData : this.trelloData.filter(c => c.list === filter);
                    
                    const totalHeight = filtered.length * this.itemHeight;
                    const minScroll = Math.min(0, -(totalHeight - viewHeight) - 20);
                    this.scrollY = Math.max(minScroll, Math.min(0, this.scrollY));
                    this.setDirtyCanvas(true);
                    return true;
                }
            };

            // Custom Drawing
            nodeType.prototype.onDrawForeground = function (ctx) {
                const margin = 10;
                const w = this.size[0];
                const h = this.size[1];
                let y = this.listStartY;
                const viewHeight = h - y;

                ctx.save();
                
                // Clip
                ctx.beginPath();
                ctx.rect(0, y, w, viewHeight);
                ctx.clip();

                if (this.statusMessage) {
                    ctx.fillStyle = "#aaa";
                    ctx.font = "italic 14px Arial";
                    ctx.fillText(this.statusMessage, margin, y + 25 + this.scrollY);
                }

                const filterWidget = this.widgets.find(w => w.name === "list_filter");
                const filter = filterWidget ? filterWidget.value : "All";
                const filtered = filter === "All" ? this.trelloData : this.trelloData.filter(c => c.list === filter);
                const totalHeight = filtered.length * this.itemHeight;

                for (let i = 0; i < filtered.length; i++) {
                    const item = filtered[i];
                    const itemY = y + (i * this.itemHeight) + this.scrollY;
                    if (itemY + this.itemHeight < y || itemY > h) continue;

                    const isSelected = item.id === this.selectedId;

                    // Background
                    ctx.fillStyle = isSelected ? "#445" : (i % 2 === 0 ? "#1a1a1a" : "#222");
                    const itemW = w - margin * 2 - (totalHeight > viewHeight ? 12 : 0);
                    ctx.fillRect(margin, itemY, itemW, this.itemHeight - 4);

                    if (isSelected) {
                        ctx.strokeStyle = "#4a9";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(margin, itemY, itemW, this.itemHeight - 4);
                    }

                    // Image Preview
                    if (item.image_url) {
                        if (!item._img) {
                            item._img = new Image();
                            item._img.src = item.image_url;
                        }
                        if (item._img.complete) {
                            ctx.drawImage(item._img, margin + 5, itemY + 5, 110, 110);
                        } else {
                            ctx.fillStyle = "#333";
                            ctx.fillRect(margin + 5, itemY + 5, 110, 110);
                        }
                    }

                    // Text
                    ctx.fillStyle = isSelected ? "#fff" : "#eee";
                    ctx.font = "bold 13px Arial";
                    const textX = item.image_url ? margin + 125 : margin + 15;
                    
                    let name = item.name;
                    if (ctx.measureText(name).width > itemW - 130) {
                        name = name.substring(0, 50) + "...";
                    }
                    ctx.fillText(name, textX, itemY + 30);
                    
                    ctx.fillStyle = "#aaa";
                    ctx.font = "11px Arial";
                    ctx.fillText("List: " + item.list, textX, itemY + 50);
                }

                ctx.restore();

                // Draw Scrollbar
                if (totalHeight > viewHeight) {
                    const sbWidth = 8;
                    const sbX = w - sbWidth - 4;
                    const sbY = y;
                    const thumbHeight = Math.max(30, (viewHeight / totalHeight) * viewHeight);
                    const scrollRatio = -this.scrollY / (totalHeight - viewHeight);
                    const thumbY = sbY + scrollRatio * (viewHeight - thumbHeight);

                    ctx.fillStyle = "#111";
                    ctx.fillRect(sbX, sbY, sbWidth, viewHeight);
                    ctx.fillStyle = "#555";
                    ctx.fillRect(sbX, thumbY, sbWidth, thumbHeight);
                }
            };
        }
    }
});
