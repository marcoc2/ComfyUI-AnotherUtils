import { app } from "../../scripts/app.js";

// Helper to hide sync widgets
function hideWidget(node, widget) {
    if (!widget) return;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget";
    if (widget.element) {
        widget.element.style.display = "none";
    }
}

console.log("[PointCollector] Loading PointCollector extension...");

app.registerExtension({
    name: "AnotherUtils.PointCollector",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "PointCollectorSAM2") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                console.log("[PointCollector] Node created:", this.id);
                const result = onNodeCreated?.apply(this, arguments);

                // Initial size
                this.size = [400, 400];

                // Create container
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; background: #111; display: flex; align-items: center; justify-content: center;";

                // Info bar
                const infoBar = document.createElement("div");
                infoBar.style.cssText = "position: absolute; top: 5px; left: 5px; right: 5px; z-index: 10; display: flex; justify-content: space-between;";
                container.appendChild(infoBar);

                const counter = document.createElement("div");
                counter.style.cssText = "background: rgba(0,0,0,0.8); color: #0f0; padding: 2px 8px; font-size: 11px; font-family: monospace; border-radius: 3px;";
                counter.textContent = "Points: 0p / 0n";
                infoBar.appendChild(counter);

                const btn = document.createElement("button");
                btn.textContent = "Clear";
                btn.style.cssText = "background: #622; color: #fff; border: none; padding: 2px 8px; cursor: pointer; border-radius: 3px;";
                btn.onclick = (e) => {
                    e.preventDefault();
                    this.canvasWidget.pos = [];
                    this.canvasWidget.neg = [];
                    this.updatePoints();
                    this.draw();
                };
                infoBar.appendChild(btn);

                // Canvas
                const canvas = document.createElement("canvas");
                canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; cursor: crosshair;";
                container.appendChild(canvas);

                this.canvasWidget = {
                    canvas, ctx: canvas.getContext("2d"),
                    pos: [], neg: [], image: null, counter
                };

                const widget = this.addDOMWidget("canvas", "pointsPreview", container);
                widget.computeSize = (width) => [width, this.canvasWidget.height || 300];
                console.log("[PointCollector] DOM widget added");

                // Hide strings
                const wCoords = this.widgets?.find(w => w.name === "coordinates");
                const wNegCoords = this.widgets?.find(w => w.name === "neg_coordinates");
                console.log("[PointCollector] Widgets found to hide:", { wCoords, wNegCoords });
                hideWidget(this, wCoords);
                hideWidget(this, wNegCoords);

                // Click Handlers
                canvas.addEventListener("mousedown", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    const x = (e.clientX - rect.left) * scaleX;
                    const y = (e.clientY - rect.top) * scaleY;

                    if (e.button === 0 && !e.shiftKey) {
                        this.canvasWidget.pos.push({x, y});
                    } else {
                        this.canvasWidget.neg.push({x, y});
                    }
                    this.updatePoints();
                    this.draw();
                });

                canvas.addEventListener("contextmenu", (e) => e.preventDefault());

                this.onExecuted = (msg) => {
                    if (msg.bg_image?.[0]) {
                        const img = new Image();
                        img.onload = () => {
                            this.canvasWidget.image = img;
                            canvas.width = img.width;
                            canvas.height = img.height;
                            const aspectRatio = img.height / img.width;
                            this.canvasWidget.height = (this.size[0] - 20) * aspectRatio;
                            this.draw();
                        };
                        img.src = "data:image/jpeg;base64," + msg.bg_image[0];
                    }
                };

                this.updatePoints = () => {
                    const wPos = this.widgets.find(w => w.name === "coordinates");
                    const wNeg = this.widgets.find(w => w.name === "neg_coordinates");
                    if (wPos) wPos.value = JSON.stringify(this.canvasWidget.pos);
                    if (wNeg) wNeg.value = JSON.stringify(this.canvasWidget.neg);
                    this.canvasWidget.counter.textContent = `Points: ${this.canvasWidget.pos.length}p / ${this.canvasWidget.neg.length}n`;
                };

                this.draw = () => {
                    const {canvas, ctx, image, pos, neg} = this.canvasWidget;
                    ctx.clearRect(0,0, canvas.width, canvas.height);
                    if (image) ctx.drawImage(image, 0, 0);
                    
                    ctx.lineWidth = 2;
                    pos.forEach(p => {
                        ctx.fillStyle = "#0f0"; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI*2); ctx.fill();
                    });
                    neg.forEach(p => {
                        ctx.fillStyle = "#f00"; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI*2); ctx.fill();
                    });
                };

                return result;
            };
        }
    }
});
