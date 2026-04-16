import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AnotherUtils.AnotherShowList",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AnotherShowList") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.displayText = "";
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);
                if (message?.text) {
                    this.displayText = message.text[0];
                    this.setDirtyCanvas(true);
                }
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                if (onDrawForeground) onDrawForeground.apply(this, arguments);
                
                if (this.displayText) {
                    const margin = 10;
                    ctx.save();
                    ctx.font = "14px monospace";
                    ctx.fillStyle = "#fff";
                    
                    const words = this.displayText.split(" ");
                    let line = "";
                    let y = 60;
                    const maxWidth = this.size[0] - margin * 2;

                    // Support multi-line wrap if the list is very long
                    for (let n = 0; n < words.length; n++) {
                        const testLine = line + words[n] + " ";
                        const metrics = ctx.measureText(testLine);
                        if (metrics.width > maxWidth && n > 0) {
                            ctx.fillText(line, margin, y);
                            line = words[n] + " ";
                            y += 20;
                        } else {
                            line = testLine;
                        }
                    }
                    ctx.fillText(line, margin, y);
                    
                    // Auto-adjust node height if text exceeds current size
                    if (y + 20 > this.size[1]) {
                        this.size[1] = y + 30;
                    }
                    
                    ctx.restore();
                }
            };
        }
    },
});
