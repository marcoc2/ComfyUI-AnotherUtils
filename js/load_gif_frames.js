import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "AnotherUtils.LoadGifFrames",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "LoadGifFrames") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const node = this;
            const gifWidget = this.widgets.find((w) => w.name === "gif");

            // Hidden file input that only accepts GIF
            const fileInput = document.createElement("input");
            Object.assign(fileInput, {
                type: "file",
                accept: ".gif,image/gif",
                style: "display: none",
                onchange: async () => {
                    const file = fileInput.files[0];
                    if (!file) return;

                    // Upload to ComfyUI input directory
                    const body = new FormData();
                    body.append("image", new File([file], file.name, { type: file.type, lastModified: file.lastModified }));
                    const resp = await api.fetchApi("/upload/image", { method: "POST", body });

                    if (resp.status !== 200) {
                        alert("GIF upload failed: " + resp.statusText);
                        return;
                    }

                    const data = await resp.json();
                    const filename = data.name;

                    // Add to dropdown options and select it
                    if (gifWidget) {
                        if (!gifWidget.options.values.includes(filename)) {
                            gifWidget.options.values.push(filename);
                        }
                        gifWidget.value = filename;
                        if (gifWidget.callback) gifWidget.callback(filename);
                    }
                },
            });
            document.body.append(fileInput);

            // Upload button
            const uploadBtn = this.addWidget("button", "Upload GIF", "gif_upload", () => {
                app.canvas.node_widget = null;
                fileInput.click();
            });
            uploadBtn.options.serialize = false;

            // Drag & drop support
            this.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
            this.onDragDrop = async (e) => {
                if (!e?.dataTransfer?.types?.includes?.("Files")) return false;
                const file = e.dataTransfer?.files?.[0];
                if (!file || file.type !== "image/gif") return false;
                fileInput.files = e.dataTransfer.files;
                fileInput.dispatchEvent(new Event("change"));
                return true;
            };

            // Remove fileInput when node is deleted
            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                fileInput?.remove();
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };

            return r;
        };
    },
});
