import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DEBUG = true;
function log(...args) {
	if (DEBUG) console.log("[InteractiveCrop]", ...args);
}

function getImageUrl(imageWidgetValue) {
	if (!imageWidgetValue) return null;
	const url = api.apiURL(`/view?filename=${encodeURIComponent(imageWidgetValue)}&type=input`);
	log("getImageUrl:", imageWidgetValue, "->", url);
	return url;
}

app.registerExtension({
	name: "AnotherUtils.InteractiveCrop",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		log("beforeRegisterNodeDef called for:", nodeData.name);
		if (nodeData.name === "InteractiveCrop") {
			log("Registering InteractiveCrop extension");
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				log("onNodeCreated called, node id:", this.id);
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				// Debug: list all widgets
				log("All widgets:", this.widgets?.map(w => ({ name: w.name, type: w.type, value: w.value })));

				// Find the widgets by name (they are real widgets now)
				this.cropWidgets = {};
				for (const w of this.widgets || []) {
					if (["image", "roi_width", "roi_height", "crop_x", "crop_y"].includes(w.name)) {
						this.cropWidgets[w.name] = w;
						log("Found widget:", w.name, "=", w.value, "type:", w.type);
					}
				}
				log("cropWidgets keys:", Object.keys(this.cropWidgets));

				// CRITICAL: Disable the built-in image preview from the image widget
				// The image widget with image_upload:true adds its own preview
				// We use defineProperty to intercept and block any attempts to set imgs
				let _blockedImgs = null;
				Object.defineProperty(this, 'imgs', {
					get: () => null,  // Always return null
					set: (val) => {
						log("BLOCKED attempt to set this.imgs to:", val);
						_blockedImgs = val;  // Store but don't use
					},
					configurable: true
				});

				let _blockedImages = null;
				Object.defineProperty(this, 'images', {
					get: () => null,  // Always return null
					set: (val) => {
						log("BLOCKED attempt to set this.images to:", val);
						_blockedImages = val;
					},
					configurable: true
				});

				this.imageIndex = null;

				// Also check for the IMAGEUPLOAD widget that gets added
				log("Looking for IMAGEUPLOAD widget...");
				for (const w of this.widgets || []) {
					log("Widget:", w.name, w.type, w);
					if (w.name === "upload" || w.type === "IMAGEUPLOAD" || w.type === "button") {
						log("Found upload widget:", w);
					}
				}

				// Set callbacks for redraw
				if (this.cropWidgets.image) {
					this.cropWidgets.image.callback = () => {
						// Clear any built-in preview that may have been set
						this.imgs = null;
						this.images = null;
						this.imageIndex = null;
						this.updateImage();
					};
				}

				for (const name of ["roi_width", "roi_height", "crop_x", "crop_y"]) {
					if (this.cropWidgets[name]) {
						// Don't chain to original callback - it causes errors with new ComfyUI
						this.cropWidgets[name].callback = () => {
							this.setDirtyCanvas(true, true);
						};
					}
				}

				// Image element for drawing
				this.img = new Image();
				this.img.onload = () => {
					log("Image loaded:", this.img.src, "size:", this.img.naturalWidth, "x", this.img.naturalHeight);
					this.setDirtyCanvas(true, true);
				};
				this.img.onerror = (e) => {
					log("Image load ERROR:", e, this.img.src);
				};

				// Drag state
				this.isDragging = false;
				this.dragStartMouse = { x: 0, y: 0 };
				this.dragStartWidget = { x: 0, y: 0 };
				this.imageRect = null;

				// Initial load
				if (this.cropWidgets.image && this.cropWidgets.image.value) {
					this.updateImage();
				}

				return r;
			};

			nodeType.prototype.updateImage = function () {
				log("updateImage called, image widget value:", this.cropWidgets.image?.value);
				if (this.cropWidgets.image && this.cropWidgets.image.value) {
					const url = getImageUrl(this.cropWidgets.image.value);
					if (url) {
						log("Setting image src to:", url);
						this.img.src = url;
					}
				}
			};

			// --- Drawing ---
			const onDrawForeground = nodeType.prototype.onDrawForeground;
			nodeType.prototype.onDrawForeground = function (ctx) {
				// Log every 30 frames to see more
				this._drawCount = (this._drawCount || 0) + 1;
				const shouldLog = this._drawCount % 30 === 1;

				if (shouldLog) {
					log("onDrawForeground called, node size:", this.size,
						"img:", this.img?.src ? "loaded" : "not loaded",
						"complete:", this.img?.complete,
						"naturalSize:", this.img?.naturalWidth, "x", this.img?.naturalHeight);
					log("Has original onDrawForeground?", !!onDrawForeground);
				}

				// Check what the node has that could draw images
				if (shouldLog) {
					log("Node properties check - imgs:", this.imgs, "images:", this.images,
						"imageIndex:", this.imageIndex, "overIndex:", this.overIndex);
				}

				// DO NOT call original - we handle all drawing ourselves
				// if (onDrawForeground) {
				// 	if (shouldLog) log("Calling original onDrawForeground");
				// 	onDrawForeground.apply(this, arguments);
				// }

				if (!this.img || !this.img.src || !this.img.complete || this.img.naturalWidth === 0) {
					if (shouldLog) log("Image not ready, skipping draw");
					this.imageRect = null;
					return;
				}

				const maxWidth = this.size[0] - 20;
				// Calculate topPadding based on actual widget height
				// Each widget is roughly 20-25px, plus some margin
				const widgetCount = this.widgets ? this.widgets.length : 5;
				const topPadding = Math.max(150, widgetCount * 28 + 20); // Dynamic based on widget count
				const maxHeight = this.size[1] - topPadding - 10;

				if (this._drawCount % 30 === 1) {
					log("Widget count:", widgetCount, "topPadding:", topPadding);
				}

				if (shouldLog) {
					log("Draw calculations:", { maxWidth, topPadding, maxHeight, nodeSize: this.size });
				}

				if (maxHeight <= 0 || maxWidth <= 0) {
					if (shouldLog) log("Not enough space to draw");
					this.imageRect = null;
					return;
				}

				const scale = Math.min(maxWidth / this.img.naturalWidth, maxHeight / this.img.naturalHeight);
				const drawWidth = this.img.naturalWidth * scale;
				const drawHeight = this.img.naturalHeight * scale;

				const drawX = (this.size[0] - drawWidth) / 2;
				const drawY = topPadding + (maxHeight - drawHeight) / 2;

				this.imageRect = { x: drawX, y: drawY, w: drawWidth, h: drawHeight, scale: scale };

				if (shouldLog) {
					log("imageRect:", this.imageRect);
				}

				// Draw image
				try {
					ctx.drawImage(this.img, drawX, drawY, drawWidth, drawHeight);
					if (shouldLog) log("Drew image at", drawX, drawY, drawWidth, drawHeight);
				} catch (e) {
					log("ERROR drawing image:", e);
					return;
				}

				// Get ROI values safely
				const cropX = this.cropWidgets.crop_x ? Number(this.cropWidgets.crop_x.value) || 0 : 0;
				const cropY = this.cropWidgets.crop_y ? Number(this.cropWidgets.crop_y.value) || 0 : 0;
				const roiW = this.cropWidgets.roi_width ? Number(this.cropWidgets.roi_width.value) || 100 : 100;
				const roiH = this.cropWidgets.roi_height ? Number(this.cropWidgets.roi_height.value) || 100 : 100;

				if (shouldLog) {
					log("ROI values:", { cropX, cropY, roiW, roiH });
				}

				if (roiW > 0 && roiH > 0) {
					const boxX = drawX + cropX * scale;
					const boxY = drawY + cropY * scale;
					const boxW = roiW * scale;
					const boxH = roiH * scale;

					if (shouldLog) {
						log("Drawing ROI box at:", { boxX, boxY, boxW, boxH });
					}

					// Fill semi-transparent
					ctx.fillStyle = "rgba(255, 255, 0, 0.25)";
					ctx.fillRect(boxX, boxY, boxW, boxH);

					// Stroke
					ctx.setLineDash([]);
					ctx.lineWidth = 2;
					ctx.strokeStyle = "#FFFF00";
					ctx.strokeRect(boxX, boxY, boxW, boxH);
				}

				// Draw resolution caption below the image
				const caption = `${this.img.naturalWidth} x ${this.img.naturalHeight}`;
				ctx.font = "12px Arial";
				ctx.fillStyle = "#AAA";
				ctx.textAlign = "center";
				ctx.fillText(caption, this.size[0] / 2, drawY + drawHeight + 16);
			};

			// --- Mouse Interaction ---
			const onMouseDown = nodeType.prototype.onMouseDown;
			nodeType.prototype.onMouseDown = function (event, pos, graphCanvas) {
				log("onMouseDown called, pos:", pos, "imageRect:", this.imageRect);

				// pos is relative to the node's top-left corner
				if (!this.imageRect) {
					log("No imageRect, passing to original handler");
					return onMouseDown ? onMouseDown.apply(this, arguments) : false;
				}

				const x = pos[0];
				const y = pos[1];

				// Check if click is inside the image area
				const isInsideImage = (
					x >= this.imageRect.x &&
					x <= this.imageRect.x + this.imageRect.w &&
					y >= this.imageRect.y &&
					y <= this.imageRect.y + this.imageRect.h
				);

				log("Click at", x, y, "isInsideImage:", isInsideImage,
					"imageRect bounds:", this.imageRect.x, "-", this.imageRect.x + this.imageRect.w,
					",", this.imageRect.y, "-", this.imageRect.y + this.imageRect.h);

				if (isInsideImage) {
					log("Starting drag!");
					this.isDragging = true;

					const scale = this.imageRect.scale;
					const roiW = this.cropWidgets.roi_width ? Number(this.cropWidgets.roi_width.value) || 100 : 100;
					const roiH = this.cropWidgets.roi_height ? Number(this.cropWidgets.roi_height.value) || 100 : 100;

					// Convert mouse position to image coordinates
					const imgX = (x - this.imageRect.x) / scale;
					const imgY = (y - this.imageRect.y) / scale;

					// Center the ROI on the click position
					let newX = Math.floor(imgX - roiW / 2);
					let newY = Math.floor(imgY - roiH / 2);

					// Clamp to image bounds
					const maxX = Math.max(0, this.img.naturalWidth - roiW);
					const maxY = Math.max(0, this.img.naturalHeight - roiH);
					newX = Math.max(0, Math.min(newX, maxX));
					newY = Math.max(0, Math.min(newY, maxY));

					// Update widgets
					log("Setting crop_x to", newX, "crop_y to", newY);
					if (this.cropWidgets.crop_x) this.cropWidgets.crop_x.value = newX;
					if (this.cropWidgets.crop_y) this.cropWidgets.crop_y.value = newY;

					// Store for drag
					this.dragStartMouse = { x: x, y: y };
					this.dragStartWidget = { x: newX, y: newY };

					this.setDirtyCanvas(true, true);

					// CRITICAL: Capture pointer to prevent node dragging
					if (graphCanvas && graphCanvas.canvas) {
						this._capturedCanvas = graphCanvas.canvas;
						// Mark the graph canvas as not dragging
						if (graphCanvas.dragging_canvas) {
							graphCanvas.dragging_canvas = false;
						}
					}

					log("Returning true to consume event");
					return true; // Consume the event
				}

				log("Not inside image, passing to original handler");
				return onMouseDown ? onMouseDown.apply(this, arguments) : false;
			};

			const onMouseMove = nodeType.prototype.onMouseMove;
			nodeType.prototype.onMouseMove = function (event, pos, graphCanvas) {
				if (this.isDragging && this.imageRect) {
					const x = pos[0];
					const y = pos[1];
					const scale = this.imageRect.scale;

					const dx = (x - this.dragStartMouse.x) / scale;
					const dy = (y - this.dragStartMouse.y) / scale;

					let newX = Math.floor(this.dragStartWidget.x + dx);
					let newY = Math.floor(this.dragStartWidget.y + dy);

					// Clamp
					const roiW = this.cropWidgets.roi_width ? Number(this.cropWidgets.roi_width.value) || 100 : 100;
					const roiH = this.cropWidgets.roi_height ? Number(this.cropWidgets.roi_height.value) || 100 : 100;
					const maxX = Math.max(0, this.img.naturalWidth - roiW);
					const maxY = Math.max(0, this.img.naturalHeight - roiH);

					newX = Math.max(0, Math.min(newX, maxX));
					newY = Math.max(0, Math.min(newY, maxY));

					log("onMouseMove dragging, new position:", newX, newY);
					if (this.cropWidgets.crop_x) this.cropWidgets.crop_x.value = newX;
					if (this.cropWidgets.crop_y) this.cropWidgets.crop_y.value = newY;

					this.setDirtyCanvas(true, true);
					return true;
				}

				return onMouseMove ? onMouseMove.apply(this, arguments) : false;
			};

			const onMouseUp = nodeType.prototype.onMouseUp;
			nodeType.prototype.onMouseUp = function (event, pos, graphCanvas) {
				if (this.isDragging) {
					log("onMouseUp, ending drag");
					this.isDragging = false;
					this._capturedCanvas = null;
					this.setDirtyCanvas(true, true);
					return true;
				}

				return onMouseUp ? onMouseUp.apply(this, arguments) : false;
			};

			// Ensure node is large enough to show the image
			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (info) {
				log("onConfigure called", info);
				if (onConfigure) onConfigure.apply(this, arguments);
				// Reload image on configure (workflow load)
				if (this.cropWidgets && this.cropWidgets.image && this.cropWidgets.image.value) {
					this.updateImage();
				}
			};

			// Set minimum size
			const computeSize = nodeType.prototype.computeSize;
			nodeType.prototype.computeSize = function () {
				const size = computeSize ? computeSize.apply(this, arguments) : [200, 200];
				// Ensure enough height for widgets + image preview
				const widgetCount = this.widgets ? this.widgets.length : 5;
				const minHeight = widgetCount * 28 + 200; // widgets + image area
				return [Math.max(size[0], 320), Math.max(size[1], minHeight)];
			};

			// Debug: Check if node has images property (used for preview)
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (output) {
				log("onExecuted called, output:", output);
				log("this.images before:", this.images);
				if (onExecuted) onExecuted.apply(this, arguments);
				log("this.images after:", this.images);

				// IMPORTANT: ComfyUI may set this.imgs for preview. Let's check:
				log("this.imgs:", this.imgs);
			};

			// OVERRIDE onDrawBackground completely to prevent built-in image preview
			nodeType.prototype.onDrawBackground = function (ctx) {
				// Force clear any preview images that ComfyUI might have set
				if (this.imgs || this.images) {
					log("Clearing this.imgs/images that was set by ComfyUI, imgs:", this.imgs?.length, "images:", this.images?.length);
					this.imgs = null;
					this.images = null;
					this.imageIndex = null;
				}
				// DO NOT call any original - we don't want the built-in preview
			};
		}
	},
});
