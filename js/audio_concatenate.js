import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AnotherUtils.AudioConcatenate",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "AudioConcatenate") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            var self = this;
            this.addWidget("button", "Update Inputs", null, function () {
                if (!self.inputs) {
                    self.inputs = [];
                }
                var targetCount = self.widgets.find(function (w) { return w.name === "inputcount"; });
                if (!targetCount) return;

                var target = targetCount.value;
                var currentAudioInputs = self.inputs.filter(function (inp) { return inp.type === "AUDIO"; });
                var currentCount = currentAudioInputs.length;

                if (target === currentCount) return;

                if (target < currentCount) {
                    // Remove from the end
                    var toRemove = currentCount - target;
                    for (var i = 0; i < toRemove; i++) {
                        // Find last AUDIO input
                        for (var j = self.inputs.length - 1; j >= 0; j--) {
                            if (self.inputs[j].type === "AUDIO") {
                                self.removeInput(j);
                                break;
                            }
                        }
                    }
                } else {
                    // Add new inputs
                    for (var i = currentCount + 1; i <= target; i++) {
                        self.addInput("audio_" + i, "AUDIO");
                    }
                }

                self.setSize(self.computeSize());
                self.setDirtyCanvas(true, true);
            });

            return r;
        };
    },
});
