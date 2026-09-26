// Fills in the lyrics when a song is picked, and the style/strengths when a LoRA is picked.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function onChange(node, name, fn) {
  const w = widget(node, name);
  if (!w) return;
  const original = w.callback;
  w.callback = function (value, ...rest) {
    const r = original?.apply(this, [value, ...rest]);
    fn(w.value);
    return r;
  };
}

async function fetchJson(route, params) {
  const r = await api.fetchApi(`${route}?${new URLSearchParams(params)}`);
  return r.ok ? r.json() : {};
}

app.registerExtension({
  name: "AnotherUtils.YuE2Cover",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "YuE2CoverSong") return;
    // The core upload button looks up the "audioUI" player, which core only creates for LoadAudio and friends.
    // The player has to exist before the upload widget, so it goes ahead of it in the widget order.
    const req = nodeData.input.required;
    const upload = req.upload;
    delete req.upload;
    req.audioUI = ["AUDIO_UI", {}];
    if (upload) req.upload = upload;

    // After a run the box shows the lyrics that ran (structured by DeepSeek when they had no tags),
    // so the next run starts from the tagged version and does not call DeepSeek again.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const lyrics = message?.lyrics?.[0];
      const w = widget(this, "lyrics");
      if (typeof lyrics === "string" && w) {
        w.value = lyrics;
        this.setDirtyCanvas(true, true);
      }
    };
  },
  nodeCreated(node) {
    if (node.comfyClass === "YuE2CoverSong") {
      onChange(node, "audio", async (song) => {
        const { lyrics } = await fetchJson("/another_utils/yue2cover/lyrics", { song });
        // No saved lyrics: clear the box, so the previous song's lyrics are not saved to this one.
        widget(node, "lyrics").value = lyrics ?? "";
        node.setDirtyCanvas(true, true);
      });
    }
    if (node.comfyClass === "YuE2CoverLoras") {
      onChange(node, "lora", async (lora) => {
        const { profile } = await fetchJson("/another_utils/yue2cover/profile", { lora });
        widget(node, "style").value = profile?.style ?? "";
        if (profile) {
          widget(node, "strength_model").value = profile.strength_model;
          widget(node, "strength_clip").value = profile.strength_clip;
          widget(node, "instrumental").value = !!profile.instrumental;
        }
        node.setDirtyCanvas(true, true);
      });
    }
  },
});
