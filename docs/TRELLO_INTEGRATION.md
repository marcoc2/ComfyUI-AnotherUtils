# 📋 Trello Integrations & Prompt Management

A specialized benchmarking and prompt fetching suite designed to use Trello boards as active logic or dataset drivers.

### 🗃️ Trello Browser
A rich graphical node providing an advanced embedded web interface (via JavaScript). The Trello Browser lists active cards on a specific board/list, displaying their descriptions as prompts, and dynamically loads their image attachments natively into ComfyUI. 
- You can manually select what Prompt-Image pair to load into an ongoing generation.
- Highly useful for inspecting dataset tests before starting automated jobs.

### 🔄 Trello Prompt Loader
A headless, index-based iterator meant for automated benchmarking scripts.
- Designed to run systematically in bulk using ComfyUI batch queues or API integrations.
- Steps through a specified Trello list, extracting card names and desc payloads one-by-one by index.
