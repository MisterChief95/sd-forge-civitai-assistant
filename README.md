# Civitai Assistant for Stable Diffusion Forge WebUI

![Window](https://github.com/user-attachments/assets/fb74c2ba-3c26-4241-92ad-96cab6145b46)

## Overview

Civitai Assistant is an extension for the Stable Diffusion Forge WebUI. Its primary purpose is to fetch metadata for various models, including checkpoints, Loras, and embeddings.

## Features

- Supports the following model types:
  1. Checkpoints
  1. LoRA/DoRA/LyCORIS
  1. Textual Inversions (Embeddings)
- Update model tags
  - LoRA activation text
  - SD version
  - SHA-256 hash for API
  - Listing ID/model ID
- Clean UI

## Work in Progress

🚧 **Currently being worked on:**
- Improved metadata fetching speed
- Checking for new model versions

📝 **Planned features:**
- Downloading models
  - By URL
  - Simple integrated browser
- More model support
  - Hypernetwork
  - AestheticGradient
  - Controlnet
  - Poses

## Installation

1. Navigate to the extension directory in your WebUI installation
1. Clone the repository:
    ```sh
    git clone https://github.com/MisterChief95/sd-forge-civitai-Assistant.git
    ```
1. Start WebUI

## Usage

Civitai Assistant adds a new tab to the UI where you can select which models to have their metadata updated.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

```
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/MisterChief95/sd-forge-civitai-Assistant/issues).
