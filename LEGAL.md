# Legal Compliance Guidelines for Sharing Project Content

This file outlines the legal restrictions and guidelines for sharing any content related to this Autopilot project. **Always refer to this file before sharing code, data, results, or media.** Failure to comply may violate licenses, especially for the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

## Key Licenses and Restrictions

### NVIDIA PhysicalAI-Autonomous-Vehicles Dataset
- **Source**: Hugging Face dataset (https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
- **License**: Research License Agreement (see https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/blob/main/LICENSE.pdf).
- **Confidential Information**: Includes the dataset, any derivatives, and information that should reasonably be understood as confidential.
- **Prohibited Actions**:
  - Redistribute, disclose, publish, or make available the dataset or any part of it to third parties.
  - Share derivatives, including training results (e.g., CSVs, charts, model weights), videos, or visualizations based on the dataset.
  - Use for commercial purposes without permission.
- **Allowed Uses**: Internal research and development only. No public sharing of outputs derived from the dataset.

### Other Datasets (e.g., GoPro/GPS Legacy Data)
- If using your own or public data, standard open-source licenses apply (e.g., MIT for code).
- No restrictions on sharing results or demos from non-NVIDIA data.

## What Can Be Shared
- **Code**: The source code (Python scripts, models) is yours and can be shared publicly (e.g., on GitHub) under your chosen license (e.g., MIT).
- **Qualitative Demos**: Videos or images of the model performing on your own recorded footage (not from NVIDIA data).
- **Documentation**: README, architecture diagrams, and general project descriptions.
- **Anonymized Results**: If sharing training trends, use normalized/qualitative data without specific numbers or NVIDIA-derived metrics.

## What Cannot Be Shared
- **NVIDIA Dataset Content**: Any raw data, processed data, or excerpts (e.g., videos, images, parquet files).
- **Training Outputs**: CSVs (e.g., training_history.csv), charts (e.g., loss/ADE-FDE plots), model weights (e.g., .pth files), or any quantitative results from training on NVIDIA data.
- **Videos/Images from NVIDIA Data**: Demos or visualizations using NVIDIA footage, even with overlays.
- **Derivatives**: Anything that reveals insights into the NVIDIA dataset's content or performance.

## Best Practices
- **Before Sharing**: Review this file and the full NVIDIA license. If unsure, consult legal counsel or NVIDIA.
- **Version Control**: Sensitive files (e.g., training results) should be excluded from Git (see .gitignore).
- **Attribution**: Always credit sources appropriately, but do not share restricted content.
- **Presentations**: When sharing results internally (e.g., to a school board), add a subtle watermark to slides containing NVIDIA-derived content with text like "Confidential - Derived from NVIDIA Dataset - No Redistribution Allowed" to visually reinforce restrictions.
- **Contact**: If you need clarification, reach out to NVIDIA or a legal expert.