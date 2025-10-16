// ...existing code...
# Art Style Classification

This repository contains code and notebooks for training multimodal Vision Transformer (ViT) and CLIP-based models on the ArtBench 10-way art-style dataset.

Quick overview
- The pipeline:
  1. Prepare dataset using the custom dataset class `CustomImageTextDataset` (see `multimodal_vit_and_clip_based_art_classification.py`).  
  2. Create ViT feature-extractor via `create_vit_model` (see `multimodal_vit_and_clip_based_art_classification.py`).  
  3. Train and evaluate with `train_and_evaluate` (see `multimodal_vit_and_clip_based_art_classification.py`).  
  4. Save model with `save_model` and results with `save_results_to_CSV` (see `multimodal_vit_and_clip_based_art_classification.py`).  
  5. Measure prediction latency via `pred_and_timing` (see `multimodal_vit_and_clip_based_art_classification.py`).  
  6. Helper utilities: `get_model_summary`, `plot_loss_curves`, and accuracy calculation `accuracy_fn` (see `multimodal_vit_and_clip_based_art_classification.py`).

Key functions / classes
- `CustomImageTextDataset` — Dataset that pairs images with precomputed text embeddings.
- `create_vit_model` — Builds the ViT_B16 model and transforms (optionally with TrivialAugmentWide).
- `train_and_evaluate` — Full training loop and evaluation.
- `pred_and_timing` — Runs predictions on sample images and computes average inference time.
- `save_model` — Saves the model state_dict.
- `get_model_summary` — Uses torchinfo to print model stats.
- `plot_loss_curves` — Plots training/test loss and accuracy.
- `save_results_to_CSV` — Export results to CSV for later analysis.

Where outputs are saved
- Results and graphs: `kaggle/working/results` (created by the scripts).
- Trained models: `kaggle/working/models`.

How to run
1. Open and run the notebook:
   - All/Art-Style-Classification-/Multimodal_ViT_and_Clip_based_Art_classification.ipynb
2. Or run the script (example):
   - python All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py
   - The script is exported from Colab; adjust FLAGS and paths at the top of the file before running.

Notes / tips
- The dataset used is the ArtBench imagefolder split (256x256). See data path variable and download block in `multimodal_vit_and_clip_based_art_classification.py`.
- For experiments, check saved CSVs in the `results` folder and compare feature-extraction vs fine-tuning using `compare_results`.
