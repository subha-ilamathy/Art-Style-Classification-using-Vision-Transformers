# Art Style Classification

This repository contains code and notebooks for training multimodal Vision Transformer (ViT) and CLIP-based models on the ArtBench 10-way art-style dataset.

Quick overview
- The pipeline:
  1. Prepare dataset using the custom dataset class [`CustomImageTextDataset`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py).  
  2. Create ViT feature-extractor via [`create_vit_model`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py).  
  3. Train and evaluate with [`train_and_evaluate`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py).  
  4. Save model with [`save_model`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py) and results with [`save_results_to_CSV`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py).  
  5. Measure prediction latency via [`pred_and_timing`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py).  
  6. Helper utilities: [`get_model_summary`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py), [`plot_loss_curves`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py), and accuracy calculation [`accuracy_fn`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py).

Key functions / classes
- [`CustomImageTextDataset`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py) — Dataset that pairs images with precomputed text embeddings.
- [`create_vit_model`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py) — Builds the ViT_B16 model and transforms (optionally with TrivialAugmentWide).
- [`train_and_evaluate`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py) — Full training loop and evaluation.
- [`pred_and_timing`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py) — Runs predictions on sample images and computes average inference time.
- [`save_model`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py) — Saves the model state_dict.
- [`get_model_summary`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py) — Uses torchinfo to print model stats.
- [`plot_loss_curves`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py) — Plots training/test loss and accuracy.
- [`save_results_to_CSV`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py) — Export results to CSV for later analysis.

Where outputs are saved
- Results and graphs: `kaggle/working/results` (created by the scripts).
- Trained models: `kaggle/working/models`.

How to run
1. Open and run the notebook:
   - [All/Art-Style-Classification-/Multimodal_ViT_and_Clip_based_Art_classification.ipynb](All/Art-Style-Classification-/Multimodal_ViT_and_Clip_based_Art_classification.ipynb)
2. Or run the script (example):
   - python All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py
   - The script is exported from Colab; adjust FLAGS and paths at the top of the file before running.

Notes / tips
- The dataset used is the ArtBench imagefolder split (256x256). See data path variable and download block in [`multimodal_vit_and_clip_based_art_classification.py`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py).
- For experiments, check saved CSVs in the `results` folder and compare feature-extraction vs fine-tuning using [`compare_results`](All/Art-Style-Classification-/multimodal_vit_and_clip_based_art_classification.py).
