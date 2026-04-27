# CAPI: Coordinated Autoencoders for Persona Inference

Neural Clustering Architecture for Tabular Data (Customer Segmentation/Persona Creation)
This repo contains the official PyTorch implementation for CAPI: Coordinated Autoencoders for Persona Inference. CAPI is designed for clustering tabular data, leveraging autoencoders and a routing network. This work expands on existing work that follows a Mixture of Experts approach for clustering images and text. 

One key contribution from this work is the novel training schedule, which dynamically adjusts its own hyperparameters throughout training in a self-learning loop. This approach prevents model collapse and eliminates the need for manual hyperparameter tuning. This is specially valauable for practitioners, since clustering tasks contain no labels for efficient parameter tuning. 

## 1.1: How to use on your own dataset
1. Clone the repo
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Edit `main.py` for your own dataset. If you don't have one, a default toy dataset is set already
4. Run:
   ```
   python main.py
   ```

For an interactive walkthrough, open `usage_example.ipynb` at the repo root.

## TO DO:
1. Functions need comments across the code
2. Output should return learned embeddings
2. Make the resulting column from the inference function return the ORIGINAL feature values along with the cluster assignments. Currently it returns the ENCODED features with the cluster assignments
