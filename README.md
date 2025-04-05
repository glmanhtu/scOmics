# SC Omics network
### A Transformer-based network for single-cell omics modality prediction

Please refer to the accompanying jupyter notebook for the details of the intuition and 
motivation behind the model.

### Installation
```bash
pip install -r requirements.txt
```

### Model training
```bash
python train.py --data_path /path/to/dataset/Lung --batch_size 256 --learning_rate 0.0001 --n_bins 21 --enable-amp --n_epochs 50
```