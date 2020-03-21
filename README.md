# dlminiproj

# Files:
parser.py – Configurations and hyperparameters

data.py – Dataset class for PASCAL VOC dataset, to pre-process data from data directory and return image with a binary encoding of target classes.

utils.py – Helper functions to get class-wise and mean average precision score, loss, tail accuracy and top 50 images

main.py – Main method, to run training or to reproduce results.

# To train, run:
python main.py --run train --data_dir <"path to PASCAL VOC data directory">

# To reproduce results, run:
python main.py --run results --data_dir <"path to PASCAL VOC data directory">
