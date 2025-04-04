# README

## Installation

Navigate to your project directory in the terminal and execute the following command to install the required packages:

```
pip install -r requirements.txt
```

This will install all the packages listed in the file.

Updating requirements.txt
If you add new packages during the project, update the requirements.txt file with:

```
pip freeze > requirements.txt
```
This will list all currently installed packages and their versions.


## Install new package
1. add package to ```requirements.in```
2. run ```pip-compile requirements.in```
3. run ```pip-sync```


## Training the ResNet18 Model

To train the ResNet18 model on the *Flowers102* dataset, follow these steps:

1. **Configuration**: Use the provided YAML configuration file to set your training parameters. Key settings include:
   - `epochs`: Number of training epochs (e.g., 100).
   - `batch_size`: Size of each training batch (e.g., 64).
   - `learning_rate`: Initial learning rate (e.g., 0.001).
   - `lr_scheduler`: Learning rate scheduler (e.g., `CosineAnnealingLR`).
   - `num_workers`: Number of data loading workers (e.g., 2).
   - `output_folder`: Directory to save logs and checkpoints (e.g., `logging`).
   - `model`: Specify `resnet18` to train this model.
   - `pretrained`: Set to `true` to use a pretrained model.
   - `dataset_name`: Use `flowers102` for this dataset.
   - `max_classes`: Use `-1` to include all classes
   - `checkpoint`: Path to a checkpoint file, or leave empty if not using.

2. **Execute Training**: Run the training script with the YAML configuration:

   ```
   python train_classifier.py --config configs/train_resnet18_flowers102.yaml
   ```
3. **Output:** Training artifacts such as the config YAML, best checkpoint, last checkpoint, and the logfile will be saved in the logging folder specified in output_folder.


## t-SNE Visualization

Use the `tsne.py` script to visualize your model's output using t-SNE.

### Parameters

- `-c`, `--config`: Path to the training configuration file (required).
- `-cp`, `--checkpoint-path`: Path to the model checkpoint file (required).
- `--max`: Limit the number of classes for the t-SNE plot. Set `--max=0` to use all classes (default is 10).
- `--plot`: Choose between a `2d` or `3d` plot (default is `2d`).

### Usage

Run the script as follows:

```
python tsne.py -c path/to/config.yaml -cp path/to/checkpoint.pth --max=10 --plot=2d
```

### Output
- 2D Plot: Creates a 2D t-SNE plot using Bokeh. Hovering over a point displays the image and label.
- 3D Plot: Displays only the label when hovering over a point.