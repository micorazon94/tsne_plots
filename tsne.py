import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import yaml
from io import BytesIO
from PIL import Image
import os

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show, output_file, save
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
from bokeh.layouts import column, row
from bokeh.models import CustomJS, Div
import plotly.graph_objects as go
import base64

from load_dataset import CustomDataset, create_dataloader
from utils.model_utils import load_model
from utils.json_files import get_class_labels

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def extract_features(dataloader, model, idx_to_class):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            # Convert remapped label indices to original class indices
            original_labels = [idx_to_class[target.item()] for target in targets]
            labels.extend(original_labels)
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    assert features.shape[0] == labels.shape[0], "Number of features and labels do not match"
    
    return features, labels

# Function to convert Numpy arrays to Base64-encoded images
def array_to_base64(arr):
    # Ensure the array is in the range [0, 255] and in uint8 format
    arr = ((arr + 1) * 127.5).clip(0, 255).astype('uint8')
    # Convert from (C, H, W) to (H, W, C)
    arr = arr.transpose(1, 2, 0)
    # Convert to RGB image
    img = Image.fromarray(arr, 'RGB')
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def create_2d_tsne_plot(testset, features, labels, output_file_path):
    """Create a 2D t-SNE plot using Bokeh.
    The plot will be interactive and can be zoomed and panned.
    By hovering over a point, the class number, the label and the image will be displayed."""

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    # Map numeric labels to class names
    class_names = get_class_labels(labels)

    # create a DataFrame with the t-SNE results and the labels
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'label': labels.astype(str),  # convert labels to strings
        'labelname': class_names,  # Add class names for display
        'image': [array_to_base64(testset[i][0].numpy()) for i in range(len(testset))]
    })

    source = ColumnDataSource(df)

    p = figure(width=800, height=600, title="t-SNE Visualization")

    # create a scatter plot with a color mapping based on the label
    p.scatter('x', 'y', source=source, size=5,
              color=factor_cmap('label', palette=Category10[10], factors=sorted(df.label.unique())))

    # create a div to display the image
    image_div = Div(text="", width=200, height=200)

    # add a HoverTool with a CustomJS callback to display the image on hover
    hover = HoverTool(tooltips=[
        ("Label", "@label"),
        ("Class Name", "@labelname"),
        ("(x, y)", "(@x, @y)"),
    ])
    p.add_tools(hover)

    hover.callback = CustomJS(args=dict(source=source, image_div=image_div), code="""
        const index = cb_data.index.indices[0];
        if (index !== undefined) {
            const image = source.data['image'][index];
            image_div.text = '<img src="' + image + '" width=200 height=200>';
        } else {
            image_div.text = '';
        }
    """)

    # show the plot
    layout = row(p, image_div)
    
    # show the plot in a browser
    show(layout)

    # Optionally, save the plot as an HTML file
    # output_file(output_file_path)
    # save(layout)
    # print(f"t-SNE plot saved to {output_file_path}")
    


def create_3d_tsne_plot(features, labels, output_file_path):
    """Create a 3D t-SNE plot using Plotly. 
    The plot will be interactive and can be rotated and zoomed.
    By hovering over a point, the label will be displayed."""

    # Map numeric labels to class names
    class_names = get_class_labels(labels)

    # Perform t-SNE with 3 components
    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Create a DataFrame with the t-SNE results and the labels
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'z': tsne_results[:, 2],
        'label': labels.astype(str),
        'labelname': class_names,  # Add class names for display
    })

    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['label'].astype('category').cat.codes,  # Color by label
            colorscale='Rainbow',
            opacity=0.8
        ),
        text=df['labelname'],  # This will be shown in the hover text
        hoverinfo='text'
    )])

    # Update the layout for better viewing
    fig.update_layout(
        title='3D t-SNE Visualization',
        autosize=False,
        width=900,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the plot
    fig.show()

    # Optionally, save the plot as an HTML file
    # output_file(output_file_path)
    # save(fig)
    # print(f"t-SNE plot saved to {output_file_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network on CIFAR-10')
    parser.add_argument('-c', '--config', type=str, help='path to train config file', required=True)
    parser.add_argument('-cp', '--checkpoint-path', type=str, help='path to the model checkpoint', required=True)
    parser.add_argument('--max', type=int, help="Limit number of classes for tsne plot. For using all classes, set --max=0", default=10)
    parser.add_argument('--plot', choices=['2d', '3d'], help="Select 2d or 3d plot", default='2d')
    return parser.parse_args()
    
    # -c configs/train_resnet18_flowers102.yaml -cp logging/2025-03-26_18-49-28/best.pth


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = args.checkpoint_path
    config_path = args.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)
    max_classes = args.max if args.max > 0 else num_classes
    print(f"Number of classes: {num_classes}")
    chosen_classes = list(class_to_idx.keys())

    # LOAD MODEL WITH CHECKPOINT AND SET TO EVAL MODE
    model = load_model(model_name=config['model'],
                       num_classes=num_classes,
                       pretrained=False,
                       checkpoint_path=checkpoint_path)
    model.eval()
    model = model.to(device)

    # LOAD DATA
    if num_classes >= max_classes:
        print(f"Select {max_classes} of {num_classes} classes.")
        # Randomly select classes
        chosen_classes = chosen_classes[:max_classes]
        print(f"Chosen classes: {chosen_classes}")
    elif num_classes < max_classes:
        raise ValueError(f"Number of classes {num_classes} is less than max_classes {max_classes}."
                         f"Please set max_classes to a value less than or equal to the number of classes.")
        
    testset = CustomDataset(dataset_name=config['dataset_name'], 
                            split="test",
                            max_classes=config['max_classes'],
                            chosen_classes=chosen_classes)
    
    print(f"Testdataset size: {len(testset)}")
    
    print(f"Created test dataset with {len(chosen_classes)} classes.")
    testloader = create_dataloader(testset, 
                                   batch_size=config["batch_size"], 
                                   num_workers=config['num_workers'], 
                                   train=False)

    # EXTRACT FEATURES AND CREATE TSNE PLOT
    # Reverse the mapping to get idx_to_class
    idx_to_class = {v: k for k, v in testset.class_to_idx.items()}
    print(idx_to_class)
    features, labels = extract_features(testloader, model, idx_to_class)

    # Create t-SNE plot
    output_folder = "tsne_output"
    os.makedirs(output_folder, exist_ok=True)

    if args.plot == '2d':
        # 2D plot
        output_file_path = os.path.join(output_folder, "tsne_2d_plot.html")
        create_2d_tsne_plot(testset, features, labels, output_file_path)
    else:
        # 3D plot
        output_file_path = os.path.join(output_folder, "tsne_3d_plot.html")
        create_3d_tsne_plot(features, labels, output_file_path)

