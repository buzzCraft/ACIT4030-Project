import torch
import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import time

def load_and_evaluate_models(models, model_paths, valid_loader):
    all_model_preds = {}
    all_model_labels = {}
    metrics_data = []

    for model_name, model in models.items():
        start_time = time.time()

        model.load_state_dict(torch.load(model_paths[model_name]))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                print(f'Batch [{i+1:4} / {len(valid_loader):4}] for {model_name}')

                inputs, labels = data['pointcloud'].float(), data['category']
                outputs, _, _ = model(inputs.transpose(1,2))
                _, preds = torch.max(outputs.data, 1)
                all_preds += list(preds.numpy())
                all_labels += list(labels.numpy())

        end_time = time.time()
        elapsed_time = end_time - start_time

        accuracy = accuracy_score(all_labels, all_preds)
        metrics_data.append({"Model": model_name, "Accuracy": accuracy, "Time (seconds)": elapsed_time})

        all_model_preds[model_name] = all_preds
        all_model_labels[model_name] = all_labels

    metrics_df = pd.DataFrame(metrics_data)
    return all_model_preds, all_model_labels, metrics_df

def plot_confusion_matrix_plotly(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    z = cm
    x = classes
    y = classes

    # Change the colorscale if needed
    colorscale = [[0, '#a6bdf5'], [.5, '#9067d6'], [1, '#c74848']]

    layout = {
        "title": title,
        "xaxis": {"title": "Predicted label"},
        "yaxis": {"title": "True label"}
    }

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z, colorscale=colorscale)
    fig.update_layout(layout)
    fig.show()

def generate_confusion_matrices(models, all_model_preds, all_model_labels, classes):
    for model_name in models.keys():
        cm = confusion_matrix(all_model_labels[model_name], all_model_preds[model_name])
        plot_confusion_matrix_plotly(cm, list(classes.keys()), normalize=True, title=f'Normalized Confusion Matrix for {model_name}')
        plot_confusion_matrix_plotly(cm, list(classes.keys()), normalize=False, title=f'Confusion Matrix for {model_name}')

if __name__ == '__main__':
    # Sample usage:
    from src.pointnet.pointnet import PointNet
    from pathlib import Path
    from src.trainer import dataload

    path = Path('../../data/ModelNet10')
    print("Loading data...")
    _, valid_loader, classes = dataload(path)
    print("Creating models...")
    models = {'PointNet': PointNet()}  # You can add more models here as needed
    model_paths = {'PointNet': '../../save_14.pth'}  # Paths for each model's weights
    print("Evaluating models...")
    all_model_preds, all_model_labels, metrics_df = load_and_evaluate_models(models, model_paths, valid_loader)
    print("Generating confusion matrices...")
    generate_confusion_matrices(models, all_model_preds, all_model_labels, classes)
    print(metrics_df.head(20))
