# src/data_loader.py
import os
import json
import numpy as np
import requests

def download_dataset(url):
    """Download dataset from `url` to a local directory."""
    save_dir = './data'
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, url.split('/')[-1])
    
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    return filename

def load_data(url, headline_title=None, label_title=None):
    # Define the local file name based on the URL
    save_dir = './data'
    filename = os.path.join(save_dir, url.split('/')[-1])

    # Check if the file exists locally, if not, download it
    if not os.path.exists(filename):
        filename = download_dataset(url)

    # Load data from the downloaded file
    with open(filename, 'r') as f:
        datastore = json.load(f)

    # Dynamically identify the titles
    if not headline_title or not label_title:
        sample_item = datastore[0]
        keys = list(sample_item.keys())
        headline_title = headline_title or keys[0]
        label_title = label_title or keys[1]

    # Allow the user to specify the titles as arguments
    dataset = []
    label_dataset = []

    for item in datastore:
        dataset.append(item[headline_title])
        label_dataset.append(item[label_title])

    return np.array(dataset), np.array(label_dataset)