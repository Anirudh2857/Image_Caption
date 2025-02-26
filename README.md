# Image to Caption Generation

## Overview
This project implements an image captioning model that generates descriptive captions for images using deep learning techniques. It utilizes a Convolutional Neural Network (CNN) for feature extraction and a Transformer-based language model for text generation. The dataset used is the Flickr8K dataset.

## Features
- Uses ConvNeXt Small for image feature extraction.
- Implements a Transformer-based language model (GPT-2 tokenizer) for caption generation.
- Provides a Gradio-based UI for easy interaction.
- Includes data preprocessing, dataset loading, and model training.

## Dataset
The project uses the [Flickr8K dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k), which consists of 8,000 images with corresponding captions.

## Installation
To run the project, install the required dependencies:

```bash
pip install gradio torch torchvision torchmetrics torchtext transformers pandas numpy pillow
```

## Usage
### Running the Notebook
1. Open the `image_to_caption.ipynb` notebook in Google Colab.
2. Mount Google Drive to access the dataset.
3. Execute the cells to preprocess data and train the model.

### Running the Gradio Interface
To launch the web interface:

```python
import gradio as gr
def generate_caption(image):
    # Implement caption generation logic here
    return "Generated caption"

demo = gr.Interface(fn=generate_caption, inputs="image", outputs="text")
demo.launch()
```

## Model Training
The model consists of:
- A feature extractor (ConvNeXt Small) that encodes images into feature vectors.
- A Transformer-based text generation model that decodes features into captions.

To train the model:
```python
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

# Define and train your model here
```

## Demo Video
A demonstration video of the project can be found 

## License
This project is open-source and available under the MIT License.

## Contributors
Anirudh Jeevan

## Acknowledgments
- Flickr8K Dataset
- PyTorch and Hugging Face Transformers

Feel free to contribute to the project by submitting pull requests or reporting issues.

