# Unreasonable Effectiveness of RNNs

This project implements character-level text generation models using Recurrent Neural Networks (RNNs), specifically focusing on Long Short-Term Memory (LSTM) architectures. Inspired by Andrej Karpathy's article [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), the models are trained to generate coherent text sequences, learning from large textual datasets.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model Architectures](#model-architectures)
- [Usage](#usage)
  - [Training](#training)
  - [Text Generation](#text-generation)
- [Dependencies](#dependencies)
- [Results](#results)
- [References](#references)

## Overview

The repository focuses on building and training two types of models from scratch:

- **CharCNN**: A character-level Convolutional Neural Network.
- **CharLSTM**: A character-level Long Short-Term Memory network.

These models are trained on textual data from Sherlock Holmes stories and Shakespearean plays. The primary objective is to evaluate the models' ability to generate coherent text sequences, starting from an initial prompt or even from scratch.

## Project Structure

The repository contains the following files:

- `main.py`: The main script to train and evaluate the models.
- `lstm_runner.py`: Contains the implementation details for the CharLSTM model.
- `rnn_runner.py`: Contains the implementation details for the CharCNN model.
- `data/`: Directory containing the training datasets.
- `Readme.md`: Project documentation.

## Data

The models are trained on two primary datasets:

1. **Sherlock Holmes Stories**: A collection of detective stories by Arthur Conan Doyle.
2. **Shakespearean Plays**: Works by William Shakespeare, encompassing tragedies, comedies, and histories.

These datasets are placed within the `data/` directory. Each text file is preprocessed to remove unwanted characters and to standardize formatting.

## Model Architectures

### Character-Level Convolutional Neural Network (CharCNN)

The CharCNN model processes input text at the character level using convolutional layers. This approach captures local dependencies and patterns within the text.

**Architecture Details**:

- **Embedding Layer**: Maps each character to a dense vector representation.
- **Convolutional Layers**: Multiple 1D convolutional layers with varying kernel sizes to capture n-gram features.
- **Pooling Layers**: Max-pooling layers to reduce dimensionality and capture the most salient features.
- **Fully Connected Layers**: Dense layers to project the features into the desired output space.
- **Output Layer**: Softmax layer to predict the probability distribution over the next character.

### Character-Level Long Short-Term Memory Network (CharLSTM)

The CharLSTM model leverages the capabilities of LSTM units to capture long-range dependencies in text.

**Architecture Details**:

- **Embedding Layer**: Similar to CharCNN, maps characters to dense vectors.
- **LSTM Layers**: One or more stacked LSTM layers to process the sequence of embeddings.
- **Fully Connected Layers**: Dense layers to transform the LSTM outputs.
- **Output Layer**: Softmax layer to predict the next character in the sequence.

## Usage

### Training

To train either the CharCNN or CharLSTM model, execute the `main.py` script with the appropriate arguments.

**Example**:

  ```bash
  python main.py --model lstm --data data/sherlock.txt --epochs 50 --batch_size 64

Arguments:

--model: Specify the model type (lstm or cnn).

--data: Path to the training data file.

--epochs: Number of training epochs.

--batch_size: Size of each training batch.

### Text Generation
After training, the models can generate text based on an initial prompt. Use the main.py script with the --generate flag.

Example:
  ```bash
  python main.py --model lstm --generate "Once upon a time" --length 200

Arguments:

--generate: Initial text prompt for generation.

--length: Number of characters to generate.

Dependencies
Ensure the following Python libraries are installed:

numpy

torch

torchtext

argparse

You can install them using pip:
  ```bash
  pip install numpy torch torchtext argparse

Results
The trained models demonstrate the ability to generate text that mimics the style and structure of the training data. For instance, after training on Shakespearean plays, the CharLSTM model can produce outputs resembling Shakespeare's writing style, complete with archaic language and iambic pentameter.

Sample Output:
  ```bash
  To be, or not to be: that is the question:
  Whether 'tis nobler in the mind to suffer
  The slings and arrows of outrageous fortune,
  Or to take arms against a sea of troubles,
  And by opposing end them?

