# Numbers Speech Recognizer

A simple speech-to-text / digit recognition project — recognizing spoken digits (0–9) from audio input.

## 📋 Table of Contents

- [About](#about)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Getting Started](#getting-started)  
  - [Clone the Repo](#clone-the-repo)  
  - [Install Dependencies](#install-dependencies)  
  - [Training / Using the Model](#training--using-the-model)  
- [Usage Examples](#usage-examples)  
- [Project Structure](#project-structure)  
- [License](#license)  
- [Credits / Acknowledgments](#credits--acknowledgments)  

## About

This project takes audio recordings of spoken digits (0–9) and uses a machine learning / signal processing model to convert them into textual representation. It’s useful for simple speech recognition tasks focused on digits (e.g. recognizing phone numbers, PINs, etc.).

## Features

- Recognizes single spoken digits (0 through 9)  
- Simple architecture, easily extensible  
- Pretrained model (if provided)  
- Ability to train on custom audio datasets  

## Requirements

- Python 3.7+  
- Libraries
  - `numpy`
  - `scipy` or `librosa` (for audio feature extraction)
  - `skikit-learn` or other ML framework used
  - `sounfile` or `wave` (for audio loading)  
- (Optional) GPU if you expand the model
-  A trained model (provided as model.pkl)

## Instalation

### Clone this Repo

```bash
git clone https://github.com/NorikHovhannisyan/Numbers_Speech_Recognizer.git
cd Numbers_Speech_Recognizer
