# LLM Evaluation Framework

This repository provides a comprehensive framework for evaluating Large Language Models (LLMs) using various metrics such as toxicity, bias, and factuality. The framework is built to easily incorporate Hugging Face models and datasets, facilitating streamlined inference and evaluation workflows.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Implemented Metrics](#implemented-metrics)


## Overview

This project offers an evaluation pipeline specifically designed for binary and multiclass classification tasks typically performed by Hugging Face Transformers. It includes implementations of key evaluation metrics and supports easy integration with Hugging Face datasets and models.


## Installation

Install the required libraries using:

``pip install torch transformers datasets numpy pyyaml``

## Usage

## Implemented Metrics
1. Toxicity Metric
Evaluates the model's accuracy in classifying text as toxic or non-toxic.

2. Bias Metric
Analyzes demographic biases by examining toxicity predictions for texts containing specific demographic keywords (e.g., "women", "men", "black", "white", "gay", "straight").

3. Factuality Metric
Compares model predictions against a reference, calculating a simple accuracy. This can be extended to incorporate more sophisticated fact-checking measures such as BLEU or ROUGE scores.
