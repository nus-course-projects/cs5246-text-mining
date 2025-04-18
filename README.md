# Composite Model Architecture for Information Extraction from News Articles

This project describes an approach for identifying and quantifying the impact of natural and man-made disasters on stock prices. The methodology involves scraping the web for news articles and stock market data, applying text mining techniques to process and extract relevant information from the articles, and conducting a correlation analysis between the extracted data and stock price movements. Among all the disasters assessed, the Covid-19 pandemic exhibited the strongest correlation with stock prices.


## Setup

We use poetry to manage project dependencies. To get started, create a virtual environment and run the following:

```shell
python -m pip install poetry
```

Then install project dependencies
```shell
poetry install
```

## Running

The entire project pipeline can be ran from [pipeline.ipynb](./pipeline.ipynb).
For simplicity, let the `BUILD_DATASET` flag remain `False`. The code will automatically download the compiled dataset from our Github Releases.

Since individual tasks of the project require different pre-processings, the entire notebook will take a while to run (a few hours). We have already ran the notebook and compiled the outputs for observation at the end of each cell's execution.
