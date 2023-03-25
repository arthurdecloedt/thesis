# Master Thesis: Multimodal Tweet Sentiment for ImpliedVolatility Predictions

This repository contains code for my master thesis project

## Abstract
This thesis researches the extraction of multimodal sentiment from tweets referencing
an equity and approaches on using it for predicting future implied volatility on that
equity. A scalable and efficient data collection procedure is defined and implemented.
The constructed dataset contains both textual and visual information with a sub day
level granularity. Emotional features from both modalities are extracted using pretrained models with a decision level fusion approach, resulting in a dataset reflecting
popular sentiment towards a specific equity over multiple years with sentiment
vectors for each tweet. We present several lightweight, size agnostic architectures,
derived from convolutional deep averaging networks, that outperform out of the box
machine learning approaches. While the used dataset is relatively small and rough,
the neural networks show promise in this task. We finally demonstrate the models
outperforming baselines set by the lagged value when relaxing the constraint of only
using the multimodal data.
