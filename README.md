# SDGFNet

This is an official implementation of [SDGF: Fusing Static and Multi-Scale Dynamic Correlations for Multivariate Time Series Forecasting].

## Usage

- Train and evaluate SDGF
  - You can use the following command:`sh ./scripts/ETTh1.sh`.

- Train your model
  - Add model file in the folder `./models/your_model.py`.
  - Add model in the ***class*** Exp_Main.

## Model

Our proposed SDGF Network consists of three key modules: Graph Structure Learning module that uses RevIN normalization and Multi-level Wavelet Decomposition to construct static and dynamic inter-series graphs, an Attention Gated Fusion module that adaptively integrates static and dynamic graph features, and Temporal Feature Learning module that employs multi-kernel dilated convolutions and an MLP-based output layer to capture temporal dependencies and generate predictions.

<div align=center>
<img src="https://github.com/WSX378350448/SDGFNet/blob/main/pic/model.png" width='50%'>
</div>


## Citation

If you find this repo useful, please cite our paper as follows:
```
SDGF: Fusing Static and Multi-Scale Dynamic Correlations for Multivariate Time Series Forecasting
https://doi.org/10.48550/arXiv.2509.18135
```

## Contact
If you have any questions, please contact us or submit an issue.

## Acknowledgement

We appreciate the valuable contributions of the following GitHub.

- TimesNet (https://github.com/thuml/TimesNet)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- MSGNet (https://github.com/YoZhibo/MSGNet)
- MTGnn (https://github.com/nnzhan/MTGNN)
- Autoformer (https://github.com/thuml/Autoformer)
