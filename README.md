# MAD-GAN in PyTorch

An implementation of [MAD-GAN (Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks)](https://arxiv.org/pdf/1901.04997.pdf) in **PyTorch**.

**Disclaimer: The only reason I open-sourced this is because I spent time reproducing the results in PyTorch so you don't have to. I do not advocate the usage of MAD-GAN for intrusion detection on SWaT. In fact, preliminary experiments suggest that MAD-GAN is not better than using residual sum of squares.**

MAD-GAN is a generative adversarial network designed to perform unsupervised anomaly detection on time-series data. 
The model can detect anomalies in various domains without prior knowledge of the specific domain or data distribution.

## Requirements
- Python 3.8 or later
- PyTorch 1.7.1 or later
- NumPy
- pandas
- scikit-learn
- tqdm

## Usage

To train and test the MAD-GAN model on your own dataset, you can use the `main.py` script. You can customize various settings, such as the data file paths, training epochs, learning rate, and more using command-line arguments.

For example:
```sh
python main.py --train_data data/train.csv --test_data data/test.csv --train_epochs 100 --train_lr 1e-4
```

Refer to the [`main.py`](main.py) file for a complete list of available command-line arguments.

## Model

The MAD-GAN model consists of two main components: a generator and a discriminator. The generator is responsible for creating synthetic time-series data, while the discriminator is responsible for distinguishing between real and generated data. During training, the generator and discriminator are optimized simultaneously to improve their performance.

After training, the MAD-GAN model can be used to detect anomalies in time-series data. The model assigns an anomaly score to each input data point based on the reconstruction error between the input data and the corresponding generated data.

## Data

The `data` folder contains two example datasets but is not provided here:

- `SWaT_Dataset_Normal_v0.csv`: The normal dataset used for training the model.
- `SWaT_Dataset_Attack_v0.csv`: The attack dataset used for testing the model.

These datasets are from the Secure Water Treatment (SWaT) testbed and contain time-series data from various sensors in a water treatment facility.

**To use the dataset, request access from [iTrust's website](https://itrust.sutd.edu.sg/itrust-labs_datasets/).**

## Acknowledgements

Thanks for the [original MAD-GAN implementation](https://github.com/imperial-qore/TranAD) in TensorFlow and [TranAD](https://github.com/imperial-qore/TranAD) in PyTorch.

## Citation

If you find this implementation useful in your research, please consider citing the original paper and this repo:

```bibtex
@inproceedings{10.1007/978-3-030-30490-4_56,
  title        = {MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks},
  author       = {Li, Dan and Chen, Dacheng and Jin, Baihong and Shi, Lei and Goh, Jonathan and Ng, See-Kiong},
  year         = 2019,
  booktitle    = {Artificial Neural Networks and Machine Learning -- ICANN 2019: Text and Time Series},
  publisher    = {Springer International Publishing},
  address      = {Cham},
  pages        = {703--716},
  isbn         = {978-3-030-30490-4},
  editor       = {Tetko, Igor V. and K{\r{u}}rkov{\'a}, V{\v{e}}ra and Karpov, Pavel and Theis, Fabian}
}
@software{Dai_MAD-GAN_in_PyTorch_2023,
  title        = {{MAD-GAN in PyTorch}},
  author       = {Dai, Zhihao},
  year         = 2023,
  month        = may,
  url          = {https://github.com/daidahao/MAD-GAN-PyTorch},
  version      = {0.0.1}
}
```