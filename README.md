## Cross-Modal Fine-Tuning: Align then Refine

Original PyTorch implementation of ORCA proposed in the paper "[Cross-Modal Fine-Tuning: Align then Refine](https://arxiv.org/abs/)." 
ORCA is developed for effectively solving  ML problems in diverse modalities using large-scale pretrained transformers. 
It adapts to a target task  via an align-then-refine workflow: given the target input, ORCA first learns an embedding network  that aligns the embedded feature distribution with the pretraining modality. The pretrained model is then fine-tuned  on the embedded data to exploit the  knowledge shared across modalities. 

This repo specifically supports
- transferring RoBERTa and Swin transformers (Hugging Face implementation) to downstream tasks;
- minimizing the l2 distance, Maximum Mean Descrepancy (MMD), or optimal transport dataset distance ([OTDD](https://github.com/microsoft/otdd)) for distributional alignment;
- replicate experiments on [NAS-Bench-360](https://nb360.ml.cmu.edu), [PDEBench](https://github.com/pdebench/PDEBench), and [OpenML](https://www.openml.org/) tabular tasks.

## Requirements

The Docker image needed for each task can be found in the configuration files under the `./src/configs' directory. Then, run `./src/start-up.sh' to install the dependencies.

## Experiment with NAS-Bench-360, PDEBench, OpenML tabular data, and GDSC/CTRP drug-response datasets

1. Download required datasets and precomputed language features [text_xs.py](https://www.dropbox.com/s/yhlf25n8rzmdrtp/text_xs.npy?dl=0) and [text_ys.py](https://www.dropbox.com/s/16lj1vprg1pzckt/text_ys.npy?dl=0) (if you are using RoBERTa models) to  `./src/datasets'
2. Run the following command:
```
python3 ./src/main.py --config ./src/configs/task.yaml
```

## Experiment with Your Own Pretrained Transformers and Datasets

#### For new transformer model bodies:
Place the corresponding implementation in `./src/embedders.py` and complete the `get_tgt_model` function.

#### For new datasets:
1. Add the data loaders to `./src/data_loaders.py` and complete the `get_data` function in `./src/task_configs.py`.
2. Add the loss functions and evaluation metrics to `./src/utils.py` and complete the `get_metric` function in `./src/task_configs.py`.
3. Modify the `get_config` function in `./src/task_configs.py`.
4. Add the yaml file to  `./src/configs'.


## Citation
If you find this project helpful, please consider citing our paper:
```bibtex
@inproceedings{shen2023orca,
  title={Cross-Modal Fine-Tuning: Align then Refine},
    year={2023}
}
```
Thanks!
