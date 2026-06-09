## Prepare Dastaset

```bash
conda activate .dino
python smiths/split_dataset.py /path/to/classes_labled_dataset /path/to/destibation_dino 
```

## Pre-Training

- Copy or edit `config/smiths/privatedata_dinov3_pretrain.yaml`

### Configure Dataset

- Open the `.yaml` file and modify 
    - `train > dataset_path : PRIVATE_DATA:root=/path/to/destibation_dino:split=TRAIN`
 
### Configure Model Weights

> **Important Notes:**
> - Available Models check [Pretrained models](../README.md#pretrained-models)
> - Download the checkpoints from: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

- Open the `.yaml` file and modify 

    - `student > arch : vit_small` <- model architecture 16bit
    - `MODEL > WEIGHTS : /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
    - `student > pretrained_weights : /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` OFFICIAL_EPOCH_LENGTH
    - Calculate and Set `crops > rgb_mean` and `crops > rgb_std` for your dataset
        - For `HIF` file this is
        ```
            rgb_mean:
            - 0.9208470582962036
            - 0.8263269662857056
            - 0.7000124454498291
            rgb_std:
            - 0.1979101151227951
            - 0.2202170044183731
            - 0.33237993717193604
        ```
    - Configure `train > OFFICIAL_EPOCH_LENGTH` as per your need


### Training
```bash
conda activate .dino
python -m torch.distributed.run --nproc_per_node=2 dinov3/train/train.py \
        --config dinov3/configs/smiths/your_privatedata_dinov3_pretrain.yaml \
        --output-dir outputs/privatedata_dinov3-small/ \
        --seed 1
```

## Export PyTorch

```bash
conda activate .dino
python smiths/export_dinov3.py outputs/privatedata_dinov3-small outputs/privatedata_dinov3-small.pth
```