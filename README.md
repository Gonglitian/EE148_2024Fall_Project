# EE148_2024Fall_Project
## TODO List
- [ ] Data Preprocess
  - [ ] Load
  - [ ] Augmentation
- [ ] Model
  - [ ] DeepLabV3
    - dataset
    - num_classes
    - [ ] Feauture map shape in each layer
    - bigger lr
    - unbalance class, maybe should use weight
  - [ ] FCN
- [ ] Train Loop
- [ ] Evaluation
- [ ] Visulization
- [ ] Comparison
- [ ] Report

## DeepLabV3
>  raise ValueError("Expected more than 1 value per channel when training, got input size {}".format(size))
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])