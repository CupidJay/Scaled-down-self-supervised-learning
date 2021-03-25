
## Transferring to classification benchmarks

We provide fine-tuning and linear evaluation training scripts for aircraft, caltech, cars, cub, pets, cifar and voc. We organized cub, aircraft, caltech, cars and pets in train and val with labeled subfolders as the official ImageNet dataset. 

### Instruction

1. For linear evaluation (aircraft as an example), set --pretrained to the model you want to evaluate:
   ```
   bash main_train_aircraft.sh
   ```

1. For linear evaluation  (aircraft as an example):
   ```
   bash main_linear_aircraft.sh
	```
