# Grad-Prune
This is the code required to implement and benchmark the Grad-Prune paper proposed in XXXXXXXXXXXXXXXXXXXX (replace with a link when published). Importantly, this repository contains all of the scripts that require modification of BackdoorBench provided by https://github.com/SCLBD/BackdoorBench/tree/main.

To get the repository to work, please download the BackdoorBench repository and then replace the files in the attack and utils folder with the provided files. Then add grad-prune.py file to the defence folder. Below we have provided an example of how to train a BadNet model using CIFAR-10 and then use Grad-Prune to remove the backdoor.

# BadNet Example
Once you have followed the installation, train a BadNet model using the same command provided by BackdoorBench.
```
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml --patch_mask_path ../resource/badnet/trigger_image.png  --save_folder_name badnet_0_1
```
Then use the following command to use Grad-Prune. Note, the cifar10.yaml file needs to be added to a new config folder in config/defense/grad-prune. This uses Grad-Prune with 100 SPC, 10% accuracy reduction threshold and a 10% validation ratio.
```
python ./defense/grad-prune.py --result_file badnet_0_1 --yaml_path ../config/defense/grad-prune/cifar10.yaml --dataset cifar10
```

# Modification Attacks
The attacks have been modified to create a complete copy of the training data with the backdoor trigger added. This is saved to a variable called "bd_train_dataset_all." This variable then needs to be added to the "save_attack_result" function call at the end of the attack method. This will ensure that Grad-Prune has access a backdoor copy of all training samples when sampling.

# !!! IMPORTANT !!!
NOTE: This repository is NOT self-contained. Moreover, only the BadNet, Blended, LF and BPP attacks have been modified to be compatible with Grad-Prune. 
