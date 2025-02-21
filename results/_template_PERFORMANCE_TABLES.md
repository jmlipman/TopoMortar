We study the following challenges: small datasets, noisy labels, pseudo labels, and out-of-distribution test-set images, and we tackled them with deliberately-simple strategies.

Table summarizing the challenges and the solutions:
| Challenge | Solution
|--|--|
| OOD test set images (*Study case 2*) | Focus on In-Distr. test set images (*Study case 1*)
| OOD test set images (*Study case 2*) | Data augmentation (*Study case 6*)
| Small training set (*Study case 3*) | Larger training set (*Study case 1*)
| Pseudo-labels (*Study case 4*) | Self-distillation (*Study case 8*)
| Noisy labels (*Study case 5*) | Self-distillation (*Study case 9*)

### Study case 1: An ideal scenario
An ideal scenario consists of: 1) A sufficiently large training set, 2) Accurate labels for training, 3) Long enough training time, 4) A large deep learning model, 5) Strong data augmentation, and 6) An in-distribution test set.
In this ideal scenario, **increases in topology accuracy are very hard to obtain**.

### Study case 2: Dataset Challenge: Out-of-distribution test-set images
We evaluate the models trained in *Study case 1: An ideal scenario* on these images.
We investigate **which topology loss functions are specifically advantageous when the test-set images are OOD**.

### Study case 3: Dataset Challenge: Small datasets
Departing from the experimental setting of the ideal scenario (Study case 1), we reduce the training set size from 50 to 10 images.
We investigate **which topology loss functions are specifically advantageous when training data is limited**.

### Study case 4: Dataset Challenge: Pseudo-labels
Departing from the experimental setting of the ideal scenario (Study case 1), we train on automatically-generated pseudo-labels instead of on accurate, manual annotations (see them [here](/dataset/train/pseudo/)).
We investigate **which topology loss functions are specifically advantageous when training on pseudo-labels**.

### Study case 5: Dataset Challenge: Noisy labels
This is similar to Study case 4 but with noisy labels. You can see them [here](/dataset/train/noisy/).
These noisy labels simulate annotations done by a person quickly, with some errors, and a pen of a fixed size.
We investigate **which topology loss functions are specifically advantageous when training on noisy labels**.

### Study case 6: Addressing the Dataset Challenge of OOD test-set images
We investigate **if/which topology loss functions stop being advantageous after applying a simple data augmentation strategy (RandHue) that increases the diversity of brick colors** (see example in Figure 13 of Supplementary Material; code [here](lib/transforms.py)).
Note that, all the experiments were run with strong data augmentation (10 transformations described in Table 2 of the Supplementary Material).
The table compares the ideal scenario in Study case 1 vs. applying RandHue.

### Study case 7: Addressing the Dataset Challenge of Small datasets
We investigate **if/which topology loss functions stop being advantageous after simply increasing the training set size**, which, in many scenarios, can be easily done.
The table compares training on a small training set vs. training on a large training set (the ideal scenario from Study case 1).

### Study case 8: Addressing the Dataset Challenge of Pseudo-labels
We investigate **if/which topology loss functions stop being advantageous after utilizing a simple framework to learn from pseudo-labels (self-distillation)**, which can be done easily, although at the cost of a few more hyper-parameters.
The table compares the scenario in Study case 4 training standard supervised learning vs. self-distillation.

### Study case 9: Addressing the Dataset Challenge of Noisy labels
Same as in Study case 8 but with noisy labels.

# Results
The following measurements are averages across 10 random seeds.
The training included strong data augmentation (details in Table 2 of Supplementary Material).
**Bold**: Better than Cross Entropy + Dice Loss (CEDice) and significantly different (paired permutation tests, p < 0.05).

### Study case 1: An ideal scenario
[TABLE-1]

### Study case 2: Dataset Challenge: Out-of-distribution test-set images
[TABLE-2]

### Study case 3: Dataset Challenge: Small datasets
[TABLE-3]

### Study case 4: Dataset Challenge: Pseudo-labels
[TABLE-4]

### Study case 5: Dataset Challenge: Noisy labels
[TABLE-5]

### Study case 6: Addressing the Dataset Challenge of OOD test-set images
[TABLE-6]

### Study case 7: Addressing the Dataset Challenge of Small datasets
[TABLE-7]

### Study case 8: Addressing the Dataset Challenge of Pseudo-labels
[TABLE-8]

### Study case 9: Addressing the Dataset Challenge of Noisy labels
[TABLE-9]

