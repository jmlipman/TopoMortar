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
<table>
<thead>
<tr>
	<th>Loss</th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th>
</tr>
</thead>
<tbody>
<tr>
	<td>CEDice</td><td>3.31 ± 6.6</td><td>2.13 ± 3.89</td><td>0.91 ± 0.04</td><td>1.87 ± 0.87</td>
</tr>
<tr>
	<td>RWLoss</td><td>5.72 ± 9.94</td><td>6.03 ± 12.09</td><td>0.91 ± 0.05</td><td><b>1.84 ± 0.84</b></td>
</tr>
<tr>
	<td>TopoLoss</td><td>3.14 ± 6.11</td><td>2.73 ± 7.24</td><td>0.91 ± 0.04</td><td>1.87 ± 0.88</td>
</tr>
<tr>
	<td>TOPO</td><td>33.69 ± 22.06</td><td>43.35 ± 23.34</td><td>0.86 ± 0.06</td><td>2.68 ± 1.24</td>
</tr>
<tr>
	<td>clDice</td><td><b>1.17 ± 2.26</b></td><td><b>1.41 ± 2.53</b></td><td>0.91 ± 0.05</td><td><b>1.83 ± 0.84</b></td>
</tr>
<tr>
	<td>Warping</td><td>4.2 ± 6.95</td><td>3.62 ± 6.25</td><td>0.91 ± 0.05</td><td><b>1.84 ± 0.83</b></td>
</tr>
<tr>
	<td>SkelRecall</td><td>3.08 ± 5.3</td><td><b>1.73 ± 2.81</b></td><td>0.91 ± 0.05</td><td>1.9 ± 0.89</td>
</tr>
<tr>
	<td>cbDice</td><td>4.24 ± 7.12</td><td>3.81 ± 7.65</td><td>0.91 ± 0.04</td><td><b>1.85 ± 0.86</b></td>
</tr>
</tbody>
</table>


### Study case 2: Dataset Challenge: Out-of-distribution test-set images
<table>
<thead>
<tr>
	<th>Loss</th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th>
</tr>
</thead>
<tbody>
<tr>
	<td>CEDice</td><td>197.25 ± 188.68</td><td>87.81 ± 84.29</td><td>0.65 ± 0.19</td><td>37.65 ± 27.76</td>
</tr>
<tr>
	<td>RWLoss</td><td>219.55 ± 172.14</td><td>87.74 ± 86.38</td><td>0.64 ± 0.19</td><td>37.31 ± 27.63</td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>193.29 ± 190.19</b></td><td>99.33 ± 95.53</td><td>0.65 ± 0.19</td><td><b>37.33 ± 27.58</b></td>
</tr>
<tr>
	<td>TOPO</td><td><b>117.73 ± 106.3</b></td><td>105.09 ± 115.16</td><td>0.65 ± 0.17</td><td>40.21 ± 28.65</td>
</tr>
<tr>
	<td>clDice</td><td><b>81.33 ± 92.37</b></td><td><b>51.64 ± 48.97</b></td><td><b>0.66 ± 0.19</b></td><td><b>36.9 ± 28.19</b></td>
</tr>
<tr>
	<td>Warping</td><td>204.26 ± 179.53</td><td><b>80.23 ± 75.14</b></td><td>0.65 ± 0.19</td><td><b>37.07 ± 27.29</b></td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>180.23 ± 170.31</b></td><td>96.79 ± 98.12</td><td><b>0.67 ± 0.18</b></td><td>38.46 ± 27.66</td>
</tr>
<tr>
	<td>cbDice</td><td>233.49 ± 202.28</td><td>106.08 ± 104.07</td><td>0.65 ± 0.18</td><td>37.84 ± 27.71</td>
</tr>
</tbody>
</table>


### Study case 3: Dataset Challenge: Small datasets
<table>
<thead>
<tr>
<th></th>
	<th>Loss</th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">ID</td>	<td>CEDice</td><td>9.57 ± 26.49</td><td>4.63 ± 13.21</td><td>0.9 ± 0.06</td><td>2.06 ± 1.17</td>
</tr>
<tr>
	<td>RWLoss</td><td>12.35 ± 33.74</td><td>6.42 ± 16.75</td><td>0.9 ± 0.06</td><td><b>2.04 ± 1.13</b></td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>8.72 ± 25.53</b></td><td>4.81 ± 16.27</td><td>0.9 ± 0.06</td><td>2.09 ± 1.28</td>
</tr>
<tr>
	<td>TOPO</td><td>59.6 ± 70.31</td><td>43.29 ± 27.57</td><td>0.86 ± 0.07</td><td>2.64 ± 1.91</td>
</tr>
<tr>
	<td>clDice</td><td><b>6.18 ± 23.81</b></td><td><b>4.01 ± 12.22</b></td><td>0.9 ± 0.06</td><td>2.04 ± 1.15</td>
</tr>
<tr>
	<td>Warping</td><td>11.37 ± 33.09</td><td>4.53 ± 11.88</td><td>0.9 ± 0.06</td><td>2.06 ± 1.14</td>
</tr>
<tr>
	<td>SkelRecall</td><td>9.89 ± 27.72</td><td>5.18 ± 15.87</td><td>0.9 ± 0.06</td><td>2.1 ± 1.3</td>
</tr>
<tr>
	<td>cbDice</td><td>9.6 ± 26.65</td><td>4.92 ± 13.94</td><td>0.9 ± 0.06</td><td>2.08 ± 1.23</td>
</tr>
<tr>
<td rowspan="8">OOD</td>	<td>CEDice</td><td>182.01 ± 128.12</td><td>61.51 ± 52.63</td><td>0.55 ± 0.2</td><td>39.57 ± 28.9</td>
</tr>
<tr>
	<td>RWLoss</td><td>201.0 ± 142.69</td><td><b>56.53 ± 56.51</b></td><td><b>0.57 ± 0.2</b></td><td>39.46 ± 29.43</td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>152.05 ± 108.79</b></td><td><b>58.13 ± 50.52</b></td><td><b>0.56 ± 0.2</b></td><td>39.86 ± 29.06</td>
</tr>
<tr>
	<td>TOPO</td><td>312.12 ± 190.99</td><td>109.61 ± 107.83</td><td><b>0.59 ± 0.17</b></td><td>40.54 ± 29.47</td>
</tr>
<tr>
	<td>clDice</td><td><b>127.32 ± 99.03</b></td><td><b>38.7 ± 45.49</b></td><td><b>0.56 ± 0.2</b></td><td>39.83 ± 30.19</td>
</tr>
<tr>
	<td>Warping</td><td>190.2 ± 142.46</td><td><b>43.59 ± 48.43</b></td><td><b>0.56 ± 0.2</b></td><td>39.56 ± 29.15</td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>160.08 ± 115.51</b></td><td>69.23 ± 56.09</td><td><b>0.57 ± 0.19</b></td><td>39.68 ± 28.32</td>
</tr>
<tr>
	<td>cbDice</td><td><b>156.88 ± 113.95</b></td><td><b>54.75 ± 48.32</b></td><td><b>0.56 ± 0.2</b></td><td>39.62 ± 28.76</td>
</tr>
</tbody>
</table>


### Study case 4: Dataset Challenge: Pseudo-labels
<table>
<thead>
<tr>
<th></th>
	<th>Loss</th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">ID</td>	<td>CEDice</td><td>12.26 ± 9.71</td><td>12.0 ± 10.14</td><td>0.86 ± 0.07</td><td>3.17 ± 1.96</td>
</tr>
<tr>
	<td>RWLoss</td><td><b>11.38 ± 9.29</b></td><td><b>11.11 ± 10.4</b></td><td><b>0.87 ± 0.07</b></td><td><b>3.09 ± 1.98</b></td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>10.43 ± 8.22</b></td><td><b>10.13 ± 9.07</b></td><td>0.86 ± 0.07</td><td>3.18 ± 1.94</td>
</tr>
<tr>
	<td>TOPO</td><td>36.09 ± 25.6</td><td>31.1 ± 24.7</td><td>0.81 ± 0.07</td><td>4.4 ± 2.21</td>
</tr>
<tr>
	<td>clDice</td><td><b>1.8 ± 2.23</b></td><td><b>4.3 ± 5.37</b></td><td><b>0.87 ± 0.07</b></td><td>3.17 ± 2.09</td>
</tr>
<tr>
	<td>Warping</td><td>13.57 ± 10.78</td><td>13.26 ± 11.74</td><td><b>0.87 ± 0.07</b></td><td><b>3.12 ± 1.97</b></td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>10.4 ± 9.31</b></td><td><b>10.43 ± 9.27</b></td><td>0.86 ± 0.07</td><td>3.29 ± 2.06</td>
</tr>
<tr>
	<td>cbDice</td><td>13.12 ± 12.76</td><td>14.66 ± 13.15</td><td>0.86 ± 0.07</td><td><b>3.13 ± 1.98</b></td>
</tr>
<tr>
<td rowspan="8">OOD</td>	<td>CEDice</td><td>126.21 ± 126.54</td><td>83.92 ± 68.47</td><td>0.68 ± 0.2</td><td>35.19 ± 29.88</td>
</tr>
<tr>
	<td>RWLoss</td><td><b>114.65 ± 115.08</b></td><td>96.49 ± 82.71</td><td>0.67 ± 0.18</td><td>36.65 ± 26.79</td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>117.98 ± 108.17</b></td><td>110.32 ± 97.64</td><td>0.68 ± 0.2</td><td>36.33 ± 32.16</td>
</tr>
<tr>
	<td>TOPO</td><td>225.37 ± 191.59</td><td>132.21 ± 152.82</td><td>0.66 ± 0.17</td><td>40.53 ± 26.88</td>
</tr>
<tr>
	<td>clDice</td><td><b>32.7 ± 42.79</b></td><td><b>61.57 ± 71.51</b></td><td>0.67 ± 0.17</td><td>36.87 ± 28.15</td>
</tr>
<tr>
	<td>Warping</td><td>127.08 ± 141.36</td><td>95.47 ± 81.2</td><td>0.67 ± 0.19</td><td>36.43 ± 28.29</td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>113.63 ± 124.15</b></td><td>85.75 ± 77.06</td><td><b>0.7 ± 0.19</b></td><td>34.91 ± 27.81</td>
</tr>
<tr>
	<td>cbDice</td><td>141.42 ± 129.71</td><td>122.91 ± 103.54</td><td>0.68 ± 0.2</td><td>35.72 ± 32.0</td>
</tr>
</tbody>
</table>


### Study case 5: Dataset Challenge: Noisy labels
<table>
<thead>
<tr>
<th></th>
	<th>Loss</th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">ID</td>	<td>CEDice</td><td>6.24 ± 6.64</td><td>5.08 ± 5.21</td><td>0.63 ± 0.12</td><td>3.84 ± 1.66</td>
</tr>
<tr>
	<td>RWLoss</td><td>17.9 ± 30.69</td><td>11.93 ± 13.81</td><td><b>0.66 ± 0.13</b></td><td><b>3.57 ± 1.73</b></td>
</tr>
<tr>
	<td>TopoLoss</td><td>7.19 ± 9.93</td><td>6.62 ± 6.85</td><td>0.62 ± 0.13</td><td>3.88 ± 1.73</td>
</tr>
<tr>
	<td>TOPO</td><td>24.14 ± 30.55</td><td>10.93 ± 13.42</td><td><b>0.84 ± 0.07</b></td><td><b>2.65 ± 0.93</b></td>
</tr>
<tr>
	<td>clDice</td><td><b>5.01 ± 4.95</b></td><td>6.77 ± 5.81</td><td><b>0.69 ± 0.12</b></td><td><b>3.58 ± 1.68</b></td>
</tr>
<tr>
	<td>Warping</td><td>8.48 ± 9.78</td><td>9.49 ± 8.26</td><td>0.62 ± 0.13</td><td>3.85 ± 1.7</td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>3.79 ± 4.8</b></td><td><b>1.49 ± 2.19</b></td><td><b>0.76 ± 0.09</b></td><td><b>3.24 ± 1.47</b></td>
</tr>
<tr>
	<td>cbDice</td><td>10.5 ± 10.6</td><td>9.74 ± 8.39</td><td>0.63 ± 0.13</td><td><b>3.82 ± 1.71</b></td>
</tr>
<tr>
<td rowspan="8">OOD</td>	<td>CEDice</td><td>175.17 ± 109.47</td><td>18.72 ± 24.36</td><td>0.33 ± 0.14</td><td>40.17 ± 33.05</td>
</tr>
<tr>
	<td>RWLoss</td><td>427.46 ± 230.97</td><td><b>17.87 ± 21.87</b></td><td>0.31 ± 0.14</td><td><b>36.35 ± 27.0</b></td>
</tr>
<tr>
	<td>TopoLoss</td><td>205.05 ± 112.25</td><td><b>14.84 ± 18.31</b></td><td>0.26 ± 0.14</td><td>42.99 ± 39.85</td>
</tr>
<tr>
	<td>TOPO</td><td><b>157.22 ± 125.65</b></td><td>140.45 ± 105.8</td><td><b>0.58 ± 0.16</b></td><td><b>39.27 ± 27.05</b></td>
</tr>
<tr>
	<td>clDice</td><td><b>126.52 ± 89.23</b></td><td>48.2 ± 48.63</td><td><b>0.46 ± 0.14</b></td><td><b>38.9 ± 26.86</b></td>
</tr>
<tr>
	<td>Warping</td><td>285.2 ± 170.07</td><td><b>18.01 ± 20.78</b></td><td><b>0.35 ± 0.12</b></td><td><b>37.61 ± 25.5</b></td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>113.33 ± 87.74</b></td><td>44.6 ± 36.58</td><td><b>0.51 ± 0.16</b></td><td>40.16 ± 29.33</td>
</tr>
<tr>
	<td>cbDice</td><td>275.69 ± 148.81</td><td>27.5 ± 27.25</td><td>0.31 ± 0.14</td><td><b>39.26 ± 32.83</b></td>
</tr>
</tbody>
</table>


### Study case 6: Addressing the Dataset Challenge of OOD test-set images
<table>
<thead>
<tr>
<th colspan=2></th><th colspan=4>Ideal scenario</th><th>vs.</th><th colspan=4>RandHue</th></tr>
<tr>
<th></th>
	<th>Loss</th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th><th></th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">ID</td>	<td>CEDice</td><td>3.31 ± 6.6</td><td>2.13 ± 3.89</td><td>0.91 ± 0.04</td><td>1.87 ± 0.87</td><td></td><td>1.9 ± 5.92</td><td>1.99 ± 8.94</td><td>0.91 ± 0.04</td><td>1.88 ± 1.19</td>
</tr>
<tr>
	<td>RWLoss</td><td>5.72 ± 9.94</td><td>6.03 ± 12.09</td><td>0.91 ± 0.05</td><td><b>1.84 ± 0.84</b></td><td></td><td>3.3 ± 6.07</td><td>2.93 ± 6.1</td><td>0.91 ± 0.05</td><td><b>1.8 ± 0.82</b></td>
</tr>
<tr>
	<td>TopoLoss</td><td>3.14 ± 6.11</td><td>2.73 ± 7.24</td><td>0.91 ± 0.04</td><td>1.87 ± 0.88</td><td></td><td>2.87 ± 11.4</td><td>3.34 ± 17.14</td><td>0.91 ± 0.04</td><td>1.9 ± 1.37</td>
</tr>
<tr>
	<td>TOPO</td><td>33.69 ± 22.06</td><td>43.35 ± 23.34</td><td>0.86 ± 0.06</td><td>2.68 ± 1.24</td><td></td><td>19.71 ± 13.75</td><td>40.56 ± 21.89</td><td>0.86 ± 0.06</td><td>2.64 ± 1.48</td>
</tr>
<tr>
	<td>clDice</td><td><b>1.17 ± 2.26</b></td><td><b>1.41 ± 2.53</b></td><td>0.91 ± 0.05</td><td><b>1.83 ± 0.84</b></td><td></td><td><b>0.56 ± 1.18</b></td><td><b>0.95 ± 1.44</b></td><td>0.91 ± 0.04</td><td><b>1.77 ± 0.81</b></td>
</tr>
<tr>
	<td>Warping</td><td>4.2 ± 6.95</td><td>3.62 ± 6.25</td><td>0.91 ± 0.05</td><td><b>1.84 ± 0.83</b></td><td></td><td>2.5 ± 5.53</td><td>1.88 ± 3.76</td><td>0.91 ± 0.04</td><td><b>1.8 ± 0.84</b></td>
</tr>
<tr>
	<td>SkelRecall</td><td>3.08 ± 5.3</td><td><b>1.73 ± 2.81</b></td><td>0.91 ± 0.05</td><td>1.9 ± 0.89</td><td></td><td>2.0 ± 6.44</td><td>2.01 ± 8.44</td><td>0.91 ± 0.05</td><td>1.91 ± 1.19</td>
</tr>
<tr>
	<td>cbDice</td><td>4.24 ± 7.12</td><td>3.81 ± 7.65</td><td>0.91 ± 0.04</td><td><b>1.85 ± 0.86</b></td><td></td><td>2.71 ± 8.31</td><td>3.29 ± 14.07</td><td>0.91 ± 0.05</td><td>1.89 ± 1.39</td>
</tr>
<tr>
<td rowspan="8">OOD</td>	<td>CEDice</td><td>197.25 ± 188.68</td><td>87.81 ± 84.29</td><td>0.65 ± 0.19</td><td>37.65 ± 27.76</td><td></td><td>69.73 ± 96.57</td><td>53.32 ± 62.06</td><td>0.74 ± 0.16</td><td>36.1 ± 27.65</td>
</tr>
<tr>
	<td>RWLoss</td><td>219.55 ± 172.14</td><td>87.74 ± 86.38</td><td>0.64 ± 0.19</td><td>37.31 ± 27.63</td><td></td><td><b>62.75 ± 59.68</b></td><td><b>49.84 ± 43.58</b></td><td>0.72 ± 0.17</td><td>36.54 ± 27.81</td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>193.29 ± 190.19</b></td><td>99.33 ± 95.53</td><td>0.65 ± 0.19</td><td><b>37.33 ± 27.58</b></td><td></td><td>85.25 ± 112.1</td><td>80.09 ± 92.3</td><td>0.74 ± 0.16</td><td>36.19 ± 28.21</td>
</tr>
<tr>
	<td>TOPO</td><td><b>117.73 ± 106.3</b></td><td>105.09 ± 115.16</td><td>0.65 ± 0.17</td><td>40.21 ± 28.65</td><td></td><td>97.24 ± 74.51</td><td><b>49.91 ± 55.28</b></td><td>0.72 ± 0.15</td><td>42.31 ± 29.35</td>
</tr>
<tr>
	<td>clDice</td><td><b>81.33 ± 92.37</b></td><td><b>51.64 ± 48.97</b></td><td><b>0.66 ± 0.19</b></td><td><b>36.9 ± 28.19</b></td><td></td><td><b>16.54 ± 21.6</b></td><td><b>15.12 ± 19.93</b></td><td>0.74 ± 0.16</td><td><b>35.42 ± 28.44</b></td>
</tr>
<tr>
	<td>Warping</td><td>204.26 ± 179.53</td><td><b>80.23 ± 75.14</b></td><td>0.65 ± 0.19</td><td><b>37.07 ± 27.29</b></td><td></td><td><b>57.07 ± 66.24</b></td><td><b>40.02 ± 39.33</b></td><td>0.73 ± 0.17</td><td>35.94 ± 27.97</td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>180.23 ± 170.31</b></td><td>96.79 ± 98.12</td><td><b>0.67 ± 0.18</b></td><td>38.46 ± 27.66</td><td></td><td><b>65.46 ± 82.18</b></td><td>62.72 ± 69.3</td><td><b>0.75 ± 0.15</b></td><td>37.83 ± 28.27</td>
</tr>
<tr>
	<td>cbDice</td><td>233.49 ± 202.28</td><td>106.08 ± 104.07</td><td>0.65 ± 0.18</td><td>37.84 ± 27.71</td><td></td><td>90.26 ± 111.28</td><td>83.22 ± 89.04</td><td>0.74 ± 0.16</td><td>36.5 ± 28.26</td>
</tr>
</tbody>
</table>


### Study case 7: Addressing the Dataset Challenge of Small datasets
<table>
<thead>
<tr>
<th colspan=2></th><th colspan=4>Small training set</th><th>vs.</th><th colspan=4>Large training set</th></tr>
<tr>
<th></th>
	<th>Loss</th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th><th></th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">ID</td>	<td>CEDice</td><td>9.57 ± 26.49</td><td>4.63 ± 13.21</td><td>0.9 ± 0.06</td><td>2.06 ± 1.17</td><td></td><td>3.31 ± 6.6</td><td>2.13 ± 3.89</td><td>0.91 ± 0.04</td><td>1.87 ± 0.87</td>
</tr>
<tr>
	<td>RWLoss</td><td>12.35 ± 33.74</td><td>6.42 ± 16.75</td><td>0.9 ± 0.06</td><td><b>2.04 ± 1.13</b></td><td></td><td>5.72 ± 9.94</td><td>6.03 ± 12.09</td><td>0.91 ± 0.05</td><td><b>1.84 ± 0.84</b></td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>8.72 ± 25.53</b></td><td>4.81 ± 16.27</td><td>0.9 ± 0.06</td><td>2.09 ± 1.28</td><td></td><td>3.14 ± 6.11</td><td>2.73 ± 7.24</td><td>0.91 ± 0.04</td><td>1.87 ± 0.88</td>
</tr>
<tr>
	<td>TOPO</td><td>59.6 ± 70.31</td><td>43.29 ± 27.57</td><td>0.86 ± 0.07</td><td>2.64 ± 1.91</td><td></td><td>33.69 ± 22.06</td><td>43.35 ± 23.34</td><td>0.86 ± 0.06</td><td>2.68 ± 1.24</td>
</tr>
<tr>
	<td>clDice</td><td><b>6.18 ± 23.81</b></td><td><b>4.01 ± 12.22</b></td><td>0.9 ± 0.06</td><td>2.04 ± 1.15</td><td></td><td><b>1.17 ± 2.26</b></td><td><b>1.41 ± 2.53</b></td><td>0.91 ± 0.05</td><td><b>1.83 ± 0.84</b></td>
</tr>
<tr>
	<td>Warping</td><td>11.37 ± 33.09</td><td>4.53 ± 11.88</td><td>0.9 ± 0.06</td><td>2.06 ± 1.14</td><td></td><td>4.2 ± 6.95</td><td>3.62 ± 6.25</td><td>0.91 ± 0.05</td><td><b>1.84 ± 0.83</b></td>
</tr>
<tr>
	<td>SkelRecall</td><td>9.89 ± 27.72</td><td>5.18 ± 15.87</td><td>0.9 ± 0.06</td><td>2.1 ± 1.3</td><td></td><td>3.08 ± 5.3</td><td><b>1.73 ± 2.81</b></td><td>0.91 ± 0.05</td><td>1.9 ± 0.89</td>
</tr>
<tr>
	<td>cbDice</td><td>9.6 ± 26.65</td><td>4.92 ± 13.94</td><td>0.9 ± 0.06</td><td>2.08 ± 1.23</td><td></td><td>4.24 ± 7.12</td><td>3.81 ± 7.65</td><td>0.91 ± 0.04</td><td><b>1.85 ± 0.86</b></td>
</tr>
<tr>
<td rowspan="8">OOD</td>	<td>CEDice</td><td>182.01 ± 128.12</td><td>61.51 ± 52.63</td><td>0.55 ± 0.2</td><td>39.57 ± 28.9</td><td></td><td>197.25 ± 188.68</td><td>87.81 ± 84.29</td><td>0.65 ± 0.19</td><td>37.65 ± 27.76</td>
</tr>
<tr>
	<td>RWLoss</td><td>201.0 ± 142.69</td><td><b>56.53 ± 56.51</b></td><td><b>0.57 ± 0.2</b></td><td>39.46 ± 29.43</td><td></td><td>219.55 ± 172.14</td><td>87.74 ± 86.38</td><td>0.64 ± 0.19</td><td>37.31 ± 27.63</td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>152.05 ± 108.79</b></td><td><b>58.13 ± 50.52</b></td><td><b>0.56 ± 0.2</b></td><td>39.86 ± 29.06</td><td></td><td><b>193.29 ± 190.19</b></td><td>99.33 ± 95.53</td><td>0.65 ± 0.19</td><td><b>37.33 ± 27.58</b></td>
</tr>
<tr>
	<td>TOPO</td><td>312.12 ± 190.99</td><td>109.61 ± 107.83</td><td><b>0.59 ± 0.17</b></td><td>40.54 ± 29.47</td><td></td><td><b>117.73 ± 106.3</b></td><td>105.09 ± 115.16</td><td>0.65 ± 0.17</td><td>40.21 ± 28.65</td>
</tr>
<tr>
	<td>clDice</td><td><b>127.32 ± 99.03</b></td><td><b>38.7 ± 45.49</b></td><td><b>0.56 ± 0.2</b></td><td>39.83 ± 30.19</td><td></td><td><b>81.33 ± 92.37</b></td><td><b>51.64 ± 48.97</b></td><td><b>0.66 ± 0.19</b></td><td><b>36.9 ± 28.19</b></td>
</tr>
<tr>
	<td>Warping</td><td>190.2 ± 142.46</td><td><b>43.59 ± 48.43</b></td><td><b>0.56 ± 0.2</b></td><td>39.56 ± 29.15</td><td></td><td>204.26 ± 179.53</td><td><b>80.23 ± 75.14</b></td><td>0.65 ± 0.19</td><td><b>37.07 ± 27.29</b></td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>160.08 ± 115.51</b></td><td>69.23 ± 56.09</td><td><b>0.57 ± 0.19</b></td><td>39.68 ± 28.32</td><td></td><td><b>180.23 ± 170.31</b></td><td>96.79 ± 98.12</td><td><b>0.67 ± 0.18</b></td><td>38.46 ± 27.66</td>
</tr>
<tr>
	<td>cbDice</td><td><b>156.88 ± 113.95</b></td><td><b>54.75 ± 48.32</b></td><td><b>0.56 ± 0.2</b></td><td>39.62 ± 28.76</td><td></td><td>233.49 ± 202.28</td><td>106.08 ± 104.07</td><td>0.65 ± 0.18</td><td>37.84 ± 27.71</td>
</tr>
</tbody>
</table>


### Study case 8: Addressing the Dataset Challenge of Pseudo-labels
<table>
<thead>
<tr>
<th colspan=2></th><th colspan=4>Standard supervised learning + Pseudo-labels</th><th>vs.</th><th colspan=4>Self-distillation + Pseudo-labels</th></tr>
<tr>
<th></th>
	<th>Loss</th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th><th></th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">ID</td>	<td>CEDice</td><td>12.26 ± 9.71</td><td>12.0 ± 10.14</td><td>0.86 ± 0.07</td><td>3.17 ± 1.96</td><td></td><td>3.76 ± 4.0</td><td>3.99 ± 4.68</td><td>0.87 ± 0.07</td><td>3.11 ± 2.01</td>
</tr>
<tr>
	<td>RWLoss</td><td><b>11.38 ± 9.29</b></td><td><b>11.11 ± 10.4</b></td><td><b>0.87 ± 0.07</b></td><td><b>3.09 ± 1.98</b></td><td></td><td>4.43 ± 4.36</td><td>4.27 ± 4.56</td><td>0.87 ± 0.07</td><td>3.11 ± 2.01</td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>10.43 ± 8.22</b></td><td><b>10.13 ± 9.07</b></td><td>0.86 ± 0.07</td><td>3.18 ± 1.94</td><td></td><td>7.57 ± 6.02</td><td><b>3.55 ± 3.22</b></td><td>0.86 ± 0.07</td><td>3.43 ± 2.23</td>
</tr>
<tr>
	<td>TOPO</td><td>36.09 ± 25.6</td><td>31.1 ± 24.7</td><td>0.81 ± 0.07</td><td>4.4 ± 2.21</td><td></td><td>64.52 ± 26.38</td><td>73.08 ± 56.86</td><td>0.76 ± 0.08</td><td>5.56 ± 3.07</td>
</tr>
<tr>
	<td>clDice</td><td><b>1.8 ± 2.23</b></td><td><b>4.3 ± 5.37</b></td><td><b>0.87 ± 0.07</b></td><td>3.17 ± 2.09</td><td></td><td><b>2.13 ± 2.54</b></td><td><b>2.69 ± 3.51</b></td><td><b>0.88 ± 0.08</b></td><td>3.24 ± 2.24</td>
</tr>
<tr>
	<td>Warping</td><td>13.57 ± 10.78</td><td>13.26 ± 11.74</td><td><b>0.87 ± 0.07</b></td><td><b>3.12 ± 1.97</b></td><td></td><td>4.58 ± 4.33</td><td>4.82 ± 5.04</td><td><b>0.88 ± 0.07</b></td><td><b>3.01 ± 1.97</b></td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>10.4 ± 9.31</b></td><td><b>10.43 ± 9.27</b></td><td>0.86 ± 0.07</td><td>3.29 ± 2.06</td><td></td><td>4.72 ± 5.46</td><td>5.44 ± 5.91</td><td>0.86 ± 0.08</td><td>3.4 ± 2.17</td>
</tr>
<tr>
	<td>cbDice</td><td>13.12 ± 12.76</td><td>14.66 ± 13.15</td><td>0.86 ± 0.07</td><td><b>3.13 ± 1.98</b></td><td></td><td>4.5 ± 4.8</td><td>4.79 ± 5.31</td><td>0.87 ± 0.07</td><td><b>3.06 ± 2.02</b></td>
</tr>
<tr>
<td rowspan="8">OOD</td>	<td>CEDice</td><td>126.21 ± 126.54</td><td>83.92 ± 68.47</td><td>0.68 ± 0.2</td><td>35.19 ± 29.88</td><td></td><td>68.35 ± 81.14</td><td>47.43 ± 54.34</td><td>0.69 ± 0.19</td><td>36.34 ± 29.07</td>
</tr>
<tr>
	<td>RWLoss</td><td><b>114.65 ± 115.08</b></td><td>96.49 ± 82.71</td><td>0.67 ± 0.18</td><td>36.65 ± 26.79</td><td></td><td>72.95 ± 82.68</td><td><b>43.88 ± 50.22</b></td><td>0.67 ± 0.19</td><td>36.09 ± 27.9</td>
</tr>
<tr>
	<td>TopoLoss</td><td><b>117.98 ± 108.17</b></td><td>110.32 ± 97.64</td><td>0.68 ± 0.2</td><td>36.33 ± 32.16</td><td></td><td><b>34.87 ± 35.12</b></td><td><b>31.74 ± 38.98</b></td><td>0.68 ± 0.19</td><td>37.93 ± 32.34</td>
</tr>
<tr>
	<td>TOPO</td><td>225.37 ± 191.59</td><td>132.21 ± 152.82</td><td>0.66 ± 0.17</td><td>40.53 ± 26.88</td><td></td><td>131.26 ± 107.06</td><td>57.51 ± 84.87</td><td>0.63 ± 0.15</td><td>39.74 ± 29.12</td>
</tr>
<tr>
	<td>clDice</td><td><b>32.7 ± 42.79</b></td><td><b>61.57 ± 71.51</b></td><td>0.67 ± 0.17</td><td>36.87 ± 28.15</td><td></td><td><b>32.55 ± 38.26</b></td><td>55.92 ± 66.45</td><td>0.63 ± 0.18</td><td>37.82 ± 28.83</td>
</tr>
<tr>
	<td>Warping</td><td>127.08 ± 141.36</td><td>95.47 ± 81.2</td><td>0.67 ± 0.19</td><td>36.43 ± 28.29</td><td></td><td><b>66.29 ± 69.64</b></td><td><b>42.68 ± 49.32</b></td><td>0.66 ± 0.19</td><td>36.48 ± 28.36</td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>113.63 ± 124.15</b></td><td>85.75 ± 77.06</td><td><b>0.7 ± 0.19</b></td><td>34.91 ± 27.81</td><td></td><td>81.54 ± 97.27</td><td>61.77 ± 64.07</td><td>0.68 ± 0.18</td><td>36.44 ± 27.7</td>
</tr>
<tr>
	<td>cbDice</td><td>141.42 ± 129.71</td><td>122.91 ± 103.54</td><td>0.68 ± 0.2</td><td>35.72 ± 32.0</td><td></td><td>78.42 ± 79.94</td><td>61.37 ± 65.34</td><td>0.68 ± 0.2</td><td>37.29 ± 30.67</td>
</tr>
</tbody>
</table>


### Study case 9: Addressing the Dataset Challenge of Noisy labels
<table>
<thead>
<tr>
<th colspan=2></th><th colspan=4>Standard supervised learning + Noisy labels</th><th>vs.</th><th colspan=4>Self-distillation + Noisy labels</th></tr>
<tr>
<th></th>
	<th>Loss</th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th><th></th>
<th>Betti0</th><th>Betti1</th><th>Dice</th><th>HD95</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">ID</td>	<td>CEDice</td><td>6.24 ± 6.64</td><td>5.08 ± 5.21</td><td>0.63 ± 0.12</td><td>3.84 ± 1.66</td><td></td><td>2.43 ± 2.93</td><td>2.79 ± 3.1</td><td>0.65 ± 0.12</td><td>3.72 ± 1.74</td>
</tr>
<tr>
	<td>RWLoss</td><td>17.9 ± 30.69</td><td>11.93 ± 13.81</td><td><b>0.66 ± 0.13</b></td><td><b>3.57 ± 1.73</b></td><td></td><td>13.48 ± 21.05</td><td>16.62 ± 19.25</td><td><b>0.66 ± 0.12</b></td><td><b>3.62 ± 1.82</b></td>
</tr>
<tr>
	<td>TopoLoss</td><td>7.19 ± 9.93</td><td>6.62 ± 6.85</td><td>0.62 ± 0.13</td><td>3.88 ± 1.73</td><td></td><td>5.57 ± 6.59</td><td>8.28 ± 7.7</td><td>0.6 ± 0.15</td><td>4.09 ± 2.72</td>
</tr>
<tr>
	<td>TOPO</td><td>24.14 ± 30.55</td><td>10.93 ± 13.42</td><td><b>0.84 ± 0.07</b></td><td><b>2.65 ± 0.93</b></td><td></td><td>67.23 ± 103.18</td><td>9.25 ± 14.23</td><td><b>0.83 ± 0.08</b></td><td><b>3.52 ± 2.63</b></td>
</tr>
<tr>
	<td>clDice</td><td><b>5.01 ± 4.95</b></td><td>6.77 ± 5.81</td><td><b>0.69 ± 0.12</b></td><td><b>3.58 ± 1.68</b></td><td></td><td><b>0.91 ± 1.44</b></td><td><b>1.64 ± 2.08</b></td><td><b>0.78 ± 0.1</b></td><td><b>2.98 ± 1.48</b></td>
</tr>
<tr>
	<td>Warping</td><td>8.48 ± 9.78</td><td>9.49 ± 8.26</td><td>0.62 ± 0.13</td><td>3.85 ± 1.7</td><td></td><td>2.75 ± 3.13</td><td>4.0 ± 4.59</td><td><b>0.67 ± 0.13</b></td><td><b>3.56 ± 1.74</b></td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>3.79 ± 4.8</b></td><td><b>1.49 ± 2.19</b></td><td><b>0.76 ± 0.09</b></td><td><b>3.24 ± 1.47</b></td><td></td><td><b>1.43 ± 2.28</b></td><td><b>0.97 ± 1.58</b></td><td><b>0.79 ± 0.07</b></td><td><b>3.08 ± 1.34</b></td>
</tr>
<tr>
	<td>cbDice</td><td>10.5 ± 10.6</td><td>9.74 ± 8.39</td><td>0.63 ± 0.13</td><td><b>3.82 ± 1.71</b></td><td></td><td>4.74 ± 5.84</td><td>6.25 ± 7.12</td><td><b>0.69 ± 0.13</b></td><td><b>3.44 ± 1.73</b></td>
</tr>
<tr>
<td rowspan="8">OOD</td>	<td>CEDice</td><td>175.17 ± 109.47</td><td>18.72 ± 24.36</td><td>0.33 ± 0.14</td><td>40.17 ± 33.05</td><td></td><td>113.55 ± 87.91</td><td>13.51 ± 17.81</td><td>0.35 ± 0.14</td><td>42.12 ± 35.22</td>
</tr>
<tr>
	<td>RWLoss</td><td>427.46 ± 230.97</td><td><b>17.87 ± 21.87</b></td><td>0.31 ± 0.14</td><td><b>36.35 ± 27.0</b></td><td></td><td>288.58 ± 183.94</td><td>17.93 ± 21.14</td><td>0.3 ± 0.14</td><td><b>36.97 ± 29.49</b></td>
</tr>
<tr>
	<td>TopoLoss</td><td>205.05 ± 112.25</td><td><b>14.84 ± 18.31</b></td><td>0.26 ± 0.14</td><td>42.99 ± 39.85</td><td></td><td>112.54 ± 70.03</td><td>15.73 ± 20.22</td><td>0.25 ± 0.14</td><td>71.88 ± 93.34</td>
</tr>
<tr>
	<td>TOPO</td><td><b>157.22 ± 125.65</b></td><td>140.45 ± 105.8</td><td><b>0.58 ± 0.16</b></td><td><b>39.27 ± 27.05</b></td><td></td><td>118.35 ± 111.84</td><td>81.03 ± 73.9</td><td><b>0.54 ± 0.16</b></td><td>41.72 ± 28.55</td>
</tr>
<tr>
	<td>clDice</td><td><b>126.52 ± 89.23</b></td><td>48.2 ± 48.63</td><td><b>0.46 ± 0.14</b></td><td><b>38.9 ± 26.86</b></td><td></td><td><b>62.84 ± 63.16</b></td><td>22.54 ± 24.62</td><td><b>0.49 ± 0.16</b></td><td>42.41 ± 33.64</td>
</tr>
<tr>
	<td>Warping</td><td>285.2 ± 170.07</td><td><b>18.01 ± 20.78</b></td><td><b>0.35 ± 0.12</b></td><td><b>37.61 ± 25.5</b></td><td></td><td>153.22 ± 108.2</td><td>16.17 ± 19.84</td><td><b>0.37 ± 0.13</b></td><td>42.07 ± 33.03</td>
</tr>
<tr>
	<td>SkelRecall</td><td><b>113.33 ± 87.74</b></td><td>44.6 ± 36.58</td><td><b>0.51 ± 0.16</b></td><td>40.16 ± 29.33</td><td></td><td><b>62.77 ± 63.92</b></td><td>36.82 ± 32.2</td><td><b>0.55 ± 0.16</b></td><td><b>39.94 ± 29.55</b></td>
</tr>
<tr>
	<td>cbDice</td><td>275.69 ± 148.81</td><td>27.5 ± 27.25</td><td>0.31 ± 0.14</td><td><b>39.26 ± 32.83</b></td><td></td><td>163.05 ± 111.21</td><td>23.56 ± 49.91</td><td>0.35 ± 0.15</td><td>42.83 ± 35.83</td>
</tr>
</tbody>
</table>


