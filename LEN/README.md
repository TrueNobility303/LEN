# LEN (Lazy Extra Newton)

Before running the codes, create the following folders if they do not exist
```
mkdir img
mkdir result
```

## Synthetic Problem

Experiment on the cubic regularized bilinear min-max problem of the form

$$
\min_{x \in \mathbb{R}^n} \max_{y \in \mathbb{R}^n} f(x,y) = \frac{\rho}{6} \Vert x \Vert^3  + y^\top (A x - b).
$$

Reproduce our results via
```
python -u Synthetic.py --n 10 --training_time 10.0
python -u Synthetic.py --n 100 --training_time 50.0
python -u Synthetic.py --n 100 --training_time 80.0
```
You are also encouraged to vary the hyper-parameters to see their performance.

## Fairness Machine Learning 

The objective is given by

$$
    \min_{x \in \mathbb{R}^{d_x}} \max_{y \in \mathbb{R}} \frac{1}{n} \sum_{i=1}^n \ell(b_i a_i^\top x) - \beta \ell(c_i y a_i^\top x) + \lambda \Vert x \Vert^2 -\gamma y^2.
$$

Create a new folder
```
mkdir Data
```
Download the dataset from the links: [adult](https://github.com/7CCLiu/Partial-Quasi-Newton/blob/main/a9a.mat) , [lawschool](https://github.com/7CCLiu/Partial-Quasi-Newton/blob/main/LSTUDENT_DATA1.mat), and put them in the created folder.

Reproduce our results via
```
python -u Fairness.py --rho 10.0 --dataset heart --training_time 10.0
python -u Fairness.py --rho 10.0 --dataset adult --training_time 500.0
python -u Fairness.py --rho 10.0 --dataset lawschool --training_time 5000.0
```

When the program is completed, the results will be stored in the folder `./result`.
