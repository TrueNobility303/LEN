# LEN

Beforing runing the codes, create the following folders if not exist
```
mkdir img
mkdir result
```

## Synthetic Problem

Reproduce the result via

```
python -u Synthetic.py --n 100 --training_time 10.0
python -u Synthetic.py --n 200 --training_time 20.0
python -u Synthetic.py --n 500 --training_time 100.0
```


## Fairness Machine Learning 

Create a new folder
```
mkdir Data
```
Download the dataset from the links: [adult](https://github.com/7CCLiu/Partial-Quasi-Newton/blob/main/a9a.mat) [lawschool](https://github.com/7CCLiu/Partial-Quasi-Newton/blob/main/LSTUDENT_DATA1.mat)
,and put them in the created folder.

Reproduce the result via
```
python -u Fairness.py --M 10.0 --LazyM 10.0  --dataset adult --training_time 10.0
python -u Fairness.py --M 10.0 --LazyM 100.0 --dataset lawschool --training_time 100.0
```
