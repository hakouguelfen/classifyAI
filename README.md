### Installation

create virtual enviroment
```bash
python3 -m venv env
```

activate virtual enviroment
```bash
source env/bin/activate
```

install requirement packages
```bash
pip install -r requirements.txt
```

start the app
```bash
python main.py
```

### TODO:
- check if there is outliers for report, plot them
- show K input when knn is selected
### NOTES:
- there are alot of outliers in `insu` attribute (374)

## Introduction
- ...

## Preprocessing
### Initial view
- no missing values
- Dataset size and structure
- Description of features


### Data Scaling
- explain each one how it works, use math too

+ Log transformation for 'insu' and 'pedi' (good for handling skewness and large values)
+ StandardScaler for 'plas' and 'pres' (normal variables)
+ RobustScaler for variables that might have outliers ('preg', 'skin', 'mass', 'age')


## Models
### KNN
- random_state: 121
- k: 91 
- using these parameters gave the best percision: 94%

### Naive Bayes
- random_state: 452
- using these parameters gave the best percision: 85%

### Decision Tree
- random_state: 147
- max_features: sqrt  | Using max_features="sqrt" introduces randomness, speeds up training
- min_samples_leaf: 4 |
- max_depth: 5        |
- using these parameters gave the best percision: 91%
