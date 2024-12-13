### Installation

create virtual enviroment
```python3
python3 -m venv env
```

activate virtual enviroment
```python3
source env/bin/activate
```

install requirement packages
```python3
pip install -r requirements.txt
```

start the app
```python3
python main.py
```

### TODO:
-  check if there is outliers for report, plot them
### NOTES:
- there are alot of outliers in insu attribute (374)
- use log transformation
- don't use fill missing values, because they are not missing, they are 0
- good for KNN:{'rand': 121, 'k': 91, 'precision': 0.94} 
- Naive bayesian: {'rand': 452, 'precision': 0.85}

