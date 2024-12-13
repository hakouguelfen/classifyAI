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
-  check if there is outliers for report, plot them
### NOTES:
- there are alot of outliers in insu attribute (374)
- use log transformation
- don't use fill missing values, because they are not missing, they are 0
- good for KNN:{'rand': 121, 'k': 91, 'precision': 0.94} 
- Naive bayesian: {'rand': 452, 'precision': 0.85}

