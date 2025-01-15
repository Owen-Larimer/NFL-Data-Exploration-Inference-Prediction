# NFL-Data-Exploration-Inference-Prediction

### Project Overview

A project dedicated to taking in NFL data, cleaning it, exploring it, and drawing conclusions and predictions from it. The full project can be found in the "NFL_Explore_Infer_Predict.ipynb" file. In there you will find all three sections. The notebook is annotated at major points, including longer sectional discoveries at the end of each of the three portions. This page will be simpler and easier to read, and will be useful for those trying to skim what I've done here, but will not grasp the full extent of the project.

### Data Sources

Player Data: The main data source used for this project was the dataset was the file "players.csv" taken from [this kaggle page.](https://www.kaggle.com/datasets/aryashah2k/beginners-sports-analytics-nfl-dataset?select=players.csv)

College Data: For the inference section, we also used the file "cfb17.csv" taken from [its own kaggle page.](https://www.kaggle.com/datasets/jeffgallini/college-football-team-stats-2019/data?select=cfb17.csv)

### Tools

Jupyter Notebook
- **Pandas & Numpy** - Cleaning and exploration
- **Matplotlib & Seaborn** - Visualizations and plots
- **Scipy.stats** - One Way ANOVA Test
- **Sklearn.ensemble** - Random forest Classifer
- **Sklearn.metrics** - Log Loss and Confusion Matricies
- **Sklearn.neighbors** - KNN Classifier
- **Sklearn.model_selection** - Train Test Split and KFold

### EDA Phase

#### Cleaning
After doing "df.dtypes" to look at each column's type, it was clear that the data was already close to clean. The main issue I encountered was just that the heights of the players were not standardized. It was either in inches, like "72" or in a foot-inch format such as "6-0." Obviously the latter made it so that the data type of height was not numerical, which it had to be to be used as a feature in our later models. I decided to standardize the height to an inch format. This was done like so:

```python
nfl['height'] = nfl['height'].str.strip()
heights_unfinished = nfl['height'].str[1] == '-'#nfl['height'].str[0] if nfl['height'][1] = '-'
nfl['fix_height'] = heights_unfinished
nfl

height_to_fix = nfl[nfl['fix_height'] == True]

int_feet = np.array([int(foot) for foot in height_to_fix['height'].str[0]])
int_feet = int_feet * 12
int_inches = np.array([int(inches) for inches in height_to_fix['height'].str[2:]])
#int_inches
int_height = int_feet + int_inches
int_height

height_to_fix['height'] = int_height
height_to_fix

good_heights = nfl[nfl['fix_height'] == False]
good_heights['height'] = good_heights['height'].astype(int)#.dtypes
good_heights.dtypes
good_heights

fixed_nfl = pd.concat([height_to_fix, good_heights])
fixed_nfl = fixed_nfl.sort_index(ascending=True)
fixed_nfl
```

#### Visualizing
W.I.P.

### Inference 
W.I.P.

### Prediction
W.I.P.

### Findings
W.I.P.

### Limitations
W.I.P.

### From Here
W.I.P.
