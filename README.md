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
We started by loading in the player data from "players.csv."

After doing "df.dtypes" to look at each column's type, it was clear that the data was already close to clean. The main issue I encountered was just that the heights of the players were not standardized. It was either in inches, like "72" or in a foot-inch format such as "6-0." Obviously the latter made it so that the data type of height was not numerical, which it had to be to be used as a feature in our later models. I decided to standardize the height to an inch format. This was done like so: (Details and information regarding each section of code can be found in the actual notebook).

This was the initial height column:
![image](https://github.com/user-attachments/assets/4c425453-d2be-4911-a351-9bbd05bc0f7e)

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

And this is what our height column looked like after this code:
![image](https://github.com/user-attachments/assets/187b03d8-3ebb-4e05-a443-bea748f65346)

Secondly, I turned the birthdate column into an actual datettime type so that I could use it to get ages for every player and use it in my later models IF wanted:

```python
fixed_bdates = pd.to_datetime(fixed_nfl['birthDate'], format='mixed')

fixed_nfl['birthDate'] = fixed_bdates
```

I then made it into an age column. This dataset is from the 2018 NFL season, so our ages are determined from that:
```python
fixed_nfl['player_age'] = 2018 - fixed_nfl['birthDate'].dt.year
fixed_nfl
```

Now running fixed_nfl.describe() gave us:
![image](https://github.com/user-attachments/assets/69703b35-45df-4ee4-b2e7-7c32686bea97)

And we could now use our numeric columns in the dataframe to get correlation coefficients for relevant columns:
![image](https://github.com/user-attachments/assets/65b53c83-3947-40d7-8dc0-bacfc2191e4d)

And we generated some pairplots to visualize the associations:
![image](https://github.com/user-attachments/assets/8a20b565-657f-4011-98a3-f220a82bb928)

Initially, I was expecting there to be some greater associations between player age and the physical columns (height and weight), but the associations are almost impossible to see due to how many factors go into each player. Position, initial size, and side of ball are all figures that influence a player's specimen.

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
