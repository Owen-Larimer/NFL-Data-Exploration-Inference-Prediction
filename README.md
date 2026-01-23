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
Moving onto initial visualization, I just want some basic figures to see relationships across different variables.

For example, this heights by position figure:
![image](https://github.com/user-attachments/assets/23cd5141-a3f0-44e6-a9d7-395109d598fc)

Using the code:
```python
heights_by_position = fixed_nfl.groupby('position')['height'].mean()
heights_by_position = heights_by_position.sort_values(ascending=True)
sns.barplot(x=heights_by_position.values, y=heights_by_position.index, hue=heights_by_position.index)
plt.ylabel('Position')
plt.xlabel('Average Height (in)')
plt.title('Average Height by Position')

for index, value in enumerate(heights_by_position.values):
    plt.text(value + 0.2, index, f'{value:.2f}', va='center', color='black', fontsize=8)
```
The groupby just makes it much easier to chart the figure.

**NOTE:** Normally we would not want a bar chart with such large bars and close values. The data-to-ink ratio is very small and overall it just does too much. Best practice would be to instead have points or markers, but with the large amount of positional groups it can become hard to correlate a ytick value with it's dot, and legends become clunky with so many values. Also, the color palette here is UGLY. We want something better than a rainbow, like a sequential palette.

We could also shorten the x-axis intentionally with something like:

```python
ax = sns.barplot(x=heights_by_position.values, y=heights_by_position.index, hue=heights_by_position.index)
ax.set_xlim(65, 85)
```

for a result more like:
![image](https://github.com/user-attachments/assets/1c39bda9-e060-450a-b45b-2100429efc92)

This allows us to see differences across the observations very easily, but they are exaggerated by not including 0.

We can also do a similar chart for weight (pictured below) or age:

![image](https://github.com/user-attachments/assets/82cc9e0f-9152-4b72-aba2-79f3269cd3df)

From these two figures it's clear that cornerbacks and wide receivers are our lightest positions, nose and defensive tackles are the heaviest, runningbacks are our shortest, and tight ends are our tallest.

If we also want to better visualize the association between height and weight, we can do something like:
![image](https://github.com/user-attachments/assets/aa3c6884-f5f1-4462-b445-eff1ac7ddfe6)



Looking lastly at the CollegeName column of our fixed_nfl dataframe:
![image](https://github.com/user-attachments/assets/8b8ceaae-1470-45c3-a021-0dce64f8258f)

and we can look at the values of the colleges:
<img width="601" height="553" alt="image" src="https://github.com/user-attachments/assets/068564e6-b301-4cfe-8ea2-4ea7c31be6ba" />

There are well over 200 colleges here. Makes sense, football players come from all over the country. About half of them have only 1, 2, or 3 players that come from them, though. Let's see if we can find the colleges that have more NFL recruited players, we'll strike a balance. Trying 12 first.

<img width="1151" height="523" alt="image" src="https://github.com/user-attachments/assets/1402b95b-7883-4445-85bf-fe34744a3ebe" />


So, there are way too many schools to properly analyze them all individually, but if we reduce the schools then we lose player data. How can we remedy this?

Perhaps we should look at a conferences instead. Most of the schools NFL players come from are part of some larger NCAA conference, such as PAC-12, BIG-10, or SEC. If we can get data on the schools and what conference they are part of, we can join that data with our fixed_nfl to get a conference associated with each player. Then, we have fewer groups to compare and more data in each group. And schools that are not part of specific conferences or from more obscure ones can be grouped together as "other" or "unaffiliated."

Our NFL data is from the 2018 portion of the 2018-19 season, so let's get our college football conference data from the 2017-18 year. It is important for us to keep in mind that some conferences do change with restructuring, so these conferences might not be accurate to today's structure, and some of these players' schools may have been in different conferences when they were there or drafted. Let's do that below:




#### College Football Dataset:
**Also** in this cleaning section, we took another dataset, the "cfb17.csv" set. Here it is loaded into Jupyter:
![image](https://github.com/user-attachments/assets/24d74f2c-6238-4a36-993e-2be59a96845c)

My goal with this set is to use it's "Team" column (which is just the college's name and it's conference), along with the collegeName column in the "fixed_nfl" dataframe to get college conferences for each player in the NFL. It'll take some work, and some nifty cleaning tricks, to extract the conference and put it into the correct places for each player. Once again, the full process is available in the file, but I'll go through some steps here:

Looking at the value counts for cfb:
```python
cfb['Team'].value_counts()
```

It appears to be in a specific format: "<School>" "(<conference>)". We can take advantage of that with RegEx:

```python
cfb['Team'] = cfb['Team'].replace('Miami (FL) (ACC)', 'Miami (ACC)')
cfb['Team'] = cfb['Team'].replace('Miami (OH) (MAC)', 'Miami, O. (MAC)')

cfb[['school', 'conference']] = cfb['Team'].str.extract(r'^(.*)\((.*)\)$')
```

And now our cfb dataframe has a school column and a conference column.

They were added to the back of the cfb dataset. Let's make sure the parsing went well by checking for spaces:
```python
cfb['school'].str.len()
```
It says Air Force has 10 and Akron has 6, 1 more than is true, let's strip that column and the conference column:

```python
cfb['school'] = cfb['school'].str.strip()
cfb['school'].str.len()
cfb['conference'] = cfb['conference'].str.strip()
```

And we only want the conferences and schools to join on our original set, so let's make it easier by just grabbing those in it's own dataframe. We can always come back to the numerical data later on.

```python
cfb_conferences = cfb[['school', 'conference']]
cfb_conferences
```

<img width="485" height="838" alt="image" src="https://github.com/user-attachments/assets/176ce12f-269a-4268-bc7b-113b8c5ee564" />

It looks like not all of our schools match in the fixed_nfl and the cfb_conferences sets. Some are shortened. We might have to change some manually, but let's start by extending all of the state schools to "State" instead of "St."

```python
cfb_conferences['school'] = cfb_conferences['school'].str.replace("St.", "State")

# Let's check that
cfb_conferences[cfb_conferences['school'].str.contains("St.", case=True, na=False)]
```
<img width="469" height="537" alt="image" src="https://github.com/user-attachments/assets/fbdecb25-9124-4218-aa60-9338e9032557" />

Now let's check the two school columns against each other, this will make it much easier to manually change the data if needed:

```python
nfl_schools = pd.Series(fixed_nfl['collegeName'].unique())
nfl_schools_frame = nfl_schools.to_frame()
nfl_schools_frame['in_both'] = nfl_schools_frame[0].isin(cfb_conferences['school'])
nfl_schools_frame[nfl_schools_frame['in_both'] == False].head(50)
```
<img width="459" height="595" alt="image" src="https://github.com/user-attachments/assets/9b67f9f5-0d33-47a7-9af0-c968cafc8ebc" />


Seems like most aren't in both...It looks like we will have to change a number of colleges. We'll mostly focus on the most prominent ones, like LSU and Ole Miss and the Michigan schools. If we discover any later on, we can always change them.

We have to change them within the cfb_conferences dataframe. We'll check back on the nfl_schools_frame every once in a while to make sure it works (Will just have to rerun the cell above).

```python

#cfb_conferences['school'] = cfb_conferences['school'].replace('LSU', 'Louisiana State') - Special Case where both are present in nfl_fixed
cfb_conferences['school'] = cfb_conferences['school'].replace('Ole Miss', 'Mississippi')
#cfb_conferences[cfb_conferences['school'].str.contains("Mich", case=True, na=False)]
cfb_conferences['school'] = cfb_conferences['school'].str.replace("Mich.$", "Michigan", regex=True)
cfb_conferences['school'] = cfb_conferences['school'].replace('UNLV', 'Nevada-Las Vegas')
cfb_conferences['school'] = cfb_conferences['school'].replace('SMU', 'Southern Methodist')
cfb_conferences['school'] = cfb_conferences['school'].replace('South Fla.', 'South Florida')
cfb_conferences['school'] = cfb_conferences['school'].replace('Western Ky.', 'Western Kentucky')
cfb_conferences['school'] = cfb_conferences['school'].replace('UConn', 'Connecticut')
cfb_conferences['school'] = cfb_conferences['school'].replace('NC State', 'North Carolina State')
cfb_conferences['school'] = cfb_conferences['school'].replace('Southern Miss.', 'Southern Mississippi')

fixed_nfl['collegeName'] = fixed_nfl['collegeName'].replace('Louisiana State', 'LSU')

```

Perfect. Check the cfb_conferences dataframe:
<img width="475" height="797" alt="image" src="https://github.com/user-attachments/assets/9ef99961-ff67-4d2e-aedd-04c52d3745ce" />


And it looks good. Let's merge them!

```python
fixed_nfl = fixed_nfl.merge(cfb_conferences, left_on = 'collegeName', right_on = 'school', how='left').drop(columns=['school'])
fixed_nfl
fixed_nfl.rename({'conference': 'collegeConference'}, axis=1, inplace=True)
```

<img width="1850" height="431" alt="image" src="https://github.com/user-attachments/assets/1011ffcf-4df0-4417-9c1d-fe44431d9e65" />


We'll change the value of all NAs to 'Other' moving forward, since we have a good amount of schools, and we'll visualize our players now by conference:

```python
fixed_nfl.loc[fixed_nfl['collegeConference'].isna(), 'collegeConference'] = 'Other'

conference_values = fixed_nfl['collegeConference'].value_counts()

sns.barplot(x=conference_values.index, y=conference_values.values)

plt.xlabel('College Conference')
plt.ylabel('Number of Players')
plt.title('Number of NFL Players per College Conference')
plt.xticks(rotation=45)
```

<img width="1142" height="1048" alt="image" src="https://github.com/user-attachments/assets/f2beefe0-910f-43a2-8129-e9c8c6e90418" />


I did some more brute force work to fix some more schools:

```python
fixed_nfl.loc[fixed_nfl["collegeName"] == "Central Florida", "collegeConference"] = "AAC" #Were in AAC up until 2021
fixed_nfl.loc[fixed_nfl["collegeName"] == "North Dakota State", "collegeConference"] = "MVFC"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Texas Christian", "collegeConference"] = "Big 12"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Northern Illinois", "collegeConference"] = "MAC"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Illinois State", "collegeConference"] = "MVFC"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Florida Atlantic", "collegeConference"] = "AAC"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Eastern Washington", "collegeConference"] = "Big Sky"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Georgia Southern", "collegeConference"] = "Sun Belt"
fixed_nfl.loc[fixed_nfl["collegeName"] == "James Madison", "collegeConference"] = "Sun Belt"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Coastal Carolina", "collegeConference"] = "Sun Belt"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Portland State", "collegeConference"] = "Big Sky"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Miami (Fla.)", "collegeConference"] = "ACC"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Middle Tennessee", "collegeConference"] = "C-USA"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Northern Iowa", "collegeConference"] = "MVFC"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Samford", "collegeConference"] = "SoCon"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Western Carolina", "collegeConference"] = "SoCon"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Alabama-Birmingham", "collegeConference"] = "AAC"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Louisiana-Lafayette", "collegeConference"] = "Sun Belt"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Florida International", "collegeConference"] = "C-USA"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Sacramento State", "collegeConference"] = "Big Sky"
fixed_nfl.loc[fixed_nfl["collegeName"] == "Chattanooga", "collegeConference"] = "SoCon"
```

Which leaves us with a finished dataset as so:
<img width="1675" height="418" alt="image" src="https://github.com/user-attachments/assets/5d84a398-ada8-4dea-8880-16c29746376f" />

And values:
<img width="376" height="626" alt="image" src="https://github.com/user-attachments/assets/03d7a0ab-b316-4bb5-8326-ef2820c0f336" />

#### Initial Impressions:

I have a fair amount of categorical data. But I also some good numerical data that might be useful in pairing with some of those categories. I am particularly curious about heights/weights grouped by positions, and with the new conferences I pulled in, I wonder about which conferences have the most height or weight at certain positions. Furthermore, I am curious whether you could predict the college or more likely conference of a player in the NFL based on their size, age, and position. That might reveal what colleges look for in recruitment for specific positions. Perhaps tall, yet light receivers in the NFL are more likely to come from the Pac-12, and heavier, shorter running backs come more often from the SEC. We could get more position specific in our precictions as well. Someone who is older in our dataset has been in the league a long time. Perhaps their height/weight at their position is a good reason, and maybe that can be used to determine which conferences have longest lasting NFL players.

The positions themselves are particularly interesting to me. What makes a good player at each position? Are there any trends for younger vs older players at each position? etc.

I don't think I will, but if I struggle to really find something to predict, I also have backup college football team statistics from the year prior to this, which can possibly be used by itself or in conjuction with the NFL data I have in other ways.



### Inference 
Moving onto inference. I want to make a hypothesis; something I'm able to learn from, evaluate a relevant association, and then bootstrap to create a confidence interval.

I'm interested in what conferences produce what size of players. For instance, the SEC is widely considered the football powerhouse conference. I might expect players in the NFL who played in the SEC to be significantly bigger than NFL players who played in other conferences. I could combine height and weight into its own variable "size", but I would rather look at both of them separately to see their differences.

**Alternative Hypothesis:** At least one college conference produces players with significantly different mean heights (or weights) compared to the other conferences.

**Null Hypothesis:** The average height and weight of NFL players does not vary significantly across college conferences.



### Prediction
W.I.P.

### Findings
W.I.P.

### Limitations
W.I.P.

### From Here
W.I.P.
