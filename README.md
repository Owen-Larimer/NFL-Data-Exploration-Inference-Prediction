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



<img width="895" height="147" alt="image" src="https://github.com/user-attachments/assets/7f81cae2-40eb-41a8-b682-5e10308712fa" />  

We'll start with conferences that have more than 20 players from them. 11 + 1 'Other' group for us to look at.  

We will use an ANOVA (Analysis of Variance) test to compare our groups means/variances against each other, looking for any notable differences. This test does not have high variance in itself, making it good for inference. We will do a one-way ANOVA test with each of height and weight.. I found information on what the test is here: [ANOVA test info page](https://www.scribbr.com/statistics/one-way-anova/#:~:text=ANOVA%2C%20which%20stands%20for%20Analysis,ANOVA%20uses%20two%20independent%20variables).  

First, let's filter out only the conferences with 20+ players

```python
confs = fixed_nfl['collegeConference'].value_counts()
valid_confs = confs[confs >= 20].index
filtered_nfl = fixed_nfl[fixed_nfl['collegeConference'].isin(valid_confs)]
#confs
#valid_confs
filtered_nfl.head()
```  

<img width="1595" height="372" alt="image" src="https://github.com/user-attachments/assets/0580a25c-93c4-47d1-825f-2639ebe02b50" />  

And let's do some plots so it's clear what relationships we're trying to present:

```python
mean_vals = filtered_nfl.groupby('collegeConference')[['height', 'weight']].mean().reset_index()

plt.figure(figsize=(15, 5))

# One for Height
plt.subplot(1, 2, 1)
plt1 = sns.barplot(data=mean_vals, x='collegeConference', y='height', hue='collegeConference', palette='rocket')
plt.title('Mean Heights in College Conferences With Over 20 NFL Players')
plt.xlabel('NCAA Conference')
plt.ylabel('Mean Height (Inches)')
plt.xticks(rotation=45)

# Adding ticks
for val in plt1.patches:
    plt1.annotate(f'{val.get_height():.2f}', 
                 (val.get_x() + val.get_width() / 2., val.get_height()), 
                 xytext=(0, 5),  # 5 points vertical offset
                 textcoords='offset points', 
                 ha='center', va='bottom', fontsize=8, color='black')

# One for Weight
plt.subplot(1, 2, 2)
plt2 = sns.barplot(data=mean_vals, x='collegeConference', y='weight', hue='collegeConference', palette='mako')
plt.title('Mean Weights in College Conferences With Over 20 NFL Players')
plt.xlabel('NCAA Conference')
plt.ylabel('Mean Weight (Pounds)')
plt.xticks(rotation=45)

for val in plt2.patches:
    plt2.annotate(f'{val.get_height():.2f}', 
                 (val.get_x() + val.get_width() / 2., val.get_height()), 
                 xytext=(0, 5),  # 5 points vertical offset
                 textcoords='offset points', 
                 ha='center', va='bottom', fontsize=8)

```  
<img width="2473" height="1084" alt="image" src="https://github.com/user-attachments/assets/e00c98c6-bc34-4a03-b9c4-0184613ad18a" />  


And now we can begin our analysis with **height**. Let's do the one-way with that first:

```python
from scipy.stats import f_oneway

#filtered_nfl.groupby('collegeConference')['height'].mean()
height_groups = [group['height'].values for _, group in filtered_nfl.groupby('collegeConference')]
one_way_results_height = f_oneway(*height_groups)
one_way_results_height
```

Which resulted in:
```python
F_onewayResult(statistic=1.4288627207256122, pvalue=0.15359133079569412)
```

In this first part of our analysis, we tested whether the average height of NFL players differed significantly based on their college's athletic conference. Let us first recognize that our null hypothesis (for height portion) was that the mean height of NFL players does not vary significantly by college conference. In this case, we got an f-statistic, not a t-statistic. Where a t-statistic compares means between two groups for statistical differences, an f-statistic can compare between multiple different groups and (according to the resources I was using) will be able to tell you if there is significant difference between any pair of the groups.

For height, we got an f-statistic of ~1.429. This is relatively low, as it is very close to 1, and it indicates that even if the variability within groups somewhat large (like as a result of positions), the variability between groups (conference averages) is rather low in comparison. By that statistic, we can likely say that there is no huge difference in NFL players heights based upon what conference they were in in college.

The p-value is a low 0.154. The most typical p-value cutoff we use is 0.05. Because our p-value is much larger than the cutoff, it can be reasonably said that any variation between the groups is likely due to chance. These results suggest that we have low confidence in rejecting the null hypothesis, and thus we fail to reject it.

In other words: The difference in average height of NFL players is not statistically significant across all college conferences.

And we can now do the same thing with **weight**:

```python
weight_groups = [group['weight'].values for _, group in filtered_nfl.groupby('collegeConference')]
one_way_results_weight = f_oneway(*weight_groups)
one_way_results_weight
```

Results in: 
```python
F_onewayResult(statistic=0.916356722706692, pvalue=0.5235975579978585)
```

In this section of our analysis, we tested whether college conference has any bearing on weights of NFL players. Let us restate that our null hypothesis (for weight portion) was that the mean weight of NFL players does not vary significantly by college conference. Again we got an f-statistic and a p-value from our ANOVA one way test.

Our weight f-statistic is ~0.9164. This is even closer to 1, and it indicates that our weight variability between conferences is lower than that of within the individual groups, and even moreso than in our height analysis. We obtained a corresponding p-value of around 0.524, which is much larger than the standard cutoff of 0.05. This p-value again indicates a low confidence in rejecting the null hypothesis.

Given such results, we again fail to reject the null hypothesis, or, in other words, our findings are strong evidence that the average weight of NFL players does not appear to vary significantly by college conference, as any observed numerical differences are most likely a result of chance.

And thus we fail to reject the overall null hypothesis that the average height and weight of NFL players does not differ significantly across college conferences.


#### Thoughts:
Honestly, I was actually expecting there to be a significant difference in means for at least some of the conferences. One of my prediction was that the SEC would dominate the others, especially in weight category. It seems as though SEC players, especially linemen, tend to be much heavier than the other conferences' players. Not only was that not true, but weight had even less variation between groups than height did, which I thought was strange since weights vary so much more than heights ever can. I suppose my findings indicate that conferences do not really get large streams of recruits that are much bigger than the recruits of others. This makes sense, since teams are generally made up of the same amount of each position, and each position group will have its own mean heights and weights that don't greatly vary. In a greater scale, this does not prove but may indicate that roughly all positions are represented from all conferences in the NFL. Otherwise, if for example many NFL lineman come from the PAC-12, we might see a greater f-statistic in the weight analysis in favor of PAC-12 (since our initial grouping was by conference, not position).

#### Assumptions:
In terms of assumptions:

Our ANOVA test assumes that our observations are independent of each other. This is a valid assumption for our set, since players' heights and weights do not rest upon the heights and weights of others.
Our data should be approximately normal, which both of our NFL heights and weights variables are (See histograms below). Weights skew right since there more heavier players like linemen on a single team, but that should be representative of all teams and ideally across all conferences.
Random sampling is an assumption of an ANOVA test, and this may not always be true. Certain positions will always be represented more than others, and players are drafted based on a team's needs, not necessarily the makeup of that player. In our case though, there are a number of representations of many sizes at each position, and the distribution of positions in the NFL will remain relatively constant, so that's the population we have to work with.

#### Considerations:
I think it is important to note that a test such as this may be inherently flawed by the idea that the NFL will tend to draft guys that fit the "perfect" size/frame for their position. A heavier or lighter player might not be in the dataset because they were not drafted into the NFL league. Additionally, it is possible and even likely that players drafted into the NFL might grow more (if they are young) and will almost certainly fill out their frames even more, increasing their weight in their first few NFL years. Thus these given heights, but more importantly weights may not vary as much as when these players were in college. It is still very much possible that the SEC does have significant sizae advantage, it's just that we discovered with this test that we cannot glean that just from those players who made it into the NFL. In a dataset with a huge amount of college athletes, we could do a similar test and find very different findings with that data. Lastly, I think it's important to remember that this dataset is for the 2018 NFL season. Players change, conferences realign (especially in recent years), and certain strategies leave in the NFL along with the players. As such, it is very possible for these findings to be different now, 6 years later.


### Prediction

Now onto our prediction section:

##### Question: Can we predict a player's position based on their age, height, weight, and perhaps conference as well?
We'll try a few different models, select features, and do our best to reduce error. We'll need to subset into a train-test split.  

I am somewhat concerned about the AMOUNT of groups our model needs to choose between when predicting. There are so many positions, and some are certainly very closely related to others.

Additionally, some of the positions, especially the bottom 3 (HB, NT, and K) have very few observations in my data, and thus may not provide enough reliable data to accurately classify those positions. I am unsure whether this will effect the model as a whole though.

I have 3 possible solutions assuming that it does hurt the model:

I could abolish those positions altogether, K, HB, and NT are not hugely important positions to begin with and I could just take those (and possibly others if necessary) out of my dataframe completely. The issue with that is I am not sure how it will effect the predictions of the other positions. Would those change?
I could merge those positions. This wouldn't work for Kicker, so solution 1 might still have to apply there, but Halfback (HB) and Nose Tackle (NT) are essentially just subtypes of greater positions (Runningbacks and Defensive Tackles). I could merge those subtypes with their parent positions, which would solve the sample size problem. However, these subtypes exist for a reason, and could have notable differences from their parents. Merging them with their parents will taint the parents' group data, effecting the model, perhaps not by much, but almost certainly affecting it.
Like opion 1, I could reduce the number of positions I am classifying between as a whole. I might choose to only try and classify between WR, CB, RB, TE, QB, OLB, and maybe a safety. Those are generally the most notable and important positions on the field + LB, and they encompass the vast majority of body types and skills in football. This is what I think will end up being done.
I think I will try the model as it stands first, and see how it goes. If my accuracy is 95% normally then none of this matters. I'll adjust my approach as I see fit.

Start with features:
Let's get all our possible features in a dataframe and our positions we want to predict. Lets first look at what our clusters might look like. We will also have to convert our positions to numbers in order to get our random forest classifier to work:

```python
pred_df = fixed_nfl[['height','weight', 'player_age', 'position']]
sns.scatterplot(x=pred_df['player_age'], y=pred_df['weight'], hue=pred_df['position'])
plt.legend()
pred_df.loc[:,"position"] = pred_df["position"].replace(["WR","CB", "RB", "TE", "OLB", 'QB',
                                                    'FS', 'SS', 'LB', 'ILB', 'DE', 'DB',
                                                    'MLB', 'DT', 'FB', 'P', 'LS', 'S', 'HB',
                                                    'NT', 'K'], [0, 1, 2, 3, 4,
                                                                5, 6, 7, 8, 9,
                                                                10, 11, 12, 13, 14,
                                                                15, 16, 17, 18, 19, 20])
X = pred_df.drop("position", axis = 1)
y = pred_df['position']
```
Giving us: 
<img width="1140" height="943" alt="image" src="https://github.com/user-attachments/assets/50dc9eb0-ef97-4c2f-b236-750c8a6250e0" />

It looks like our clusters are not very clear. There is not a lot of clear sections of data points, and it's pretty scattered. It is possible though that we can't see some clear clusters because the points are too close together. Let's do our classifier to see.

Now let's fit the model. I will start without the train-test split because I want to see the results without it first:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

rfc = RandomForestClassifier()

rfc_fit = rfc.fit(X, y.astype(int))
y_preds = rfc_fit.predict(X)
```

Let's see how accurate our model was: 
<img width="643" height="170" alt="image" src="https://github.com/user-attachments/assets/c686dbbc-63a6-4082-ae11-5f0ddcce6a8d" />

90% accuracy is pretty good from the general perspective. However, the class imbalance still concerns me. If our model classified everything as a WR (0), the model would still be 5% correct. It doesn't seem like much, but what if the model is bias towards the top 3 or 4 classes, then only guessing those classes the model could get up to 40-55% correct. Let's try and check our bias.

We can use a confusion matrix to actually look at which values were misidentified: (see ISLR pg. 171 and https://scikit-learn.org/dev/modules/generated/sklearn.metrics.confusion_matrix.html) 
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y, y_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=rfc.classes_)
disp.plot(cmap='rocket', xticks_rotation=45)
``` 

<img width="1021" height="878" alt="image" src="https://github.com/user-attachments/assets/9b4dfe00-5077-4255-bebb-01751390f3e7" />

Let's use our position legend to actually make sense of it:

WR (Wide Receiver): 0 CB (Cornerback): 1 RB (Running Back): 2 TE (Tight End): 3 OLB (Outside Linebacker): 4 QB (Quarterback): 5 FS (Free Safety): 6 SS (Strong Safety): 7 LB (Linebacker): 8 ILB (Inside Linebacker): 9 DE (Defensive End): 10 DB (Defensive Back): 11 MLB (Middle Linebacker): 12 DT (Defensive Tackle): 13 FB (Fullback): 14 P (Punter): 15 LS (Long Snapper): 16 S (Safety): 17 HB (Halfback): 18 NT (Nose Tackle): 19 K (Kicker): 20

It looks like our most common misprediction was predicting a wide receiver as a corner back. This happened 10 times. It also seems we mispredict wide receivers as free and strong safeties sometimes. These midpredictions would make sense however, since CBs, FS, and SS are the ones covering the receivers and thus need to somewhat match their characteristics.

The worrying thing here might be that we didn't misidentify the bottom 3 (and even 5) positions at all. For us to not misinterpret any of those makes me think there might be something wrong with our model, such as overfitting.

Let's train-test split our data with k-fold cross-validation to see.

```
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#Let's reset our X and y just to make sure it's the same as before:
X = pred_df.drop("position", axis = 1)
y = pred_df['position']
y = y.astype(int) #Wanted?

X_train, X_holdout, Y_train, Y_holdout = train_test_split(X, y, test_size = 0.15)

max_depths = np.arange(1, 15)

kf = KFold(n_splits = 5)
depth_accuracy = np.empty(0)

for i in max_depths:
    current_rfc = RandomForestClassifier(max_depth = i)
    
    total_accuracy = np.empty(0)
    for train_idx, acc_idx in kf.split(X_train):

        split_X_train, split_X_acc = X_train.iloc[train_idx,:], X_train.iloc[acc_idx,:]
        split_Y_train, split_Y_acc = Y_train.iloc[train_idx], Y_train.iloc[acc_idx]

        current_rfc.fit(split_X_train, split_Y_train)
        current_y_preds = current_rfc.predict(split_X_acc)

        total_accuracy = np.append(total_accuracy, (np.sum(current_y_preds == split_Y_acc) / len(split_Y_acc)))

    depth_accuracy = np.append(depth_accuracy, np.mean(total_accuracy))
```








###### 

### Findings
W.I.P.

### Limitations
W.I.P.

### From Here
W.I.P.
