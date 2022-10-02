# Using-Machine-Learning-to-Predict-English-Premier-League-Fixtures-with-Scraped-Data
<p align="justify"> Using web scraping and machine learning classification models to predict to outcome of fixtures in the English Premier League. </p> 
<br/>

# Overview  

<p align="justify"> This project involved using machine learning to predict the outcome of football matches in the English Premier League. The data that was used was obtained through web scraping using Python. Random-Forest and XGBoost classification models were backtested and optimized on the 2021/2022 season, using data as far back as 2018/19 and then finally applied to the current 2022/23 season. As the season goes on more match results can be added to the training set to improve the accuracy of the model.</p>  
<br/>

# Scraping the Data  

Data was scraped from two different sources using the ***Requests*** library in Python and the HTML was parsed using ***BeautifulSoup***. The data for Premier League fixtures and team statistics was scraped from [FBref.com](https://fbref.com/en/comps/9/Premier-League-Stats), while another website called [TransferMarkt](https://www.transfermarkt.co.uk/) was used to scrape data on teams’ transfer expenses for each season. Transfer data was not available on FBref and it was desirable to see if this is also a significant predictor of match outcomes.  

<p align="justify"> The following code was used to find the fixtures table for each season. The only input that changes is the curr_season value and the rest of the code runs dynamically. 
 </p>
<br/>

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning
from lxml import html
import time
import difflib

# Turn off SSL Warning
requests.packages.urllib3.disable_warnings()

# Define URL & Header
curr_season = '2022-2023'
transfer_season = curr_season[0:4] 
url = 'https://fbref.com/en/comps/9/{0}/{1}-Premier-League-Stats'.format(curr_season, curr_season)
header_ = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36", "Accept-Language": "en-US,en;q=0.9"}

# Make Request to URL
req = requests.get(url, headers = header_, verify = False)

# Parse HTML With BeautifulSoup
dat = BeautifulSoup(req.text)
league_table = dat.select('table.stats_table')[0]

# Find Links in HTML Text
team_links = league_table.find_all('a')
team_links = [link.get('href') for link in team_links]
team_links = [link for link in team_links if '/squads/' in link]

# Complete Each Link By Adding Prefix
pre = 'https://fbref.com'
team_links = [pre + link for link in team_links]
```

<p align="justify">The table for each season looks like this: </p>
<br/>
# IMAGE

<p align="justify">The links for each team can be taken from this fixtures table so that we can later loop through each team link to get the team-specific stats. The next section of code scrapes data from TransferMarkt, parses it, cleans it and stores it in a dataframe for later use. </p>
<br/>

```python
# Scrape Data from Transfermarkt.co.uk
urlT = 'https://www.transfermarkt.co.uk/premier-league/transfers/wettbewerb/GB1/plus/?saison_id={}&s_w=&leihe=0&intern=0&intern=1'.format(transfer_season)
trans_req = requests.get(urlT, headers = header_, verify = False)

# Parse HTML & Find Span Classes Containing Expenditures
transers = BeautifulSoup(trans_req.text)
transfer_data = transers.find_all("span", {"class":"transfer-einnahmen-ausgaben redtext"})
title_tags = transers.find_all("h2")[0:20]
tnames = [str(tag).split('"')[-2] for tag in title_tags]
expenditures = []

# Clean-Up Expenditure Data
for x in transfer_data:
    expenditures.append(x.text[17:22])
for i in range(0, 20):
    if expenditures[i] == '\t\t\t\t\t':
        expenditures[i] = 0
    if 'm' in str(expenditures[i]):
        expenditures[i] = expenditures[i].replace("m", "")
    if 'Th' in str(expenditures[i]):
        expenditures[i] = expenditures[i].replace("Th", "")
        expenditures[i] = float(expenditures[i])/1000
    expenditures[i] = float(expenditures[i])

# Create Dataframe for Each Team's Transfer Expenses
team_transfers = pd.DataFrame(data = tnames, index = range(0,20), columns = ["Team"])
team_transfers["Expenditures"] = expenditures

# Sort & Match Team Names
tn = []
names = [u.split("/")[-1].replace("-Stats", "").replace("-", " ") for u in team_links]

for i in range(0, 20):
    check_word = team_transfers["Team"][i]
    n = 1
    cutoff = 0.8
    close_match = difflib.get_close_matches(check_word, names, n, cutoff)
    team_transfers["Team"].iat[i] = close_match[0]
```

<p align="justify"> The webpage for each season contains tables for each team, their transfers for that season and the amount spent and received. The expenditure for each team was isolated in the script, an example of which can be seen below. </p>
<br/>

# IMAGE

<p align="justify"> Finally, we loop through each team on FBref to collect the bulk of the data. Here’s an example of what the Scores & Fixtures table looks like for each team: </p>
<br/>

# IMAGE

<p align="justify"> This scores and fixtures table is scraped to give us the team’s fixtures for the season, the results of matches that have already been played and some other stats such as goals scored, goals against, possession and more. Above the Scores & Fixtures table there is also more links to other stats tables such as Shooting, Goalkeeping, Passing etc. These links were also scraped with a loop that was written to pass through each link, scrape the data for each table and combined these in a dataframe for each team.  </p>
<br/>

<p align="justify"> Given the large number of features in the raw data it was necessary to filter the data as many of these variables were of no use. This was done manually by selecting 24 features that were deemed to have the potential to have some impact on match results. Some examples include shots, shots on target, passes completed, possession, types of passes, defensive actions, penalties and free kicks. The full list can be seen in the code below or in the Excel files. </p>
<br/>

<p align="justify"> Finally, the Scores & Fixtures table for each team was merged with the combined stats tables and then concatenated into one dataframe containing data for all teams in that season, which was saved to CSV.  </p>
<br/>

```python
all_stats = []

# Loop Through Each Team URL
for url in team_links:
    team_req = requests.get(url, headers = header_, verify = False)
    team_name = url.split("/")[-1].replace("-Stats", "").replace("-", " ")
    for i in range(0, len(team_transfers)):
        if team_name == team_transfers["Team"][i]:
            team_exp = team_transfers["Expenditures"][i]
    time.sleep(1)

    # Use Pandas to Read the Scores & Fixtures Table for Each Team
    try:
        scores_fixtures = pd.read_html(team_req.text, match = "Scores & Fixtures")[0]
    except:
        scores_fixtures = pd.read_html(team_req.text, match = "Scores & Fixtures ")[0]
    scores_fixtures = scores_fixtures[scores_fixtures["Comp"] == "Premier League"]
    scores_fixtures.insert(loc = 9, column = "Team", value = team_name)
    scores_fixtures.insert(loc = 19, column = "Transfers - Home", value = team_exp)
    scores_fixtures = scores_fixtures.drop(columns = ["Comp", "xG", "xGA", "Attendance", "Captain", "Referee", "Match Report", "Notes"])

    # Parse HTML With BeautifulSoup & Find Links
    data = BeautifulSoup(team_req.text)
    squad_links = data.find_all('a')
    squad_links = [link.get('href') for link in squad_links]
    squad_links = [link for link in squad_links if link and 'matchlogs/all_comps/' in link]
    squad_links = list(dict.fromkeys(squad_links))
    squad_links = squad_links[1:]

    # Loop Through Extra Tables
    stats_list = []
    for link in squad_links:
        table_page = requests.get(pre + link, headers = header_, verify = False)
        table = pd.read_html(table_page.text)[0]
        table.columns = table.columns.droplevel()
        table = table[table["Comp"] == "Premier League"]
        if "/shooting/" not in link:
            table = table.drop(columns = "Date")
        if "/shooting/" in link:
            table = table.rename(columns={'Sh':'Shots', 'FK':'FK Shots'})
        if "/passing_types/" in link:
            table = table.rename(columns={'Live':'Live Passes', 'Cmp':'Passes Completed', 'Press':'Passes Under Pressure'})
        stats_list.append(table)
        time.sleep(1)

    # Combine Tables for Current Team
    combined = pd.concat(stats_list, axis = 1)
    combined = combined[["Date","Shots", "SoT%", "G/SoT", "Dist", "FK Shots", "PK", "SoTA", "Save%", "CS", "PKA", "#OPA", "Live Passes", "Passes Completed", "Passes Under Pressure", 
                        "Ground", "High", "SCA", "Press", "Succ%", "CrdR", "CrdY", "Fls", "OG", "Won%"]]
    team_stats = scores_fixtures.merge(combined, on = "Date")
    all_stats.append(team_stats)

# Create Dataframe for All Teams and Save to CSV
df = pd.concat(all_stats)
df = df.rename(columns = {'Poss':'Possession', 'SoTA':'SoT Against', 'Dist':'Avg. Distance of Shots', 'CS':'Clean Sheet', 'PKA':'PK Allowed', '#OPA':'# Defensive Actions OPA', 'Ground':'Ground Passes',
                        'High':'High Passes', 'SCA':'Shot-Creating Actions', 'Press':'Presses (Applied)','Succ%':'Successful Dribbles', 'CrdR':'Red Cards', 'CrdY':'Yellow Cards',
                        'Fls':'Fouls', 'OG':'Own Goals', 'Won%':'Aerial Battles Won (%)'})
df.to_csv("{}.csv".format(curr_season))
```  
<br/>

# Data Cleaning
<p align="justify"> After all the data was scraped the next step was to check the completeness of the data. The dataframes were checked for missing values using the code below, which prints out the dataframe (season) and columns that are missing some data. </p>
<br/>

```python
# Create Function to Get Variable Names
def get_name(df_):
    name = [x for x in globals() if globals()[x] is df_][0]
    return name
    
# Check Datasets for Missing Values
for df in dfx:
    for col in df.columns:
        if df[col].isnull().values.any():
            print(get_name(df), col)
```
df18_19 SoT%
df18_19 G/SoT
df18_19 Avg. Distance of Shots
df18_19 Save%
df19_20 G/SoT
df19_20 Save%
df20_21 SoT%
df20_21 G/SoT
df20_21 Avg. Distance of Shots
df20_21 Save%
df21_22 G/SoT
df21_22 Save%
df22_23 G/SoT
df22_23 Save%

<p align="justify"> There was indeed some missing data but it can be deduced from the listed variables that contain missing entries that this was for technical reasons rather than data not being available. For example, in some matches a team may not take any shots and therefore could not have a value for Shot on Target (SoT) or Goals per Shot on Target (G/SoT). If this was the case then the values were simply imputed with a zero. The check was done again after imputation and found no missing values. </p>
<br/>

```python
# Impute Missing Values
for df in dfx:
    for i in range(0, len(df)):
        if pd.isnull(df["G/SoT"][i]):
            if df["SoT%"][i] == 0:
                df["G/SoT"].iat[i] = 0
            else:
                df["G/SoT"].iat[i] = df["GF"][i]/(df["Shots"][i] * df["SoT%"][i])
        if pd.isnull(df["Save%"][i]):
            if df["SoT Against"][i] == 0:
                df["Save%"].iat[i] = 0
        if df["Save%"][i] < 0:
            df["Save%"].iat[i] = 0
        if df["Shots"][i] == 0:
            df["SoT%"].iat[i] = 0
            df["G/SoT"].iat[i] = 0
            df["Avg. Distance of Shots"].iat[i] = 0

# Check Again For NaN After Imputation
for df in dfx:
    for col in df.columns:
        if df[col].isnull().values.any():
            print(get_name(df), col)
```
<br/>

# Feature Engineering

<p align="justify"> The names in the lists of opponent teams were slightly different from the list of home teams in some cases and needed to be mapped by using a dictionary. This was the case for certain clubs with names that could be spelled differently, for example Newcastle Utd instead of Newcastle United or Wolves instead of Wolverhampton Wanderers.  </p>
<br/>

<p align="justify"> The following code performed this mapping and also created two new variables, one that indicated the season and another for the transfer expenses of the away team. Therefore we can have one column for the home team transfer expenses and another for the away team’s transfer expenses.  </p>
<br/>

```python
# Create Dictionary for Inconsistent Names
name_dict = {'Brighton':'Brighton and Hove Albion', 'Manchester Utd':'Manchester United', 'Newcastle Utd':'Newcastle United', 'Sheffield Utd':'Sheffield United', 
            "Nott'ham Forest":'Nottingham Forest', 'Tottenham':'Tottenham Hotspur', 'West Brom':'West Bromwich Albion', 'West Ham':'West Ham United', 'Wolves':'Wolverhampton Wanderers'}

# Replace Inconsistent Names in 'Opponent' & Add New Columns
for df in dfx:
    teams = list(dict.fromkeys(df["Team"]))
    exps = list(dict.fromkeys(df["Transfers - Home"]))
    df.insert(loc = 13, column = "Transfers - Away", value = ' ')
    season = get_name(df).replace('df', '')
    df["Season"] = season
    for i in range(0, len(df)):
        for key, value in name_dict.items():
            if df["Opponent"][i] == key:
                df["Opponent"].iat[i] = value
        for j in range(0, len(teams)):
            if df["Opponent"][i] == teams[j]:
                df["Transfers - Away"].iat[i] = exps[j]
```
<br/>

## Creating Moving Averages

<p align="justify"> The data cannot be used for machine learning in its current form because it only tells us how a team performed during a given match. If we want to be able to predict the outcome of a match we will not have this data available to us until after the match has ended. Therefore we need to create moving averages for each relevant statistic, which capture the past performance of the team over the previous games.  </p>
<br/>

<p align="justify"> For example, if we take the shots on target metric, when predicting the outcome for an upcoming game it makes sense to look at how many shots on target the team managed in its three previous games. This type of moving average metric helps to give us a measure of their ‘form’ in recent matches.  </p>
<br/>

<p align="justify"> A function was defined in order to take each statistic and calculate its moving average over the three previous games. The function takes a dataframe that is grouped by team, calculates the moving average and creates new columns containing this data, then finally removes the first three rows for each team. This is done because the moving average cannot be calculated for the first three rows given that it needs three values to do the calculation. Therefore the 4th match of the season for each team becomes our new starting point for each dataset.  </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>

<p align="justify"> </p>
<br/>
