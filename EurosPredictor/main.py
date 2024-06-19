import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

#Get matches.csv
matches = pd.read_csv("matches.csv", index_col=0)
#Convert Date column into datetime datatype
matches["Date"] = pd.to_datetime(matches["Date"])

#Converting values into numerical data so that machine learning algo can work with it
#Create new column which contains a numeric value for home or away | .cat.codes converts to int
matches["venue_code"] = matches["Venue"].astype("category").cat.codes
#Do same for opponent column
matches["opp_code"] = matches["Opponent"].astype("category").cat.codes
#Create a column for hour the game was played. :.+ is reg ex to remove colon and minutes after hour field in time col
matches["hour"] = matches["Time"].str.replace(":.+", "", regex=True).astype("int")
#Create a column that has the day of the week represented as a number
matches["day_code"] = matches["Day"].astype("category").cat.codes
#Win is 1, Draw/Loss is 0. Creating target for model
matches["target"] = (matches["Result"] == "W").astype("int")

#Using RandomForestClassifier as there are non linearities in the data (e.g. different opp codes are simply arbitary values for each team)
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
predictors = ["venue_code", "opp_code", "hour", "day_code"]


#Group the dataframe by teams
grouped_matches = matches.groupby("Team")
#Computer average performance of each individual team
def get_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = stats
    group = group.dropna(subset=new_cols)
    return group

#These are the stats we are finding averages of
cols = ["GF", "GA","Sh","SoT","Dist","PK","PKatt"]
matches[cols] = matches[cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
new_cols = [f"{c}avg" for c in cols] #Creates new col of averages
numeric_cols = matches.select_dtypes(include=[np.number]).columns

#Applies the rolling averagets function to every team
matches_rolling = matches.groupby("Team").apply(lambda x: get_averages(x, cols, new_cols)).reset_index(drop=False)
matches_rolling = matches_rolling.drop('Team', axis=1)

def make_predictions(data, predictors):
    train = data[data["Date"] < '2021-01-01']
    test = data[data["Date"] > '2021-01-01']
    # .ft trains rf model with predictors to predict target
    rf.fit(train[predictors], train["target"])
    predictions = rf.predict(test[predictors])
    # Produces 60% accuracy first time
    accuracy = accuracy_score(test["target"], predictions)
    # Create DF to display actual results and predictions
    combined = pd.DataFrame(dict(actual=test["target"], predicted=predictions), index=test.index)
    # Get precision of original model
    precision = precision_score(test["target"], predictions)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
combined = combined.merge(matches_rolling[["Date", "Team", "Opponent", "Result"]], left_index=True, right_index=True)

#Make sure the names of teams are the same
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"pt Portugal": "Portugal",
            "fr France": "France",
            "de Germany": "Germany",
            "hu Hungary": "Hungary",
            "eng England": "England",
            "sct Scotland": "Scotland",
              "hr Croatia": "Croatia",
              "nl Netherlands": "Netherlands",
              "dk Denmark": "Denmark",
              "ua Ukraine": "Ukraine",
              "at Austria": "Austria",
              "mk N. Macedonia": "North-Macedonia",
              "cz Czechia": "Czechia",
              "es Spain": "Spain",
              "fi Finland": "Finland",
              "it Italy": "Italy",
              "ru Russia": "Russia",
              "be Belgium": "Belgium",
              "tr Türkiye": "Turkiye",
              "ch Switzerland": "Switzerland",
              "wls Wales": "Wales",
              "pl Poland": "Poland",
              "se Sweden": "Sweden",
              "sk Slovakia": "Slovakia",
              "al Albania": "Albania",
              "ie Rep. of Ireland": "Republic-of-Ireland",
              "nir Northern Ireland": "Northern-Ireland",
              "is Iceland": "Iceland",
              "ro Romania": "Romania",
              "gr Greece": "Greece",
              "lv Latvia": "Latvia",
              "rs Yugoslavia": "Yugoslavia",
              "si Slovenia": "Slovenia",
              "no Norway": "Norway"
}
mapping = MissingDict(**map_values)
combined["new_team"] = combined["Team"].map(mapping)
merged = combined.merge(combined, left_on=["Date", "new_team"], right_on=["Date", "Opponent"])
print(merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts())
