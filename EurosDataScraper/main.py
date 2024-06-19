import time
import pandas as pd
from bs4 import BeautifulSoup
import requests
from io import StringIO

urls = ["https://fbref.com/en/comps/676/2021/2021-European-Championship-Stats",
"https://fbref.com/en/comps/676/2016/2016-European-Championship-Stats",
"https://fbref.com/en/comps/676/2012/2012-European-Championship-Stats",
"https://fbref.com/en/comps/676/2008/2008-European-Championship-Stats",
"https://fbref.com/en/comps/676/2004/2004-European-Championship-Stats",
"https://fbref.com/en/comps/676/2000/2000-European-Championship-Stats"]

all_games = []

for url in urls:
    # Make request using ScrapingBee premium proxy
    euros_page = requests.get(url)

    soup = BeautifulSoup(euros_page.text, 'html.parser')
    squad_links = set()  # Use a set to store unique links
    stats_tables = soup.find_all('table', class_='stats_table')
    for table in stats_tables:
        all_a_tags = table.find_all('a', href=True)
        squad_links.update({f"https://fbref.com{a['href']}" for a in all_a_tags if a['href'].startswith('/en/squads/')})

    for link in squad_links:
        try:
            data = requests.get(link)
            print(f"HTML content received for {link}")
            matches = pd.read_html(StringIO(data.text), match="Scores & Fixtures")

            if not matches:
                print(f"No tables found for {link}")
                continue

            # Scrape the data from the Shooting Table and put it into a panda dataframe
            soup = BeautifulSoup(data.text, 'html.parser')
            links = soup.findAll('a')
            links = [l.get("href") for l in links]
            links = [l for l in links if l and 'all_comps/shooting/' in l]  # Getting the link for shooting table page from original page
            data = requests.get(f"https://fbref.com{links[0]}")
            shooting_data = pd.read_html(StringIO(data.text), match="Shooting")[0]
            # Currently there is another top header which we want to remove. Following code removes it
            shooting_data.columns = shooting_data.columns.droplevel()

            # Merge Match and Shooting data into the same dataframe picking out those specific stats from shooting data
            try:
                squad_data = matches[0].merge(shooting_data[["Date", "Sh", "SoT", "Dist", "PK", "PKatt"]].fillna(0), on="Date")
            except ValueError:
                print(f"Merge failed for {link}")
                continue
            squad_data = squad_data[squad_data["Comp"] == "UEFA Euro"]
            squad_name = link.split("/")[-1].replace("-Stats","").replace("-Men","")
            squad_data["Team"] = squad_name
            all_games.append(squad_data)

        except Exception as e:
            print(f"Error for {link}: {e}")

        time.sleep(60)
    time.sleep(60)

if all_games:
    main_df = pd.concat(all_games)
    main_df.to_csv("matches.csv")
else:
    print("No data to save")