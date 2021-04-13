import pandas as pd
import numpy as np

# Read the winemag-data-65k-v2.json file and just have a peek at the first 10 rows so you know what you are dealing with.
# CP: The layout of this file is in a typical csv format and not a json format, therefore I have coverted this file into a dataframe, and then
# reconverted into json format:
df = pd.read_csv(r"data/winemag-data-65k-v2.csv")
df.to_json(r"data/winemag-data-65k-v2.json", orient="records", lines=True)

# CP: Read data from the json file:
wine_df = pd.read_json("data/winemag-data-65k-v2.json", lines=True)
print("Wine data \n", wine_df.head(n=10))  # View first 10 rows

# Read the countryContinent.csv file and again have a peek at the data.
# CP: Read data from csv file:
cc_df = pd.read_csv(r"data/countryContinent.csv")
print("Country-continent data \n", cc_df.head(n=10))  # View first 10 rows

# The wine dataset includes country but not continent. Enrich the wine dataset with the country continent reference dataset, by joining/adding a
# column of continent onto the wine dataset.

result1 = wine_df.copy()
result1["continent"] = result1["country"].map(cc_df.set_index("country")["continent"])
result1["continent"].fillna(
    "North America", inplace=True
)  # The NaN values appear for USA, this row replaces NaN with "North America"
print("Pull in continent information \n", result1.head(n=10))

# Let’s aim to create a cut of the data, call it result2, with just the columns 'points', 'title', 'description', 'price', 'variety', 'province',
# 'country', 'continent'
result2 = result1[
    [
        "points",
        "title",
        "description",
        "price",
        "variety",
        "province",
        "country",
        "continent",
    ]
]
result2.columns = (
    result2.columns.str.strip()
)  # Remove leading and trailing whitespace, which can effect grouping in the later tasks
print("New dataframe with required columns \n", result2.head(n=10))

# In this dataset, result2, filter for/find the rows that include the word ‘chocolate’ (you should find 3825 rows)
chocolate = result2[
    result2["title"].str.contains("chocolate")
    | result2["description"].str.contains("chocolate")
]
print("Total rows of data with a reference to 'chocolate': ", chocolate.shape[0])
print("Chocolatey wine \n", chocolate.head(n=10))

# CP: The total rows with a reference to chocolate appears to be higher than 3,825 in my dataframe, I saved dataframe to a csv and had a look
# through and all the descriptions included appeared to include 'chocolate'.
chocolate.to_csv("data/chocolate.csv")

# Group the filtered data by country, sort in descending order and show the top 5 countries that produce ‘chocolatey’ wine!
top_countries = chocolate[
    ["country", "variety"]
]  # Create new dataframe with just required columns
top_countries = (
    top_countries.groupby("country")
    .count()
    .sort_values("variety", ascending=False)
    .head(n=5)
)  # Group by country, sort based on highest number of varieties and keep top 5
print("Top 5 countries to product chocolatey wine: \n", top_countries)

# Finally filter the result2 dataset for rows where the country is ‘Italy’, and then pivot that so that you show a summed breakdown of province
# down the rows and variety across the columns.
df = result2[
    ["country", "variety", "province"]
]  # Create new dataframe with required columns
italy_df = df[df["country"].str.contains("Italy", na=False)]
italy_df = italy_df.groupby(
    ["variety", "province"]
).count()  # Group by province and variety
province_variety_df = pd.pivot_table(
    italy_df, values="country", index="province", columns="variety"
)  # Pivot dataframe
province_variety_df.fillna("", inplace=True)
print(
    "Pivoted data for Italy, based on province and variety of wine: \n",
    province_variety_df.head(10),
)
