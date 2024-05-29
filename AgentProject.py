import streamlit as st
import os
import shutil
import sqlite3
import pandas as pd
import requests

# Title of the Streamlit app
st.title("Travel Database Adjustment App")

# Database URL and local file paths
db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
backup_file = "travel2.backup.sqlite"

# Checkbox to overwrite existing database file
overwrite = st.checkbox("Overwrite existing database file?", value=False)

# Download the database file if it does not exist or if overwrite is checked
if overwrite or not os.path.exists(local_file):
    st.write("Downloading database...")
    response = requests.get(db_url)
    response.raise_for_status()
    with open(local_file, "wb") as f:
        f.write(response.content)
    shutil.copy(local_file, backup_file)
    st.write("Database downloaded and backup created.")

# Connect to the SQLite database
conn = sqlite3.connect(local_file)
cursor = conn.cursor()

# Retrieve the table names from the database
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
st.write(f"Tables found in the database: {tables}")

# Dictionary to store the dataframes for each table
tdf = {}
for t in tables:
    tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

# Adjust the time fields in the flights and bookings tables
example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)).max()
current_time = pd.to_datetime("now").tz_localize(example_time.tz)
time_diff = current_time - example_time

tdf["bookings"]["book_date"] = pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True) + time_diff

datetime_columns = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
for column in datetime_columns:
    tdf["flights"][column] = pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff

# Save the adjusted data back to the database
for table_name, df in tdf.items():
    df.to_sql(table_name, conn, if_exists="replace", index=False)

conn.commit()
conn.close()

# Display the updated tables
st.write("Adjusted Tables:")
for table_name, df in tdf.items():
    st.write(f"**{table_name}**")
    st.dataframe(df)

st.success("Database adjustment complete!")
