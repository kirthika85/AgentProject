import os
import shutil
import sqlite3
import streamlit as st
import pandas as pd
import requests

# Streamlit app title
st.set_page_config(page_title="Travel Database App")

# Sidebar for user inputs
with st.sidebar:
    st.title("Options")
    overwrite = st.checkbox("Overwrite Local Database")

# Download and setup the database
db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
backup_file = "travel2.backup.sqlite"

if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_file, "wb") as f:
        f.write(response.content)
    # Backup - we will use this to "reset" our DB in each section
    shutil.copy(local_file, backup_file)

# Convert the flights to present time
conn = sqlite3.connect(local_file)
cursor = conn.cursor()

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';", conn
).name.tolist()
tdf = {}
for t in tables:
    tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

example_time = pd.to_datetime(
    tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
).max()
current_time = pd.to_datetime("now").tz_localize(example_time.tz)
time_diff = current_time - example_time

tdf["bookings"]["book_date"] = (
    pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
    + time_diff
)

datetime_columns = [
    "scheduled_departure",
    "scheduled_arrival",
    "actual_departure",
    "actual_arrival",
]
for column in datetime_columns:
    tdf["flights"][column] = (
        pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
    )

for table_name, df in tdf.items():
    df.to_sql(table_name, conn, if_exists="replace", index=False)
del df
del tdf
conn.commit()
conn.close()

db = local_file  # We'll be using this local file as our DB in this tutorial

# Display the tables in the database
st.title("Travel Database")
conn = sqlite3.connect(db)
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
st.write(f"Tables in the database: {', '.join(tables)}")

# Allow the user to select a table and display its contents
selected_table = st.selectbox("Select a table", tables)
if selected_table:
    df = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
    st.write(f"Contents of {selected_table} table:")
    st.dataframe(df)

conn.close()
