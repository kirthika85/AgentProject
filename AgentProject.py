import streamlit as st
import os
import shutil
import sqlite3
import pandas as pd
import requests

# Function to download and prepare the database
def download_and_prepare_database(db_url, local_file, backup_file, overwrite=False):
    if overwrite or not os.path.exists(local_file):
        response = requests.get(db_url)
        response.raise_for_status()  # Ensure the request was successful
        with open(local_file, "wb") as f:
            f.write(response.content)
        # Backup - we will use this to "reset" our DB in each section
        shutil.copy(local_file, backup_file)

# Function to convert the flight times to present time
def convert_to_present_time(local_file):
    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    if "flights" in tables:
        tdf = {}
        for t in tables:
            tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

        example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)).max()
        current_time = pd.to_datetime("now").tz_localize(example_time.tz)
        time_diff = current_time - example_time

        tdf["bookings"]["book_date"] = pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True) + time_diff

        datetime_columns = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
        for column in datetime_columns:
            tdf["flights"][column] = pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff

        for table_name, df in tdf.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        conn.commit()
        conn.close()
        st.success("Database downloaded and prepared.")
    else:
        st.error("The 'flights' table does not exist in the database.")

# Download and prepare the database
db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
backup_file = "travel2.backup.sqlite"
overwrite = False
download_and_prepare_database(db_url, local_file, backup_file, overwrite)

# Convert the flight times to present time
convert_to_present_time(local_file)

# Connect to the database
conn = sqlite3.connect(local_file)

# Check if the 'flights' table exists
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='flights'")
table_exists = cursor.fetchone() is not None

if table_exists:
    # Query the 'flights' table and display sample data
    query = "SELECT * FROM flights LIMIT 5"
    df = pd.read_sql_query(query, conn)
    st.write("Sample Data from Flights Table:")
    st.write(df)
else:
    st.error("The 'flights' table does not exist in the database.")

# Close the database connection
conn.close()
