import os
import shutil
import sqlite3
import pandas as pd
import requests
import streamlit as st
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic

# Define the TravelAgent class with debugging messages
class TravelAgent:
    def __init__(self):
        self.db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
        self.local_file = "travel2.sqlite"
        self.backup_file = "travel2.backup.sqlite"
        st.write("TravelAgent initialized.")

    @tool
    def download_database(self, overwrite: bool = False) -> str:
        """Download the database from the given URL and create a backup."""
        st.write(f"Attempting to download database. Overwrite: {overwrite}")
        if overwrite or not os.path.exists(self.local_file):
            st.write("Downloading database...")
            response = requests.get(self.db_url)
            response.raise_for_status()
            with open(self.local_file, "wb") as f:
                f.write(response.content)
            shutil.copy(self.local_file, self.backup_file)
            st.write("Database downloaded and backup created.")
        else:
            st.write("Database already exists. Skipping download.")
        return "Database downloaded and backup created."

    @tool
    def convert_to_present_time(self) -> str:
        """Convert flight times to present time."""
        st.write("Converting flight times to present time...")
        conn = sqlite3.connect(self.local_file)
        cursor = conn.cursor()

        st.write("Fetching table names...")
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
        st.write(f"Tables found: {tables}")
        
        tdf = {}
        for t in tables:
            st.write(f"Reading table: {t}")
            tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

        st.write("Calculating time difference...")
        example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)).max()
        current_time = pd.to_datetime("now").tz_localize(example_time.tz)
        time_diff = current_time - example_time
        st.write(f"Time difference: {time_diff}")

        st.write("Updating booking dates...")
        tdf["bookings"]["book_date"] = (
            pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True) + time_diff
        )

        datetime_columns = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
        for column in datetime_columns:
            st.write(f"Updating column: {column}")
            tdf["flights"][column] = (
                pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
            )

        st.write("Writing updated data back to database...")
        for table_name, df in tdf.items():
            st.write(f"Writing table: {table_name}")
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        conn.commit()
        conn.close()
        st.write("Flight times converted to present time.")
        return "Flight times converted to present time."

# Create an instance of the agent
travel_agent = TravelAgent()

# Streamlit UI
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = st.sidebar.text_input(var, type="password")
        st.write(f"Environment variable {var} set.")

_set_env("ANTHROPIC_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("LANGCHAIN_API_KEY")

# Recommended
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"
st.write("Environment variables set.")

# Define the Streamlit app
st.title("Travel Data Processing")

# Define Streamlit buttons to trigger tool functions
if st.button("Download Database"):
    st.write("Download Database button clicked.")
    result = travel_agent.download_database()
    st.write(result)

if st.button("Convert to Present Time"):
    st.write("Convert to Present Time button clicked.")
    result = travel_agent.convert_to_present_time()
    st.write(result)

# Display data from the database
conn = None
try:
    st.write("Connecting to the database...")
    conn = sqlite3.connect("travel2.sqlite")
    cursor = conn.cursor()
    st.write("Checking if 'flights' table exists...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='flights'")
    table_exists = cursor.fetchone() is not None

    if table_exists:
        st.write("'flights' table exists. Fetching data...")
        query = "SELECT * FROM flights LIMIT 5"
        df = pd.read_sql_query(query, conn)
        st.write("Sample Data from Flights Table:")
        st.write(df)
    else:
        st.error("The 'flights' table does not exist in the database.")
except Exception as e:
    st.error(f"Error: {e}")
finally:
    if conn:
        conn.close()
        st.write("Database connection closed.")
