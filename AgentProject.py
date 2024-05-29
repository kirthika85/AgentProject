import os
import shutil
import sqlite3
import pandas as pd
import requests
import streamlit as st

from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic

# Sidebar inputs for environment variables
anthropic_api_key = st.sidebar.text_input("ANTHROPIC_API_KEY", type="password")
tavily_api_key = st.sidebar.text_input("TAVILY_API_KEY", type="password")
langchain_api_key = st.sidebar.text_input("LANGCHAIN_API_KEY", type="password")

# Set environment variables
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Recommended settings
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"

class AssistantAgent:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)

    @tool
    def download_database(self, db_url: str, local_file: str, backup_file: str, overwrite: bool = False) -> None:
        """Download the database from the given URL and create a backup."""
        if overwrite or not os.path.exists(local_file):
            response = requests.get(db_url)
            response.raise_for_status()
            with open(local_file, "wb") as f:
                f.write(response.content)
            shutil.copy(local_file, backup_file)

    @tool
    def convert_to_present_time(self, local_file: str) -> None:
        """Convert flight times to present time."""
        conn = sqlite3.connect(local_file)
        cursor = conn.cursor()

        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
        if "flights" not in tables:
            st.error("The 'flights' table does not exist in the database.")
            return

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

agent = AssistantAgent()

# Download the database
if st.button("Download Database"):
    agent.download_database(db_url="https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite", "travel2.sqlite", "travel2.backup.sqlite")

# Convert flight times to present time
if st.button("Convert to Present Time"):
    agent.convert_to_present_time("travel2.sqlite")

# Display data from the database
conn = None
try:
    conn = sqlite3.connect("travel2.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='flights'")
    table_exists = cursor.fetchone() is not None

    if table_exists:
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
