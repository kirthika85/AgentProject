import os
import shutil
import sqlite3
import pandas as pd
import requests
import streamlit as st

from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain.tools import tool, agent
from langchain_anthropic import ChatAnthropic

# Define the agent class
class TravelAgent:
    def __init__(self):
        self.db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
        self.local_file = "travel2.sqlite"
        self.backup_file = "travel2.backup.sqlite"

    @tool
    def download_database(self, overwrite: bool = False) -> None:
        """Download the database from the given URL and create a backup."""
        if overwrite or not os.path.exists(self.local_file):
            response = requests.get(self.db_url)
            response.raise_for_status()
            with open(self.local_file, "wb") as f:
                f.write(response.content)
            shutil.copy(self.local_file, self.backup_file)

    @tool
    def convert_to_present_time(self) -> None:
        """Convert flight times to present time."""
        conn = sqlite3.connect(self.local_file)
        cursor = conn.cursor()

        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
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

        conn.commit()
        conn.close()

    @agent
    def run_travel_agent(self):
        chat_history = []
        self.download_database(overwrite=True)
        chat_history.append("Database downloaded.")
        self.convert_to_present_time()
        chat_history.append("Flight times converted to present time.")
        return chat_history

# Create an instance of the agent
travel_agent = TravelAgent()

# Run the travel agent
if st.button("Run Travel Agent"):
    chat_history = travel_agent.run_travel_agent()
    st.write("Chat History:")
    for entry in chat_history:
        st.write(entry)
