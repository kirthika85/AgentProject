import os
import shutil
import sqlite3
import pandas as pd
import requests
import streamlit as st
import logging
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the TravelAgent class with tool functions
class TravelAgent:
    def __init__(self):
        self.db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
        self.local_file = "travel2.sqlite"
        self.backup_file = "travel2.backup.sqlite"
        logger.debug("TravelAgent initialized.")

    @tool
    def download_database(self, overwrite: bool = False) -> str:
        """Download the database from the given URL and create a backup."""
        logger.debug(f"Attempting to download database. Overwrite: {overwrite}")
        if overwrite or not os.path.exists(self.local_file):
            logger.debug("Downloading database...")
            response = requests.get(self.db_url)
            response.raise_for_status()
            with open(self.local_file, "wb") as f:
                f.write(response.content)
            shutil.copy(self.local_file, self.backup_file)
            logger.debug("Database downloaded and backup created.")
        else:
            logger.debug("Database already exists. Skipping download.")
        return "Database downloaded and backup created."

    @tool
    def display_table(self, table_name: str) -> pd.DataFrame:
        """Display the contents of the specified table."""
        logger.debug(f"Fetching data from table: {table_name}")
        conn = sqlite3.connect(self.local_file)
        query = f"SELECT * FROM {table_name} LIMIT 10"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

# Create an instance of the agent
travel_agent = TravelAgent()

# Define Streamlit UI
st.title("Travel Data Processing")

# Example: Set environment variables (optional)
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = st.sidebar.text_input(var, type="password")
        st.write(f"Environment variable {var} set.")

_set_env("ANTHROPIC_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("LANGCHAIN_API_KEY")

# Recommended for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"
st.write("Environment variables set.")

# Streamlit buttons to trigger tool functions
if st.button("Download Database"):
    try:
        st.write("Download Database button clicked.")
        result = travel_agent.download_database(overwrite=True)
        st.write(result)
    except Exception as e:
        st.error(f"An error occurred while downloading the database: {e}")
        logger.exception("An error occurred while downloading the database")

# Get list of tables in the database
tables = []
try:
    conn = sqlite3.connect(travel_agent.local_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
except Exception as e:
    st.error(f"Error fetching tables: {e}")
    logger.exception("Error fetching tables")

# Display table selection dropdown if tables are available
if tables:
    selected_table = st.selectbox("Select Table to Display", tables)
    if st.button("Display Table"):
        try:
            st.write(f"Displaying contents of table: {selected_table}")
            result = travel_agent.display_table(table_name=selected_table)
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred while displaying the table: {e}")
            logger.exception("An error occurred while displaying the table")
else:
    st.error("No tables found in the database.")
