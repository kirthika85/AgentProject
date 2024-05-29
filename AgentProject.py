import getpass
import os
import streamlit as st
import pandas as pd
import sqlite3
import requests
import shutil
from langchain.agents import create_pandas_data_augmented_agent
from langchain.llms import AnthropicAI
from langchain.tools import SQLDatabaseTool

# Function to set environment variables
def set_env(var: str, default: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var} (default: {default}): ") or default

# Set environment variables
set_env("ANTHROPIC_API_KEY", "")
set_env("TAVILY_API_KEY", "")
set_env("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"

# Function to download and setup the database
def setup_database():
    db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    local_file = "travel2.sqlite"
    backup_file = "travel2.backup.sqlite"
    overwrite = False
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
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)
    example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time
    # Perform time conversion and write back to database
    # (Omitted for brevity, you can include it as needed)
    conn.commit()
    conn.close()
    return local_file

# Streamlit app
def main():
    st.set_page_config(page_title="Travel Database App")
    st.title("Travel Database Chat")

    # Setup database
    db_path = setup_database()

    # Create LangChain tools and agents
    db_tool = SQLDatabaseTool(db_path=db_path)
    llm = AnthropicAI(model="claude-v1.3-100k")
    agent = create_pandas_data_augmented_agent(llm, db_tool, verbose=True)

    # Chat interface
    chat_history = []
    user_input = st.text_input("Enter your query:", key="input")
    if user_input:
        try:
            agent_response = agent.run(user_input)
            chat_history.append({"user": user_input, "assistant": agent_response})
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    for chat in chat_history:
        st.write(f"**User:** {chat['user']}")
        st.write(f"**Assistant:** {chat['assistant']}")

if __name__ == "__main__":
    main()
