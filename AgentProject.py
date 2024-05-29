import os
import pandas as pd
import sqlite3
import requests
import shutil
import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI  # Make sure this import matches the actual LangChain module structure

# Set the page configuration at the top
st.set_page_config(page_title="Travel Database App")
st.title("Travel Database Chat")

# Function to set environment variables
def set_env(var: str, value: str):
    if not os.environ.get(var):
        os.environ[var] = value

# Function to download and setup the database
def setup_database():
    db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    local_file = "travel2.sqlite"
    backup_file = "travel2.backup.sqlite"
    overwrite = False
    if overwrite or not os.path.exists(local_file):
        response = requests.get(db_url)
        response.raise_for_status()
        with open(local_file, "wb") as f:
            f.write(response.content)
        shutil.copy(local_file, backup_file)

    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)
    example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time
    datetime_columns = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
    for column in datetime_columns:
        tdf["flights"][column] = (pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff)
    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    return local_file

# Sidebar input for API keys
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
tavily_api_key = st.sidebar.text_input('Tavily API Key', type='password')

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
elif openai_api_key.startswith('sk-') and tavily_api_key:
    set_env('OPENAI_API_KEY', openai_api_key)
    set_env('TAVILY_API_KEY', tavily_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"

    # Setup database
    db_path = setup_database()

    # Load data from the database into a DataFrame
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM flights", conn)
    conn.close()

    # Create LangChain tools and agents
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, agent_type="tool-calling", verbose=True)

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
