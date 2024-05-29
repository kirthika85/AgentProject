import os
import pandas as pd
import sqlite3
import requests
import shutil
import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI  

# Set the page configuration at the top
st.set_page_config(page_title="Travel Database App")
st.title("Travel Database Chat")

# Function to set environment variables
def set_env(var: str, value: str):
    if not os.environ.get(var):
        os.environ[var] = value

# Function to download and setup the database
def setup_database():
    # Your existing setup_database function goes here...

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
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    selected_table = st.sidebar.selectbox("Select a table", tables)
    df = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
    conn.close()

    # Display selected table
    st.write(f"### {selected_table} Table")
    st.write(df)

    # Create LangChain tools and agents
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # Chat interface and history
    chat_history = st.sidebar.empty()
    user_input = st.text_input("Enter your query:")
    if user_input:
        try:
            agent_response = agent.run(user_input)
            chat_history.write(f"User: {user_input}")
            chat_history.write(f"Assistant: {agent_response}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
