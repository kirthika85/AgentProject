import os
import pandas as pd
import sqlite3
import requests
import shutil
import streamlit as st
from langchain.agents import create_openai_functions_agent, AgentExecutor, Tool
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time

# Set the page configuration at the top
st.set_page_config(page_title="Travel Database App")
st.title("Travel Database Chat")

# Function to set environment variables
def set_env(var: str, value: str):
    if not os.environ.get(var):
        os.environ[var] = value

# Function to download and setup the database
def setup_database(overwrite=False):
    db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    local_file = "travel2.sqlite"
    backup_file = "travel2.backup.sqlite"
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

# Function to safely query the database with retry logic
def safe_query_db(query, params=(), retries=5):
    db_path = "travel2.sqlite"
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            return df
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < retries - 1:
                time.sleep(1)  # Wait a bit before retrying
            else:
                raise e

# Sidebar input for API keys
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
tavily_api_key = st.sidebar.text_input('Tavily API Key', type='password')

# Define the setup_database tool
setup_database_tool = Tool(
    name="setup_database",
    func=setup_database,
    description="Download and set up the travel database."
)

tools = [setup_database_tool]

model = ChatOpenAI(
    model='gpt-3.5-turbo-1106',
    temperature=0.7,
    api_key=openai_api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
elif openai_api_key.startswith('sk-') and tavily_api_key:
    set_env('OPENAI_API_KEY', openai_api_key)
    set_env('TAVILY_API_KEY', tavily_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"

    # Create LangChain tools and agents
    try:
        agent = create_openai_functions_agent(llm=model, prompt=prompt, tools=tools)
        agent_executor = AgentExecutor(agent=agent, tools=tools)

        # Initialize chat history in session state if not already
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Enter your query:", key="input")
        if st.button("Query"):
            if user_input:
                st.session_state.chat_history.append({"user": user_input, "assistant": ""})
                try:
                    # Pass the user's query to the agent executor and get the response
                    if 'setup database' in user_input.lower():
                        agent_response = setup_database()
                    else:
                        agent_response = agent_executor.invoke(user_input)
                        st.write(f"Debug: agent_response: {agent_response}")

                    # Add the agent's response to the chat history
                    st.session_state.chat_history[-1]["assistant"] = agent_response
                    st.write(f"User: {user_input}")
                    st.write(f"Assistant: {agent_response}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.write(f"Debug: {str(e)}")

        for chat in st.session_state.chat_history:
            st.write(f"*User:* {chat['user']}")
            st.write(f"*Assistant:* {chat['assistant']}")

        # After setting up the database, load data from the database into a DataFrame
        db_path = setup_database()
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = safe_query_db(query).name.tolist()
        selected_table = st.selectbox("Select a table", tables)
        df = safe_query_db(f"SELECT * FROM {selected_table}")
        st.write(f"### {selected_table} Table")
        st.write(df)
    except Exception as e:
        st.error(f"An error occurred during the setup of the agent or the execution of the query: {e}")
