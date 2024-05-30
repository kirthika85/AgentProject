import os
import shutil
import sqlite3
import requests
import pandas as pd
import streamlit as st
from datetime import date, datetime
from typing import Optional
import pytz
import openai
from langchain.agents import create_openai_functions_agent, AgentExecutor, Tool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ensure_config
from langchain_core.tools import tool
import numpy as np
import re

# Set page configuration
st.set_page_config(page_title="Travel Database App")
st.title("Travel Database Chat")

# Function to set environment variables
def set_env(var: str, value: str):
    if not os.environ.get(var):
        os.environ[var] = value

# Sidebar inputs for API keys
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
anthropic_api_key = st.sidebar.text_input('Anthropic API Key', type='password')
tavily_api_key = st.sidebar.text_input('Tavily API Key', type='password')
langchain_api_key = st.sidebar.text_input('Langchain API Key', type='password')

# Set environment variables
if openai_api_key:
    set_env('OPENAI_API_KEY', openai_api_key)
if anthropic_api_key:
    set_env('ANTHROPIC_API_KEY', anthropic_api_key)
if tavily_api_key:
    set_env('TAVILY_API_KEY', tavily_api_key)
if langchain_api_key:
    set_env('LANGCHAIN_API_KEY', langchain_api_key)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"

# Database setup
db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
backup_file = "travel2.backup.sqlite"

if not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()
    with open(local_file, "wb") as f:
        f.write(response.content)
    shutil.copy(local_file, backup_file)

# Check if the local database file exists
if os.path.exists(local_file):
    st.write(f"Database file {local_file} found.")
else:
    st.write(f"Database file {local_file} not found. Downloading...")
    response = requests.get(db_url)
    response.raise_for_status()
    with open(local_file, "wb") as f:
        f.write(response.content)
    st.write("Download complete.")

try:
    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    st.write("Tables in the database:", tables)
except Exception as e:
    st.error(f"An error occurred while connecting to the database: {e}")
    st.stop()

# Convert the flights to present time
try:
    tdf = {t: pd.read_sql(f"SELECT * from {t}", conn) for t in tables}
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
except Exception as e:
    st.error(f"An error occurred while processing the database: {e}")
    st.stop()
finally:
    conn.close()

db = local_file  # We'll be using this local file as our DB in this tutorial

# FAQ document setup
response = requests.get("https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md")
response.raise_for_status()
faq_text = response.text
docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(model="text-embedding-ada-002", input=[doc["page_content"] for doc in docs])
        #vectors = [emb["embedding"] for emb in embeddings["data"]]
        vectors = [emb.embedding for emb in embeddings.data]
        print("Vectors:", vectors)
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(model="text-embedding-ada-002", input=[query])
        scores = np.array(embed["data"][0]["embedding"]) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [{**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted]

retriever = VectorStoreRetriever.from_docs(docs, openai.Client(api_key=openai_api_key))

@tool
def lookup_policy(query: str) -> str:
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])

@tool
def fetch_user_flight_information() -> list[dict]:
    config = ensure_config()
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]
    cursor.close()
    conn.close()
    return results

@tool
def search_flights(departure_airport: Optional[str] = None, arrival_airport: Optional[str] = None,
                   start_time: Optional[date | datetime] = None, end_time: Optional[date | datetime] = None, limit: int = 20) -> list[dict]:
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []
    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)
    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)
    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)
    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)
    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]
    cursor.close()
    conn.close()
    return results

@tool
def update_ticket_to_new_flight(ticket_no: str, new_flight_id: int) -> str:
    config = ensure_config()
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?", (new_flight_id,))
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "Invalid new flight ID provided."
    column_names = [column[0] for column in cursor.description]
    new_flight = dict(zip(column_names, new_flight))
    cursor.execute("SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."
    cursor.execute("SELECT flight_id FROM tickets WHERE ticket_no = ? AND passenger_id = ?", (ticket_no, passenger_id))
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"
    cursor.execute("UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?", (new_flight_id, ticket_no))
    conn.commit()
    cursor.close()
    conn.close()
    return "Ticket successfully updated to new flight."

@tool
def cancel_ticket(ticket_no: str) -> str:
    config = ensure_config()
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."
    cursor.execute("SELECT flight_id FROM tickets WHERE ticket_no = ? AND passenger_id = ?", (ticket_no, passenger_id))
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"
    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()
    cursor.close()
    conn.close()
    return "Ticket successfully cancelled."

# Define tools for the agent
tools = [
    Tool(name="lookup_policy", func=lookup_policy),
    Tool(name="fetch_user_flight_information", func=fetch_user_flight_information),
    Tool(name="search_flights", func=search_flights),
    Tool(name="update_ticket_to_new_flight", func=update_ticket_to_new_flight),
    Tool(name="cancel_ticket", func=cancel_ticket),
]

# Create the OpenAI functions agent
llm = OpenAI(temperature=0, api_key=openai_api_key)
agent = create_openai_functions_agent(llm, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to handle chatbot interaction
def handle_interaction(user_input):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = agent_executor({"tool": "lookup_policy", "args": {"query": user_input}})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Streamlit interface
st.header("Travel Database Management Chatbot")
st.subheader("Chat with the Bot")

user_input = st.text_input("You:")
if st.button("Send"):
    handle_interaction(user_input)

# Display chat history
st.subheader("Chat History")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.text(f"User: {message['content']}")
    else:
        st.text(f"Bot: {message['content']}")

st.subheader("User Flight Information")
passenger_id = st.text_input("Passenger ID")
if st.button("Fetch Flight Information"):
    response = agent_executor({"tool": "fetch_user_flight_information"})
    st.write(response)

st.subheader("Search Flights")
departure_airport = st.text_input("Departure Airport")
arrival_airport = st.text_input("Arrival Airport")
start_time = st.date_input("Start Time")
end_time = st.date_input("End Time")
if st.button("Search Flights"):
    response = agent_executor({"tool": "search_flights", "args": {"departure_airport": departure_airport, "arrival_airport": arrival_airport, "start_time": start_time, "end_time": end_time}})
    st.write(response)

st.subheader("Update Ticket")
ticket_no = st.text_input("Ticket Number")
new_flight_id = st.number_input("New Flight ID", step=1)
if st.button("Update Ticket"):
    response = agent_executor({"tool": "update_ticket_to_new_flight", "args": {"ticket_no": ticket_no, "new_flight_id": new_flight_id}})
    st.write(response)

st.subheader("Cancel Ticket")
ticket_no_cancel = st.text_input("Ticket Number to Cancel")
if st.button("Cancel Ticket"):
    response = agent_executor({"tool": "cancel_ticket", "args": {"ticket_no": ticket_no_cancel}})
    st.write(response)
