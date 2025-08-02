# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
import os

# --- 1. NLP2SQL Agent Setup ---
# This section contains the core logic of your NLP2SQL project.
# The code is based on your provided examples.

# Initialize the database connection.
# Make sure your database server is running and accessible.
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:root@localhost:5432/dvdrental")

# Initialize the Ollama LLM model.
# Ensure that Ollama is running and the specified model is pulled.
# Example: ollama run llama3
llm = OllamaLLM(model="llama3", temperature=0.1)

# Create the SQL database toolkit.
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Filter out the 'sql_db_query_checker' tool as it is not needed for this simple setup.
tools = [tool for tool in toolkit.get_tools() if tool.name != "sql_db_query_checker"]

db_schema = """
- Table `actor`: `actor_id`, `first_name`, `last_name`, `last_update`
- Table `actor_info`: `actor_id`, `first_name`, `last_name`, `film_info`
- Table `address`: `address_id`, `address`, `address2`, `district`, `city_id`, `postal_code`, `phone`, `last_update`
- Table `category`: `category_id`, `name`, `last_update`
- Table `city`: `city_id`, `city`, `country_id`, `last_update`
- Table `country`: `country_id`, `country`, `last_update`
- Table `customer`: `customer_id`, `store_id`, `first_name`, `last_name`, `email`, `address_id`, `activebool`, `create_date`, `last_update`, `active`
- Table `customer_list`: `id`, `name`, `address`, `zip code`, `phone`, `city`, `country`, `notes`, `sid`
- Table `film`: `film_id`, `title`, `description`, `release_year`, `language_id`, `rental_duration`, `rental_rate`, `length`, `replacement_cost`, `rating`, `last_update`, `special_features`, `fulltext`
- Table `film_actor`: `actor_id`, `film_id`, `last_update`
- Table `film_category`: `film_id`, `category_id`, `last_update`
- Table `inventory`: `inventory_id`, `film_id`, `store_id`, `last_update`
- Table `language`: `language_id`, `name`, `last_update`
- Table `nicer_but_slower_film_list`: `fid`, `title`, `description`, `category`, `price`, `length`, `rating`, `actors`
- Table `payment`: `payment_id`, `customer_id`, `staff_id`, `rental_id`, `amount`, `payment_date`
- Table `rental`: `rental_id`, `rental_date`, `inventory_id`, `customer_id`, `return_date`, `staff_id`, `last_update`
- Table `sales_by_film_category`: `category`, `total_sales`
- Table `sales_by_store`: `store`, `manager`, `total_sales`
- Table `staff`: `staff_id`, `first_name`, `last_name`, `address_id`, `email`, `store_id`, `active`, `username`, `password`, `last_update`, `picture`
- Table `staff_list`: `id`, `name`, `address`, `zip code`, `phone`, `city`, `country`, `sid`
- Table `store`: `store_id`, `manager_staff_id`, `address_id`, `last_update`
"""

# Place your few-shot examples here as a list of strings.
# This helps the LLM generate more accurate SQL queries.
examples = [
    (
        "Question: What is the total number of actors?\n"
        "Thought: The user wants to count the total number of actors. The `actor` table is the correct place to find this. I will count all rows in the table.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT COUNT(*) FROM actor;\n"
        "Observation: [(200,)]\n"
        "Thought: The query returned a single value. I now know the final answer.\n"
        "Final Answer: There are 200 actors in the database."
    ),
    (
        "Question: List all films with the least rental duration.\n"
        "Thought: I need to find the film with the shortest rental duration. The `film` table has a `rental_duration` column. I can query this table directly to find the film(s) with the minimum duration. I will use ORDER BY and LIMIT to get the result efficiently, selecting only the film title.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT title FROM film ORDER BY rental_duration ASC LIMIT 1;\n"
        "Observation: [('WONDERFUL FISH',)]\n"
        "Thought: The query returned a single value. I now know the final answer.\n"
        "Final Answer: The film with the least rental duration is 'WONDERFUL FISH'."
    ),
    (
        "Question: Which country has the most cities?\n"
        "Thought: To find out which country has the most cities, I need to join the `city` table with the `country` table on `country_id`. Then, I will count the number of cities for each country and order the results to find the highest count. I must remember to use the correct alias for the country name in the `SELECT` and `GROUP BY` clauses.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT co.country, COUNT(c.city) AS num_cities FROM city c JOIN country co ON c.country_id = co.country_id GROUP BY co.country ORDER BY num_cities DESC LIMIT 1;\n"
        "Observation: [('India', 60)]\n"
        "Thought: The query returned a single value. I now know the final answer.\n"
        "Final Answer: India has the most cities, with a total of 60 cities."
    ),
    (
        "Question: Which customer has rented the most films?\n"
        "Thought: To find the customer who has rented the most films, I need to count the number of rentals for each customer. This requires joining the customer table with the rental table on customer_id. I will then group the results by customer and order them to find the top customer.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT c.first_name, c.last_name, COUNT(r.rental_id) AS rental_count FROM customer AS c JOIN rental AS r ON c.customer_id = r.customer_id GROUP BY c.customer_id ORDER BY rental_count DESC LIMIT 1;\n"
        "Observation: [('ELEANOR', 'HUNT', 46)]\n"
        "Thought: The query returned a single value. I now know the final answer.\n"
        "Final Answer: The customer who has rented the most films is Eleanor Hunt, with 46 rentals."
    ),
    (
        "Question: what is the lowest rental duration for a film\n"
        "Thought: The user wants to find the minimum rental duration. The `film` table has a `rental_duration` column. I will use the MIN aggregate function to find the lowest value.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT MIN(rental_duration) FROM film;\n"
        "Observation: [(3,)]\n"
        "Thought: The query returned a single value. I now know the final answer.\n"
        "Final Answer: The lowest rental duration for a film is 3 days."
    ),
    (
        "Question: what is the highest rental duration for a film\n"
        "Thought: The user wants to find the maximum rental duration. The `film` table has a `rental_duration` column. I will use the MAX aggregate function to find the highest value.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT MAX(rental_duration) FROM film;\n"
        "Observation: [(7,)]\n"
        "Thought: The query returned a single value. I now know the final answer.\n"
        "Final Answer: The highest rental duration for a film is 7 days."
    )
]

example_string = "\n\n".join(examples)

# Define the PromptTemplate.
# This template guides the LLM to act as a SQL agent.
template = """
You are an agent designed to interact with a PostgreSQL database.
Given an input question, you are to generate a valid SQL query to answer it.
The database schema is as follows: {db_schema}
The user is asking the following question: {input}
You have access to the following tools: {tools}
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: The query returned a single value. I now know the final answer.
Final Answer: the final answer to the original input question

**IMPORTANT: Once you have a result that directly answers the question, you MUST provide the Final Answer and stop. Do not loop or attempt further actions.**

Begin!

{examples}

Question: {input}
{agent_scratchpad}
"""
full_prompt = PromptTemplate(
    template=template,
    input_variables=['input', 'tools', 'tool_names', 'agent_scratchpad'],
    partial_variables={'examples': example_string, 'db_schema': db_schema}
)

# Create the ReAct agent and the AgentExecutor.
agent = create_react_agent(llm=llm, tools=tools, prompt=full_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# --- 2. Flask App Configuration ---

app = Flask(__name__)

# Enable CORS for all origins. This is crucial for allowing your
# React frontend (which runs on a different port) to communicate
# with this backend API without security errors.
CORS(app)

# --- 3. Define the API Endpoint ---

# This route handles POST requests to the '/query' endpoint.
@app.route('/query', methods=['POST'])
def process_query():
    """
    Handles incoming POST requests from the frontend,
    processes the natural language query using the NLP2SQL agent,
    and returns a JSON response.
    """
    # 3.1. Extract the question from the incoming JSON data.
    data = request.json
    question = data.get('question', '')

    # 3.2. Handle cases where the question is empty.
    if not question:
        return jsonify({'error': 'No question provided.'}), 400

    try:
        # 3.3. Invoke your NLP2SQL agent with the user's question.
        response = agent_executor.invoke({"input": question})
        final_answer = response.get('output', 'Could not find an answer.')

        # 3.4. Return a successful JSON response to the frontend.
        return jsonify({
            'question': question,
            'answer': final_answer,
            'status': 'success'
        })
    except Exception as e:
        # 3.5. Catch any errors during the process and return
        # a detailed error message with a 500 status code.
        print(f"An error occurred: {e}")
        return jsonify({
            'question': question,
            'answer': str(e),
            'status': 'error'
        }), 500

# --- 4. Run the Flask Application ---

if __name__ == '__main__':
    # The application will run in debug mode, which is helpful
    # for development. The server will be accessible at
    # http://127.0.0.1:5000
    app.run(debug=True, port=5000)
