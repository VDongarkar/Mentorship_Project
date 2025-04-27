#!/usr/bin/env python
# coding: utf-8

# In[20]:


import requests
from typing import List, Dict
from langchain_core.tools import tool
from typing import TypedDict,List, Dict, Any,Annotated
import json
import re
import operator
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage ,HumanMessage,BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    """State for the Stack Overflow search agent."""
    messages: Annotated[list[BaseMessage], operator.add]

def search_stackoverflow(tags: List[str], query: str, max_results: int = 5) -> Dict:
    """
    Search Stack Overflow questions using tags and a user query.

    Args:
        tags (List[str]): List of tags like ["python", "api"]
        query (str): The search string or error message
        max_results (int): Number of results to return

    Returns:
        Dict: Contains search status and list of questions (if any)
    """
    url = "https://api.stackexchange.com/2.3/search"
    params = {
        "order": "desc",
        "sort": "relevance",
        "site": "stackoverflow",
        "pagesize": max_results,
        "tagged": ";".join(tags),
        "intitle": query
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])
        if not items:
            return {"status": "no_results", "message": "No Stack Overflow questions found."}

        questions = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "score": item.get("score", 0),
                "is_answered": item.get("is_answered", False),
                "tags": item.get("tags", [])
            }
            for item in items
        ]

        return {"status": "success", "questions": questions}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# In[21]:


from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai

# os.environ["GOOGLE_API_KEY"] = "AIzaSyA4Zn7MaXiJ9bV81NEen9z91O8ePjf5mic"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
@tool
def tag_search_tool(user_query: str,tags : List[str]) -> Dict[str,Any]:
    """
    Extract relevant error tags from a user's error description using an LLM,
    then find matching Stack Overflow questions.
    
    Args:
        user_query (str): The user's description of their error or issue
        tags(List[str]) : relevant tags for the query
    
    Returns:
        dict: A dictionary containing the extracted tags and Stack Overflow questions
    """

    
    
    try:
        
        
        # import json
        # import re
        
        # json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        # if json_match:
        #     result = json.loads(json_match.group(0))
        # else:
            
        #     result = json.loads(response)
        
        
        # if "tags" not in result:
        #     return {
        #         "status": "error",
        #         "message": "Could not extract proper tags from LLM response"
        #     }
        
        
        stackoverflow_results = search_stackoverflow(
            tags=tags,
            query=user_query
        )
        
        print("Params for search_stackoverflow:", params)
        print("Tags:", tags)
        print("Query:", user_query)

        return {
            "status": "success",
            "tags": tags,
            
            "stackoverflow_results": stackoverflow_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing: {str(e)}"
        }
    
    


# In[22]:


# @tool
# def get_stackoverflow_answers(question_id: int) -> List[Dict]:
#     """
#     Get answers for a specific StackOverflow question.
#     Args:
#         question_id: ID of the StackOverflow question
#     Returns:
#         List of answers for the question
#     """
#     response = requests.get(
#         f"https://api.stackexchange.com/2.3/questions/{question_id}/answers",
#         params={
#             "order": "desc",
#             "sort": "votes",
#             "site": "stackoverflow",
#             "filter": "withbody"
#         }
#     )
#     return response.json().get("items", [])


# In[23]:


@tool
def fetch_stackoverflow_answers(question_id: int, max_answers: int = 3) -> Dict:
    """
    Fetch answers for a specific Stack Overflow question using its ID.

    Args:
        question_id (int): The ID of the Stack Overflow question
        max_answers (int): Maximum number of answers to return

    Returns:
        Dict: Contains answers status and list of answers (if any)
    """
    url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
    params = {
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "withbody",  
        "pagesize": max_answers
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])
        if not items:
            return {"status": "no_answers", "message": "No answers found for this question."}

        answers = [
            {
                "answer_id": item.get("answer_id", 0),
                "body": item.get("body", ""),
                "score": item.get("score", 0),
                "is_accepted": item.get("is_accepted", False),
                "creation_date": item.get("creation_date", 0),
                "link": f"https://stackoverflow.com/a/{item.get('answer_id', 0)}"
            }
            for item in items
        ]
        

        return {"status": "success", "answers": answers}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# In[24]:


class Agent:
    def __init__(self, model,tools, system=""):
        self.system = system
        
        print("The system is:",self.system)
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)

        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        ai_message = state["messages"][-1]
        
        return bool(getattr(ai_message, "tool_calls", [])) 

    def call_gemini(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        message = self.model.invoke(messages)  
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for t in tool_calls:
            print(f"Calling tool: {t}")
            if t["name"] not in self.tools: 
                print("\n ....bad tool name....")
                result = "Invalid tool name, retry"
            else:
                result = self.tools[t["name"]].invoke(t["args"])  
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))

        print("Back to model!")
        return {"messages": results}


# In[25]:


tools = [fetch_stackoverflow_answers,tag_search_tool]

system = """You are a Stack Overflow Search Assistant. Help users find solutions to their 
    programming errors by analyzing their descriptions, extracting relevant tags,
    and searching Stack Overflow for matching questions and answers."""

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

stack_overflow = Agent(model,tools,system)


# In[26]:


def interact_with_agent():
    # Initial history
    history = []
    
    print("Starting conversation with the agent. Type 'exit' to end the conversation.")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Ending conversation.")
            break
        
        # Add user message to history
        history.append(HumanMessage(content=user_input))
        
        # Run the agent graph with the current history
        result = stack_overflow.graph.invoke({"messages": history})
        
        # Extract the latest messages from the result and add them to history
        new_messages = result["messages"]
        for msg in new_messages:
            print(f"\nAgent: {msg.content}")
            history.append(msg)
        
        


# In[27]:


interact_with_agent()


# In[ ]:




