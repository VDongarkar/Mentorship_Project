import time
from langchain_groq import ChatGroq
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

def search_stackoverflow(tags: List[str],  max_results: int = 5) -> Dict:
    """
    Search Stack Overflow questions using tags.

    Args:
        tags (List[str]): List of tags like ["python", "api"]
        
        max_results (int): Number of results to return

    Returns:
        Dict: Contains search status and list of questions 
    """
    url = "https://api.stackexchange.com/2.3/search"
    params = {
        "order": "desc",
        "sort": "relevance",
        "site": "stackoverflow",
        "pagesize": max_results,
        "tagged": ";".join(tags),
        # "intitle": query
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()


        items = data.get("items", [])
        print(items)
        if not items:
            return {"status": "no_results", "message": "No Stack Overflow questions found."}

        questions = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "score": item.get("score", 0),
                "question_id": item.get("question_id", 0),
                "is_answered": item.get("is_answered", False),
                "tags": item.get("tags", [])
            }
            for item in items
        ]

        return {"questions": questions}

    except Exception as e:
        return {"message": str(e)}


# In[21]:


from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai

model = ChatGroq(
    
    model_name="llama3-8b-8192"  
)

# model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
@tool
def tag_search_tool(tags : List[str]) -> Dict[str,Any]:
    """
    Extract relevant error tags from a user's error description using the LLM,
    then find matching Stack Overflow questions.
    
    Args:
        user_query (str) : The user's error description to be searched on stack overflow 
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
            tags=tags
            # query=user_query
        )
        
        # print("Params for search_stackoverflow:", params)
        # print("Tags:", tags)
        # print("Query:", user_query)

        return stackoverflow_results
        
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
def llm_solution_reflection(user_query: str, current_solution: str, max_iterations: int = 5) -> Dict[str, Any]:
    """
    An improved reflection tool that uses LLM to iteratively generate better solutions
    and check for flaws until the optimal solution is found.
    
    Process:
    1. Analyze current solution and generate a better one using LLM
    2. Check the new solution for flaws using LLM
    3. If flaws found, use the new solution as current and repeat
    4. Continue until LLM determines solution is optimal or max iterations reached
    
    Args:
        user_query (str): The original programming question or error from the user
        current_solution (str): The initial solution provided by the agent
        max_iterations (int): Maximum number of reflection iterations to perform
        
    Returns:
        Dict: Contains complete reflection history and final optimal solution
    """
    import json
    import re
    from typing import Dict, Any, List
    
    # Initialize tracking variables
    iterations = 0
    reflection_history = []
    current_solution_text = current_solution
    final_solution = current_solution
    
    while iterations < max_iterations:
        print(f"\n--- Starting Reflection Iteration {iterations + 1} ---")
        
        # STEP 1: Generate a better solution using LLM
        solution_improvement_prompt = f"""
        # Solution Improvement Task
        
        You are tasked with analyzing a current solution and generating a better, more comprehensive solution.
        
        ## Original User Query
        ```
        {user_query}
        ```
        
        ## Current Solution (Iteration {iterations + 1})
        ```
        {current_solution_text}
        ```
        
        ## Instructions
        
        Please analyze the current solution and create an improved version that:
        - Addresses any gaps or missing information
        - Provides more comprehensive explanations
        - Includes better code examples if applicable
        - Offers clearer step-by-step guidance
        - Considers edge cases and best practices
        - Improves overall quality and completeness
        
        Format your response as a JSON object:
        {{
          "analysis_of_current": "Brief analysis of what could be improved in current solution",
          "improved_solution": "Your complete improved solution that addresses the original query more effectively"
        }}
        """
        
        try:
            # Get improved solution from LLM
            response = model.invoke([HumanMessage(content=solution_improvement_prompt)])
            improvement_content = response.content
            
            # Parse JSON response for improvement
            json_match = re.search(r'\{.*\}', improvement_content, re.DOTALL)
            if json_match:
                improvement_result = json.loads(json_match.group(0))
                improved_solution = improvement_result.get("improved_solution", current_solution_text)
                improvement_analysis = improvement_result.get("analysis_of_current", "No analysis provided")
            else:
                # Fallback if parsing fails
                improved_solution = current_solution_text
                improvement_analysis = "Failed to parse improvement response"
            
            print(f"Generated improved solution (Length: {len(improved_solution)} chars)")
            
            # STEP 2: Check the improved solution for flaws using LLM
            flaw_detection_prompt = f"""
            # Solution Flaw Detection Task
            
            You need to thoroughly examine a solution and identify any flaws or areas for improvement.
            
            ## Original User Query
            ```
            {user_query}
            ```
            
            ## Previous Solution
            ```
            {current_solution_text}
            ```
            
            ## New Improved Solution to Analyze
            ```
            {improved_solution}
            ```
            
            ## Instructions
            
            Carefully analyze the new improved solution and identify:
            1. Any technical errors or mistakes
            2. Missing information or incomplete explanations
            3. Unclear or confusing parts
            4. Areas that could be more comprehensive
            5. Any other issues that prevent this from being the optimal solution
            
            Be thorough and critical in your analysis. Only mark as "NO_FLAWS_FOUND" if the solution is truly comprehensive and optimal.
            
            Format your response as a JSON object:
            {{
              "flaws_found": ["specific flaw 1", "specific flaw 2", ...],
              "flaw_severity": "HIGH/MEDIUM/LOW or NONE if no flaws",
              "detailed_analysis": "Detailed explanation of each flaw and why it's problematic",
              "is_solution_optimal": true/false,
              "confidence_score": 0.0-1.0
            }}
            
            If the solution is optimal with no significant flaws, use ["NO_FLAWS_FOUND"] for flaws_found.
            """
            
            # Get flaw analysis from LLM
            flaw_response = model.invoke([HumanMessage(content=flaw_detection_prompt)])
            flaw_content = flaw_response.content
            
            # Parse JSON response for flaw detection
            flaw_json_match = re.search(r'\{.*\}', flaw_content, re.DOTALL)
            if flaw_json_match:
                flaw_result = json.loads(flaw_json_match.group(0))
                flaws_found = flaw_result.get("flaws_found", [])
                flaw_severity = flaw_result.get("flaw_severity", "UNKNOWN")
                detailed_analysis = flaw_result.get("detailed_analysis", "No analysis provided")
                is_optimal = flaw_result.get("is_solution_optimal", False)
                confidence_score = flaw_result.get("confidence_score", 0.5)
            else:
                # Fallback if parsing fails
                flaws_found = ["Failed to parse flaw detection response"]
                flaw_severity = "UNKNOWN"
                detailed_analysis = "Error in flaw detection"
                is_optimal = False
                confidence_score = 0.0
            
            print(f"Flaw analysis complete. Flaws found: {len(flaws_found)}")
            print(f"Severity: {flaw_severity}, Optimal: {is_optimal}")
            
            # Record this iteration
            iteration_record = {
                "iteration": iterations + 1,
                "previous_solution": current_solution_text,
                "previous_solution_length": len(current_solution_text),
                "improvement_analysis": improvement_analysis,
                "improved_solution": improved_solution,
                "improved_solution_length": len(improved_solution),
                "flaws_identified": flaws_found,
                "flaw_severity": flaw_severity,
                "detailed_flaw_analysis": detailed_analysis,
                "is_solution_optimal": is_optimal,
                "confidence_score": confidence_score,
                "timestamp": iterations + 1
            }
            
            reflection_history.append(iteration_record)
            
            # STEP 3: Decide whether to continue or stop
            if "NO_FLAWS_FOUND" in flaws_found or is_optimal or confidence_score >= 0.9:
                print(f"Optimal solution found at iteration {iterations + 1}")
                final_solution = improved_solution
                break
            elif not flaws_found or len(flaws_found) == 0:
                print(f"No flaws detected, solution is good at iteration {iterations + 1}")
                final_solution = improved_solution
                break
            else:
                print(f"Flaws detected, continuing to next iteration...")
                # Use the improved solution as the current solution for next iteration
                current_solution_text = improved_solution
                final_solution = improved_solution
            
            iterations += 1
            
        except json.JSONDecodeError as je:
            print(f"JSON parsing error in iteration {iterations + 1}: {str(je)}")
            error_record = {
                "iteration": iterations + 1,
                "error_type": "JSON_PARSE_ERROR",
                "error_message": str(je),
                "previous_solution": current_solution_text,
                "final_solution": current_solution_text
            }
            reflection_history.append(error_record)
            break
            
        except Exception as e:
            print(f"General error in iteration {iterations + 1}: {str(e)}")
            error_record = {
                "iteration": iterations + 1,
                "error_type": "GENERAL_ERROR",
                "error_message": str(e),
                "previous_solution": current_solution_text,
                "final_solution": current_solution_text
            }
            reflection_history.append(error_record)
            break
    
    # Calculate improvement metrics
    original_length = len(current_solution)
    final_length = len(final_solution)
    total_iterations_performed = iterations + 1 if iterations < max_iterations else max_iterations
    
    # Prepare comprehensive final result
    final_result = {
        "status": "completed" if iterations < max_iterations else "max_iterations_reached",
        "original_solution": current_solution,
        "final_solution": final_solution,
        "solution_improved": final_solution != current_solution,
        "improvement_metrics": {
            "original_length": original_length,
            "final_length": final_length,
            "length_change_ratio": final_length / original_length if original_length > 0 else 1.0,
            "total_iterations": total_iterations_performed,
            "successful_improvements": len([r for r in reflection_history if not r.get("error_type")])
        },
        "reflection_history": reflection_history,
        "summary": {
            "iterations_performed": total_iterations_performed,
            "final_confidence": reflection_history[-1].get("confidence_score", 0.0) if reflection_history else 0.0,
            "final_optimality": reflection_history[-1].get("is_solution_optimal", False) if reflection_history else False,
            "total_flaws_addressed": sum(len(r.get("flaws_identified", [])) for r in reflection_history if not r.get("error_type")),
            "improvement_trajectory": [r.get("confidence_score", 0.0) for r in reflection_history if not r.get("error_type")]
        }
    }
    
    print(f"\n--- Reflection Complete ---")
    print(f"Status: {final_result['status']}")
    print(f"Iterations: {total_iterations_performed}")
    print(f"Solution Improved: {final_result['solution_improved']}")
    print(f"Final Confidence: {final_result['summary']['final_confidence']:.2f}")
    
    return final_result


# # Helper function to display reflection results in a readable format
# def display_reflection_summary(reflection_result: Dict[str, Any]) -> None:
#     """
#     Display a formatted summary of the reflection process.
    
#     Args:
#         reflection_result: The result dictionary from llm_solution_reflection
#     """
#     print("\n" + "="*60)
#     print("REFLECTION SUMMARY")
#     print("="*60)
    
#     summary = reflection_result.get("summary", {})
#     print(f"Status: {reflection_result.get('status', 'Unknown')}")
#     print(f"Total Iterations: {summary.get('iterations_performed', 0)}")
#     print(f"Solution Improved: {'Yes' if reflection_result.get('solution_improved') else 'No'}")
#     print(f"Final Confidence Score: {summary.get('final_confidence', 0.0):.2f}")
#     print(f"Solution Deemed Optimal: {'Yes' if summary.get('final_optimality') else 'No'}")
#     print(f"Total Flaws Addressed: {summary.get('total_flaws_addressed', 0)}")
    
#     print(f"\nImprovement Metrics:")
#     metrics = reflection_result.get("improvement_metrics", {})
#     print(f"  Original Length: {metrics.get('original_length', 0)} characters")
#     print(f"  Final Length: {metrics.get('final_length', 0)} characters")
#     print(f"  Length Change: {metrics.get('length_change_ratio', 1.0):.2f}x")
    
#     print(f"\nConfidence Trajectory: {summary.get('improvement_trajectory', [])}")
    
#     print("\n" + "="*60)

@tool
def fetch_stackoverflow_answers(question_id: int, max_answers: int = 3) -> Dict:
    """
    Fetch answers for a specific Stack Overflow question using its question_id.From the given question_id,
    fetch the answers from Stack Overflow.Get the most relevant answers based on votes.

    Args:
        question_id (int): The ID of the Stack Overflow question
        max_answers (int): Maximum number of answers to return

    Returns:
        Dict: Contains answers status and list of answers 
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
        

        return {"status" : "success","answers": answers}

    except Exception as e:
        return {"status" : "error","message": str(e)}


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
        time.sleep(4)
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


tools = [fetch_stackoverflow_answers,tag_search_tool,llm_solution_reflection]

system = """You are a Stack Overflow Search Assistant. Your goal is to help users solve their programming errors by analyzing their queries, finding relevant questions and answers on Stack Overflow, and returning the most accurate and refined solution.

- When a user asks a question, use `tag_search_tool` to find relevant Stack Overflow questions by extracting appropriate tags from the query.
- From the search results, extract the `question_id` from each question URL (the number after "questions/").
- For each `question_id`, call `fetch_stackoverflow_answers` and retrieve at least 3 answers for the most relevant questions.
- Summarize the key information from these answers and generate an initial solution based on the combined insights.

Once the initial solution is generated:

- Immediately evaluate it using the `llm_solution_reflection` tool by providing the original user query, the current solution, the tags used, and the question IDs.
- The reflection tool will return an assessment score, reasoning, flaws (gaps), tag suggestions, and an improved solution if needed.

If the reflection result indicates the solution needs improvement:

- First, clearly display the flaws in the previous answer based on the reflection output, along with reasoning and gaps identified.
- Then, use those flaws and reasoning to generate a new, improved answer.
- Re-run the reflection process on the new solution and repeat this loop up to 3 times or until the reflection score is high enough and the answer is considered sufficient.

At the end of the process, always present:
- A list of flaws (if any) from the previous version,
- The final improved and validated solution,


If no relevant questions or answers can be found, respond with: "I don't know."

Your response must always include a self-assessed and improved solution validated by the reflection tool.

"""

# model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

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
        # history.append(HumanMessage(content=user_input))
        
        # Run the agent graph with the current history
        result = stack_overflow.graph.invoke({"messages": [HumanMessage(content=user_input)]})
        
        # Extract the latest messages from the result and add them to history
        new_messages = result["messages"][-1]
        print("Agent:", new_messages.content)
        
# In[27]:

# search_stackoverflow(["docker"],"docker run error",5)
interact_with_agent()


#print(stack_overflow.graph)
