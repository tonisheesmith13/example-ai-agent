#!/usr/bin/env python3
import os
import sys
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Define terminal colors
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_HEADER = "\033[1;35m"  # Magenta
COLOR_AGENT = "\033[1;34m"   # Blue
COLOR_USER = "\033[1;32m"    # Green
COLOR_SEARCH = "\033[1;36m"  # Cyan
COLOR_PLAN = "\033[1;33m"    # Yellow
COLOR_ERROR = "\033[1;31m"   # Red

# Define the Pydantic schema for structured output during the interview phase
class TurnResponse(BaseModel):
    ready_to_plan: bool = Field(
        description="Set to True if you have sufficient details to recommend a destination, hotels, and activities, or if you must stop because 5 questions have already been asked."
    )
    next_question: str = Field(
        description="The next friendly, targeted clarifying question to ask the user. Keep it brief. Empty if ready_to_plan is True."
    )
    search_query: str = Field(
        description="A search query for Google to find specific recommendations (e.g. 'best luxury eco-resorts and hikes in Costa Rica'). Empty if ready_to_plan is False."
    )

def load_api_key():
    # Attempt to load the Gemini API Key
    key_path = "/home/tonisheesmith/gemini_key.txt"
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            api_key = f.read().strip()
        os.environ["GEMINI_API_KEY"] = api_key
    elif "GEMINI_API_KEY" not in os.environ:
        print(f"{COLOR_ERROR}Error: GEMINI_API_KEY environment variable not set, and gemini_key.txt not found.{COLOR_RESET}")
        sys.exit(1)

def main():
    load_api_key()
    
    try:
        client = genai.Client()
    except Exception as e:
        print(f"{COLOR_ERROR}Failed to initialize GenAI Client: {e}{COLOR_RESET}")
        sys.exit(1)
        
    print(f"{COLOR_HEADER}=====================================================")
    print(f"        🌴  WELCOME TO YOUR AI TRAVEL AGENT  🌴")
    print(f"====================================================={COLOR_RESET}")
    print("Let's design your perfect custom vacation! I will ask you")
    print("up to 5 clarifying questions, search Google for real-time")
    print("options, and present your tailored travel plan.")
    print("-----------------------------------------------------\n")
    
    # Initialize history and state
    conversation_history = []
    questions_asked = 0
    max_questions = 5
    
    try:
        # Prompt user for initial input
        print(f"{COLOR_BOLD}Tell me: What kind of trip are you dreaming of?{COLOR_RESET}")
        user_input = input(f"{COLOR_USER}> {COLOR_RESET}").strip()
        while not user_input:
            user_input = input(f"{COLOR_USER}> {COLOR_RESET}").strip()
        
        conversation_history.append({"role": "user", "text": user_input})
        
        ready_to_plan = False
        search_query = ""
        
        # Interview phase
        while not ready_to_plan and questions_asked < max_questions:
            # Prepare instructions for structured agent assessment
            history_str = ""
            for turn in conversation_history:
                role_label = "User" if turn["role"] == "user" else "Agent"
                history_str += f"{role_label}: {turn['text']}\n"
                
            prompt = (
                f"You are a friendly, professional travel agent.\n"
                f"Your task is to review the conversation history and decide if you have enough details "
                f"to create a comprehensive travel plan (including destination, 3 activities, and 2 hotel options), "
                f"or if you need to ask another clarifying question.\n\n"
                f"Constraints:\n"
                f"- You can ask at most {max_questions} questions.\n"
                f"- Currently, you have asked {questions_asked} questions.\n"
                f"- If questions_asked is {max_questions}, you MUST set ready_to_plan to true.\n"
                f"- Only ask one question at a time.\n\n"
                f"Conversation history:\n"
                f"{history_str}\n"
                f"Evaluate and output the appropriate next step."
            )
            
            # Request decision from Gemini with structured outputs
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TurnResponse,
                    temperature=0.2
                )
            )
            
            turn_data = response.parsed
            
            # Enforce hard limit on number of questions
            if questions_asked >= max_questions:
                ready_to_plan = True
                search_query = turn_data.search_query or f"best destinations and hotels for {user_input}"
                break
                
            if turn_data.ready_to_plan:
                ready_to_plan = True
                search_query = turn_data.search_query
                break
                
            # If agent wants to ask a question
            questions_asked += 1
            question = turn_data.next_question
            
            print(f"\n{COLOR_AGENT}Agent (Question {questions_asked}/{max_questions}):{COLOR_RESET} {COLOR_BOLD}{question}{COLOR_RESET}")
            user_input = input(f"{COLOR_USER}> {COLOR_RESET}").strip()
            while not user_input:
                user_input = input(f"{COLOR_USER}> {COLOR_RESET}").strip()
                
            conversation_history.append({"role": "agent", "text": question})
            conversation_history.append({"role": "user", "text": user_input})
            
        # Compile preferences for the search phase
        print(f"\n{COLOR_SEARCH}🔍 Compiling preferences and searching Google...{COLOR_RESET}")
        if not search_query:
            search_query = f"top vacation destinations hotels and activities matching user preferences"
            
        print(f"{COLOR_SEARCH}   Target search: \"{search_query}\"{COLOR_RESET}\n")
        
        # Build comprehensive planning prompt
        history_summary = ""
        for turn in conversation_history:
            role_label = "User" if turn["role"] == "user" else "Agent"
            history_summary += f"{role_label}: {turn['text']}\n"
            
        planning_prompt = (
            f"You are a professional travel agent presenting a highly personalized vacation recommendation.\n"
            f"Review the conversation history and user preferences below, and search Google to find "
            f"the most appropriate real-world options.\n\n"
            f"User Preferences:\n"
            f"{history_summary}\n"
            f"Please structure your response in beautiful Markdown, including:\n"
            f"1. **Destination**: A single, specific recommended city or region with a compelling explanation of why it fits the user.\n"
            f"2. **Top 3 Recommended Activities**: At least three distinct activities or attractions complete with short descriptions and practical tips.\n"
            f"3. **Hotel / Stay Options**: At least two specific hotel recommendations (e.g. range of prices/styles) with brief descriptions and why they are recommended.\n\n"
            f"Ensure the formatting is rich, polished, and exciting, ready to present directly to a client!"
        )
        
        # Call Gemini with Google Search Grounding
        planning_response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=planning_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.7
            )
        )
        
        plan_text = planning_response.text
        
        # Display the final plan
        print(f"{COLOR_PLAN}=====================================================")
        print(f"               🌴 YOUR TAILORED TRAVEL PLAN 🌴")
        print(f"====================================================={COLOR_RESET}\n")
        print(plan_text)
        print(f"\n{COLOR_PLAN}====================================================={COLOR_RESET}")
        
        # Print sources if available
        if planning_response.candidates and planning_response.candidates[0].grounding_metadata:
            metadata = planning_response.candidates[0].grounding_metadata
            if metadata.grounding_chunks:
                print(f"\n{COLOR_SEARCH}Grounding Sources & References:{COLOR_RESET}")
                seen_uris = set()
                for chunk in metadata.grounding_chunks:
                    if chunk.web and chunk.web.uri not in seen_uris:
                        print(f"🔗 {chunk.web.title}: {chunk.web.uri}")
                        seen_uris.add(chunk.web.uri)
                        
        # Save to a local markdown file
        plan_file_path = "/home/tonisheesmith/example-ai-agent/travel_plan.md"
        with open(plan_file_path, "w") as f:
            f.write("# 🌴 Your Tailored Travel Plan 🌴\n\n")
            f.write(plan_text)
            if 'seen_uris' in locals() and seen_uris:
                f.write("\n\n## Grounding Sources & References\n")
                for chunk in metadata.grounding_chunks:
                    if chunk.web:
                        f.write(f"- [{chunk.web.title}]({chunk.web.uri})\n")
                        
        print(f"\n{COLOR_BOLD}💾 Travel plan saved successfully to {plan_file_path}!{COLOR_RESET}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n{COLOR_ERROR}👋 Wave goodbye! Exiting travel agent planner...{COLOR_RESET}\n")
    except Exception as e:
        print(f"\n{COLOR_ERROR}An error occurred: {e}{COLOR_RESET}\n")

if __name__ == "__main__":
    main()
