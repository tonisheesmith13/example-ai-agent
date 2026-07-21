#!/usr/bin/env python3
import os
import sys
from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)

# Load API key
key_path = os.path.expanduser("~/gemini_key.txt")
if os.path.exists(key_path):
    with open(key_path, "r") as f:
        api_key = f.read().strip()
    os.environ["GEMINI_API_KEY"] = api_key
elif "GEMINI_API_KEY" not in os.environ:
    print("Warning: GEMINI_API_KEY environment variable not set, and ~/gemini_key.txt not found.")

class TurnResponse(BaseModel):
    ready_to_plan: bool = Field(
        description="Set to True if you have sufficient details to recommend a destination and activities, or if you must stop because 3 questions have already been asked."
    )
    next_question: str = Field(
        description="The next friendly, targeted clarifying question to ask the user. Keep it brief. Empty if ready_to_plan is True."
    )
    search_query: str = Field(
        description="A search query for Google to find specific recommendations (e.g. 'best luxury eco-resorts and hikes in Costa Rica'). Empty if ready_to_plan is False."
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: print(f"⚠️ Flask API: Gemini API call failed or stalled. Retrying in {retry_state.next_action.sleep:.1f}s (Attempt {retry_state.attempt_number}/3)...", file=sys.stderr),
    reraise=True
)
def generate_content_with_retry(client, model, contents, config):
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        history = data.get('history', [])
        
        # Determine how many agent questions have been asked so far
        questions_asked = sum(1 for turn in history if turn.get('role') == 'agent')
        max_questions = 3
        
        client = genai.Client(http_options=types.HttpOptions(timeout=120_000))
        
        # Turn history list into text block for model evaluation
        history_str = ""
        for turn in history:
            role_label = "User" if turn.get("role") == "user" else "Agent"
            history_str += f"{role_label}: {turn.get('text')}\n"
            
        # If we have reached the hard question limit, immediately proceed to planning
        if questions_asked >= max_questions:
            # Generate search query based on user's initial dream
            initial_user_input = next((turn.get('text') for turn in history if turn.get('role') == 'user'), "nature holiday")
            search_query = f"top travel destinations and activities matching {initial_user_input}"
            return generate_final_plan(client, history_str, search_query)
            
        # Interview evaluation prompt
        prompt = (
            f"You are a friendly, professional travel agent.\n"
            f"Your task is to review the conversation history and decide if you have enough details "
            f"to create a comprehensive travel plan (including destination and 3 activities), "
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
        
        # Request evaluation from Gemini with structured outputs
        response = generate_content_with_retry(
            client=client,
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TurnResponse,
                temperature=0.2
            )
        )
        
        turn_data = response.parsed
        
        # If the model is ready to plan, or if we hit the limit
        if turn_data.ready_to_plan or questions_asked >= max_questions:
            search_q = turn_data.search_query or "top travel destinations and hotels matching user preferences"
            return generate_final_plan(client, history_str, search_q)
            
        # Otherwise, return the next clarifying question
        return jsonify({
            "ready_to_plan": False,
            "next_question": turn_data.next_question,
            "questions_asked": questions_asked + 1
        })
        
    except Exception as e:
        print(f"Error in /api/chat: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

def generate_final_plan(client, history_str, search_query):
    # Comprehensive planning prompt with search grounding
    planning_prompt = (
        f"You are a professional travel agent presenting a highly personalized vacation recommendation.\n"
        f"Review the conversation history and user preferences below, and search Google to find "
        f"the most appropriate real-world options.\n\n"
        f"User Preferences:\n"
        f"{history_str}\n"
        f"Please structure your response in beautiful Markdown, including:\n"
        f"1. **Destination**: A single, specific recommended city or region with a compelling explanation of why it fits the user.\n"
        f"2. **Top 3 Recommended Activities**: At least three distinct activities or attractions complete with short descriptions and practical tips.\n\n"
        f"Ensure the formatting is rich, polished, and exciting, ready to present directly to a client!"
    )
    
    # Call Gemini with Google Search Grounding
    planning_response = generate_content_with_retry(
        client=client,
        model="gemini-3.5-flash",
        contents=planning_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.7
        )
    )
    
    sources = []
    if planning_response.candidates and planning_response.candidates[0].grounding_metadata:
        metadata = planning_response.candidates[0].grounding_metadata
        if metadata.grounding_chunks:
            seen_uris = set()
            for chunk in metadata.grounding_chunks:
                if chunk.web and chunk.web.uri not in seen_uris:
                    sources.append({
                        "title": chunk.web.title or "Reference",
                        "uri": chunk.web.uri
                    })
                    seen_uris.add(chunk.web.uri)
                    
    return jsonify({
        "ready_to_plan": True,
        "plan_text": planning_response.text,
        "sources": sources,
        "search_query": search_query
    })

if __name__ == '__main__':
    # Default to port 5000 or Cloud Shell standard environments
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
