#!/usr/bin/env python3
import os
import sys
import sqlite3
import threading
from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)

DATABASE = "travel_planner.db"
WELCOME_MESSAGE = "Welcome! I am ready to design your perfect custom travel plan. Let's start with the basics: What kind of trip are you dreaming of?"

# Load API key
key_path = os.path.expanduser("~/gemini_key.txt")
if os.path.exists(key_path):
    with open(key_path, "r") as f:
        api_key = f.read().strip()
    os.environ["GEMINI_API_KEY"] = api_key
elif "GEMINI_API_KEY" not in os.environ:
    print("Warning: GEMINI_API_KEY environment variable not set, and ~/gemini_key.txt not found.")

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def get_session_history(session_id):
    history = []
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM summaries WHERE session_id = ?", (session_id,))
        summary_row = cursor.fetchone()
        if summary_row:
            history.append({"role": "system", "text": f"[Summary of previous preferences: {summary_row[0]}]"})
            
        cursor.execute("SELECT role, text FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
        for role, text in cursor.fetchall():
            history.append({"role": role, "text": text})
    return history

def ensure_session_initialized(session_id):
    history = get_session_history(session_id)
    if not history:
        with sqlite3.connect(DATABASE) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, text) VALUES (?, ?, ?)",
                (session_id, "agent", WELCOME_MESSAGE)
            )
            conn.commit()
        return [{"role": "agent", "text": WELCOME_MESSAGE}]
    return history

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

def search_vacation_activities(query: str) -> str:
    """Searches Google for vacation destinations, attractions, and activities matching a query.
    
    Args:
        query: The search query (e.g., 'top outdoor activities and sights in Patagonia').
        
    Returns:
        A text summary of top attractions and activities with citation sources.
    """
    if not query or len(query.strip()) < 3:
        return "Error: The search query is too short or empty. Guided resolution: Please retry with a specific, descriptive search query (e.g., 'adventure sports in Queenstown New Zealand')."
        
    try:
        search_client = genai.Client()
        response = search_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=f"Perform a Google Search and summarize the top 3 activities and key travel attractions for: {query}. Keep it factual, descriptive, and return reference URLs.",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2
            )
        )
        return response.text
    except Exception as e:
        return f"Error occurred during search: {str(e)}. Guided resolution: The search service is currently experiencing high load. Please try re-formulating the search query to be more specific or try again shortly."

def check_travel_advisories_and_restrictions(destination: str) -> str:
    """Retrieves travel advisories, entry requirements, or safety guidelines for a destination.
    
    Args:
        destination: The target country or city (e.g., 'Peru' or 'Costa Rica').
        
    Returns:
        Safety advisories, entry requirements, or a healthy travel overview.
    """
    if not destination or len(destination.strip()) < 2:
        return "Error: Destination name is too short or empty. Guided resolution: Please provide a valid country or city name (e.g., 'Colombia')."
        
    try:
        advisory_client = genai.Client()
        response = advisory_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=f"Provide any essential travel safety tips, entry/visa requirements, or health advisories for travelers visiting: {destination}.",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2
            )
        )
        return response.text
    except Exception as e:
        return f"Error retrieving travel advisories: {str(e)}. Guided resolution: Unable to contact government advisory databases. Please proceed with standard precautions or try again."

def compact_session_history(session_id):
    """Executes history compaction and summarization in a background worker thread."""
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, role, text FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
            msg_rows = cursor.fetchall()
            
        # We only compact if we have more than 5 messages (welcome + 2 user + 2 agent questions)
        # to ensure we keep the 3 most recent turns completely unsummarized
        if len(msg_rows) <= 5:
            return
            
        to_compact = msg_rows[:-3]
        
        compaction_content = ""
        for _, role, text in to_compact:
            role_label = "User" if role == "user" else "Agent"
            compaction_content += f"{role_label}: {text}\n"
            
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM summaries WHERE session_id = ?", (session_id,))
            existing_summary_row = cursor.fetchone()
            existing_summary = existing_summary_row[0] if existing_summary_row else ""
            
        prompt = (
            f"You are a background helper for an AI travel agent.\n"
            f"Your job is to update a running, concise summary of the traveler's vacation preferences based on new details.\n\n"
            f"Existing Summary:\n{existing_summary or 'No prior summary.'}\n\n"
            f"New conversational details to integrate:\n{compaction_content}\n\n"
            f"Output a single, highly concise paragraph summarizing the user's specific travel preferences, styles, and interests gathered so far. "
            f"Do NOT include greeting, preamble, or formatting. Be direct and fact-based."
        )
        
        client = genai.Client()
        response = generate_content_with_retry(
            client=client,
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        
        new_summary = response.text.strip()
        
        with sqlite3.connect(DATABASE) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO summaries (session_id, text) VALUES (?, ?)",
                (session_id, new_summary)
            )
            compact_ids = [row[0] for row in to_compact]
            placeholders = ",".join("?" for _ in compact_ids)
            conn.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                tuple(compact_ids)
            )
            conn.commit()
            
        print(f"📝 Background Compaction: Successfully consolidated session {session_id}.", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ Background Compaction Error: {e}", file=sys.stderr)

def run_async_compaction(session_id):
    thread = threading.Thread(target=compact_session_history, args=(session_id,))
    thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/history', methods=['GET'])
def history():
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
            
        init_db()
        history_list = ensure_session_initialized(session_id)
        return jsonify({"history": history_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_session():
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
            
        init_db()
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
            conn.commit()
            
        ensure_session_initialized(session_id)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        user_text = data.get('text')
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        if not user_text:
            return jsonify({"error": "text is required"}), 400
            
        init_db()
        
        # Save user message to database
        with sqlite3.connect(DATABASE) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, text) VALUES (?, ?, ?)",
                (session_id, "user", user_text)
            )
            conn.commit()
            
        history = get_session_history(session_id)
        
        # Determine number of clarifying questions asked
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'agent'", (session_id,))
            questions_asked = cursor.fetchone()[0]
            
        max_questions = 3
        client = genai.Client(http_options=types.HttpOptions(timeout=120_000))
        
        history_str = ""
        for turn in history:
            role_label = "User" if turn.get("role") == "user" else "Agent"
            if turn.get("role") == "system":
                history_str += f"{turn.get('text')}\n"
            else:
                history_str += f"{role_label}: {turn.get('text')}\n"
                
        if questions_asked >= max_questions:
            initial_user_input = next((turn.get('text') for turn in history if turn.get('role') == 'user'), "nature holiday")
            search_query = f"top travel destinations and activities matching {initial_user_input}"
            plan_response = generate_final_plan(client, history_str, search_query)
            plan_data = plan_response.get_json()
            with sqlite3.connect(DATABASE) as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, text) VALUES (?, ?, ?)",
                    (session_id, "agent", plan_data.get("plan_text"))
                )
                conn.commit()
            return plan_response
            
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
        
        if turn_data.ready_to_plan or questions_asked >= max_questions:
            search_q = turn_data.search_query or "top travel destinations and activities matching user preferences"
            plan_response = generate_final_plan(client, history_str, search_q)
            plan_data = plan_response.get_json()
            with sqlite3.connect(DATABASE) as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, text) VALUES (?, ?, ?)",
                    (session_id, "agent", plan_data.get("plan_text"))
                )
                conn.commit()
            return plan_response
            
        next_q = turn_data.next_question
        with sqlite3.connect(DATABASE) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, text) VALUES (?, ?, ?)",
                (session_id, "agent", next_q)
            )
            conn.commit()
            
        run_async_compaction(session_id)
            
        return jsonify({
            "ready_to_plan": False,
            "next_question": next_q,
            "questions_asked": questions_asked + 1
        })
        
    except Exception as e:
        print(f"Error in /api/chat: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

def generate_final_plan(client, history_str, search_query):
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
    
    planning_response = generate_content_with_retry(
        client=client,
        model="gemini-3.5-flash",
        contents=planning_prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(google_search=types.GoogleSearch()),
                search_vacation_activities,
                check_travel_advisories_and_restrictions
            ],
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
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
