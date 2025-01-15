import requests
import json
import os
from typing import Optional
from openai import OpenAI

###############################################################################
#  Step 1: High-Level Prompt
###############################################################################
HIGH_LEVEL_PROMPT = """
Make me a tic-tac-toe master by learning how to always win, then code a bot in Python 
that can play against me in the console. Also provide instructions to run it.
"""

###############################################################################
#  Step 2: DuckDuckGo Search (Replacing AI_SEARCH)
#     - We fetch raw HTML or text from DuckDuckGo for relevant info.
#     - In real use, you'd parse or summarize this before sending to GPT.
###############################################################################
def fetch_duckduckgo(query: str) -> Optional[str]:
    """
    Fetches HTML or text results from DuckDuckGo using the 'html.duckduckgo.com' endpoint.
    Returns the raw HTML on success or None on failure.
    """
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    " AppleWebKit/537.36 (KHTML, like Gecko)"
                ),
            },
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching DuckDuckGo data for query '{query}': {e}")
        return None

###############################################################################
#  Step 3: GPT Helper (Pseudocode)
#     - We define a helper function for GPT ChatCompletion.
#     - Adjust model, temperature, etc. as needed.
###############################################################################
def call_gpt_system(system_instructions: str, user_message: str, model: str = "gpt-4o"):
    """
    Calls GPT with system instructions and user message using the new OpenAI client.
    """
    # Make sure your environment has OPENAI_API_KEY set
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content

###############################################################################
#  Step 4: Orchestrating Function
###############################################################################
def multi_step_orchestration(high_level_prompt: str):
    """
    1. Break down the user's high-level prompt.
    2. Search DuckDuckGo for relevant info.
    3. Summarize or plan with GPT.
    4. Possibly refine with another GPT call.
    5. Final GPT call to produce solution code & instructions in JSON.
    """
    print("\n🔄 Starting orchestration process...")

    # ---- Step 4.1: Ask GPT how to approach the query
    print("\n📋 Step 1: Breaking down the high-level prompt...")
    system_msg = (
        "You are an AI orchestrator. The user gave a high-level prompt. "
        "Please break it down into sub-steps to gather knowledge, code, etc."
    )
    breakdown = call_gpt_system(system_msg, high_level_prompt, model="gpt-4o")

    print("=== GPT's Proposed Breakdown of Steps ===")
    print(breakdown)

    # ---- Step 4.2: DuckDuckGo search based on sub-steps
    print("\n🔍 Step 2: Searching DuckDuckGo for strategies...")
    duckduckgo_data = fetch_duckduckgo("how to always win at tic tac toe")
    if not duckduckgo_data:
        print("⚠️  Warning: Could not fetch DuckDuckGo data")

    # ---- Step 4.3: Summarize search results with GPT
    print("\n📝 Step 3: Summarizing search results...")
    system_msg_summary = (
        "You are an AI summarizer. Summarize the following raw HTML or text from DuckDuckGo. "
        "Focus on tic-tac-toe winning strategies."
    )
    raw_search_summary = call_gpt_system(system_msg_summary, duckduckgo_data[:5000], model="gpt-4o-mini")

    print("=== GPT's Summary of DuckDuckGo Search ===")
    print(raw_search_summary)

    # ---- Step 4.4: Ask GPT to combine high-level prompt + search summary
    print("\n🤔 Step 4: Generating final approach...")
    system_msg_plan = (
        "You are an AI developer. Combine the user's high-level prompt with the search summary. "
        "Propose a final approach or plan to meet the user's request."
    )
    final_plan = call_gpt_system(system_msg_plan, f"Prompt:\n{high_level_prompt}\n\nSearch Summary:\n{raw_search_summary}", model="gpt-4o")
    
    print("=== GPT's Final Proposed Plan ===")
    print(final_plan)

    # ---- Step 4.5: Ask GPT to produce code solution
    print("\n💻 Step 5: Generating code solution...")
    system_msg_code = (
        "You are an AI that generates code solutions. Return a JSON with keys "
        "'explanation', 'code', 'installation', 'execution'."
    )
    code_solution_json_str = call_gpt_system(system_msg_code, final_plan, model="gpt-4o")

    # Attempt to parse JSON. If it fails, just return the string.
    try:
        code_solution = json.loads(code_solution_json_str)
    except json.JSONDecodeError:
        code_solution = {
            "explanation": "Could not parse JSON from GPT. Here is raw content.",
            "code": code_solution_json_str,
            "installation": "",
            "execution": "",
        }

    return code_solution

###############################################################################
#  Step 5: Entry Point
###############################################################################
def main():
    # Here we pass the single HIGH_LEVEL_PROMPT into our orchestrator.
    final_output = multi_step_orchestration(HIGH_LEVEL_PROMPT)

    # Print or return the final JSON from GPT
    print("\n=== Final JSON Output ===")
    print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    main()
