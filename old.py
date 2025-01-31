import requests
import json
import os
from typing import Optional
from openai import OpenAI

###############################################################################
#  Step 1: High-Level Prompt
###############################################################################
HIGH_LEVEL_PROMPT = """
Research ICLR, NeurIPS,and ICML and teach it to me simply in a way where I can understand it.
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
def call_gpt_system(system_instructions: str, user_message: str, model: str):
    """
    Calls GPT with system instructions and user message using the new OpenAI client.
    """
    try:
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
    except Exception as e:
        print(f"Error calling GPT system: {e}")
        return None

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
    if not breakdown:
        print("⚠️ Failed to get a breakdown from GPT. Exiting...")
        return {"error": "Failed to generate breakdown from GPT."}

    print("=== GPT's Proposed Breakdown of Steps ===")
    print(breakdown)

    searches = (
        "Using the following breakdown, create a list of google search queries to hit all of the steps: "
        + breakdown
        + " Return them in a list, such as search('sample search query here')."
    )
    search_query = call_gpt_system(system_msg, searches, model="gpt-4o")
    if not search_query:
        print("⚠️ Failed to generate search queries from GPT. Exiting...")
        return {"error": "Failed to generate search queries from GPT."}

    # Convert search(query) format to fetch_duckduckgo calls
    search_results = []
    print("\n🔎 Search queries being executed:")
    for line in search_query.split('\n'):
        if 'search(' in line:
            # Extract query from between parentheses
            query = line[line.find('(')+1:line.find(')')].strip('"\'')
            print(f"  • Searching for: {query}")
            duckduckgo_data = fetch_duckduckgo(query)
            if duckduckgo_data:
                search_results.append(duckduckgo_data)
            else:
                print(f"    ⚠️ No results found for: {query}")

    # Combine all search results
    duckduckgo_data = '\n'.join(search_results) if search_results else None

    # ---- Step 4.2: DuckDuckGo search based on sub-steps
    print("\n🔍 Step 2: Searching DuckDuckGo for strategies...")
    if not duckduckgo_data:
        print("⚠️  Warning: Could not fetch DuckDuckGo data")
        return {"error": "No DuckDuckGo data found."}

    # ---- Step 4.3: Summarize search results with GPT
    print("\n📝 Step 3: Summarizing search results...")
    system_msg_summary = (
        "You are an AI summarizer. Summarize the following raw HTML or text from DuckDuckGo. "
        "Focus on useful strategies and community improvement ideas."
    )
    raw_search_summary = call_gpt_system(system_msg_summary, duckduckgo_data[:5000], model="gpt-4o-mini")
    if not raw_search_summary:
        print("⚠️ Failed to summarize search results.")
        return {"error": "Failed to summarize search results."}

    print("=== GPT's Summary of DuckDuckGo Search ===")
    print(raw_search_summary)

    # ---- Step 4.4: Ask GPT to combine high-level prompt + search summary
    print("\n🤔 Step 4: Generating final approach...")
    system_msg_plan = (
        "You are an AI developer. Combine the user's high-level prompt with the search summary. "
        "Propose a final approach or plan to meet the user's request."
    )
    final_plan = call_gpt_system(system_msg_plan, f"Prompt:\n{high_level_prompt}\n\nSearch Summary:\n{raw_search_summary}", model="gpt-4o")
    if not final_plan:
        print("⚠️ Failed to generate final plan.")
        return {"error": "Failed to generate final plan."}

    print("=== GPT's Final Proposed Plan ===")
    print(final_plan)

    # ---- Step 4.5: Ask GPT to produce code solution
    print("\n💻 Step 5: Generating code solution...")
    system_msg_code = (
        "You are an AI that generates code solutions. For installation - example: pip install ... Return a JSON with keys "
        "'explanation', 'code', 'installation'."
    )
    code_solution_json_str = call_gpt_system(system_msg_code, final_plan, model="gpt-4o")
    if not code_solution_json_str:
        print("⚠️ Failed to generate code solution.")
        return {"error": "Failed to generate code solution."}

    # Log the raw response before attempting to parse
    print("\n💾 Logging raw GPT response...")
    with open('gpt_response.txt', 'w', encoding='utf-8') as f:
        f.write(code_solution_json_str)

    # Attempt to parse JSON with better error handling
    try:
        # First, find and extract just the JSON content
        json_start = code_solution_json_str.find('```json')
        if json_start != -1:
            # Remove everything before ```json
            code_solution_json_str = code_solution_json_str[json_start + 7:]
            
            # Remove everything after the next ```
            json_end = code_solution_json_str.find('```')
            if json_end != -1:
                code_solution_json_str = code_solution_json_str[:json_end]
        
        code_solution = json.loads(code_solution_json_str.strip())
        
        if not isinstance(code_solution, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        
        required_keys = ['explanation', 'code', 'installation']
        if not all(key in code_solution for key in required_keys):
            raise ValueError(f"Missing required keys. Expected: {required_keys}")
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ Error parsing JSON: {e}")
        print("📝 Raw response has been saved to 'gpt_response.txt'")
        return {
            "error": "JSON parsing failed",
            "raw_content": code_solution_json_str
        }

    # If parsing succeeded, write the code to ai.py
    if 'code' in code_solution:
        try:
            # Clean and format the code before writing
            clean_code = (code_solution['code']
                .replace('\\n', '\n')  # Replace escaped newlines
                .replace('\"', '"')    # Replace escaped quotes
                .strip()               # Remove leading/trailing whitespace
            )
            
            # Write the cleaned code
            with open('ai.py', 'w', encoding='utf-8') as f:
                f.write(clean_code)
            print("✅ Successfully wrote code to ai.py")
        except Exception as e:
            print(f"⚠️ Error writing to ai.py: {e}")

    return code_solution

###############################################################################
#  Step 5: Entry Point
###############################################################################
def pretty_print_json(json_data):
    """
    Pretty prints JSON data with proper formatting for nested structures
    """
    def format_value(value, indent=0):
        if isinstance(value, dict):
            return pretty_print_dict(value, indent)
        elif isinstance(value, list):
            return pretty_print_list(value, indent)
        elif isinstance(value, str):
            # Handle code blocks or long text
            if "```" in value:
                return f"\n{'  ' * indent}" + value.replace("\n", f"\n{'  ' * indent}")
            return value
        return str(value)

    def pretty_print_dict(d, indent=0):
        output = []
        for key, value in d.items():
            output.append(f"{'  ' * indent}{key}:")
            output.append(format_value(value, indent + 1))
        return '\n'.join(output)

    def pretty_print_list(lst, indent=0):
        output = []
        for item in lst:
            output.append(f"{'  ' * indent}- {format_value(item, indent + 1)}")
        return '\n'.join(output)

    print("\n=== Final Output ===")
    print(pretty_print_dict(json_data))

def test_duckduckgo_connection():
    """
    Tests the DuckDuckGo connection with a simple search.
    Returns True if successful, False otherwise.
    """
    print("\n🔍 Testing DuckDuckGo connection...")
    test_result = fetch_duckduckgo("python programming")
    if test_result:
        print("✅ DuckDuckGo connection test successful")
        return True
    else:
        print("❌ DuckDuckGo connection test failed")
        return False

def main():
    # Test DuckDuckGo connection before proceeding
    if not test_duckduckgo_connection():
        print("⚠️  Exiting due to DuckDuckGo connection failure")
        return
        
    final_output = multi_step_orchestration(HIGH_LEVEL_PROMPT)
    # Comment out or remove the pretty print since we're handling output differently
    # pretty_print_json(final_output)

if __name__ == "__main__":
    main()
