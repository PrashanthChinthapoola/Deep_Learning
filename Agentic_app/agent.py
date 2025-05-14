import streamlit as st
import json
import openai  # Import the openai module
# Assuming openai_client is initialized in config and accessible
from config import openai_client
from session_manager import get_current_session
from tool_definitions import tools, available_functions, system_prompt

def process_chat_message(prompt: str):
    """Processes a user's chat message using the OpenAI agent logic."""
    current_session = get_current_session()
    current_messages = current_session["messages"]
    session_id = current_session.get("name", "Current Session")

    # Append user message to session history
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for the API call
    # The API needs a list of message dictionaries in a specific format
    messages_for_api = [{"role": "system", "content": system_prompt}]

    for m in current_messages:
        if m["role"] == "system":
            continue # Skip the system prompt we just added
        # Ensure message format is compatible with OpenAI API
        msg = {"role": m["role"], "content": m.get("content", "")}
        if m["role"] == "assistant" and m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        elif m["role"] == "tool" and m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
            msg["name"] = m["name"]
            # Ensure tool content is a string for the API
            msg["content"] = str(m.get("content", ""))
        messages_for_api.append(msg)

    try:
        # First API call: Ask model to respond or call tool(s)
        response = openai_client.chat.completions.create(
            model="gpt-4o", # Or another suitable model
            messages=messages_for_api,
            tools=tools,
            tool_choice="auto", # Allows the model to decide whether to call a tool
            temperature=0.7
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # --- Handle Tool Calls (if any) ---
        if tool_calls:
            # Append the model's response (which includes tool_calls) to history
            current_messages.append(response_message.model_dump(exclude_unset=True))

            if st.session_state.get("show_tool_messages"):
                 st.info(f"[{session_id}] 🤖 Model decided to use tool(s)...")

            # Prepare messages for the next API call, including tool outputs
            messages_after_tool_calls = messages_for_api + [response_message.model_dump(exclude_unset=True)]

            # Execute each requested tool
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)

                if function_to_call:
                    try:
                        # Parse arguments from the tool call
                        # Using json.loads is necessary as function.arguments is a string
                        function_args = json.loads(tool_call.function.arguments)

                        with st.spinner(f"[{session_id}] Executing tool: `{function_name}`..." if st.session_state.get("show_tool_messages") else "..."):
                             # Execute the tool function
                             # Special handling for summarize_section which needs the client
                             if function_name == "summarize_section":
                                 function_response_content = function_to_call(openai_client=openai_client, **function_args)
                             else:
                                 function_response_content = function_to_call(**function_args)

                        # Format the tool's output as a tool message
                        tool_output_message = {
                            "tool_call_id": tool_call.id,
                            "role": "tool", # Role is 'tool' for tool outputs
                            "name": function_name,
                            "content": function_response_content, # The output from the tool function
                        }
                        # Append tool output to history
                        current_messages.append(tool_output_message)
                        # Add tool output to messages for the next API call
                        messages_after_tool_calls.append(tool_output_message)

                        if st.session_state.get("show_tool_messages"):
                             st.success(f"[{session_id}] Tool `{function_name}` executed.")

                    except json.JSONDecodeError:
                        error_msg_content = f"TOOL_ERROR: Invalid JSON arguments provided by model for tool `{function_name}`. Args: {tool_call.function.arguments}"
                        st.error(f"[{session_id}] {error_msg_content}")
                        # Append error as a tool output message
                        error_tool_output = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": error_msg_content,
                        }
                        current_messages.append(error_tool_output)
                        messages_after_tool_calls.append(error_tool_output)


                    except Exception as e:
                        # Handle any other exceptions during tool execution
                        error_msg_content = f"TOOL_ERROR: An error occurred during execution of `{function_name}`: {e}"
                        st.error(f"[{session_id}] {error_msg_content}")
                         # Append error as a tool output message
                        error_tool_output = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": error_msg_content,
                        }
                        current_messages.append(error_tool_output)
                        messages_after_tool_calls.append(error_tool_output)

                else:
                    # Handle cases where the model requested an unknown tool
                    error_msg_content = f"TOOL_ERROR: Model attempted to call an unimplemented or unknown tool: {function_name}"
                    st.error(f"[{session_id}] {error_msg_content}")
                     # Append error as a tool output message
                    error_tool_output = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": error_msg_content,
                    }
                    current_messages.append(error_tool_output)
                    messages_after_tool_calls.append(error_tool_output)

            # Second API call: Get the final response based on tool results
            if st.session_state.get("show_tool_messages"):
                 st.info(f"[{session_id}] 🧠 Getting final response based on tool results...")
            try:
                final_response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages_after_tool_calls, # Include tool outputs
                )
                final_answer = final_response.choices[0].message.content
                # Append final answer to history and display
                current_messages.append({"role": "assistant", "content": final_answer})
                with st.chat_message("assistant"):
                    st.markdown(final_answer)

            except Exception as e:
                st.error(f"[{session_id}] An error occurred during the second API call (synthesizing response): {e}")
                error_message = f"Sorry, I encountered an error while trying to formulate my response based on the tool results: {e}"
                current_messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.markdown("Sorry, I encountered an error while trying to formulate my response.")


        # --- Handle No Tool Calls ---
        else:
            # If no tool calls, the model's initial response is the final answer
            final_answer = response_message.content
            # Append final answer to history and display
            current_messages.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                st.markdown(final_answer)


    except openai.AuthenticationError as e:
        st.error(f"[{session_id}] Authentication Error: {e}. Please check your OpenAI API key.")
        error_message = "Authentication failed. Please check your OpenAI API key."
        current_messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown("Authentication failed. Please check your OpenAI API key.")

    except Exception as e:
        st.error(f"[{session_id}] An error occurred during the API call: {e}")
        error_message = f"An error occurred during the conversation: {e}"
        current_messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown("Sorry, an unexpected error occurred.")

    # Streamlit will automatically rerun and display updated messages