from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain, LLMMathChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import os
from langchain.memory import ConversationBufferWindowMemory
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Access your API keys
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.getenv('SERPAPI_API_KEY')

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0, streaming=True)

llm_math_chain = LLMMathChain(llm=llm)
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="Useful for when you need to answer in depth questions about Sales."
    ),
    Tool(
        name="LLMMath",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math"
    )

]

template = """You are a friendly assistant that helps the user with queries related to Sales only.
Determine what user wants from the following options:
1. Value proposition: Focus on the unique selling points and value proposition that the user can work on that will set it apart from competitors.
2. Needs analysis: Understand the user's queries and ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.
3. Solution presentation: Based on the prospect's needs, present the solution that can address their pain points.
4. Product/Service Information: Check if the user wants information regarding a particular product or Service. If they do, provide them an appropriate answer.
5. Personalized Recommendation: The user may want a personalized recommendation for something, always help the user with this.
6. Future prospects: A user may want to ask queries about the future aspects of some product or service. 


After getting the user Intent, answer the user query with respect to sales only. Give attention to the company the user mentions
and then answer accordingly to the company's work and needs. End the execution when you want the user to clarify something.
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to help people with thorough research.

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt_with_history = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"])


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

memory=ConversationBufferWindowMemory(k=2)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools,verbose = True, memory=memory)

agent_executor.run('''What is Chelsea''')
