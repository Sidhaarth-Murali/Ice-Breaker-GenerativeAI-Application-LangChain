from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedIn import scrape_linkedin_profile
from agents.linkedIn_lookup_agent import lookup

if __name__ == "__main__":
    print("Hello langChain")

    linkedin_profile_url = lookup(name = "Eden Marco")
    summary_template = """
        given LinkedIn information{information} about person i want you to create
        1. Short Summary
        2. two interesting facts about them
    """

    sample_prompt = PromptTemplate(
        input_variables=["information"], template=summary_template
    )


    llm = ChatOpenAI(temperature=0, tiktoken_model_name="gpt-3.5-turbo")
    # temperature decides if the model will be creative; 0 means 0 creativity
    linkedIn_data = scrape_linkedin_profile(
        linkedin_profile_url= linkedin_profile_url
    )

    chain = LLMChain(prompt=sample_prompt, llm=llm)
    print(chain.run(information=linkedIn_data))
