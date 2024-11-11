from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

def generate_restaurant_name_items(cuisine):

    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a single fancy name."
    )
    print(prompt_template_name.format(cuisine=cuisine))

    chain_name = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest menu items for {restaurant_name}. Return it as a comma seperetaed list."
    )

    food_item_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[chain_name, food_item_chain],
        input_variables=['cuisine'],
        output_variables=["restaurant_name", "menu_items"]
    )

    response = chain({"cuisine": cuisine})
    print(response)
    return response

# if __name__ == "__main__":
#     print(generate_restaurant_name_items("Italian"))