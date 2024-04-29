# from langchain_openai import ChatOpenAI
import os
from langchain_community.chat_models import ChatZhipuAI
from langchain.chains import LLMChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from utils import LOG

class TranslationChain:
    def __init__(self, model_name: str = "glm-4", verbose: bool = True):
        
        # 翻译任务指令始终由 System 角色承担
        template = (
                "You are a translation expert, proficient in various languages.\n"
                "Translate from {source_language} to {target_language}. "
                "When translating standard text, use the specific style: {translation_style}. "
                "Do not apply any specific style when translating tables."
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        # 待翻译文本由 Human 角色输入
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        # 为了翻译结果的稳定性，将 temperature 设置为 0
        # chat = ChatOpenAI(model_name=model_name, temperature=0, verbose=verbose)
        chat = ChatZhipuAI(
                api_key=os.environ.get("ZHIPUAI_API_KEY"),
                max_token=1000,
                top_p=0.9,
                verbose=verbose)

        self.chain = LLMChain(llm=chat, prompt=chat_prompt_template, verbose=verbose)

    def run(self, text: str, source_language: str, target_language: str,translation_style: str) -> (str, bool):
        result = ""
        try:
            result = self.chain.run({
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "translation_style": translation_style,
            })
        except Exception as e:
            LOG.error(f"An error occurred during translation: {e}")
            return result, False

        return result, True