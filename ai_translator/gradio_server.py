import sys
import os
import gradio as gr
from gradio.themes.utils import (
    colors,
    fonts,
    get_matching_version,
    get_theme_assets,
    sizes,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ArgumentParser, LOG
from translator import PDFTranslator, TranslationConfig

def translation(input_file, source_language, target_language,translation_style,output_file_format):
    LOG.debug(f"[翻译任务]\n源文件: {input_file.name}\n源语言: {source_language}\n目标语言: {target_language}\n翻译风格：{translation_style}\n输出文档格式：{output_file_format}")

    output_file_path = Translator.translate_pdf(
        input_file.name, source_language=source_language, target_language=target_language,translation_style=translation_style,output_file_format=output_file_format)

    return output_file_path

def launch_gradio():

    iface = gr.Interface(
        fn=translation,
        title="ChatGlmAI-Translator v2.0(PDF 电子书翻译工具)",
        inputs=[
            gr.File(label="上传PDF文件"),
            gr.Textbox(label="源语言（默认：英文）", placeholder="English", value="English"),
            gr.Textbox(label="目标语言（默认：中文）", placeholder="Chinese", value="Chinese"),
            gr.Textbox(label="翻译风格（默认：正式'）", placeholder="Enter style description", value="formal"),  # 使用 Textbox 允许自然语言输入
            gr.Dropdown(label="输出格式", choices=["markdown", "pdf"], value="markdown") # 选择输出文档的格式
        ],
        outputs=[
            gr.File(label="下载翻译文件")
        ],
        allow_flagging="never",
    ) 

    iface.launch(share=True, server_name="0.0.0.0") 

def initialize_translator():
    # 解析命令行
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    # 初始化配置单例
    config = TranslationConfig()
    config.initialize(args)    
    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    global Translator
    Translator = PDFTranslator(config.model_name)


if __name__ == "__main__":
    # 初始化 translator
    initialize_translator()
    # 启动 Gradio 服务
    launch_gradio()
