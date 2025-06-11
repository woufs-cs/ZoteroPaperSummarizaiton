import os
import time
from dotenv import load_dotenv
from pyzotero import zotero
from typing import Union

# LangChain and LLM related imports
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
# --- 在文件顶部添加这些 ---
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import markdown # 确保这个import在全局范围内，以便GUI和核心函数都能使用

# --- CONFIGURATION ---
load_dotenv()

MD_TEMPLATE_PATH = "template.md"
TEMP_PDF_DIR = "temp_pdfs"
# 不再硬编码模型，但保留温度设置
LLM_TEMPERATURE = 0.2


# --- HELPER & CORE FUNCTIONS ---

# ... (get_item_display_name, select_collection_interactively, select_items_interactively 函数保持不变) ...
def get_item_display_name(item_data: dict) -> str:
    """格式化文献条目以便清晰显示，现在包含条目类型。"""
    title = item_data.get('title', 'No Title')
    year = item_data.get('date', 'N/A')
    item_type = item_data.get('itemType', 'Unknown Type').replace('journalArticle', 'Journal').replace(
        'conferencePaper', 'Conference')

    authors = item_data.get('creators', [])
    author_str = ""
    if authors:
        last_name = authors[0].get('lastName', '')
        if last_name:
            author_str = f"({last_name} et al.)" if len(authors) > 1 else f"({last_name})"

    return f"{title} {author_str} - {year} [{item_type}]"


# 在文件顶部，和其他 import 语句放在一起
from typing import Union

# ... 其他代码 ...

def select_collection_interactively(zot: zotero.Zotero) -> Union[str, None]:
    """从Zotero获取所有收藏夹，让用户选择一个。"""
    try:
        print("Fetching your Zotero collections...")
        collections = zot.collections()
        if not collections:
            print("[WARNING] No collections found in your Zotero library.")
            choice = input("Do you want to process items not in any collection? (yes/no): ").lower()
            return 'unfiled' if choice == 'yes' else None
    except Exception as e:
        print(f"[ERROR] Failed to fetch collections: {e}")
        return None

    print("\n--- Please select a collection to process ---")
    for i, collection in enumerate(collections):
        print(f"  [{i + 1}] {collection['data']['name']}")

    print("\n> Enter the number of the collection, or 'quit' to exit.")

    while True:
        user_input = input("Your choice: ").strip()
        if user_input.lower() == 'quit': return None

        try:
            index = int(user_input) - 1
            if 0 <= index < len(collections):
                selected_collection = collections[index]
                print(f"\n[INFO] You have selected collection: '{selected_collection['data']['name']}'")
                return selected_collection['key']
            else:
                print("[ERROR] Invalid number. Please enter a number from the list.")
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number or 'quit'.")



def select_items_interactively(items: list) -> list:
    """向用户显示文献列表并让他们选择要处理的条目。"""
    print("\n--- Please select papers to summarize ---")
    if not items:
        print("No items with PDF attachments found in this collection.")
        return []

    for i, item in enumerate(items):
        print(f"  [{i + 1}] {get_item_display_name(item['data'])}")

    print("\n> Enter the numbers of the papers (e.g., 1, 3, 5), 'all', or 'quit'.")

    while True:
        user_input = input("Your choice: ").strip()
        if user_input.lower() == 'quit': return []
        if user_input.lower() == 'all': return items

        try:
            selected_indices = [int(i.strip()) - 1 for i in user_input.split(',')]
            if all(0 <= index < len(items) for index in selected_indices):
                selected_items = [items[i] for i in selected_indices]
                print(f"\n[INFO] You have selected {len(selected_items)} paper(s) to process.")
                return selected_items
            else:
                print("[ERROR] Invalid number detected. Please use numbers from the list.")
        except ValueError:
            print("[ERROR] Invalid input. Please enter numbers, 'all', or 'quit'.")


# 在文件顶部添加导入
import tiktoken


# ... 其他代码 ...

def summarize_long_pdf(pdf_path: str, template_content: str) -> str:
    """
    使用LangChain总结PDF。
    此函数现在会智能选择策略：
    1. 尝试使用 "stuff" 方法进行快速、高质量的总结。
    2. 如果文档太长，则自动回退到 "map_reduce" 方法。
    """
    # 从环境变量加载LLM配置
    llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    api_base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY")

    print(f"  [*] Initializing LLM: [Model: {llm_model_name}, Endpoint: {api_base_url}]")

    llm = ChatOpenAI(
        model=llm_model_name,
        temperature=LLM_TEMPERATURE,
        api_key=api_key,
        base_url=api_base_url
    )

    # 加载PDF文档
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # --- 智能策略选择 ---
    try:
        # 尝试使用 "stuff" 方法
        print("  [*] Attempting fast summarization with 'stuff' method...")

        prompt_template = f"你是一名顶尖的AI研究助理，你十分熟悉自然语言处理和大模型。请仔细阅读以下整篇论文，并严格按照下面的Markdown模板格式，在‘方法’部分，请详细描述模型结构和算法流程。对于需要精确排版的数学公式，请务必使用$...$或$$...$$的LaTeX格式。对于描述算法步骤的伪代码或函数调用，请直接写出或使用反引号 `...` 来高亮，生成一份完整的、连贯的最终总结。如果信息不足，请填写 \"N/A\"。\n---\n[论文全文开始]\n{{text}}\n[论文全文结束]\n---\n[Markdown模板开始]\n{template_content}\n[Markdown模板结束]\n---\n请现在开始填充最终的Markdown模板，直接输出填充后的内容："
        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        summary = chain.invoke(docs)
        print("  [+] 'Stuff' method successful!")
        return summary.get('output_text', "Error: Could not extract summary from 'stuff' chain.")

    except Exception as e:
        # 检查是否是上下文窗口超出的错误
        if "context length" in str(e).lower():
            print("  [!] 'Stuff' method failed: Document is too long for the model's context window.")
            print("  [*] Automatically falling back to 'map_reduce' method. This will be slower.")

            # --- 回退到 Map-Reduce ---
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
            split_docs = text_splitter.split_documents(docs)

            map_prompt_template = "你是一名研究助理，请仔细阅读以下论文的一部分，并总结其核心内容，重点关注：研究问题、使用的方法、实验结果和主要结论。\n---\n{text}\n---\n请给出这部分的简明摘要："
            map_prompt = ChatPromptTemplate.from_template(map_prompt_template)

            combine_prompt_template = f"你是一名顶尖的AI研究助理。现在有一系列关于同一篇论文的摘要，它们来自论文的不同部分。你的任务是将这些摘要整合起来，并严格按照下面的Markdown模板格式，生成一份完整的、连贯的最终总结。如果信息不足，请填写 \"N/A\"。\n---\n[摘要集合开始]\n{{text}}\n[摘要集合结束]\n---\n[Markdown模板开始]\n{template_content}\n[Markdown模板结束]\n---\n请现在开始填充最终的Markdown模板，直接输出填充后的内容："
            combine_prompt = ChatPromptTemplate.from_template(combine_prompt_template)

            chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt,
                                         combine_prompt=combine_prompt)

            print("  [*] Summarization in progress with 'map_reduce'...")
            summary = chain.invoke(split_docs)
            return summary.get('output_text', "Error: Could not extract summary from 'map_reduce' chain.")
        else:
            # 如果是其他错误，则直接抛出
            print(f"  [!] An unexpected error occurred: {e}")
            raise e


# --- MAIN EXECUTION SCRIPT ---
# ... (main 函数保持不变) ...
def main():
    """
    主执行函数：连接Zotero，让用户选择收藏夹和文献，下载PDF，总结，并上传笔记。
    """
    print("--- Zotero AI Summarizer Initialized ---")

    if not os.path.exists(TEMP_PDF_DIR):
        os.makedirs(TEMP_PDF_DIR)

    try:
        zotero_user_id = os.getenv("ZOTERO_USER_ID")
        zotero_api_key = os.getenv("ZOTERO_API_KEY")
        if not all([zotero_user_id, zotero_api_key, os.getenv("OPENAI_API_KEY")]):
            raise ValueError("Zotero or LLM API keys are missing in the .env file.")
        with open(MD_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            base_template_str = f.read()
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] Setup failed: {e}")
        return

    try:
        zot = zotero.Zotero(zotero_user_id, 'user', zotero_api_key)
    except Exception as e:
        print(f"[ERROR] Could not connect to Zotero: {e}")
        return

    selected_collection_key = select_collection_interactively(zot)
    if not selected_collection_key:
        print("No collection selected. Exiting program.")
        return

    try:
        print("Fetching items from the selected collection...")
        if selected_collection_key == 'unfiled':
            # 修正：使用 collectionID=False 获取未归档条目
            all_items = zot.items(itemType='-attachment')
        else:
            all_items = zot.collection_items(selected_collection_key, itemType='-attachment')

        # 过滤出有PDF附件的条目
        items_with_pdf = [item for item in all_items if
                          item['data'].get('itemType') != 'note' and zot.children(item['key'])]

    except Exception as e:
        print(f"[ERROR] Failed to fetch items: {e}")
        return

    items_to_process = select_items_interactively(items_with_pdf)
    if not items_to_process:
        print("No items selected for processing. Exiting program.")
        return

    for item in items_to_process:
        data = item.get('data', {})
        item_key = data.get('key')
        display_name = get_item_display_name(data)

        print(f"\n--- Processing: {display_name} ---")

        attachments = zot.children(item_key)
        pdf_attachment = next(
            (att for att in attachments if att.get('data', {}).get('contentType') == 'application/pdf'), None)

        if not pdf_attachment:
            print("  [!] No PDF attachment found for this item. Skipping.")
            continue

        pdf_key = pdf_attachment['key']
        temp_pdf_path = os.path.join(TEMP_PDF_DIR, f"{pdf_key}.pdf")

        try:
            print(f"  [*] Downloading PDF (key: {pdf_key})...")
            # 修正：zot.dump 的 data_dir 参数已弃用，直接提供完整路径即可
            zot.dump(pdf_key, temp_pdf_path)
            # ... (之前的代码) ...
            pdf_key = pdf_attachment['key']
            temp_original_pdf_path = os.path.join(TEMP_PDF_DIR, f"{pdf_key}.pdf")

            print(f"  [DEBUG] Attempting to download PDF with key: {pdf_key}")
            print(f"  [DEBUG] Target path: {temp_original_pdf_path}")

            try:
                print(f"  [*] Downloading original PDF (key: {pdf_key})...")
                zot.dump(pdf_key, temp_original_pdf_path)
                print(f"  [DEBUG] PDF download successful for key: {pdf_key}")  # 如果成功会打印
                # ... (后续代码) ...
            except Exception as e:
                print(f"  [!] An error occurred while processing '{display_name}': {e}")
                import traceback
                traceback.print_exc()  # 打印完整的堆栈跟踪，这会提供更多细节

            # 修正：修复了 a.g'et 的语法错误
            authors = data.get('creators', [])
            author_str = ' and '.join([f"{a.get('firstName', '')} {a.get('lastName', '')}".strip() for a in authors])

            populated_template = base_template_str.replace("{{TITLE}}", data.get('title', 'No Title'))
            populated_template = populated_template.replace("{{AUTHORS}}", author_str)
            populated_template = populated_template.replace("{{YEAR}}", data.get('date', 'N/A'))

            summary_md = summarize_long_pdf(temp_pdf_path, populated_template)

            print("  [*] Attaching summary note to Zotero...")
            # 将Markdown转换为HTML以在Zotero笔记中获得更好的格式
            import markdown
            html_summary = markdown.markdown(summary_md)
            note_content = f"<h1>AI-Generated Summary</h1>\n{html_summary}"
            zot.create_items(
                [{'itemType': 'note', 'parentItem': item_key, 'note': note_content, 'tags': [{'tag': 'AI Summary'}]}])

            print(f"  [+] Successfully processed and added note for '{display_name}'.")

        except Exception as e:
            print(f"  [!] An error occurred while processing '{display_name}': {e}")

        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                print("  [*] Cleaned up temporary PDF file.")

        if len(items_to_process) > 1:
            print("  [*] Waiting for 5 seconds before next item to avoid rate limits...")
            time.sleep(5)

    print("\n--- All selected items have been processed. Script finished. ---")


if __name__ == "__main__":
    main()
