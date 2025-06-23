import os
import time
import sqlite3
import markdown
from dotenv import load_dotenv
from pyzotero import zotero
from typing import Union, List, Dict, Any

# LangChain and LLM related imports
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
load_dotenv()

# !!! 关键配置：请将此路径修改为你自己的Zotero数据目录 !!!
ZOTERO_DATA_DIR = r"C:\Users\89721\Zotero"

MD_TEMPLATE_PATH = "template.md"
LLM_TEMPERATURE = 0.2
AI_SUMMARY_TAG = "AI-Generated Summary"


# --- HELPER & CORE FUNCTIONS ---
import sqlite3
import os
from typing import List, Dict, Any


# 假设 ZOTERO_DATA_DIR 已经定义好了
# ZOTERO_DATA_DIR = "/path/to/your/Zotero" # 请确保这个路径是正确的

def debug_get_items_logic(cur: sqlite3.Cursor, collection_id: int):
    """
    【调试专用】分解SQL查询逻辑，一步步打印结果，以定位问题。
    """
    print("\n--- 开始分解查询逻辑 ---")
    print(f"[*] 正在为 collectionID: {collection_id} 进行调试...")
    print("--------------------------------------------------")

    # 步骤 1: 检查指定收藏夹中，到底有多少个条目（包括文献、笔记等）
    query1 = "SELECT COUNT(*) FROM collectionItems WHERE collectionID = ?;"
    cur.execute(query1, (collection_id,))
    count1 = cur.fetchone()[0]
    print(f"【步骤 1】收藏夹中的条目总数: {count1}")
    if count1 == 0:
        print(">> 结论: 失败点在步骤1。这个收藏夹是空的，或者 collection_id 不正确。后续步骤无法进行。")
        print("--------------------------------------------------\n")
        return
    print(">> 分析: 很好，收藏夹里有东西。我们继续。")
    print("-" * 20)

    # 步骤 2: 在这些条目中，有多少是未被删除的文献/项目？
    # 我们将 collectionItems 与 items 表连接，并排除已删除项
    query2 = """
    SELECT COUNT(i.itemID)
    FROM items i
    JOIN collectionItems ci ON i.itemID = ci.itemID
    WHERE ci.collectionID = ? AND i.itemID NOT IN (SELECT itemID FROM deletedItems);
    """
    cur.execute(query2, (collection_id,))
    count2 = cur.fetchone()[0]
    print(f"【步骤 2】收藏夹中 '未被删除' 的条目数: {count2}")
    if count2 == 0:
        print(">> 结论: 失败点在步骤2。收藏夹中的所有条目可能都被删除了。")
        print("--------------------------------------------------\n")
        return
    print(f">> 分析: 数量{'没有变化' if count1 == count2 else '有变化'}。我们继续筛选。")
    print("-" * 20)

    # 步骤 3: 在这些未删除的条目中，有多少是拥有“附件”的？
    # 这是关键的一步，我们加入了 itemAttachments 表
    query3 = """
    SELECT COUNT(DISTINCT i.itemID) -- 使用 DISTINCT 防止一个文献有多个附件导致重复计数
    FROM items i
    JOIN collectionItems ci ON i.itemID = ci.itemID
    JOIN itemAttachments ia ON i.itemID = ia.parentItemID
    WHERE ci.collectionID = ? AND i.itemID NOT IN (SELECT itemID FROM deletedItems);
    """
    cur.execute(query3, (collection_id,))
    count3 = cur.fetchone()[0]
    print(f"【步骤 3】其中拥有 '任何类型附件' 的条目数: {count3}")
    if count3 == 0:
        print(">> 结论: 失败点在步骤3。收藏夹中的文献都没有任何附件（PDF、网页快照等）。")
        print("--------------------------------------------------\n")
        return
    print(">> 分析: 很好，至少有文献是带附件的。现在我们来精确匹配PDF。")
    print("-" * 20)

    # 步骤 4: 在这些带附件的条目中，有多少附件的类型被正确标记为 'application/pdf'？
    # 这是最常见的失败点！
    query4 = """
    SELECT COUNT(DISTINCT i.itemID)
    FROM items i
    JOIN collectionItems ci ON i.itemID = ci.itemID
    JOIN itemAttachments ia ON i.itemID = ia.parentItemID
    WHERE ci.collectionID = ?
      AND ia.contentType = 'application/pdf' -- 核心筛选条件
      AND i.itemID NOT IN (SELECT itemID FROM deletedItems);
    """
    cur.execute(query4, (collection_id,))
    count4 = cur.fetchone()[0]
    print(f"【步骤 4】其中附件类型为 'application/pdf' 的条目数: {count4}")
    if count4 == 0:
        print(">> 结论: 失败点在步骤4。文献虽然有附件，但它们的类型不是 'application/pdf'。")
        # 运行一个辅助查询，看看它们的类型到底是什么
        helper_query = """
        SELECT DISTINCT ia.contentType
        FROM items i
        JOIN collectionItems ci ON i.itemID = ci.itemID
        JOIN itemAttachments ia ON i.itemID = ia.parentItemID
        WHERE ci.collectionID = ? AND i.itemID NOT IN (SELECT itemID FROM deletedItems);
        """
        cur.execute(helper_query, (collection_id,))
        actual_types = cur.fetchall()
        print(f">> 辅助信息: 在这个收藏夹中，附件的实际类型有: {[t[0] for t in actual_types]}")
        print(">> 你可能需要修改代码中的 'application/pdf' 为实际存在的类型，或者在Zotero中修复这些条目。")
        print("--------------------------------------------------\n")
        return
    print(">> 分析: 成功！数据库查询逻辑看起来是通的。现在检查最后的文件路径处理。")
    print("-" * 20)

    # 步骤 5: 检查Python端的文件路径处理是否正确
    print("【步骤 5】检查Python端的文件路径和文件是否存在...")
    # 我们运行完整的查询，但只取第一条结果来做示例
    final_query = """
    SELECT
        (SELECT value FROM itemDataValues WHERE valueID = (SELECT valueID FROM itemData WHERE itemID = i.itemID AND fieldID = 1)) AS title,
        ia.path AS pdfPath
    FROM items i
    JOIN collectionItems ci ON i.itemID = ci.itemID
    JOIN itemAttachments ia ON i.itemID = ia.parentItemID
    WHERE ci.collectionID = ? AND ia.contentType = 'application/pdf' AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
    LIMIT 1;
    """
    cur.execute(final_query, (collection_id,))
    first_item = cur.fetchone()
    if first_item:
        item_data = dict(first_item)
        title = item_data.get('title', 'N/A')
        pdf_path_str = item_data.get('pdfPath')
        print(f"  - 从数据库获取到的文献标题: '{title}'")
        print(f"  - 从数据库获取到的路径(pdfPath): '{pdf_path_str}'")

        if pdf_path_str and pdf_path_str.startswith('storage:'):
            pdf_filename = pdf_path_str.split(':')[1]
            # 确保 ZOTERO_DATA_DIR 在这里可用
            try:
                pdf_full_path = os.path.join(ZOTERO_DATA_DIR, 'storage', pdf_filename)
                print(f"  - Python拼接的完整文件路径: '{pdf_full_path}'")

                # 最终检查：文件是否存在？
                file_exists = os.path.exists(pdf_full_path)
                print(f"  - 检查该路径文件是否存在: {file_exists}")
                if not file_exists:
                    print(f">> 结论: 失败点可能在步骤5。数据库记录存在，但物理文件在磁盘上找不到。")
                    print(f">> 请确认你的 ZOTERO_DATA_DIR 变量 ('{ZOTERO_DATA_DIR}') 是否设置正确。")
            except NameError:
                print(">> 错误: ZOTERO_DATA_DIR 变量未定义。无法检查文件路径。")

        else:
            print(f">> 结论: 失败点可能在步骤5。PDF路径格式不正确（不是以 'storage:' 开头）。")
            print(f">> 这可能是一个 '链接文件' 而不是Zotero存储的文件。")

    print("--------------------------------------------------")
    print("--- 调试结束 ---")
    print("\n")


def get_item_display_name(item_data: dict) -> str:
    """
    格式化文献条目以便清晰显示，现在包含作者信息。
    """
    title = item_data.get('title', 'No Title')
    authors = item_data.get('authors_str', 'Unknown Authors')
    year = item_data.get('date', 'N/A')

    # 格式化作者显示，例如 "Smith et al."
    first_author = authors.split(',')[0] if authors and authors != 'N/A' else "Unknown"
    display_authors = f"{first_author} et al." if ',' in authors else first_author

    return f"{title} - {display_authors} ({year})"


def connect_local_zotero_db(data_dir: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    【最终优化版】安全地以只读模式连接到正在运行的Zotero的实时数据库，
    利用WAL模式避免数据库锁定问题，实现与Zotero应用同时运行。
    """
    db_path = os.path.join(data_dir, "zotero.sqlite")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Zotero database not found at: {db_path}\n"
                                f"Please check the 'ZOTERO_DATA_DIR' variable in the script.")

    # --- 关键改动在这里 ---
    # 使用 'immutable=1' 参数来利用Zotero的WAL模式。
    # 这会创建一个数据库的只读快照，允许脚本在Zotero运行时安全地读取数据，
    # 彻底解决 "database is locked" 的问题。
    db_uri = f'file:{db_path}?mode=ro&immutable=1'

    print("[INFO] Connecting to LIVE Zotero database in concurrent read-only mode...")
    try:
        # 我们将 timeout 设置为一个较小的值，以防万一出现连接问题
        con = sqlite3.connect(db_uri, uri=True, timeout=10)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        print("[SUCCESS] Successfully connected to the live database while Zotero is running.")
        return con, cur
    except sqlite3.OperationalError as e:
        print(f"[FATAL] Could not connect to the Zotero database: {e}")
        print("[HINT]  1. Is Zotero running? This mode works best when it is.")
        print("[HINT]  2. If Zotero is NOT running, try removing '&immutable=1' from the db_uri in the code.")
        exit()



def get_local_collections(cur: sqlite3.Cursor) -> List[Dict[str, Any]]:
    """
    【已修改】获取所有收藏夹，并使用缩进显示层级关系。
    """
    print("Fetching all collections from local database...")
    # 查询所有未被删除的收藏夹及其父收藏夹ID
    cur.execute("""
        SELECT collectionID, collectionName, parentCollectionID 
        FROM collections 
        WHERE collectionID NOT IN (SELECT collectionID FROM deletedCollections)
    """)
    all_collections = cur.fetchall()

    # 创建一个字典以便快速查找
    collection_map = {c['collectionID']: dict(c) for c in all_collections}

    # 构建层级结构
    nested_list = []
    # 先找到所有顶级收藏夹
    top_level_collections = [c for c in all_collections if c['parentCollectionID'] is None]

    def build_flat_list(collection_id, depth):
        collection = collection_map[collection_id]
        # 添加带缩进的当前收藏夹
        nested_list.append({
            'id': collection['collectionID'],
            'name': '  ' * depth + '- ' + collection['collectionName']
        })
        # 查找并递归处理子收藏夹
        children = [c for c in all_collections if c['parentCollectionID'] == collection_id]
        for child in sorted(children, key=lambda x: x['collectionName']):
            build_flat_list(child['collectionID'], depth + 1)

    # 从每个顶级收藏夹开始构建扁平化的层级列表
    for collection in sorted(top_level_collections, key=lambda x: x['collectionName']):
        build_flat_list(collection['collectionID'], 0)

    return nested_list


def select_collection_interactively(collections: List[Dict[str, Any]]) -> Union[int, None]:
    """
    【无需修改】此函数现在可以完美地显示带缩进的收藏夹列表。
    """
    if not collections:
        print("[WARNING] No collections found.")
        return None
    print("\n--- Please select a collection to process ---")
    for i, collection in enumerate(collections):
        print(f"  [{i + 1}] {collection['name']}")
    while True:
        try:
            choice = int(input("Your choice (enter the number): ").strip())
            if 1 <= choice <= len(collections):
                selected = collections[choice - 1]
                print(f"\n[INFO] You have selected collection: '{selected['name'].strip()}'")
                return selected['id']
            else:
                print("[ERROR] Invalid number.")
        except (ValueError, IndexError):
            print("[ERROR] Invalid input. Please enter a number from the list.")


def get_local_items_with_pdf(cur: sqlite3.Cursor, collection_id: int) -> List[Dict[str, Any]]:
    """
    【最终确认版】根据您的正确思路，获取文献并构建PDF的完整物理路径。
    """
    # 这个查询实现了您的思路：通过附件的itemID，在items表中找到它对应的key。
    query = """
    SELECT
        i.key AS parentKey,                     -- 父条目的key
        attachment_item.key AS attachmentKey,   -- 【您指出的关键】附件条目自己的key，用作存储文件夹名
        i.itemID,
        (SELECT value FROM itemDataValues WHERE valueID = (
            SELECT valueID FROM itemData WHERE itemID = i.itemID AND fieldID = 1
        )) AS title,
        ia.path AS pdfPath
    FROM
        items i
    JOIN
        collectionItems ci ON i.itemID = ci.itemID
    JOIN
        itemAttachments ia ON i.itemID = ia.parentItemID
    JOIN
        items attachment_item ON ia.itemID = attachment_item.itemID -- 通过附件的itemID找到它在items表中的行
    WHERE
        ci.collectionID = ?
        AND ia.contentType = 'application/pdf'
        AND i.itemID NOT IN (SELECT itemID FROM deletedItems);
    """
    cur.execute(query, (collection_id,))
    cur.row_factory = sqlite3.Row
    items = cur.fetchall()
    cur.row_factory = None
    print(f"[DEBUG] Found {len(items)} raw items with PDF from DB for collection ID {collection_id}.")
    if not items:
        return []
    formatted_items = []
    for row in items:
        item_data = dict(row)
        title = item_data.get('title', "Title not found")
        pdf_path_str = item_data.get('pdfPath')
        attachment_key = item_data.get('attachmentKey')  # 获取附件的key
        if pdf_path_str and pdf_path_str.startswith('storage:') and attachment_key:
            pdf_filename = pdf_path_str.split(':')[1]

            # 使用附件的key来构建正确的路径
            pdf_full_path = os.path.join(ZOTERO_DATA_DIR, 'storage', attachment_key, pdf_filename)
            if os.path.exists(pdf_full_path):
                formatted_items.append({
                    'data': {
                        'key': item_data.get('parentKey'),  # 这是父条目的key
                        'title': title,
                        # ... 其他信息 ...
                    },
                    'pdf_path': pdf_full_path
                })
            else:
                print(f"[WARN] DB record found for '{title}', but PDF file not found at: {pdf_full_path}")
        else:
            print(f"[WARN] Skipping item '{title}' due to missing 'attachmentKey' or invalid 'pdfPath'.")
    print(f"[INFO] Successfully formatted {len(formatted_items)} items with accessible PDFs.")
    return formatted_items


def filter_items_without_summary(items: List[Dict[str, Any]], zot_api: zotero.Zotero) -> List[Dict[str, Any]]:
    print("\n[INFO] Checking for existing AI summaries... (This may take a moment)")
    items_to_process = []
    for item in items:
        item_key = item['data']['key']
        try:
            children = zot_api.children(item_key)
            has_summary = False
            for child in children:
                if child['data'].get('itemType') == 'note':
                    tags = child['data'].get('tags', [])
                    if any(tag['tag'] == AI_SUMMARY_TAG for tag in tags):
                        has_summary = True
                        display_name = get_item_display_name(item['data'])
                        print(f"  [SKIP] '{display_name}' already has an AI summary.")
                        break
            if not has_summary:
                items_to_process.append(item)
        except Exception as e:
            print(f"  [WARN] Could not check children for item {item_key}: {e}. Including it for processing.")
            items_to_process.append(item)
    return items_to_process


def select_items_interactively(items: list) -> list:
    print("\n--- Please select papers to summarize ---")
    if not items:
        print("No new items to process in this collection.")
        return []
    for i, item in enumerate(items):
        print(f"  [{i + 1}] {get_item_display_name(item['data'])}")
    print("\n> Enter numbers (e.g., 1, 3, 5), 'all', or 'quit'.")
    while True:
        user_input = input("Your choice: ").strip().lower()
        if user_input == 'quit': return []
        if user_input == 'all': return items
        try:
            indices = [int(i.strip()) - 1 for i in user_input.split(',')]
            if all(0 <= index < len(items) for index in indices):
                return [items[i] for i in indices]
            else:
                print("[ERROR] Invalid number detected.")
        except ValueError:
            print("[ERROR] Invalid input. Please enter numbers, 'all', or 'quit'.")


def summarize_long_pdf(pdf_path: str, template_content: str) -> str:
    llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    api_base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"  [*] Initializing LLM: [Model: {llm_model_name}]")
    llm = ChatOpenAI(model=llm_model_name, temperature=LLM_TEMPERATURE, api_key=api_key, base_url=api_base_url)
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # 定义通用的提示信息，避免重复
    map_prompt_template = "You are a professional research assistant. Summarize the following section of a research paper, focusing on its key contributions, methods, and findings. Be concise and clear.\n\n---\n\n{text}"
    map_prompt = ChatPromptTemplate.from_template(map_prompt_template)

    combine_prompt_template = f"You are a master synthesizer. The following are summaries of different sections of a research paper. Your task is to integrate these summaries into a single, coherent, and comprehensive overview. The final output must strictly follow the provided Markdown template format. Do not add any extra text or explanations outside the template structure.\n\n---\n\n{{text}}\n\n---\n\n{template_content}"
    combine_prompt = ChatPromptTemplate.from_template(combine_prompt_template)

    try:
        print("  [*] Attempting fast summarization with 'stuff' method...")
        # 对于stuff，我们直接使用最终的combine prompt，因为它包含了模板
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=combine_prompt)
        summary = chain.invoke({"input_documents": docs, "text": " ".join([doc.page_content for doc in docs])})
        return summary.get('output_text', "Error: Could not extract summary.")
    except Exception as e:
        if "context length" in str(e).lower():
            print("  [!] Document too long, falling back to 'map_reduce' method...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
            split_docs = text_splitter.split_documents(docs)
            chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt,
                                         combine_prompt=combine_prompt, verbose=False)
            summary = chain.invoke(split_docs)
            return summary.get('output_text', "Error: Could not extract summary.")
        else:
            raise e


# --- MAIN EXECUTION SCRIPT ---
def main():
    """
    主执行函数：采用混合模式连接Zotero，处理文献。
    """
    print("--- Zotero AI Summarizer (Live Hybrid Mode) ---")

    try:
        zotero_user_id = os.getenv("ZOTERO_USER_ID")
        zotero_api_key = os.getenv("ZOTERO_API_KEY")
        if not all([zotero_user_id, zotero_api_key, os.getenv("OPENAI_API_KEY")]):
            raise ValueError("API keys for Zotero or your LLM are missing in the .env file.")
        with open(MD_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            base_template_str = f.read()
        zot_api = zotero.Zotero(zotero_user_id, 'user', zotero_api_key)
    except Exception as e:
        print(f"[FATAL] Setup failed: {e}")
        return

    db_con, db_cur = None, None
    try:
        db_con, db_cur = connect_local_zotero_db(ZOTERO_DATA_DIR)
        collections = get_local_collections(db_cur)
        selected_collection_id = select_collection_interactively(collections)
        if selected_collection_id is None:
            print("\nNo collection selected. Exiting.")
            return
        all_items_in_collection = get_local_items_with_pdf(db_cur, selected_collection_id)

        items_to_consider = filter_items_without_summary(all_items_in_collection, zot_api)

        items_to_process = select_items_interactively(items_to_consider)
        if not items_to_process:
            print("\nNo items selected for processing. Exiting.")
            return

        for item in items_to_process:
            data = item.get('data', {})
            item_key = data.get('key')
            display_name = get_item_display_name(data)
            pdf_path = item.get('pdf_path')

            print(f"\n--- Processing: {display_name} ---")
            print(f"  [*] PDF located at: {pdf_path}")

            try:
                # 【已修改】现在可以正确填充作者信息
                populated_template = base_template_str.replace("{{TITLE}}", data.get('title', 'No Title'))
                populated_template = populated_template.replace("{{AUTHORS}}", data.get('authors_str', 'N/A'))
                populated_template = populated_template.replace("{{YEAR}}", data.get('date', 'N/A'))

                summary_md = summarize_long_pdf(pdf_path, populated_template)

                print("  [*] Attaching summary note to Zotero via API...")
                html_summary = markdown.markdown(summary_md)
                note_content = f"<h1>AI-Generated Summary</h1>\n{html_summary}"
                zot_api.create_items([{
                    'itemType': 'note',
                    'parentItem': item_key,
                    'note': note_content,
                    'tags': [{'tag': AI_SUMMARY_TAG}]
                }])

                print(f"  [+] Successfully processed and added note for '{display_name}'.")

            except Exception as e:
                print(f"  [!] An error occurred while processing '{display_name}': {e}")

            if len(items_to_process) > 1 and item != items_to_process[-1]:
                print("  [*] Waiting 5 seconds before next item...")
                time.sleep(3)

    finally:
        if db_con:
            db_con.close()
            print("\n[INFO] Closed live database connection.")
        print("--- Script finished. ---")


if __name__ == "__main__":
    main()

