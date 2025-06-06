import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def test_llm_connection():
    """
    一个独立的脚本，用于测试与任何兼容OpenAI API的LLM服务的连接。
    它会从 .env 文件加载配置，并尝试进行一次简单的API调用。
    """
    print("--- LLM Connection Test ---")

    # 1. 加载 .env 文件中的配置
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("LLM_MODEL_NAME")

    # 2. 检查关键配置是否存在
    if not all([api_key, base_url, model_name]):
        print("\n[ERROR] Configuration missing!")
        print("Please ensure OPENAI_API_KEY, OPENAI_API_BASE, and LLM_MODEL_NAME are all set in your .env file.")
        return

    # 3. 打印出当前使用的配置（隐藏部分密钥以保安全）
    print(f"[*] Attempting to connect with the following configuration:")
    print(f"    - Endpoint (Base URL): {base_url}")
    print(f"    - Model Name:          {model_name}")
    print(f"    - API Key:             {api_key[:5]}...{api_key[-4:]}")  # 显示部分密钥以供核对

    try:
        # 4. 初始化 ChatOpenAI 实例
        # 这与我们主项目中的方式完全相同
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            max_retries=1  # 快速失败，不等重试
        )

        # 5. 发送一个简单的测试消息
        print("\n[*] Sending a test message to the model...")
        message = HumanMessage(
            content="Hello! Please reply with a short confirmation message to verify we are connected.")

        response = llm.invoke([message])

        # 6. 打印成功信息和模型的回复
        print("\n" + "=" * 30)
        print("✅ SUCCESS! Connection established.")
        print("=" * 30)
        print(f"\nModel's Response:\n---\n{response.content}\n---")

    except Exception as e:
        # 7. 如果出错，打印详细的错误信息
        print("\n" + "=" * 30)
        print("❌ FAILURE! Could not connect to the LLM service.")
        print("=" * 30)
        print(f"\n[ERROR DETAILS]: {e}")
        print("\n[TROUBLESHOOTING TIPS]:")
        print("  1. Double-check your OPENAI_API_KEY in the .env file. Is it correct and active?")
        print("  2. Verify the OPENAI_API_BASE URL. Is it exactly 'https://aihubmix.com/v1'?")
        print("  3. Check the LLM_MODEL_NAME. Is this model available on the AiHubMix service?")
        print("  4. Ensure you have an active internet connection.")


if __name__ == "__main__":
    test_llm_connection()
