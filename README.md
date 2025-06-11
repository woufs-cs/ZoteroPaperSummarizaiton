# ZoteroPaperSummarizaiton（Zotero自动文献总结）
该项目是一个agent相关的项目，通过LLM和zotero的api，可以实现ai生成文献总结，并自动附在zotero对应的条目下
Installation steps
## Step1
``` pip install pyzotero,langchai ```
## Step2
创建一个.env文件，传入LLM api baseurl和zotero api userid(https://www.zotero.org/settings/security）
等信息，示例如下:
```
ZOTERO_USER_ID="xxx"
ZOTERO_API_KEY="xxx"
OPENAI_API_KEY="xxx"
OPENAI_API_BASE="xxx"
LLM_MODEL_NAME="xxx"
```
## Step3
修改提示词和笔记模板
模板在 `template.md`中可以修改
提示词则在`zotero_summarizer.py`中修改

## Step4（可选）
测试LLM是否正常连接
运行`test_llm_connection.py`

## Step5
运行`zotero_summarizer.py`则可以进行文献总结
程序会先让你选择目标文献库，然后你可以选择对应文献库的文献进行总结，最后的笔记会保存在zotero对应条目中。