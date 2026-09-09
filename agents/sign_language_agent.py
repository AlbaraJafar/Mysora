"""
Main Mysora agentic assistant — combines RAG retrieval,
tool calling, and conversational response generation.
"""
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.provider_router import get_llm, get_current_provider
from agents.tools import ALL_TOOLS
from agents.knowledge_base import search_knowledge_base

SYSTEM_PROMPT = """أنت مساعد ميسورة الذكي — تساعد مستخدمي
تطبيق ميسورة لتعلم لغة الإشارة العربية القرآنية.

مهمتك:
- الإجابة على أسئلة حول لغة الإشارة العربية بلطف ووضوح
- استخدام الأدوات المتاحة للتحقق من دقة النموذج الحالية
  عند سؤال المستخدم عن أي حرف أو عن أدائه
- اقتراح ما يجب على المستخدم التدرب عليه بناءً على البيانات
  الفعلية، وليس التخمين
- الرد دائماً بالعربية، بأسلوب دافئ ومختصر
- إذا سأل المستخدم عن حرف ضعيف الدقة، شجعه على المساهمة
  في جمع البيانات لتحسين النموذج

لا تخترع معلومات تقنية. استخدم الأدوات المتاحة للحصول
على بيانات حقيقية قبل الإجابة على أسئلة الدقة."""

STUDENT_PROMPT_ADDITION = """
المستخدم الحالي طالب. عند سؤاله عن أدائه الشخصي، استخدم
أداة get_my_progress مع student_id الخاص به. كن مشجعاً
ودافئاً — ركّز على تقدمه الشخصي فقط ولا تقارنه بأحد.
"""

TEACHER_PROMPT_ADDITION = """
المستخدم الحالي معلم أو ولي أمر. عند سؤاله عن طلابه،
استخدم أداة get_my_students_status مع teacher_id الخاص به.
قدّم تحليلاً واضحاً ومباشراً: من يحتاج تدخلاً فورياً،
ومن يستحق التشجيع.
"""


def build_agent() -> AgentExecutor:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("system", "معلومات مرجعية ذات صلة:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        max_iterations=4,
        handle_parsing_errors=True,
    )


def chat(
    message: str,
    user_id: str = None,
    user_role: str = None,
    chat_history: list = None,
) -> dict:
    """
    Run one turn of conversation with the Mysora agent.
    Returns response text and which provider was used.
    user_id and user_role are optional — anonymous chat works unchanged.
    """
    context_snippets = search_knowledge_base(message, k=2)
    context = "\n".join(context_snippets) if context_snippets else "لا توجد معلومات إضافية"

    role_addition = ""
    if user_role == "student" and user_id:
        role_addition = STUDENT_PROMPT_ADDITION + f"\nstudent_id: {user_id}"
    elif user_role in ("teacher", "admin") and user_id:
        role_addition = TEACHER_PROMPT_ADDITION + f"\nteacher_id: {user_id}"

    full_context = context + ("\n\n" + role_addition if role_addition else "")

    executor = build_agent()
    result = executor.invoke({
        "input": message,
        "context": full_context,
        "chat_history": chat_history or [],
    })

    return {
        "response": result.get("output", ""),
        "provider": get_current_provider(),
    }
