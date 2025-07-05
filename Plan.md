                                      ┌──────────────────────┐
                                      │    Frontend (UI)     │
                                      │ Streamlit / Web App  │
                                      └────────┬─────────────┘
                                               │
                                       REST / WebSocket
                                               │
                                    ┌──────────▼──────────┐
                                    │     Backend API      │
                                    │  FastAPI / Flask     │
                                    └────────┬─────────────┘
                                     async job trigger │
                                          Kafka Topic │ (event driven)
                                                     ▼
                                  ┌────────────────────────────────────┐
                                  │         Kafka Message Broker        │
                                  │  (agent-tasks, planner-events, etc)│
                                  └─────────────┬──────────────────────┘
                                                │
                                    ┌───────────▼─────────────┐
                                    │   Planner Agent (LangGraph) │
                                    │  - Routes task by intent   │
                                    └──────────┬───────────────┘
                                               │ calls
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         │                                     │                                     │
┌────────▼────────┐                ┌───────────▼────────────┐           ┌────────────▼────────────┐
│ Market Research │                │ Landing Page Generator │           │   Email Campaign Agent  │
│  (worker agent) │                │  (DALL·E + Jinja2)      │           │ SendGrid API integrated │
└─────────────────┘                └─────────────────────────┘           └─────────────────────────┘

         ▼                                      ▼                                  ▼
 ┌────────────────┐                   ┌───────────────────┐              ┌────────────────────────┐
 │   SQLite DB    │ ◄──save/load──►   │   llm_client.py    │  ◄──shared──┤  prompt_manager.py     │
 └────────────────┘                   └───────────────────┘              └────────────────────────┘

         ▲                                      ▲                                  ▲
         │                                  LLM Call Queue (Celery / Async)       │
         └──────────────────────────────────────┬──────────────────────────────────┘
                                                │
                                         ┌──────▼───────┐
                                         │  OpenAI/GPT  │
                                         │  LLM Backend │
                                         └──────────────┘
