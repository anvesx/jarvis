# orb-discussion

The discussion agent: Jev reads each message, debt rules choose one of six personas, and a local LLM speaks.
The design is in [LLD.md](LLD.md).

```python
from orb_core.jev import make_reader
from orb_core.ollama import Ollama
from orb_discussion.agent import DiscussionAgent

agent = DiscussionAgent(make_reader(), Ollama())
session = agent.new_session("Should our assistant store passwords?", "demo-1")
async for event in session.turn("We should store them in a JSON file."):
    print(event.type, event.data)
```

Replay a saved chat with live Jev calls (about $0.001), then build the page that shows every call:
```sh
python -m orb_discussion.replay data/chats/20261012-152327.json
python -m orb_discussion.view data/debt_runs/20261012-152327.json   # writes docs/explainers/debt_run.html
```
