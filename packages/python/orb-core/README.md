# orb-core

The shared parts of every orb agent: settings, the Jev reader, the Ollama client, the agent interface and the
page loader. The design is in [LLD.md](LLD.md).

```python
from orb_core import config
from orb_core.agent import SessionStore
from orb_core.jev import make_reader
from orb_core.ollama import Ollama

config.bootstrap()                     # the repo .env, then the file named by JEV_ENV_FILE
reader = make_reader()                 # needs JEV_API_KEY
llm = Ollama(model="gemma4:e2b")
```

To write a new agent, implement `orb_core.agent.Agent` (`name`, `new_session`, `health`) and a session with
`turn(text, channel, replace_last)` that yields `Event`s, then register it in `apps/agent`.
