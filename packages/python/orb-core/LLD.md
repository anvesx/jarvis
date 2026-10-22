# LLD: orb-core (shared parts of every agent)

Code: `src/orb_core/` · Tests: `tests/` (17) · Status: built · Last update: 2026-10-21

**orb-core holds what every agent needs and no agent owns:** settings that never print a secret, the Jev
reader, a streaming Ollama client, the agent interface with its session store, and a safe page loader.

| Module | Job |
|---|---|
| `config.py` | settings from the environment, with `.env` files as a fallback |
| `jev.py` | the TypeSafe client for Jev, and the cost of a call |
| `ollama.py` | streaming chat with a local LLM |
| `agent.py` | `Agent`, `Session`, `Event`, input checks, `SessionStore` |
| `web.py` | the readable text of a web page, with a guard against private addresses |

## 1. Input
| Input | Limits | Trust |
|---|---|---|
| Environment variables and `.env` files (`KEY=VALUE`, `export`, quotes, `#` comments) | none | trusted; values can be secrets |
| Session IDs | `^[a-z0-9][a-z0-9-]{2,47}$` | untrusted (from URLs) |
| Topics | 1 to 2,000 characters | untrusted |
| Messages | 1 to 8,000 characters | untrusted |
| URLs in a topic | http or https; every hop must resolve to a public address | untrusted |

## 2. Algorithm
- **Settings:** a variable that is already set in the process wins; then the repo `.env`; then the file named
  by `ORB_SECRETS_FILE` or `JEV_ENV_FILE`. Empty values are skipped. `describe(names)` returns "set" or
  "missing" for each name, never a value.
- **Session store:** `create` checks the agent name, the topic and the ID, evicts the least recently used
  session when 200 are live, and asks the agent for a new session. `save` writes the snapshot to a temporary
  file and renames it over the old one, so a reader never sees half a file.
- **Page loader:** check the URL → GET with redirects off → for each redirect, check the new URL again (at most 5
  hops) → keep the title and the text of headings, paragraphs, list items, quotes and code → skip scripts,
  styles, navigation, headers, footers and forms → cut at 40,000 characters.

## 3. Maths
| Name | Value | Source |
|---|---|---|
| Jev price | $0.042 per million input tokens; cost = tokens / 10⁶ × 0.042 | the price that the old chat app used (2026-10-12); not checked again against TypeSafe's current price list |
| Jev retries | 6, back-off 1 s to 30 s | chosen: a turn must survive a short API outage |
| Ollama defaults | temperature 0.7, context 16,384 tokens, keep-alive 30 min, `think: false` | chosen; the persona sets the temperature per reply |
| Live sessions | at most 200, least recently used evicted | chosen |
| Page limits | 6,000,000 bytes read; 40,000 characters kept (about 10k tokens); 20 s timeout; 5 redirects | chosen: a long blog post fits Gemma's context with the chat |
| Text block kept | headings always; other blocks of 30 or more characters | chosen: drops menu items and buttons |

## 4. Test cases
| Case | Expected result | Test |
|---|---|---|
| `.env` syntax | comments, quotes, `export` and `a=b` values parse | `test_config.py::test_parse_env_text_handles_comments_quotes_and_export` |
| Precedence | a set variable is never overwritten; empty values are skipped | `test_load_env_files_never_overrides_the_process` |
| `describe` | "set" or "missing", never the value | `test_describe_never_returns_values` |
| A missing required variable | the error names the variable | `test_env_required_names_the_variable_but_not_a_value` |
| Secret names | KEY, TOKEN, SECRET, PASSWORD are secret | `test_is_secret` |
| Repo root | the nearest folder with `go.work` | `test_repo_root_finds_go_work` |
| Protocols | test doubles satisfy `Agent` and `Session` | `test_agent.py::test_fakes_satisfy_the_protocols` |
| Events | an unknown type is refused; JSON is stable | `test_event_rejects_unknown_types_and_serializes` |
| Session IDs | `..`, spaces, capitals and long IDs are refused | `test_session_ids` |
| Store | create, get, turn, save (no `.tmp` left) | `test_store_create_get_save` |
| Bad input | empty or long topic, unknown agent, duplicate ID | `test_store_rejects_bad_input` |
| Eviction | the least recently used session goes first | `test_store_get_or_create_and_eviction` |
| URL in text | found; trailing punctuation removed | `test_web.py::test_find_url_strips_trailing_punctuation` |
| Private addresses | 127.0.0.1, 10/8, 192.168/16, 169.254.169.254, ::1 refused | `test_check_public_blocks_private_and_odd_schemes` |
| Override | `ORB_ALLOW_PRIVATE_URLS=1` allows them | `test_check_public_can_be_switched_off` |
| Extraction | keeps headings and paragraphs, drops navigation and scripts | `test_extract_keeps_text_and_skips_navigation` |
| A redirect to a private address | refused; the private URL is never requested | `test_load_rechecks_every_redirect` |

## 5. Security
- **Secrets:** only read from the environment; never logged; `describe` and health pages show names only.
- **SSRF:** a topic URL could make the server fetch internal services or cloud metadata. Every hop is resolved
  and checked with `ipaddress.is_global`. **Gap:** DNS rebinding (the name resolves to a public address at the
  check and to a private one at the connection) is not blocked. Close it before the server is reachable from a
  network: connect to the checked address, not the name.
- **Path safety:** session IDs cannot contain `/` or `.`, so a snapshot path cannot leave `data/sessions/`.

## 6. Performance
Every function here is far below 1 ms, except the network calls: an Ollama stream (see orb-discussion) and a
page load (up to the 20 s timeout). The page load runs in a worker thread, so it never blocks the event loop.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|
| No `.env` file | nothing is loaded; no error | files are optional |
| A value with `=` in it | kept whole (`D=a=b` gives `a=b`) | split at the first `=` only |
| An evicted session that is mid-turn | the turn finishes; the session is gone afterwards | the turn holds a reference |
| A page that needs JavaScript | `"error": "the page has almost no readable text"` | fewer than 200 characters of text |
| A non-HTML link (PDF) | an error that names the content type | PDFs come with iteration 1 (orb-docs) |

## 8. Expected results
`make test` passes the 17 tests. `GET /health` of the agent service lists the agents and voice, with no
secret values.
