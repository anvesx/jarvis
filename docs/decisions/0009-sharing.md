# 0009 · Sharing orb: a key for remote requests, and one address for the page and the voice signal

Date: 2026-10-22 and 2026-10-23 · Status: accepted and built; the audio of a device outside the local network is open ·
Changes: the rule "local only: the agent service has no login, so no service may be reachable from a network"
(architecture page, agent LLD)

## What happened
The owner opened a Cloudflare quick tunnel to port 8100 (`cloudflared tunnel --url http://localhost:8100`, from
22:29) and tried the page on other devices. The page loaded. **Talk failed** on every other device with "could not
establish signal connection: Failed to fetch". We found two causes and one more problem:

1. **The voice address was local.** The agent service gave every browser the LiveKit address `ws://127.0.0.1:7880`.
   On another device, 127.0.0.1 is that device, and no LiveKit server runs there. A tunnel carries web pages (HTTP).
   It does not carry WebRTC audio. So the tunnel alone cannot fix this.
2. **The page did not say why.** The chip said "Voice: local dev" with no check, and the error text was the browser's.
3. **The service had no login.** The tunnel put the whole agent service on the internet. A request with a foreign
   Host header got the page, the list of sessions and an answer from the Cartesia route
   (`/v1/speech/stt?engine=cartesia`, which costs money) with no key. The link of a quick tunnel is hard to guess, but
   anyone who got it could read sessions, make the LLM and Jev work, and spend Cartesia credits.

Later the quick tunnel lost its connection to Cloudflare: its host name stopped resolving. The address of a quick
tunnel is not stable. The owner then chose their own LiveKit server, run on this computer and forwarded with their
ngrok domain (a stable address).

## Decision
**1. The service serves this computer, and a request from outside needs the key `ORB_SHARE_KEY`**
(`apps/agent/src/orb_agent/access.py`).
- A request is **remote** when its Host is not `127.0.0.1`, `localhost` or `[::1]`, or when it has a header that a
  proxy or tunnel adds (`CF-Connecting-IP`, `CF-Ray`, `X-Forwarded-For` and four more). The Host rule also stops DNS
  rebinding: a web page that points its own name at 127.0.0.1 gets a refusal.
- A remote request gets 403 unless the key is set and the request has it. **With no `ORB_SHARE_KEY`, every remote
  request is refused.** This is the default.
- The key has 16 or more characters (`openssl rand -hex 16`). A shorter key stops the service at start. The
  service compares keys in constant time.
- **The link:** open `https://<host>/?key=<the key>` once. The service sets a cookie (HttpOnly, SameSite=Lax, 30 days,
  Secure behind https) and redirects to the same address without the key, so the key stays out of the address bar and
  the history. A script sends the header `X-Orb-Key` or `Authorization: Bearer`. The key in a link works for a GET
  page load only.
- A refusal is logged as one line for each 10 seconds, with the count, the method, the path and the Host. The
  query (where a wrong key could be) is never logged.

**2. One address carries the page and the voice signal** (`apps/agent/src/orb_agent/livekit_proxy.py`).
- The LiveKit server listens on 127.0.0.1. The service forwards `wss://<host>/rtc` and `/rtc/v1` (the signal) and the
  two `validate` pages to it, when the server is on this computer. Both sit behind the key, and LiveKit still checks
  the token. The target is fixed (`LIVEKIT_URL`), so this is not an open proxy. The admin API is not forwarded.
- A device that is not this computer cannot reach `ws://127.0.0.1:7880`, so the page uses its own address for the
  signal (`signalUrl` in the page). The owner at home still connects straight to 127.0.0.1.
- The forwarder carries signal messages only. **The audio does not pass through it.**

**3. The page checks before it calls.** `GET /v1/voice/status` says whether the LiveKit server and the voice worker
answer (the server by its health page, the worker by a TCP connection) and whether the signal is forwarded. The page
adds what only the browser knows: a page that is not secure (the browser blocks the microphone). It says the reason in
words (in the chip, under the orb and in the label of the orb), and it disables Talk when voice can never work from
this page. The check runs again at each press of Talk. Through a tunnel the page asks for the session state every 3 s,
not every 0.7 s: a free ngrok account has a limit on requests.

**4. A LiveKit server with its own keys, run here** (`scripts/livekit-keys.yaml`). The dev server's keys (`devkey` and
`secret`) are public. With `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` in `.env`, `scripts/dev.sh` starts the server with
those keys (in the environment variable `LIVEKIT_KEYS`, not on the command line), and the worker and the agent service
use them. The signal port stays on 127.0.0.1.

**5. `scripts/dev.sh` runs the forwarding.** With `ORB_NGROK_DOMAIN` in `.env` (a domain of the owner's ngrok account),
it runs `ngrok http 127.0.0.1:8100 --url https://<domain>` and waits until the domain reaches the service. Its three voice
modes are `dev` (no keys: the dev server), `keys` (this computer, with keys) and `elsewhere` (`LIVEKIT_URL` on another
host: no server is started here). Before this change the script forced the dev address and the dev keys.

## The audio: what works now, and what does not
LiveKit advertises the addresses of this computer (its network address, and its IPv6 addresses), on UDP 7882 and TCP
7881. A device gets audio when it can reach one of them: on the same network, or over IPv6 when the networks allow
inbound traffic. **Measured:** a call from a browser on this computer through the ngrok domain connected, and its
selected audio path was UDP between two host addresses (IPv6).

**A device on another network, without IPv6 access to this computer, gets no audio.** The page then says "The voice
signal works, but the audio could not reach the computer that runs orb". ngrok cannot carry it on a free account: a
TCP endpoint needs a card on file (`ERR_NGROK_8013`, tested 2026-10-23), and ngrok has no UDP. Ways to carry it:

| Option | Work |
|---|---|
| Add a card to the ngrok account (no charge on the free plan) | A TCP endpoint gives a public address and port for LiveKit's TCP port. LiveKit must advertise that exact address and port, and the port changes at each start, so `dev.sh` needs a small relay and a generated config. Not built. |
| LiveKit Cloud | Make a project, put its three values in `.env` (`LIVEKIT_URL`, key, secret): the `elsewhere` mode. The worker connects out. The audio passes through LiveKit's servers. Not tested. |
| A small public host with LiveKit or a TURN server | The audio passes through that host. Not tested. |
| A VPN between the devices | Each device installs it. Not tested. |

## ngrok on a free account
- A browser sees ngrok's notice page on the first visit ("You are about to visit ... served by <IP address>") and presses
  **Visit Site** once. **The notice shows the public IP address of this computer to every visitor.** Requests that
  carry the header `ngrok-skip-browser-warning` (the voice status check does) skip the notice.
- One static domain, and limits on requests and bandwidth (see ngrok's pricing page): this is why a tunnel page polls
  every 3 s.

## Checked (2026-10-22 and 2026-10-23)
- 49 tests in `apps/agent` (189 in the whole repo): the key rules, the proxy headers, the Host rule, the cookie and
  its flags, the redirect, the log, a stream and a websocket through the gate, the voice status, the page text, the
  health probe, and the forwarder (text and binary frames, the query byte for byte, the close codes, a server that is
  down, the two `validate` pages and nothing else, the gate, and no route for a remote server).
- On the running service: a foreign Host, the Cloudflare headers with a local Host, and a rebinding Host all got 403,
  and this computer got 200. Through the real Cloudflare tunnel (while it lived): no key, a wrong key and the Cartesia
  route got 403; the link gave 303 and a cookie, and the page loaded with the cookie only.
- In headless Chrome with a host name that is not local: no key gave the private page; the page showed the right reason
  on plain http and at phone width; with the worker off, it said so and Talk started nothing.
- **A call through the ngrok domain** (headless Chrome on this computer, a fake microphone with synthetic speech): the
  page loaded with the key link, the signal went to the ngrok host, the agent joined, the captions followed both sides,
  the agent answered, and Hang up worked. The first visit without the skip header showed ngrok's notice, and after
  Visit Site the key link worked and later visits needed no key.
- The shell helpers of `dev.sh` under bash 3.2 (the macOS default).

**Not checked:** a device on another network (no second device here), the ngrok TCP endpoint (it needs a card), and the
three options for the audio that need an account or a host. Their steps come from the documentation, not from a run.

## Consequences
- **Everyone with the key can do everything that the page does:** use the LLM, Jev and Cartesia, and read every
  session. The key is one shared secret and not a user identity. To cut everyone off, change `ORB_SHARE_KEY` and
  restart (old cookies stop working). Accounts come in iteration 6.
- **The test is the Host header and the proxy headers.** A proxy that hides both (it rewrites Host to `localhost` and
  adds no header) looks like this computer. Use a tunnel that keeps them (cloudflared and ngrok do).
- **The signal of every call passes through this service and ngrok.** The audio does not, except with a relay.
  Audio that leaves the machine (LiveKit Cloud, a public host) breaks "local by default" for the media: clear it with the
  company before real conversations use it.
- **The LiveKit server listens for audio on every interface of the computer** (TCP 7881 and UDP 7882). A session needs
  the signal port (7880, loopback only) and a token, so these ports alone start nothing. LiveKit tells every client the
  addresses of this computer, so a key holder learns them.
