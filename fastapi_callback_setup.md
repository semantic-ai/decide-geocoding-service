### FastAPI eTranslation callback setup (no tunneling)

This project is configured to use the EU eTranslation REST v2 **text** API with HTTP callbacks, as described in the official documentation [`eTranslation REST v2 Text`](https://language-tools.ec.europa.eu/dev-corner/etranslation/rest-v2/text).

The callbacks are handled by the main FastAPI app instead of a standalone HTTP server or ngrok.

---

### 1. Callback flow overview

- The service sends translation requests to:
  - `ETRANSLATION_BASE_URL + "/askTranslate"`  
    (by default `https://language-tools.ec.europa.eu/etranslation/api/askTranslate`)
- Each request includes (see `src/translation_plugin_etranslation.py`):
  - `textToTranslate`
  - `sourceLanguage` (explicit, no `"auto"`)
  - `targetLanguages` (list of ISO 639‑1 codes, e.g. `["EN"]`)
  - `domain` (e.g. `"GEN"`)
  - `notifications.success.http`, `notifications.failure.http`
  - `deliveries.http`
- All three callback URLs point to:
  - `f"{ETRANSLATION_CALLBACK_URL}/callback"`
- The FastAPI route that receives these callbacks is:
  - `POST /etranslation/callback` in `web.py`
- Therefore, **ETRANSLATION_CALLBACK_URL must be the public base URL** of:
  - `https://your-host-or-domain/etranslation`

eTranslation will POST JSON payloads (success, error, delivery) to  
`https://your-host-or-domain/etranslation/callback`, matching the REST v2 spec.

---

### 2. Server requirements

Run this on a server where you can:

- Expose HTTP/HTTPS ports (80 and/or 443; 8082 is fine for testing)
- Control firewall rules (allow inbound traffic)
- Optionally have a DNS name pointing to the server (recommended)

You can use the existing `docker-compose.yaml` with `geocoding-service` as a base:

```yaml
services:
  geocoding-service:
    build:
      context: .
    ports:
      - "8082:80"       # default in this repo; change to "80:80" on a public server
    environment:
      MODE: "development"
      LOG_LEVEL: "debug"
      MU_SPARQL_ENDPOINT: http://app-decide-virtuoso-1:8890/sparql
      NOMINATIM_BASE_URL: http://nominatim:8080
      TRANSLATION_PROVIDER: "etranslation"
      # eTranslation auth (choose ONE method)
      # Option A: bearer token
      # ETRANSLATION_BEARER_TOKEN: "YOUR_BEARER_TOKEN"
      # or:
      ETRANSLATION_USERNAME: "your-username"
      ETRANSLATION_PASSWORD: "your-password"
      ETRANSLATION_BASE_URL: "https://language-tools.ec.europa.eu/etranslation/api"
      ETRANSLATION_DOMAIN: "GEN"
      ETRANSLATION_TIMEOUT: "60"
      ETRANSLATION_CALLBACK_TIMEOUT: "180"
      # IMPORTANT: base URL of /etranslation (NO trailing /callback)
      # leave empty for local dev; set on a real server, e.g.:
      # ETRANSLATION_CALLBACK_URL: "https://your-host-or-domain/etranslation"
```

> Note: `ETRANSLATION_CALLBACK_URL` **must not** contain `/callback` or `localhost`.  
> The code will append `/callback` and will reject pure localhost as eTranslation
> cannot reach it from the EU infrastructure.

---

### 3. Network and firewall configuration

On the server:

1. **Ports**
   - If you map `8082:80` (default in this repo):
     - Open inbound TCP port 8082.
   - If you change it to `80:80` on the server:
     - Ensure inbound TCP port 80 is allowed in the OS firewall (e.g. `ufw`, `firewalld`, security groups).
2. **DNS (recommended)**
   - Point a DNS record to the server IP, e.g.:
     - `A geocoding.example.com -> 203.0.113.10`
   - Then set:
     - `ETRANSLATION_CALLBACK_URL="https://geocoding.example.com/etranslation"`
   - Or for plain HTTP testing:
     - `ETRANSLATION_CALLBACK_URL="http://geocoding.example.com/etranslation"`

HTTPS is strongly recommended in production and aligns with the examples in the
official docs, but the code will work over HTTP as well if your environment
allows it.

---

### 4. FastAPI callback contract

The FastAPI route in `web.py`:

- Accepts `POST /etranslation/callback` with a JSON body.
- Mirrors the eTranslation callback structure described in the REST v2 text docs:
  - Handles:
    - Success and delivery callbacks with `requestId` and `targetLanguage` (or variants).
    - Error callbacks with `errorCode`, `errorMessage`, and `targetLanguages` list.
  - Stores the raw payload in an in‑memory dict keyed by `(requestId, targetLanguage)`.
  - Responds with a small JSON body, e.g. `{"status": "received"}`.

This matches the behaviour expected by the eTranslation backend:
callbacks are acknowledged with HTTP 200, and any errors/timeouts are handled
on the client side by waiting for the callback up to `ETRANSLATION_CALLBACK_TIMEOUT`.

---

### 5. How to test on a real server

1. **Deploy**
   - Copy the repo to the server.
   - Set the environment variables (especially the eTranslation credentials and `ETRANSLATION_CALLBACK_URL`).
   - Run:
     ```bash
     docker-compose up --build -d
     ```
2. **Verify locally on the server**
   - From the server itself:
     ```bash
     curl -X POST "http://localhost/etranslation/callback" \
       -H "Content-Type: application/json" \
       -d '{"requestId": 12345, "targetLanguage": "EN"}'
     ```
   - Expected response: a small JSON such as `{"status":"received"}`.
3. **Verify externally**
   - From another machine on the internet:
     ```bash
     curl -X POST "https://your-host-or-domain/etranslation/callback" \
       -H "Content-Type: application/json" \
       -d '{"requestId": 12345, "targetLanguage": "EN"}'
     ```
   - If this returns `{"status":"received"}`, the callback endpoint is reachable.
4. **End‑to‑end with eTranslation**
   - Trigger a translation through the application (e.g. via a `TranslationTask`).
   - Watch logs of `geocoding-service`:
     ```bash
     docker-compose logs -f geocoding-service
     ```
   - You should see:
     - A call to `/askTranslate` with a positive `requestId`.
     - A later log from the FastAPI callback handler indicating the callback was received for that `requestId` and `targetLanguage`.

If all of the above steps work, the setup is ready for others to use on any
server with a public IP and proper firewall configuration, without any tunneling
services. 

