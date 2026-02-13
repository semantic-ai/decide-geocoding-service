### FastAPI eTranslation callback setup

This service uses the EU eTranslation REST v2 API with HTTP callbacks. Callbacks are handled by the FastAPI app in `web.py` - no need for ngrok or a separate server.

---

### How it works

1. Service sends translation requests to eTranslation API
2. eTranslation processes the request asynchronously
3. eTranslation POSTs the result back to your `callback_url` + `/callback`
4. The route `POST /etranslation/callback` in `web.py` receives and processes it

The callback URL you configure must be publicly reachable by eTranslation's servers.

---

### Configuration

Set up eTranslation in `config.json`:

```json
{
  "translation": {
    "provider": "etranslation",
    "etranslation": {
      "base_url": "https://language-tools.ec.europa.eu/etranslation/api",
      "bearer_token": null,
      "username": "your-username",
      "password": "your-password",
      "domain": "GEN",
      "callback_url": "https://your-dispatcher-domain/etranslation"
    }
  }
}
```

Use either `bearer_token` OR `username`/`password` for auth.

**Important:** Don't include `/callback` in `callback_url` - the code adds it automatically. Also don't use `localhost` - eTranslation can't reach it.

---

### Use a dispatcher (recommended)

Don't open ports on your router. Instead, expose the service through your dispatcher (nginx, Traefik, etc.) that already has ports 80/443 open.

**Setup:**

1. Configure dispatcher to route `/etranslation/*` → `geocoding-service:80`

   Example nginx config:
   ```nginx
   location /etranslation {
       proxy_pass http://geocoding-service:80;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
       proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       proxy_set_header X-Forwarded-Proto $scheme;
   }
   ```

2. Set `callback_url` in `config.json` to your dispatcher's public URL:
   ```json
   "callback_url": "https://your-dispatcher-domain/etranslation"
   ```

That's it. The dispatcher forwards `/etranslation/callback` to your service internally.

**Why use a dispatcher:**
- No router port forwarding needed
- SSL termination handled centrally
- Service stays internal (better security)
- Standard setup for microservices

---

### Testing

1. **Local test** (from the server):
   ```bash
   curl -X POST "http://localhost/etranslation/callback" \
     -H "Content-Type: application/json" \
     -d '{"requestId": 12345, "targetLanguage": "EN"}'
   ```
   Should return `{"status":"received"}`.

2. **External test** (via dispatcher):
   ```bash
   curl -X POST "https://your-dispatcher-domain/etranslation/callback" \
     -H "Content-Type: application/json" \
     -d '{"requestId": 12345, "targetLanguage": "EN"}'
   ```
   Should return `{"status":"received"}`.

3. **End-to-end test:**
   - Trigger a translation task
   - Watch logs: `docker-compose logs -f geocoding-service`
   - You should see the `/askTranslate` request and later the callback being received

---

### Callback handling

The route `POST /etranslation/callback` in `web.py` handles:
- Success callbacks with `requestId` and `targetLanguage`
- Error callbacks with `errorCode` and `errorMessage`
- Stores results in memory keyed by `(requestId, targetLanguage)`
- Returns `{"status": "received"}`

The service waits up to `callback_wait_timeout` seconds (default 180) for callbacks before timing out.
