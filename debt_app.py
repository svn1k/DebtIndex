import os, json, re, time, asyncio, threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app, origins="*")

OG_OK = False
llm_client = None
og = None
WORKING_MODEL = None
_ready = False
_init_done = False
_init_lock = threading.Lock()

MODEL_PRIORITY = [
    "CLAUDE_HAIKU_4_5",
    "CLAUDE_SONNET_4_5",
    "CLAUDE_SONNET_4_6",
    "GPT_5_MINI",
]

# ── Event loop ────────────────────────────────────────────────────────────────

_loop = None

def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()

def _ensure_loop():
    global _loop
    if _loop is None:
        t = threading.Thread(target=_start_loop, daemon=True)
        t.start()
        deadline = time.time() + 10
        while _loop is None and time.time() < deadline:
            time.sleep(0.05)

def _run(coro, timeout=120):
    _ensure_loop()
    if _loop is None:
        raise RuntimeError("Event loop not ready")
    async def _with_timeout():
        return await asyncio.wait_for(coro, timeout=timeout)
    return asyncio.run_coroutine_threadsafe(_with_timeout(), _loop).result(timeout=timeout + 5)

# ── OG init ───────────────────────────────────────────────────────────────────

def _init_og():
    global OG_OK, llm_client, og, _ready, _init_done, WORKING_MODEL
    with _init_lock:
        if _init_done:
            return
        _init_done = True
    try:
        import opengradient as _og
        import ssl, urllib3
        og = _og
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        private_key = os.environ.get("OG_PRIVATE_KEY", "")
        if not private_key:
            raise ValueError("OG_PRIVATE_KEY not set")
        print(f"OG_PRIVATE_KEY found: {private_key[:6]}...")
        llm_client = og.LLM(private_key=private_key)
        try:
            approval = llm_client.ensure_opg_approval(min_allowance=0.1)
            print(f"OPG approval: {approval}")
        except Exception as e:
            print(f"Approval warning (continuing): {e}")
        OG_OK = True
        print("OG connected — selecting model...")
        _pick_model()
    except Exception as e:
        import traceback
        print(f"OG init failed: {e}\n{traceback.format_exc()}")
    finally:
        _ready = True
        print(f"OG ready. OG_OK={OG_OK}, model={WORKING_MODEL}")

def _pick_model():
    global WORKING_MODEL
    if not OG_OK or llm_client is None:
        return
    for name in MODEL_PRIORITY:
        if not hasattr(og.TEE_LLM, name):
            continue
        model = getattr(og.TEE_LLM, name)
        try:
            print(f"  Trying {name}...")
            result = _run(llm_client.chat(
                model=model,
                messages=[{"role": "user", "content": "Say: OK"}],
                max_tokens=5,
                temperature=0.0,
            ), timeout=90)
            raw = _extract_raw(result)
            if raw and raw.strip():
                WORKING_MODEL = model
                print(f"✓ Model selected: {name}")
                return
        except Exception as e:
            print(f"  {name} failed: {e}")
    print("WARNING: No working model found")

def _ensure_og():
    if not _init_done:
        t = threading.Thread(target=_init_og, daemon=True)
        t.start()
        t.join(timeout=180)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_raw(result):
    if not result:
        return ""
    for attr in ['chat_output', 'completion_output', 'content', 'text', 'output']:
        val = getattr(result, attr, None)
        if val:
            if isinstance(val, dict) and val.get('content'):
                return str(val['content'])
            if isinstance(val, str) and val.strip():
                return val
    for attr in dir(result):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(result, attr)
            if callable(val):
                continue
            if isinstance(val, str) and val.strip() and len(val) > 2:
                return val
        except:
            pass
    return ""

def _parse_json(raw):
    if not raw or not raw.strip():
        return {"error": "Empty response"}
    m = re.search(r"<JSON>(.*?)</JSON>", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception as e:
            print(f"JSON parse error: {e}")
    m = re.search(r'\{[\s\S]*?"strategy"[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return {"error": "Parse failed", "raw": raw[:300]}

def call_llm(messages, retries=2):
    global WORKING_MODEL
    _ensure_og()
    if not OG_OK or llm_client is None:
        return {"error": "OpenGradient not available"}
    if WORKING_MODEL is None:
        return {"error": "No working LLM model found — OG testnet may be down"}
    last_error = ""
    for attempt in range(retries):
        try:
            print(f"LLM attempt {attempt+1} | model: {WORKING_MODEL}")
            result = _run(llm_client.chat(
                model=WORKING_MODEL,
                messages=messages,
                max_tokens=4000,
                temperature=0.2,
            ), timeout=120)
            raw = _extract_raw(result)
            print(f"Raw (200): {repr(raw[:200])}")
            if not raw.strip():
                last_error = "Empty response"
                time.sleep(2)
                continue
            parsed = _parse_json(raw)
            if "error" in parsed:
                last_error = parsed["error"]
                time.sleep(1)
                continue
            tx = getattr(result, "transaction_hash", None) or getattr(result, "payment_hash", None)
            if tx:
                parsed["proof"] = {
                    "transaction_hash": tx,
                    "explorer_url": f"https://explorer.opengradient.ai/tx/{tx}",
                }
            return parsed
        except (asyncio.TimeoutError, TimeoutError):
            last_error = "Model timeout"
            print(f"LLM timeout attempt {attempt+1}")
        except Exception as e:
            last_error = str(e)
            print(f"LLM error attempt {attempt+1}: {e}")
            time.sleep(3)
    return {"error": f"All attempts failed: {last_error}"}

SYSTEM_PROMPT = """You are an expert financial advisor and debt strategist. Analyze the provided debts and income, then reply ONLY with valid JSON inside <JSON>...</JSON> tags. No text outside the tags.

Return this exact structure:
<JSON>
{
  "strategy": "avalanche",
  "strategy_reason": "Why this strategy is optimal for this person",
  "monthly_payment_available": 500,
  "total_debt": 25000,
  "total_interest_avalanche": 3200,
  "total_interest_snowball": 3900,
  "months_to_freedom_avalanche": 52,
  "months_to_freedom_snowball": 54,
  "recommended_savings": 700,
  "summary": "2-3 sentence overall assessment and encouragement.",
  "payment_schedule": [
    {
      "debt_name": "Credit Card A",
      "balance": 5000,
      "rate": 22.99,
      "min_payment": 100,
      "recommended_payment": 300,
      "payoff_month": 18,
      "total_interest": 820,
      "priority": 1,
      "reason": "Highest interest rate — attack first"
    }
  ],
  "monthly_breakdown": [
    {"month": 1, "total_remaining": 25000, "payment": 500, "interest_paid": 120},
    {"month": 6, "total_remaining": 22100, "payment": 500, "interest_paid": 105},
    {"month": 12, "total_remaining": 18800, "payment": 500, "interest_paid": 88},
    {"month": 24, "total_remaining": 11200, "payment": 500, "interest_paid": 55},
    {"month": 36, "total_remaining": 3100, "payment": 500, "interest_paid": 18},
    {"month": 52, "total_remaining": 0, "payment": 200, "interest_paid": 0}
  ],
  "tips": [
    "Consider balance transfer to 0% APR card for highest-rate debt",
    "Build $1000 emergency fund before aggressive paydown"
  ]
}
</JSON>

Rules:
- strategy: "avalanche" (highest rate first) or "snowball" (lowest balance first)
- monthly_breakdown: 5-8 key milestone months showing debt reduction progress
- payment_schedule: ordered by priority (1 = pay first)
- All dollar amounts as numbers without $ signs
- Be realistic and mathematically accurate
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "og": OG_OK,
        "ready": _ready,
        "model": str(WORKING_MODEL) if WORKING_MODEL else None,
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    debts = data.get("debts", [])
    income = data.get("income", 0)
    monthly_payment = data.get("monthly_payment", 0)
    goal = data.get("goal", "balanced")

    if not debts:
        return jsonify({"error": "At least one debt is required"}), 400

    debt_text = "\n".join([
        f"- {d.get('name','Debt')}: ${d.get('balance',0)} balance, {d.get('rate',0)}% APR, ${d.get('min_payment',0)}/mo minimum"
        for d in debts
    ])
    total_debt = sum(d.get('balance', 0) for d in debts)

    user_msg = f"""Please analyze these debts and create an optimal repayment strategy:

DEBTS:
{debt_text}

FINANCIAL SITUATION:
- Monthly take-home income: ${income}
- Monthly payment available for debt: ${monthly_payment}
- Total debt: ${total_debt}
- Goal: {goal}

Create a detailed repayment plan with both avalanche and snowball comparison. Return the JSON."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ]

    print(f"\nAnalyzing | debts: {len(debts)} | income: ${income} | payment: ${monthly_payment}")
    return jsonify(call_llm(messages))

def _ping():
    time.sleep(120)
    import urllib.request
    while True:
        time.sleep(240)
        try:
            url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:10000")
            urllib.request.urlopen(f"{url}/health", timeout=10)
            print("Self-ping OK")
        except Exception as e:
            print(f"Self-ping failed: {e}")

threading.Thread(target=_ping, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
