// main.js — Faith AI frontend logic

const API = "http://localhost:8000";
const SESSION_ID = `session_${Date.now()}`;

const DENOMINATIONS = {
  catholic: "⛪ Catholic",
  orthodox: "🕍 Orthodox",
  protestant: "📖 Protestant",
};

const EXAMPLE_PROMPTS = [
  "Jesus walking on water at sunrise",
  "The Good Shepherd with his flock",
  "A dove descending with golden light",
  "The empty tomb on Easter morning",
  "A prayer candle in a stone chapel",
];

// ── DOM refs ──────────────────────────────────────────
const tabChat     = document.getElementById("tab-chat");
const tabImage    = document.getElementById("tab-image");
const panelChat   = document.getElementById("panel-chat");
const panelImage  = document.getElementById("panel-image");
const messagesEl  = document.getElementById("messages");
const chatInput   = document.getElementById("chat-input");
const sendBtn     = document.getElementById("send-btn");
const imageInput  = document.getElementById("image-input");
const generateBtn = document.getElementById("generate-btn");
const imageResult = document.getElementById("image-result");
const promptChips = document.getElementById("prompt-chips");

// ── Tab switching ─────────────────────────────────────
tabChat.addEventListener("click", () => {
  tabChat.classList.add("active");
  tabImage.classList.remove("active");
  panelChat.classList.remove("hidden");
  panelImage.classList.add("hidden");
});

tabImage.addEventListener("click", () => {
  tabImage.classList.add("active");
  tabChat.classList.remove("active");
  panelImage.classList.remove("hidden");
  panelChat.classList.add("hidden");
});

// ── Populate example prompts ──────────────────────────
EXAMPLE_PROMPTS.forEach((p) => {
  const btn = document.createElement("button");
  btn.className = "chip";
  btn.textContent = p;
  btn.addEventListener("click", () => { imageInput.value = p; });
  promptChips.appendChild(btn);
});

// ── Render a message bubble ───────────────────────────
function renderMessage({ role, content, citations, denomination, warning, blocked }) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = role === "assistant" ? "avatar" : "avatar user-avatar";
  avatar.textContent = role === "assistant" ? "✝" : "👤";

  const bubbleWrap = document.createElement("div");
  bubbleWrap.className = "bubble-wrap";

  const bubble = document.createElement("div");
  bubble.className = `bubble${blocked ? " blocked" : ""}`;

  const text = document.createElement("p");
  text.textContent = content;
  bubble.appendChild(text);

  // Hallucination warning
  if (warning) {
    const w = document.createElement("div");
    w.className = "warning";
    w.textContent = `⚠️ ${warning}`;
    bubble.appendChild(w);
  }

  // Citations
  if (citations?.length) {
    const cite = document.createElement("div");
    cite.className = "citations";
    const label = document.createElement("span");
    label.className = "citations-label";
    label.textContent = "📖 Grounded in:";
    cite.appendChild(label);
    citations.forEach((c) => {
      const tag = document.createElement("span");
      tag.className = "citation-tag";
      tag.textContent = c;
      cite.appendChild(tag);
    });
    bubble.appendChild(cite);
  }

  // Denomination badge
  if (denomination && denomination !== "general" && DENOMINATIONS[denomination]) {
    const badge = document.createElement("div");
    badge.className = "denomination-badge";
    badge.textContent = DENOMINATIONS[denomination];
    bubble.appendChild(badge);
  }

  bubbleWrap.appendChild(bubble);

  if (role === "assistant") {
    wrap.appendChild(avatar);
    wrap.appendChild(bubbleWrap);
  } else {
    wrap.appendChild(bubbleWrap);
    wrap.appendChild(avatar);
  }

  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Typing indicator ──────────────────────────────────
function showTyping() {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";
  wrap.id = "typing-indicator";
  wrap.innerHTML = `
    <div class="avatar">✝</div>
    <div class="bubble-wrap">
      <div class="bubble loading-bubble">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>
    </div>`;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideTyping() {
  document.getElementById("typing-indicator")?.remove();
}

// ── Send chat message ─────────────────────────────────
async function sendMessage() {
  const message = chatInput.value.trim();
  if (!message) return;

  renderMessage({ role: "user", content: message });
  chatInput.value = "";
  sendBtn.disabled = true;
  showTyping();

  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: SESSION_ID }),
    });

    const data = await res.json();

    if (!res.ok) {
      hideTyping();
      renderMessage({ role: "assistant", content: `⚠️ ${data.detail || "Something went wrong."}` });
      return;
    }

    hideTyping();
    renderMessage({
      role: "assistant",
      content: data.blocked ? `🚫 ${data.answer}` : data.answer,
      citations: data.citations || [],
      denomination: data.denomination,
      warning: data.hallucination_warning,
      blocked: data.blocked,
    });
  } catch (err) {
    hideTyping();
    renderMessage({ role: "assistant", content: "⚠️ Could not connect to server. Is the backend running?" });
  } finally {
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Generate image ─────────────────────────────────────
generateBtn.addEventListener("click", async () => {
  const prompt = imageInput.value.trim();
  if (!prompt) return;

  generateBtn.disabled = true;
  generateBtn.textContent = "✨ Generating...";
  imageResult.innerHTML = "";

  try {
    const res = await fetch(`${API}/image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, session_id: SESSION_ID }),
    });

    const data = await res.json();

    if (data.blocked) {
      imageResult.innerHTML = `
        <div class="image-error">
          🚫 <strong>Blocked:</strong> ${data.block_reason}
        </div>`;
    } else if (data.image_url) {
      imageResult.innerHTML = `
        <div class="image-output">
          <img src="${data.image_url}" alt="Generated Christian artwork" />
          <p class="image-prompt-used">Prompt: ${data.safe_prompt}</p>
        </div>`;
    } else {
      imageResult.innerHTML = `<div class="image-error">⚠️ No image returned.</div>`;
    }
  } catch (err) {
    imageResult.innerHTML = `<div class="image-error">⚠️ Could not connect to server.</div>`;
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "✨ Generate Image";
  }
});

// ── Initial greeting ──────────────────────────────────
renderMessage({
  role: "assistant",
  content: "Peace be with you. I am your Christian AI companion, grounded in Scripture. Ask me anything about the Bible, theology, prayer, or faith — or switch to the Images tab to generate Christian artwork.",
  citations: [],
});