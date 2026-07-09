let model = {};

fetch("model.json")
  .then(res => res.json())
  .then(data => model = data);

function sendMessage() {
  const input = document.getElementById("user-input");
  const text = input.value.toLowerCase();

  addMessage("You: " + text);

  const reply = getBestMatch(text);

  setTimeout(() => addMessage("Bot: " + reply), 300);

  input.value = "";
}

// 🧠 smarter matching system
function getBestMatch(input) {
  let bestScore = 0;
  let bestResponse = "I don't understand 🤔";

  for (let key in model) {
    let score = similarity(input, key);

    if (input.includes(key)) score += 0.5; // boost direct match

    if (score > bestScore) {
      bestScore = score;
      bestResponse = model[key];
    }
  }

  return bestResponse;
}

// simple similarity function
function similarity(a, b) {
  let match = 0;
  for (let word of b.split(" ")) {
    if (a.includes(word)) match++;
  }
  return match / b.split(" ").length;
}

function addMessage(text) {
  const box = document.getElementById("chat-box");
  const msg = document.createElement("div");
  msg.innerText = text;
  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;
}