const base = localStorage.getItem('apiBase') || "http://localhost:8000";
let sessions = JSON.parse(localStorage.getItem('sessions')||"[]");
let currentSession = sessions.length ? sessions[0].id : createSessionId();
renderChatList();

function createSessionId(){ return "sess_" + Math.random().toString(36).slice(2,10); }

function newSession(){
  const id = createSessionId();
  sessions.unshift({id, messages:[]});
  saveSessions();
  currentSession = id;
  renderChatList();
  renderMessages();
}

function deleteSession(id){
  sessions = sessions.filter(s=>s.id!==id);
  if(currentSession===id){
    currentSession = sessions.length ? sessions[0].id : createSessionId();
    if(!sessions.length) sessions.push({id:currentSession, messages:[]});
  }
  saveSessions();
  renderChatList();
  renderMessages();
}

function saveSessions(){ localStorage.setItem('sessions', JSON.stringify(sessions)); }

function renderChatList(){
  const ul = document.getElementById('chatList');
  ul.innerHTML = '';
  sessions.forEach(s=>{
    const li = document.createElement('li');
    const title = document.createElement('span');
    title.className = "chat-title";
    title.textContent = s.messages[0]?.content?.slice(0,30) || "New Chat";
    title.onclick = ()=>{ currentSession=s.id; renderChatList(); renderMessages(); };

    const del = document.createElement('button');
    del.className="delete-btn";
    del.textContent="✕";
    del.onclick=(e)=>{ e.stopPropagation(); deleteSession(s.id); };

    li.className = (s.id===currentSession ? "active":"");
    li.appendChild(title);
    li.appendChild(del);
    ul.appendChild(li);
  });
}

function renderMessages(){
  const session = sessions.find(s=>s.id===currentSession);
  const box = document.getElementById('messages');
  box.innerHTML='';
  if(!session) return;
  session.messages.forEach(m=>{
    const div = document.createElement('div');
    div.className = "msg "+m.role;
    div.innerHTML = `<span>${m.content}</span>`;
    box.appendChild(div);
  });
  box.scrollTop = box.scrollHeight;
}

async function send(){
  const input = document.getElementById('input');
  const text = input.value.trim();
  if(!text) return;
  input.value = '';
  const session = sessions.find(s=>s.id===currentSession);
  session.messages.push({role:'user', content:text});
  renderMessages();
  saveSessions();

  const useAgent = document.getElementById('agentToggle').checked;
  const endpoint = useAgent ? "/agent/ask" : "/ask";
  const body = useAgent ? {question:text, session_id:currentSession} : {question:text, session_id:currentSession, top_k_ctx:8};

  const res = await fetch(base+endpoint, {method:"POST", headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await res.json();
  const reply = data.final || data.output || data.answer || JSON.stringify(data);

  session.messages.push({role:'assistant', content:reply});
  renderMessages();
  saveSessions();
}

// init
if(!sessions.length) newSession(); else renderMessages();
