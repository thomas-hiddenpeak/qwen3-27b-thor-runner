// ============================================================================
// API Base URL — auto-detect or configure
// ============================================================================
// If served from the engine's built-in static file server (port 8080),
// relative URLs work. If served from a separate file server (e.g. port 8888),
// we need to point to the engine's API port.
const API_PORT = 8080;
const API_HOST = location.hostname + ':' + API_PORT;
const API_BASE = (location.port == API_PORT) ? '' : 'http://' + API_HOST;
const WS_BASE = (location.port == API_PORT)
  ? ((location.protocol === 'https:') ? 'wss://' : 'ws://') + location.host
  : 'ws://' + API_HOST;

// ============================================================================
// State
// ============================================================================
let ws = null;
let isRecording = false;
let isProcessing = false;
let currentMode = 'text';
let streamingDiv = null;
let streamingText = '';
let responseT0 = 0;

// Streaming ASR
let audioContext = null;
let audioWorkletNode = null;
let mediaStream = null;
const STREAM_SAMPLE_RATE = 16000;

// Streaming TTS playback
let ttsPlayCtx = null;
let ttsNextTime = 0;
let ttsPreBuffer = [];
let ttsSampleRate = 24000;
let ttsStreamActive = false;
let wasRecording = false;
let ttsTotalSize = 0;
let ttsDone = false;
let hasConnectedBefore = false;  // 是否曾经连接过 (区分首次 vs 重连)

// Server-side recording (path reported by recording.saved event)
let lastRecordingPath = '';  // 最近一次录音文件路径
let recordingStartTime = 0;  // 录音开始时间戳 (ms)

// TTS model info
let ttsModelType = 'custom_voice';
let ttsAvailableVoices = ['serena'];
let ttsAvailableLanguages = [];
let ttsSpeakerDialects = {};
let ttsCloneVoices = [];
let hasAsr = false;

// Voice clone recording state
let cloneMediaStream = null;
let cloneMediaRecorder = null;
let cloneAudioChunks = [];
let cloneRecordInterval = null;
let cloneAudioBlob = null;
let cloneAudioFilename = 'audio.wav';

// ============================================================================
// Panel Toggle
// ============================================================================
function togglePanel(id) {
  const body = document.getElementById(id);
  const icon = document.getElementById(id + '_icon');
  if (!body) return;
  body.classList.toggle('collapsed');
  if (icon) icon.classList.toggle('collapsed');
}

function toggleSidebar() {
  document.getElementById('rightPanel').classList.toggle('hidden');
}

// ============================================================================
// ASR Monitor
// ============================================================================
let asrLogEntries = []; // {time, text, partial}
let asrPartialDiv = null;

function toggleAsrMonitor() {
  const body = document.getElementById('asrMonitorBody');
  const actions = document.getElementById('asrMonitorActions');
  const icon = document.getElementById('asrMonitor_icon');
  body.classList.toggle('collapsed');
  actions.classList.toggle('collapsed');
  if (icon) icon.classList.toggle('collapsed');
}

function addAsrLogEntry(text, isPartial, speaker) {
  if (!text) return;
  const list = document.getElementById('asrLogList');
  const empty = document.getElementById('asrLogEmpty');
  if (empty) empty.style.display = 'none';

  // 移除之前的 partial 条目
  if (asrPartialDiv) {
    asrPartialDiv.remove();
    asrPartialDiv = null;
  }

  const now = new Date();
  const timeStr = now.toLocaleTimeString('zh-CN', {hour12: false});

  const div = document.createElement('div');
  div.className = 'asr-log-entry' + (isPartial ? ' partial' : '');
  let html = '<span class="asr-time">[' + timeStr + ']</span>';
  if (speaker && !isPartial) {
    const isUnknown = speaker.startsWith('Speaker_');
    html += '<span class="asr-speaker' + (isUnknown ? ' unknown' : '') + '">' + escapeHtml(speaker) + '</span>';
  }
  html += '<span class="asr-text">' + escapeHtml(text) + '</span>';
  div.innerHTML = html;
  list.appendChild(div);

  if (isPartial) {
    asrPartialDiv = div;
  } else {
    asrLogEntries.push({time: timeStr, text: text, speaker: speaker || ''});
    updateAsrCount();
  }

  // 滚动到底部
  const body = document.getElementById('asrMonitorBody');
  body.scrollTop = body.scrollHeight;
}

function updateAsrPartial(text) {
  addAsrLogEntry(text, true);
}

function updateAsrCount() {
  const el = document.getElementById('asrMonitorCount');
  if (el) el.textContent = asrLogEntries.length > 0 ? '(' + asrLogEntries.length + '条)' : '';
}

function clearAsrLog() {
  asrLogEntries = [];
  asrPartialDiv = null;
  document.getElementById('asrLogList').innerHTML = '';
  document.getElementById('asrLogEmpty').style.display = 'block';
  updateAsrCount();
}

function saveAsrLog() {
  if (asrLogEntries.length === 0) return;
  let content = 'ASR 识别记录 — ' + new Date().toLocaleString('zh-CN') + '\n';
  content += '='.repeat(50) + '\n\n';
  for (const e of asrLogEntries) {
    const spkTag = e.speaker ? ' <' + e.speaker + '>' : '';
    content += '[' + e.time + ']' + spkTag + ' ' + e.text + '\n';
  }
  const blob = new Blob([content], {type: 'text/plain;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'asr_log_' + new Date().toISOString().slice(0,19).replace(/[:-]/g,'') + '.txt';
  a.click();
  URL.revokeObjectURL(url);
}

function onAsrToLlmChanged() {
  const enabled = document.getElementById('asrToLlmToggle').checked;
  // 通知后端
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({type: 'config', asr_to_llm: enabled}));
  }
}

// 下载服务端录音文件
function saveRecording() {
  if (!lastRecordingPath) {
    alert('还没有录音文件。请先录制并停止麦克风，或等待服务端保存完成。');
    return;
  }
  // 通过 HTTP 下载
  const a = document.createElement('a');
  a.href = API_BASE + '/v1/recordings/' + encodeURIComponent(lastRecordingPath.split('/').pop());
  a.download = lastRecordingPath.split('/').pop();
  a.click();
}

// 更新录音信息显示
function updateRecordingInfo() {
  const el = document.getElementById('asrLogInfo');
  if (!el) return;
  if (recordingStartTime > 0) {
    const durS = (Date.now() - recordingStartTime) / 1000;
    const min = Math.floor(durS / 60);
    const sec = Math.floor(durS % 60);
    el.textContent = (min > 0 ? min + 'm' : '') + sec + 's';
  } else {
    el.textContent = lastRecordingPath ? '✅ 已保存' : '';
  }
}

// ============================================================================
// TTS Info — 获取模型类型和可用音色
// ============================================================================
async function fetchTtsInfo() {
  try {
    const resp = await fetch(API_BASE + '/v1/tts/info');
    if (!resp.ok) return;
    const info = await resp.json();

    // ASR / Speaker encoder can be available even without TTS
    hasAsr = info.has_asr || false;
    const asrBtn = document.getElementById('cloneAsrBtn');
    if (asrBtn) {
      asrBtn.disabled = !hasAsr;
      asrBtn.title = hasAsr ? '自动识别参考音频文本' : 'ASR 未加载 (需配置 asr_enabled=true)';
    }
    const hasEncoder = info.has_speaker_encoder || false;
    document.getElementById('speakerMgmtSection').style.display = hasEncoder ? '' : 'none';
    if (hasEncoder) fetchSpeakerList();
    document.getElementById('fileTranscribeSection').style.display = hasAsr ? '' : 'none';

    if (!info.enabled) return;

    ttsModelType = info.model_type || 'custom_voice';
    ttsAvailableVoices = (info.available_voices && info.available_voices.length > 0)
        ? info.available_voices : [];
    if (info.available_languages && info.available_languages.length > 0) {
      ttsAvailableLanguages = info.available_languages;
    }
    if (info.speaker_dialects) {
      ttsSpeakerDialects = info.speaker_dialects;
    }
    if (info.sample_rate) ttsSampleRate = info.sample_rate;

    updateVoiceSelects();
    updateLangSelect();
    updateTtsLangSelect();

    const isDesign = ttsModelType === 'voice_design';
    const isClone = ttsModelType === 'voice_clone' || ttsModelType === 'base';

    // Voice Design section visibility
    document.getElementById('voiceDesignSection').style.display = isDesign ? '' : 'none';

    // VoiceDesign: load default instruct
    if (isDesign && info.default_instruct) {
      const instructEl = document.getElementById('ttsInstruct');
      if (instructEl && !instructEl.value) instructEl.value = info.default_instruct;
    }

    // Hide voice selects when no preset voices
    const hasVoices = ttsAvailableVoices.length > 0;
    // voiceSelectGroup visibility is managed by updateVoiceSelects()
    document.getElementById('synthVoiceGroup').style.display = hasVoices ? '' : 'none';

    // ASR clone button availability (uses hasAsr set above)
    // (already set at the top of fetchTtsInfo)

    // Voice clone section
    document.getElementById('voiceCloneSection').style.display = (isClone || hasEncoder) ? '' : 'none';

    // Clone voices
    if (info.clone_voices && info.clone_voices.length > 0) {
      ttsCloneVoices = info.clone_voices;
    }
    updateCloneVoiceSelect();

    // Show clone voice select in synthesis when clone voices exist
    document.getElementById('synthCloneGroup').style.display =
      (ttsCloneVoices.length > 0 || isClone || hasEncoder) ? '' : 'none';

    if (isClone) {
      setStatus('加载了克隆音色模型, 请先注册参考音频');
    }
  } catch (e) {
    console.warn('Failed to fetch TTS info:', e);
  }
}

function updateVoiceSelects() {
  const selects = [document.getElementById('voiceSelect'), document.getElementById('ttsVoiceSelect')];
  const isCloneModel = ttsModelType === 'voice_clone' || ttsModelType === 'base';
  for (const sel of selects) {
    const current = sel.value;
    sel.innerHTML = '';
    // Add preset voices
    for (const v of ttsAvailableVoices) {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = voiceDisplayNames[v] || (v.charAt(0).toUpperCase() + v.slice(1).replace(/_/g, ' '));
      sel.appendChild(opt);
    }
    // Add clone voices to chat voiceSelect (with separator)
    if (sel.id === 'voiceSelect' && isCloneModel && ttsCloneVoices.length > 0) {
      const grp = document.createElement('optgroup');
      grp.label = '── 克隆音色 ──';
      for (const v of ttsCloneVoices) {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = '🎙 ' + v;
        grp.appendChild(opt);
      }
      sel.appendChild(grp);
    }
    if (ttsAvailableVoices.includes(current) || ttsCloneVoices.includes(current)) sel.value = current;
  }
  // Show voice select group when any voice available
  const hasAnyVoice = ttsAvailableVoices.length > 0 || (isCloneModel && ttsCloneVoices.length > 0);
  document.getElementById('voiceSelectGroup').style.display = hasAnyVoice ? '' : 'none';
}

const voiceDisplayNames = {
  serena: 'Serena', vivian: 'Vivian', uncle_fu: 'Uncle Fu',
  ryan: 'Ryan', aiden: 'Aiden', ono_anna: 'Ono Anna', sohee: 'Sohee',
  eric: 'Eric (四川话)', dylan: 'Dylan (北京话)'
};

const langDisplayNames = {
  chinese: '中文', english: 'English', german: 'Deutsch', italian: 'Italiano',
  portuguese: 'Português', spanish: 'Español', japanese: '日本語', korean: '한국어',
  french: 'Français', russian: 'Русский', beijing_dialect: '北京话', sichuan_dialect: '四川话'
};

const dialectLanguages = new Set(['beijing_dialect', 'sichuan_dialect']);

function isCloneVoice(voice) {
  return !!voice && ttsCloneVoices.includes(voice);
}

function getDefaultLanguageForVoice(voice, langs) {
  const dialect = ttsSpeakerDialects[voice];
  if (dialect && langs.includes(dialect)) {
    return dialect;
  }
  if (isCloneVoice(voice) && langs.includes('chinese')) {
    return 'chinese';
  }
  return '';
}

function getFilteredLanguages(voice) {
  const dialect = ttsSpeakerDialects[voice];
  if (dialect) {
    return ttsAvailableLanguages.filter(l => l === dialect || !dialectLanguages.has(l));
  }
  return ttsAvailableLanguages.filter(l => !dialectLanguages.has(l));
}

function updateLangSelect(forceDefault = false) {
  const voice = document.getElementById('voiceSelect').value;
  const sel = document.getElementById('langSelect');
  const current = sel.value;
  const langs = getFilteredLanguages(voice);
  sel.innerHTML = '<option value="">自动</option>';
  for (const lang of langs) {
    const opt = document.createElement('option');
    opt.value = lang;
    opt.textContent = langDisplayNames[lang] || lang;
    sel.appendChild(opt);
  }
  if (!forceDefault && current && langs.includes(current)) {
    sel.value = current;
  } else {
    sel.value = getDefaultLanguageForVoice(voice, langs);
  }
}

function updateTtsLangSelect(forceDefault = false) {
  let voice = document.getElementById('ttsVoiceSelect').value;
  const cloneVoiceSel = document.getElementById('cloneVoiceSelect');
  const cloneVoice = cloneVoiceSel ? cloneVoiceSel.value : '';
  const isCloneModel = ttsModelType === 'voice_clone' || ttsModelType === 'base';
  if (isCloneModel && cloneVoice) {
    voice = cloneVoice;
  }
  const sel = document.getElementById('ttsLangSelect');
  const current = sel.value;
  const langs = getFilteredLanguages(voice);
  sel.innerHTML = '<option value="">自动</option>';
  for (const lang of langs) {
    const opt = document.createElement('option');
    opt.value = lang;
    opt.textContent = langDisplayNames[lang] || lang;
    sel.appendChild(opt);
  }
  if (!forceDefault && current && langs.includes(current)) {
    sel.value = current;
  } else {
    sel.value = getDefaultLanguageForVoice(voice, langs);
  }
}

function onVoiceChanged() {
  const voice = document.getElementById('voiceSelect').value;
  const dialect = ttsSpeakerDialects[voice];
  if (dialect) {
    document.getElementById('langSelect').value = dialect;
  } else if (isCloneVoice(voice)) {
    document.getElementById('langSelect').value = 'chinese';
  }
  updateLangSelect(true);
}

// ============================================================================
// WebSocket
// ============================================================================
let wsReconnectTimer = null;
let wsReconnectDelay = 1000;  // 起始 1s, 最大 30s
const WS_RECONNECT_MAX = 30000;

function scheduleReconnect() {
  if (wsReconnectTimer) return;  // 防重复
  setStatus('连接断开，' + (wsReconnectDelay / 1000).toFixed(0) + 's 后重连...', 'error');
  wsReconnectTimer = setTimeout(() => {
    wsReconnectTimer = null;
    connectWS();
  }, wsReconnectDelay);
  wsReconnectDelay = Math.min(wsReconnectDelay * 2, WS_RECONNECT_MAX);
}

function connectWS() {
  // 清理旧连接
  if (ws) {
    try { ws.onclose = null; ws.onerror = null; ws.close(); } catch(e) {}
    ws = null;
  }

  const url = WS_BASE + '/v1/voice';
  ws = new WebSocket(url);

  ws.onopen = () => {
    // 连接成功: 重置退避
    wsReconnectDelay = 1000;
    if (wsReconnectTimer) { clearTimeout(wsReconnectTimer); wsReconnectTimer = null; }

    document.getElementById('wsDot').classList.add('connected');
    document.getElementById('wsLabel').textContent = '已连接';
    fetchTtsInfo();
    sendConfig();
    if (hasConnectedBefore) {
      setStatus('连接已恢复');
      setTimeout(clearStatus, 2000);
    }
    hasConnectedBefore = true;
    // 如果重连前麦克风是开的，自动恢复流式录音
    if (wasRecording) {
      wasRecording = false;
      setTimeout(() => startStreaming(), 500);
    }
  };

  ws.onclose = () => {
    document.getElementById('wsDot').classList.remove('connected');
    document.getElementById('wsLabel').textContent = '已断开';
    isProcessing = false;
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('stopTtsBtn').style.display = 'none';
    if (streamingDiv) {
      const contentEl = streamingDiv.querySelector('.content');
      if (contentEl) contentEl.classList.remove('streaming-cursor');
      streamingDiv = null;
    }
    ttsStreamActive = false;
    if (ttsPlayCtx) { ttsPlayCtx.close(); ttsPlayCtx = null; }
    ttsPreBuffer = [];
    // 保留重连前的录音状态
    wasRecording = isRecording;
    stopStreaming(false);
    clearStatus();
    scheduleReconnect();
  };

  ws.onerror = () => {
    // onerror 后一定会触发 onclose, 不需要重复处理
  };

  ws.onmessage = (e) => {
    if (e.data instanceof Blob) {
      handleTtsBinary(e.data);
      return;
    }
    const msg = JSON.parse(e.data);
    handleEvent(msg);
  };
}

function sendConfig() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const voice = document.getElementById('voiceSelect').value;
  const tts = document.getElementById('ttsToggle').checked;
  const msg = {
    type: 'config', voice, tts,
    asr_to_llm: document.getElementById('asrToLlmToggle').checked,
    tts_language: document.getElementById('langSelect').value,
    tts_temperature: parseFloat(document.getElementById('ttsTemp').value),
    tts_top_k: parseInt(document.getElementById('ttsTopK').value),
    tts_top_p: parseFloat(document.getElementById('ttsTopP').value),
    tts_rep_penalty: parseFloat(document.getElementById('ttsRepPenalty').value),
    voice_max_turns: parseInt(document.getElementById('voiceMaxTurns').value),
    voice_max_output_tokens: parseInt(document.getElementById('voiceMaxOutputTokens').value)
  };
  const instruct = document.getElementById('ttsInstruct').value.trim();
  if (ttsModelType === 'voice_design') msg.tts_instruct = instruct;
  else if (instruct) msg.tts_instruct = instruct;
  ws.send(JSON.stringify(msg));
}

function resetTtsDefaults() {
  document.getElementById('ttsTemp').value = 0.9;
  document.getElementById('ttsTempVal').textContent = '0.9';
  document.getElementById('ttsTopK').value = 50;
  document.getElementById('ttsTopKVal').textContent = '50';
  document.getElementById('ttsTopP').value = 1.0;
  document.getElementById('ttsTopPVal').textContent = '1.0';
  document.getElementById('ttsRepPenalty').value = 1.05;
  document.getElementById('ttsRepVal').textContent = '1.05';
  document.getElementById('voiceMaxTurns').value = defaultVoiceMaxTurns;
  document.getElementById('voiceMaxTurnsVal').textContent = defaultVoiceMaxTurns;
  document.getElementById('voiceMaxOutputTokens').value = defaultVoiceMaxOutputTokens;
  document.getElementById('voiceMaxOutputTokensVal').textContent = defaultVoiceMaxOutputTokens;
  sendConfig();
}

let defaultSystemPrompt = '';
let defaultVoiceMaxTurns = 10;
let defaultVoiceMaxOutputTokens = 150;

function applySystemPrompt() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const text = document.getElementById('systemPromptEditor').value;
  ws.send(JSON.stringify({ type: 'config', system_prompt: text }));
}

function resetSystemPrompt() {
  document.getElementById('systemPromptEditor').value = defaultSystemPrompt;
  applySystemPrompt();
}

// ============================================================================
// TTS Binary Handling (Voice Chat)
// ============================================================================
function handleTtsBinary(blob) {
  ttsTotalSize += blob.size;

  if (ttsStreamActive) {
    blob.arrayBuffer().then(buf => {
      const pcm16 = new Int16Array(buf);
      if (pcm16.length === 0) return;

      const float32 = new Float32Array(pcm16.length);
      for (let i = 0; i < pcm16.length; i++) {
        float32[i] = pcm16[i] / 32768.0;
      }

      if (!ttsPlayCtx) {
        ttsPreBuffer.push(float32);
        let totalSamples = 0;
        for (const c of ttsPreBuffer) totalSamples += c.length;
        const totalMs = (totalSamples / ttsSampleRate) * 1000;
        if (totalMs < 600) return;

        ttsPlayCtx = new AudioContext({ sampleRate: ttsSampleRate });
        ttsNextTime = ttsPlayCtx.currentTime + 0.05;
        for (const chunk of ttsPreBuffer) {
          const ab = ttsPlayCtx.createBuffer(1, chunk.length, ttsSampleRate);
          ab.copyToChannel(chunk, 0);
          const src = ttsPlayCtx.createBufferSource();
          src.buffer = ab;
          src.connect(ttsPlayCtx.destination);
          src.start(ttsNextTime);
          ttsNextTime += ab.duration;
        }
        ttsPreBuffer = [];
        return;
      }

      const audioBuf = ttsPlayCtx.createBuffer(1, float32.length, ttsSampleRate);
      audioBuf.copyToChannel(float32, 0);
      const source = ttsPlayCtx.createBufferSource();
      source.buffer = audioBuf;
      source.connect(ttsPlayCtx.destination);

      const now = ttsPlayCtx.currentTime;
      if (ttsNextTime < now) ttsNextTime = now + 0.15;
      source.start(ttsNextTime);
      ttsNextTime += audioBuf.duration;
    });
  } else {
    const audioUrl = URL.createObjectURL(blob);
    if (streamingDiv) {
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.style.cssText = 'display:block;margin-top:8px;width:100%;height:36px;border-radius:6px;';
      audio.src = audioUrl;
      streamingDiv.appendChild(audio);
      audio.play().catch(() => {});
      scrollToBottom();
    }
  }
}

function waitForTtsPlaybackEnd(callback) {
  if (!ttsPlayCtx || ttsNextTime <= ttsPlayCtx.currentTime) {
    callback();
    return;
  }
  const remaining = (ttsNextTime - ttsPlayCtx.currentTime) * 1000;
  setTimeout(callback, remaining + 200);
}

function stopTts() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'interrupt' }));
  }
  ttsStreamActive = false;
  ttsPreBuffer = [];
  if (ttsPlayCtx) {
    ttsPlayCtx.close().catch(() => {});
    ttsPlayCtx = null;
  }
  ttsNextTime = 0;
  document.getElementById('stopTtsBtn').style.display = 'none';
  isProcessing = false;
  document.getElementById('sendBtn').disabled = false;
  if (streamingDiv) {
    const contentEl = streamingDiv.querySelector('.content');
    if (contentEl) contentEl.classList.remove('streaming-cursor');
    streamingDiv = null;
  }
  clearStatus();
}

// ============================================================================
// Event Handling (Voice Chat)
// ============================================================================
function handleEvent(msg) {
  switch (msg.type) {
    case 'session.created':
      // 确保重连后所有 stale 状态被清理
      isProcessing = false;
      document.getElementById('sendBtn').disabled = false;
      document.getElementById('stopTtsBtn').style.display = 'none';
      if (streamingDiv) {
        const contentEl = streamingDiv.querySelector('.content');
        if (contentEl) contentEl.classList.remove('streaming-cursor');
        streamingDiv = null;
      }
      clearStatus();
      break;

    case 'config.updated':
      if (msg.system_prompt !== undefined) {
        const editor = document.getElementById('systemPromptEditor');
        if (!editor.value) editor.value = msg.system_prompt;
        if (!defaultSystemPrompt) defaultSystemPrompt = msg.system_prompt;
      }
      if (msg.voice_max_turns !== undefined) {
        if (!defaultVoiceMaxTurns || defaultVoiceMaxTurns === 10) defaultVoiceMaxTurns = msg.voice_max_turns;
        document.getElementById('voiceMaxTurns').value = msg.voice_max_turns;
        document.getElementById('voiceMaxTurnsVal').textContent = msg.voice_max_turns;
      }
      if (msg.voice_max_output_tokens !== undefined) {
        if (!defaultVoiceMaxOutputTokens || defaultVoiceMaxOutputTokens === 150) defaultVoiceMaxOutputTokens = msg.voice_max_output_tokens;
        document.getElementById('voiceMaxOutputTokens').value = msg.voice_max_output_tokens;
        document.getElementById('voiceMaxOutputTokensVal').textContent = msg.voice_max_output_tokens;
      }
      break;

    case 'status':
      if (msg.stage === 'asr') setStatus('正在转录语音...');
      else if (msg.stage === 'tts') setStatus('正在合成语音...');
      break;

    case 'stream.started':
      setStatus('🎤 正在听...');
      break;

    case 'stream.vad':
      setStatus('语音结束，正在识别...');
      // 麦克风保持开启, 不停止流式录制
      break;

    case 'stream.stopped':
      if (isProcessing && !streamingDiv) {
        isProcessing = false;
        document.getElementById('sendBtn').disabled = false;
        clearStatus();
      }
      recordingStartTime = 0;
      updateRecordingInfo();
      break;

    case 'recording.saved':
      lastRecordingPath = msg.path || '';
      recordingStartTime = 0;
      updateRecordingInfo();
      break;

    case 'audio.level':
      updateAudioLevel(msg.rms || 0);
      break;

    case 'asr.partial':
      // 流式 ASR 中间结果: 显示在状态栏 + ASR监控面板
      if (msg.text) {
        setStatus('🗣️ ' + msg.text);
        updateAsrPartial(msg.text);
      }
      break;

    case 'asr':
      addAsrLogEntry(msg.text, false, msg.speaker);
      if (document.getElementById('asrToLlmToggle').checked) {
        const meta = msg.speaker ? 'ASR 识别 · ' + msg.speaker : 'ASR 识别';
        addMessage('user', msg.text, null, meta);
      }
      break;

    case 'asr.done':
      // ASR-only 模式完成 (LLM 跳过): 重置 UI 状态
      clearStatus();
      streamingDiv = null;
      isProcessing = false;
      document.getElementById('sendBtn').disabled = false;
      // 麦克风持续开启, 显示“正在听...”
      if (isRecording) setStatus('🎤 正在听...');
      break;

    case 'tts.stream_start':
      ttsStreamActive = true;
      ttsSampleRate = msg.sample_rate || 24000;
      ttsNextTime = 0;
      ttsPreBuffer = [];
      if (ttsPlayCtx) { ttsPlayCtx.close(); ttsPlayCtx = null; }
      document.getElementById('stopTtsBtn').style.display = 'inline-block';
      break;

    case 'llm.start':
      responseT0 = performance.now();
      streamingText = '';
      streamingDiv = createStreamingMessage();
      ttsTotalSize = 0;
      ttsDone = false;
      setStatus('正在生成回复...');
      document.getElementById('stopTtsBtn').style.display = 'inline-block';
      break;

    case 'llm.delta':
      if (streamingDiv) {
        streamingText += msg.delta;
        const contentEl = streamingDiv.querySelector('.content');
        contentEl.textContent = streamingText;
        contentEl.classList.add('streaming-cursor');
        scrollToBottom();
      }
      break;

    case 'llm.done':
      if (streamingDiv) {
        const contentEl = streamingDiv.querySelector('.content');
        contentEl.classList.remove('streaming-cursor');
        const llmTime = ((performance.now() - responseT0) / 1000).toFixed(1);
        let metaText = `LLM ${llmTime}s`;
        if (msg.prompt_tokens) metaText += ` · ${msg.prompt_tokens} prompt`;
        if (msg.completion_tokens) metaText += ` · ${msg.completion_tokens} gen`;
        streamingDiv.dataset.llmMeta = metaText;
        let metaEl = streamingDiv.querySelector('.meta');
        if (!metaEl) {
          metaEl = document.createElement('div');
          metaEl.className = 'meta';
          streamingDiv.appendChild(metaEl);
        }
        metaEl.textContent = metaText;
      }
      clearStatus();
      // TTS 未激活时, LLM 完成即全部完成
      if (!ttsStreamActive) {
        streamingDiv = null;
        isProcessing = false;
        document.getElementById('sendBtn').disabled = false;
        if (isRecording) setStatus('🎤 正在听...');
      }
      break;

    case 'tts.meta':
      break;

    case 'tts.done':
      ttsDone = true;
      ttsStreamActive = false;
      document.getElementById('stopTtsBtn').style.display = 'none';
      if (!ttsPlayCtx && ttsPreBuffer.length > 0) {
        ttsPlayCtx = new AudioContext({ sampleRate: ttsSampleRate });
        ttsNextTime = ttsPlayCtx.currentTime + 0.05;
        for (const chunk of ttsPreBuffer) {
          const ab = ttsPlayCtx.createBuffer(1, chunk.length, ttsSampleRate);
          ab.copyToChannel(chunk, 0);
          const src = ttsPlayCtx.createBufferSource();
          src.buffer = ab;
          src.connect(ttsPlayCtx.destination);
          src.start(ttsNextTime);
          ttsNextTime += ab.duration;
        }
        ttsPreBuffer = [];
      }
      if (streamingDiv) {
        const metaEl = streamingDiv.querySelector('.meta');
        if (metaEl) {
          const sizeKB = (ttsTotalSize / 1024).toFixed(0);
          metaEl.textContent = (streamingDiv.dataset.llmMeta || '') +
            ` · TTS ${sizeKB}KB (${msg.segments || 1}段)`;
        }
      }
      clearStatus();
      streamingDiv = null;
      waitForTtsPlaybackEnd(() => {
        isProcessing = false;
        document.getElementById('sendBtn').disabled = false;
        if (isRecording) setStatus('🎤 正在听...');
      });
      break;

    case 'error':
      setStatus(msg.message || '发生错误', 'error');
      isProcessing = false;
      document.getElementById('sendBtn').disabled = false;
      document.getElementById('stopTtsBtn').style.display = 'none';
      if (streamingDiv) {
        const contentEl = streamingDiv.querySelector('.content');
        contentEl.classList.remove('streaming-cursor');
        streamingDiv = null;
      }
      break;

    case 'history.cleared':
      break;
  }

  if (msg.type === 'llm.done' && !ttsStreamActive) {
    const ttsEnabled = document.getElementById('ttsToggle').checked;
    if (!ttsEnabled) {
      isProcessing = false;
      document.getElementById('sendBtn').disabled = false;
      document.getElementById('stopTtsBtn').style.display = 'none';
      streamingDiv = null;
      if (isRecording) setStatus('🎤 正在听...');
    }
  }
}

// ============================================================================
// UI Helpers
// ============================================================================
function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-toggle button').forEach(b => b.classList.remove('active'));
  document.getElementById('mode' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
  sendConfig();
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function setStatus(msg, type) {
  const bar = document.getElementById('statusBar');
  bar.textContent = msg;
  bar.className = 'status-bar visible' + (type ? ' ' + type : '');
}

function clearStatus() {
  document.getElementById('statusBar').className = 'status-bar';
}

function scrollToBottom() {
  const chatArea = document.getElementById('chatArea');
  chatArea.scrollTop = chatArea.scrollHeight;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function formatTimestamp(seconds) {
  if (seconds == null) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m + ':' + s.toString().padStart(2, '0');
}

function addMessage(role, content, audioUrl, meta) {
  const chatArea = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'message ' + role;
  let html = '<div class="role-tag">' + (role === 'user' ? 'You' : 'Assistant') + '</div>';
  html += '<div class="content">' + escapeHtml(content) + '</div>';
  if (audioUrl) {
    html += '<audio controls src="' + escapeHtml(audioUrl) + '" style="display:block;margin-top:8px;width:100%;height:36px;border-radius:6px;"></audio>';
  }
  if (meta) {
    html += '<div class="meta">' + escapeHtml(meta) + '</div>';
  }
  div.innerHTML = html;
  chatArea.appendChild(div);
  scrollToBottom();
}

function createStreamingMessage() {
  const chatArea = document.getElementById('chatArea');
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = '<div class="role-tag">Assistant</div><div class="content streaming-cursor"></div>';
  chatArea.appendChild(div);
  scrollToBottom();
  return div;
}

// ============================================================================
// Send Text Message
// ============================================================================
function sendMessage() {
  const input = document.getElementById('inputText');
  const text = input.value.trim();
  if (!text || isProcessing || !ws || ws.readyState !== WebSocket.OPEN) return;

  input.value = '';
  autoResize(input);

  addMessage('user', text);
  isProcessing = true;
  document.getElementById('sendBtn').disabled = true;

  ws.send(JSON.stringify({ type: 'chat', text }));
}

// ============================================================================
// Streaming ASR
// ============================================================================
async function toggleRecording() {
  // 允许随时停止麦克风 (即使在生成中)
  if (isRecording) {
    stopStreaming(true);
    return;
  }
  if (isProcessing) return;
  await startStreaming();
}

async function startStreaming() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    setStatus('WebSocket 未连接', 'error');
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: STREAM_SAMPLE_RATE, channelCount: 1,
               echoCancellation: true, noiseSuppression: true }
    });

    audioContext = new AudioContext({ sampleRate: STREAM_SAMPLE_RATE });
    const source = audioContext.createMediaStreamSource(mediaStream);

    const workletCode = `
      class PcmProcessor extends AudioWorkletProcessor {
        process(inputs) {
          const ch = inputs[0] && inputs[0][0];
          if (ch && ch.length > 0) {
            this.port.postMessage(ch);
          }
          return true;
        }
      }
      registerProcessor('pcm-processor', PcmProcessor);
    `;
    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const moduleUrl = URL.createObjectURL(blob);
    await audioContext.audioWorklet.addModule(moduleUrl);
    URL.revokeObjectURL(moduleUrl);

    audioWorkletNode = new AudioWorkletNode(audioContext, 'pcm-processor');
    audioWorkletNode.port.onmessage = (e) => {
      if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
      const float32 = e.data;
      const pcm16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      // Path: 发送到服务端做实时 ASR (服务端同时累积录音)
      ws.send(pcm16.buffer);
    };

    source.connect(audioWorkletNode);
    audioWorkletNode.connect(audioContext.destination);

    const actualRate = audioContext.sampleRate;
    ws.send(JSON.stringify({ type: 'stream.start', sample_rate: actualRate }));

    recordingStartTime = Date.now();
    lastRecordingPath = '';

    isRecording = true;
    document.getElementById('micBtn').classList.add('recording');
    document.getElementById('recordingStatus').textContent = '🔴 实时语音流...';
    updateRecordingInfo();

  } catch (err) {
    setStatus('无法访问麦克风: ' + err.message, 'error');
  }
}

function stopStreaming(sendStop) {
  if (!isRecording && !audioContext) return;

  isRecording = false;
  document.getElementById('micBtn').classList.remove('recording');
  document.getElementById('recordingStatus').textContent = '';

  if (audioWorkletNode) { audioWorkletNode.disconnect(); audioWorkletNode = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }

  if (sendStop && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'stream.stop' }));
  }
}

function updateAudioLevel(rms) {
  const statusEl = document.getElementById('recordingStatus');
  if (isRecording) {
    const bars = Math.min(5, Math.floor(rms * 100));
    const indicator = '█'.repeat(bars) + '░'.repeat(5 - bars);
    const durS = recordingStartTime > 0 ? (Date.now() - recordingStartTime) / 1000 : 0;
    const min = Math.floor(durS / 60);
    const sec = Math.floor(durS % 60);
    const timeStr = min > 0 ? min + ':' + String(sec).padStart(2, '0') : sec + 's';
    statusEl.textContent = '🔴 ' + indicator + ' ' + timeStr;
    updateRecordingInfo();
  }
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================
document.getElementById('inputText').addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    sendMessage();
  }
});

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && document.activeElement.tagName !== 'TEXTAREA'
      && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    toggleRecording();
  }
});

// ============================================================================
// TTS Synthesis (REST API)
// ============================================================================
let ttsCurrentBlob = null;
let ttsCurrentUrl = null;
let ttsHistory = [];

document.querySelectorAll('.preset').forEach(el => {
  el.addEventListener('click', () => {
    document.getElementById('ttsInputText').value = el.dataset.text;
  });
});

function setTtsStatus(msg, type) {
  const el = document.getElementById('ttsStatus');
  el.textContent = msg;
  el.className = 'tts-status visible' + (type ? ' ' + type : '');
}

function clearTtsStatus() {
  document.getElementById('ttsStatus').className = 'tts-status';
}

async function synthesizeTts() {
  const text = document.getElementById('ttsInputText').value.trim();
  if (!text) { setTtsStatus('请输入要合成的文本', 'error'); return; }

  // Determine voice: clone voice takes priority
  let voice = '';
  const cloneVoiceSel = document.getElementById('cloneVoiceSelect');
  const cloneVoice = cloneVoiceSel ? cloneVoiceSel.value : '';
  const isClone = ttsModelType === 'voice_clone' || ttsModelType === 'base';
  if (isClone && cloneVoice) {
    voice = cloneVoice;
  } else if (ttsModelType !== 'voice_design') {
    voice = document.getElementById('ttsVoiceSelect').value;
  }

  const format = document.getElementById('formatSelect').value;
  const instruct = document.getElementById('ttsInstruct')?.value.trim() || undefined;

  const btn = document.getElementById('synthesizeBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> 合成中...';

  document.getElementById('ttsPlayer').className = 'player';
  setTtsStatus('正在合成语音，请稍候...');

  const t0 = performance.now();

  try {
    const response = await fetch(API_BASE + '/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'qwen3-tts',
        input: text,
        voice: voice,
        response_format: format,
        speed: 1.0,
        language: document.getElementById('ttsLangSelect').value || undefined,
        instruct: instruct
      })
    });

    if (!response.ok) {
      let errMsg = 'HTTP ' + response.status;
      try { const errJson = await response.json(); errMsg = errJson.error?.message || errMsg; } catch {}
      throw new Error(errMsg);
    }

    const blob = await response.blob();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    if (ttsCurrentUrl) URL.revokeObjectURL(ttsCurrentUrl);

    ttsCurrentBlob = blob;
    if (format === 'pcm') {
      ttsCurrentBlob = await wrapPcmAsWav(blob, 24000);
    }

    ttsCurrentUrl = URL.createObjectURL(ttsCurrentBlob);

    const audioPlayer = document.getElementById('ttsAudioPlayer');
    audioPlayer.src = ttsCurrentUrl;

    document.getElementById('ttsPlayer').className = 'player visible';

    const sizeKB = (blob.size / 1024).toFixed(1);
    document.getElementById('ttsAudioInfo').textContent =
      sizeKB + ' KB · ' + elapsed + 's · ' + (voice || 'default');

    setTtsStatus('合成完成 (' + elapsed + 's)', 'success');
    try { await audioPlayer.play(); } catch {}

    addTtsHistory(text, voice, ttsCurrentUrl, elapsed, sizeKB);

  } catch (err) {
    setTtsStatus('合成失败: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '合成语音';
  }
}

async function wrapPcmAsWav(blob, sampleRate) {
  const pcmData = await blob.arrayBuffer();
  const wavBuffer = new ArrayBuffer(44 + pcmData.byteLength);
  const view = new DataView(wavBuffer);
  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + pcmData.byteLength, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, pcmData.byteLength, true);
  new Uint8Array(wavBuffer, 44).set(new Uint8Array(pcmData));
  return new Blob([wavBuffer], { type: 'audio/wav' });
}

function downloadAudio() {
  if (!ttsCurrentBlob) return;
  const format = document.getElementById('formatSelect').value;
  const a = document.createElement('a');
  a.href = ttsCurrentUrl;
  a.download = 'speech.' + (format === 'pcm' ? 'wav' : format);
  a.click();
}

function addTtsHistory(text, voice, audioUrl, elapsed, sizeKB) {
  ttsHistory.unshift({ text, voice, audioUrl, elapsed, sizeKB });
  if (ttsHistory.length > 20) ttsHistory.pop();
  renderTtsHistory();
}

function renderTtsHistory() {
  const card = document.getElementById('ttsHistoryCard');
  const list = document.getElementById('ttsHistoryList');
  if (ttsHistory.length === 0) { card.style.display = 'none'; return; }
  card.style.display = '';
  list.innerHTML = ttsHistory.map((item, idx) =>
    '<div class="history-item" onclick="playTtsHistory(' + idx + ')">' +
    '<div class="play-icon"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg></div>' +
    '<div class="text">' + escapeHtml(item.text) + '</div>' +
    '<div class="item-meta">' + (item.voice || 'default') + ' · ' + item.elapsed + 's</div></div>'
  ).join('');
}

function playTtsHistory(idx) {
  const item = ttsHistory[idx];
  if (!item?.audioUrl) return;
  const player = document.getElementById('ttsAudioPlayer');
  player.src = item.audioUrl;
  document.getElementById('ttsPlayer').className = 'player visible';
  document.getElementById('ttsAudioInfo').textContent =
    item.sizeKB + ' KB · ' + item.elapsed + 's · ' + (item.voice || 'default');
  player.play();
}

// ============================================================================
// Voice Clone Functions
// ============================================================================
function updateCloneVoiceSelect() {
  const sel = document.getElementById('cloneVoiceSelect');
  if (!sel) return;
  const current = sel.value;
  sel.innerHTML = '';
  if (ttsCloneVoices.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(请先注册音色)';
    sel.appendChild(opt);
  } else {
    for (const v of ttsCloneVoices) {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      sel.appendChild(opt);
    }
    if (ttsCloneVoices.includes(current)) sel.value = current;
  }
  // Show/hide clone voice group in synthesis section
  const group = document.getElementById('synthCloneGroup');
  if (group) {
    group.style.display = ttsCloneVoices.length > 0 ? '' : 'none';
  }
  // Also update chat voiceSelect to include clone voices
  updateVoiceSelects();
  updateTtsLangSelect();
}

function onCloneFileSelected(input) {
  const file = input.files[0];
  if (file) {
    document.getElementById('cloneFileInfo').textContent =
      '已选择: ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    cloneAudioBlob = file;
    cloneAudioFilename = file.name || 'audio.wav';
  }
}

async function toggleCloneRecord() {
  const btn = document.getElementById('cloneRecordBtn');
  const statusDiv = document.getElementById('cloneRecordingStatus');

  if (cloneMediaRecorder && cloneMediaRecorder.state === 'recording') {
    cloneMediaRecorder.stop();
    btn.textContent = '🎙 录音';
    btn.style.background = '';
    statusDiv.style.display = 'none';
    if (cloneRecordInterval) { clearInterval(cloneRecordInterval); cloneRecordInterval = null; }
    if (cloneMediaStream) { cloneMediaStream.getTracks().forEach(t => t.stop()); cloneMediaStream = null; }
    return;
  }

  try {
    cloneMediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    cloneMediaRecorder = new MediaRecorder(cloneMediaStream, { mimeType: 'audio/webm' });
    cloneAudioChunks = [];

    cloneMediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) cloneAudioChunks.push(e.data);
    };

    cloneMediaRecorder.onstop = async () => {
      const webmBlob = new Blob(cloneAudioChunks, { type: 'audio/webm' });
      const ctx = new AudioContext({ sampleRate: 24000 });
      const arrayBuf = await webmBlob.arrayBuffer();
      const decoded = await ctx.decodeAudioData(arrayBuf);
      const channelData = decoded.getChannelData(0);
      const wavBuf = new ArrayBuffer(44 + channelData.length * 2);
      const view = new DataView(wavBuf);
      const sr = decoded.sampleRate;
      const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
      writeStr(0, 'RIFF'); view.setUint32(4, 36 + channelData.length * 2, true); writeStr(8, 'WAVE');
      writeStr(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
      view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
      writeStr(36, 'data'); view.setUint32(40, channelData.length * 2, true);
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(44 + i * 2, s * 32767, true);
      }
      cloneAudioBlob = new Blob([wavBuf], { type: 'audio/wav' });
      document.getElementById('cloneFileInfo').textContent =
        '录音完成: ' + (cloneAudioBlob.size / 1024).toFixed(1) + ' KB, ' +
        decoded.duration.toFixed(1) + 's';
      ctx.close();
    };

    cloneMediaRecorder.start();
    btn.textContent = '⏹ 停止录音';
    btn.style.background = 'var(--recording)';
    statusDiv.style.display = 'block';
    let sec = 0;
    cloneRecordInterval = setInterval(() => {
      sec++;
      document.getElementById('cloneRecordTime').textContent = sec + 's';
    }, 1000);
  } catch (err) {
    document.getElementById('cloneRegStatus').textContent = '麦克风权限被拒绝: ' + err.message;
    document.getElementById('cloneRegStatus').style.color = 'var(--error)';
  }
}

async function recognizeCloneAudio() {
  const statusEl = document.getElementById('cloneRefTextStatus');

  if (!hasAsr) {
    statusEl.textContent = 'ASR 语音识别未加载。请在启动配置中添加 asr_enabled=true 和 asr_model 参数';
    statusEl.style.color = 'var(--error)';
    return;
  }

  if (!cloneAudioBlob) {
    statusEl.textContent = '请先上传或录制参考音频';
    statusEl.style.color = 'var(--error)';
    return;
  }

  statusEl.textContent = '正在识别...';
  statusEl.style.color = 'var(--text2)';

  try {
    const formData = new FormData();
    formData.append('file', cloneAudioBlob, cloneAudioFilename);
    formData.append('language', 'auto');
    formData.append('suppress_early_eos', 'true');

    const resp = await fetch(API_BASE + '/v1/audio/transcriptions', {
      method: 'POST',
      body: formData
    });

    if (!resp.ok) {
      const errMsg = resp.status === 404 ? 'ASR 服务未加载' : 'HTTP ' + resp.status;
      throw new Error(errMsg);
    }

    const json = await resp.json();
    if (json.text) {
      document.getElementById('cloneRefText').value = json.text;
      statusEl.textContent = '✅ 识别完成';
      statusEl.style.color = 'var(--success)';
    } else {
      statusEl.textContent = '识别结果为空';
      statusEl.style.color = 'var(--text2)';
    }
  } catch (err) {
    statusEl.textContent = '识别失败: ' + err.message;
    statusEl.style.color = 'var(--error)';
  }
}

async function registerCloneVoice() {
  const name = document.getElementById('cloneVoiceName').value.trim();
  const statusEl = document.getElementById('cloneRegStatus');

  if (!name) {
    statusEl.textContent = '请输入音色名称';
    statusEl.style.color = 'var(--error)';
    return;
  }
  if (!cloneAudioBlob) {
    statusEl.textContent = '请选择音频文件或录音';
    statusEl.style.color = 'var(--error)';
    return;
  }

  statusEl.textContent = '正在注册音色...';
  statusEl.style.color = 'var(--text2)';

  try {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('audio', cloneAudioBlob, cloneAudioFilename);

    // Include reference text if provided (for future ICL mode)
    const refText = document.getElementById('cloneRefText').value.trim();
    if (refText) {
      formData.append('reference_text', refText);
    }

    const resp = await fetch(API_BASE + '/v1/voice_clone/register', {
      method: 'POST',
      body: formData
    });

    const json = await resp.json();
    if (json.success) {
      statusEl.textContent = '✅ 音色 "' + name + '" 注册成功！';
      statusEl.style.color = 'var(--success)';
      if (!ttsCloneVoices.includes(name)) {
        ttsCloneVoices.push(name);
        ttsCloneVoices.sort();
      }
      updateCloneVoiceSelect();
      document.getElementById('cloneVoiceSelect').value = name;
      // Auto-select newly registered voice for chat and notify server
      document.getElementById('voiceSelect').value = name;
      updateLangSelect(true);
      updateTtsLangSelect(true);
      sendConfig();
      // Clear form
      document.getElementById('cloneVoiceName').value = '';
      document.getElementById('cloneFileInfo').textContent = '';
      document.getElementById('cloneRefText').value = '';
      document.getElementById('cloneRefTextStatus').textContent = '';
      cloneAudioBlob = null;
    } else {
      statusEl.textContent = '❌ 注册失败: ' + (json.error || 'unknown error');
      statusEl.style.color = 'var(--error)';
    }
  } catch (err) {
    statusEl.textContent = '❌ 请求失败: ' + err.message;
    statusEl.style.color = 'var(--error)';
  }
}

async function deleteCloneVoice() {
  const sel = document.getElementById('cloneVoiceSelect');
  const name = sel ? sel.value : '';
  if (!name) return;

  if (!confirm('确认删除音色 "' + name + '"?')) return;

  try {
    const resp = await fetch(API_BASE + '/v1/voice_clone/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const json = await resp.json();
    if (json.success) {
      ttsCloneVoices = ttsCloneVoices.filter(v => v !== name);
      updateCloneVoiceSelect();
      sendConfig();  // Update chat voice after deletion
      setTtsStatus('音色 "' + name + '" 已删除', 'success');
    } else {
      setTtsStatus('删除失败: ' + (json.error || ''), 'error');
    }
  } catch (err) {
    setTtsStatus('删除请求失败: ' + err.message, 'error');
  }
}

// ============================================================================
// Speaker Management (CAM++ 说话人识别)
// ============================================================================
let spkAudioBlob = null;
let spkAudioFilename = 'speaker.wav';
let spkMediaStream = null;
let spkMediaRecorder = null;
let spkAudioChunks = [];
let spkRecordInterval = null;

async function fetchSpeakerList() {
  try {
    const resp = await fetch(API_BASE + '/v1/speakers');
    if (!resp.ok) return;
    const json = await resp.json();
    renderSpeakerList(json.speakers || []);
  } catch (e) {
    console.warn('Failed to fetch speakers:', e);
  }
}

function renderSpeakerList(speakers) {
  const list = document.getElementById('speakerList');
  const empty = document.getElementById('speakerEmpty');
  // 清空已有条目 (保留 empty 提示)
  list.querySelectorAll('.speaker-item').forEach(el => el.remove());

  if (speakers.length === 0) {
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';
  for (const name of speakers) {
    const item = document.createElement('div');
    item.className = 'speaker-item';
    item.innerHTML = '<span class="speaker-name">👤 ' + escapeHtml(name) + '</span>' +
      '<button onclick="deleteSpeaker(\'' + escapeHtml(name).replace(/'/g, "\\'") + '\')" title="删除">✕</button>';
    list.appendChild(item);
  }
}

function onSpkFileSelected(input) {
  const file = input.files[0];
  if (file) {
    document.getElementById('spkFileInfo').textContent =
      '已选择: ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    spkAudioBlob = file;
    spkAudioFilename = file.name || 'speaker.wav';
  }
}

async function toggleSpkRecord() {
  const btn = document.getElementById('spkRecordBtn');
  const statusDiv = document.getElementById('spkRecordingStatus');

  if (spkMediaRecorder && spkMediaRecorder.state === 'recording') {
    spkMediaRecorder.stop();
    btn.textContent = '🎙 录音';
    btn.style.background = '';
    statusDiv.style.display = 'none';
    if (spkRecordInterval) { clearInterval(spkRecordInterval); spkRecordInterval = null; }
    if (spkMediaStream) { spkMediaStream.getTracks().forEach(t => t.stop()); spkMediaStream = null; }
    return;
  }

  try {
    spkMediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    spkMediaRecorder = new MediaRecorder(spkMediaStream, { mimeType: 'audio/webm' });
    spkAudioChunks = [];

    spkMediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) spkAudioChunks.push(e.data);
    };

    spkMediaRecorder.onstop = async () => {
      const webmBlob = new Blob(spkAudioChunks, { type: 'audio/webm' });
      const ctx = new AudioContext({ sampleRate: 16000 });
      const arrayBuf = await webmBlob.arrayBuffer();
      const decoded = await ctx.decodeAudioData(arrayBuf);
      const channelData = decoded.getChannelData(0);
      const wavBuf = new ArrayBuffer(44 + channelData.length * 2);
      const view = new DataView(wavBuf);
      const sr = decoded.sampleRate;
      const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
      writeStr(0, 'RIFF'); view.setUint32(4, 36 + channelData.length * 2, true); writeStr(8, 'WAVE');
      writeStr(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
      view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
      writeStr(36, 'data'); view.setUint32(40, channelData.length * 2, true);
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(44 + i * 2, s * 32767, true);
      }
      spkAudioBlob = new Blob([wavBuf], { type: 'audio/wav' });
      spkAudioFilename = 'recording.wav';
      document.getElementById('spkFileInfo').textContent =
        '录音完成: ' + (spkAudioBlob.size / 1024).toFixed(1) + ' KB, ' +
        decoded.duration.toFixed(1) + 's';
      ctx.close();
    };

    spkMediaRecorder.start();
    btn.textContent = '⏹ 停止录音';
    btn.style.background = 'var(--recording)';
    statusDiv.style.display = 'block';
    let sec = 0;
    spkRecordInterval = setInterval(() => {
      sec++;
      document.getElementById('spkRecordTime').textContent = sec + 's';
    }, 1000);
  } catch (err) {
    document.getElementById('spkRegStatus').textContent = '麦克风权限被拒绝: ' + err.message;
    document.getElementById('spkRegStatus').style.color = 'var(--error)';
  }
}

async function registerSpeaker() {
  const name = document.getElementById('spkRegName').value.trim();
  const statusEl = document.getElementById('spkRegStatus');

  if (!name) {
    statusEl.textContent = '请输入说话人名称';
    statusEl.style.color = 'var(--error)';
    return;
  }
  if (!spkAudioBlob) {
    statusEl.textContent = '请选择音频文件或录音';
    statusEl.style.color = 'var(--error)';
    return;
  }

  statusEl.textContent = '正在注册...';
  statusEl.style.color = 'var(--text2)';

  try {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', spkAudioBlob, spkAudioFilename);

    const resp = await fetch(API_BASE + '/v1/speakers/register', {
      method: 'POST',
      body: formData
    });

    const json = await resp.json();
    if (json.success) {
      statusEl.textContent = '✅ 说话人 "' + name + '" 注册成功！';
      statusEl.style.color = 'var(--success)';
      document.getElementById('spkRegName').value = '';
      document.getElementById('spkFileInfo').textContent = '';
      spkAudioBlob = null;
      fetchSpeakerList();
    } else {
      statusEl.textContent = '❌ 注册失败: ' + (json.error || 'unknown error');
      statusEl.style.color = 'var(--error)';
    }
  } catch (err) {
    statusEl.textContent = '❌ 请求失败: ' + err.message;
    statusEl.style.color = 'var(--error)';
  }
}

async function deleteSpeaker(name) {
  if (!confirm('确认删除说话人 "' + name + '"?')) return;

  try {
    const resp = await fetch(API_BASE + '/v1/speakers/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const json = await resp.json();
    if (json.success) {
      fetchSpeakerList();
    }
  } catch (err) {
    console.warn('Delete speaker failed:', err);
  }
}

async function clearAllSpeakers() {
  if (!confirm('确认清空所有已注册的说话人?')) return;

  try {
    const resp = await fetch(API_BASE + '/v1/speakers/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'all' })
    });
    const json = await resp.json();
    if (json.success) {
      fetchSpeakerList();
    }
  } catch (err) {
    console.warn('Clear speakers failed:', err);
  }
}

// ============================================================================
// File Transcription (录音转写)
// ============================================================================
let transcribeAudioBlob = null;
let transcribeAudioFilename = 'audio.wav';

function onTranscribeFileSelected(input) {
  const file = input.files[0];
  if (file) {
    document.getElementById('transcribeFileInfo').textContent =
      '已选择: ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    transcribeAudioBlob = file;
    transcribeAudioFilename = file.name || 'audio.wav';
  }
}

async function transcribeFile() {
  const statusEl = document.getElementById('transcribeStatus');
  const resultDiv = document.getElementById('transcribeResult');

  if (!transcribeAudioBlob) {
    statusEl.textContent = '请先选择音频文件';
    statusEl.style.color = 'var(--error)';
    return;
  }

  statusEl.textContent = '正在转写...';
  statusEl.style.color = 'var(--text2)';
  resultDiv.style.display = 'none';

  const btn = document.getElementById('transcribeBtn');
  btn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', transcribeAudioBlob, transcribeAudioFilename);
    formData.append('language', 'auto');

    const format = document.getElementById('transcribeFormat').value;
    formData.append('response_format', format);

    if (document.getElementById('transcribePunctuate').checked) {
      formData.append('punctuate', 'true');
    }
    if (document.getElementById('transcribeSpeaker').checked) {
      formData.append('speaker', 'true');
    }

    const resp = await fetch(API_BASE + '/v1/audio/transcriptions', {
      method: 'POST',
      body: formData
    });

    if (!resp.ok) {
      const errText = await resp.text();
      let errMsg;
      try { errMsg = JSON.parse(errText).error.message; } catch { errMsg = 'HTTP ' + resp.status; }
      throw new Error(errMsg);
    }

    const contentType = resp.headers.get('content-type') || '';

    if (format === 'text' || contentType.includes('text/plain')) {
      const text = await resp.text();
      document.getElementById('transcribeResultText').textContent = text;
      document.getElementById('transcribeResultText').style.whiteSpace = 'pre-wrap';
      document.getElementById('transcribeResultMeta').innerHTML = '';
      statusEl.textContent = '✅ 转写完成';
      statusEl.style.color = 'var(--success)';
    } else {
      const json = await resp.json();

      // 检查是否有说话人分割段
      if (json.segments && json.segments.length > 0) {
        let segHtml = '<div class="diarization-segments">';
        for (const seg of json.segments) {
          const startTime = formatTimestamp(seg.start);
          const endTime = formatTimestamp(seg.end);
          const speaker = escapeHtml(seg.speaker || 'Unknown');
          const text = escapeHtml(seg.text || '');
          segHtml += '<div class="diarization-segment">' +
            '<span class="seg-time">' + startTime + ' - ' + endTime + '</span>' +
            '<span class="seg-speaker">' + speaker + '</span>' +
            '<span class="seg-text">' + text + '</span>' +
            '</div>';
        }
        segHtml += '</div>';
        document.getElementById('transcribeResultText').innerHTML = segHtml;
        document.getElementById('transcribeResultText').style.whiteSpace = 'normal';
      } else {
        const displayText = json.text_with_punc || json.text || '';
        document.getElementById('transcribeResultText').textContent = displayText;
        document.getElementById('transcribeResultText').style.whiteSpace = 'pre-wrap';
      }

      let metaHtml = '';
      if (json.language) metaHtml += '<span>语言: ' + escapeHtml(json.language) + '</span>';
      if (json.duration != null) metaHtml += '<span>时长: ' + json.duration.toFixed(1) + 's</span>';
      if (json.segments) metaHtml += '<span>段数: ' + json.segments.length + '</span>';
      if (json.speaker && !json.segments) metaHtml += '<span>说话人: ' + escapeHtml(json.speaker) + '</span>';
      if (json.speaker_similarity != null && !json.segments) metaHtml += '<span>相似度: ' + json.speaker_similarity.toFixed(3) + '</span>';
      document.getElementById('transcribeResultMeta').innerHTML = metaHtml;

      statusEl.textContent = '✅ 转写完成';
      statusEl.style.color = 'var(--success)';
    }

    resultDiv.style.display = 'block';
  } catch (err) {
    statusEl.textContent = '❌ 转写失败: ' + err.message;
    statusEl.style.color = 'var(--error)';
  } finally {
    btn.disabled = false;
  }
}

function copyTranscribeResult() {
  const text = document.getElementById('transcribeResultText').textContent;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const btn = event.target;
    const orig = btn.textContent;
    btn.textContent = '✅ 已复制';
    setTimeout(() => btn.textContent = orig, 1500);
  });
}

// ============================================================================
// Init
// ============================================================================
connectWS();
