/* ═══════════════════════════════════════════
   StressMitigate — Frontend Application Logic
   ═══════════════════════════════════════════ */

const API = '/api/v1';
let token = localStorage.getItem('sm_token');
let currentUser = JSON.parse(localStorage.getItem('sm_user') || 'null');
let sessionCount = parseInt(localStorage.getItem('sm_sessions') || '0');

// Media state
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let capturedImageBlob = null;
let chatSessionId = null; // Persistent session ID for context windowing
let recordingStartTime = null; // For recording timer

// ════════════ AUTH ════════════
function switchTab(tab) {
    document.getElementById('tab-login').classList.toggle('active', tab === 'login');
    document.getElementById('tab-register').classList.toggle('active', tab === 'register');
    document.getElementById('form-login').classList.toggle('hidden', tab !== 'login');
    document.getElementById('form-register').classList.toggle('hidden', tab !== 'register');
    document.getElementById('auth-error').style.display = 'none';
}

function togglePasswordVisibility(inputId, btn) {
    const input = document.getElementById(inputId);
    const icon = btn.querySelector('.material-icons-round');
    if (input.type === 'password') {
        input.type = 'text';
        icon.textContent = 'visibility';
    } else {
        input.type = 'password';
        icon.textContent = 'visibility_off';
    }
}

function showAuthError(msg) {
    const el = document.getElementById('auth-error');
    el.textContent = msg;
    el.style.display = 'block';
}

async function handleLogin(e) {
    e.preventDefault();
    const btn = document.getElementById('btn-login');
    btn.disabled = true;
    btn.textContent = 'Logging in...';

    try {
        const res = await fetch(`${API}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email: document.getElementById('login-email').value,
                password: document.getElementById('login-password').value,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Login failed');

        token = data.access_token;
        currentUser = { id: data.user_id, name: data.display_name, email: document.getElementById('login-email').value };
        localStorage.setItem('sm_token', token);
        localStorage.setItem('sm_user', JSON.stringify(currentUser));
        enterApp();
    } catch (err) {
        showAuthError(err.message);
        // Clear password field on error
        document.getElementById('login-password').value = '';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Log In';
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const btn = document.getElementById('btn-register');
    btn.disabled = true;
    btn.textContent = 'Creating account...';

    try {
        const res = await fetch(`${API}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                display_name: document.getElementById('reg-name').value,
                email: document.getElementById('reg-email').value,
                password: document.getElementById('reg-password').value,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Registration failed');

        // Smart login: auto-logged in after register
        token = data.access_token;
        currentUser = { id: data.user_id, name: data.display_name, email: document.getElementById('reg-email').value.trim().toLowerCase() };
        localStorage.setItem('sm_token', token);
        localStorage.setItem('sm_user', JSON.stringify(currentUser));
        enterApp();
    } catch (err) {
        showAuthError(err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Create Account';
    }
}

function handleLogout() {
    localStorage.removeItem('sm_token');
    localStorage.removeItem('sm_user');
    token = null;
    currentUser = null;
    document.getElementById('app-shell').classList.add('hidden');
    document.getElementById('page-auth').style.display = 'flex';
}

// ════════════ APP SHELL ════════════
function enterApp() {
    document.getElementById('page-auth').style.display = 'none';
    document.getElementById('app-shell').classList.remove('hidden');
    updateUserUI();
    updateGreeting();
    fetchWellnessScore();
    // Phase 6: Load chart & insights
    loadWellnessChart(30);
    loadRecommendations();
    loadTriggers();
}

function updateUserUI() {
    if (!currentUser) return;
    const initial = (currentUser.name || 'U')[0].toUpperCase();
    document.getElementById('sidebar-avatar').textContent = initial;
    document.getElementById('sidebar-name').textContent = currentUser.name || 'User';
    document.getElementById('profile-avatar').textContent = initial;
    document.getElementById('profile-name').textContent = currentUser.name || 'User';
    document.getElementById('profile-email').textContent = currentUser.email || '';
    document.getElementById('stat-sessions').textContent = sessionCount;
}

function updateGreeting() {
    const hour = new Date().getHours();
    let greeting = 'Good evening';
    if (hour < 12) greeting = 'Good morning';
    else if (hour < 17) greeting = 'Good afternoon';
    const name = currentUser?.name || 'there';
    document.getElementById('greeting-text').textContent = `${greeting}, ${name}`;
}

// ════════════ NAVIGATION ════════════
function navigate(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    document.getElementById(`page-${page}`).classList.add('active');
    const navBtn = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (navBtn) navBtn.classList.add('active');
}

// ════════════ DASHBOARD MOOD ════════════
let selectedDashMood = null;

function selectDashMood(btn, mood) {
    document.querySelectorAll('.mood-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selectedDashMood = mood;
    // Show inline share-more box
    document.getElementById('dash-mood-expand').classList.add('show');
}

async function submitDashMoodCheckin() {
    if (!selectedDashMood || !token) return;
    const notes = document.getElementById('dash-mood-notes').value || null;
    try {
        const res = await fetch(`${API}/analyze/check-in`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body: JSON.stringify({ mood_state: selectedDashMood, optional_notes: notes }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Check-in failed');

        // Show success animation
        showCheckinSuccess(data);

        // Reset
        document.getElementById('dash-mood-expand').classList.remove('show');
        document.getElementById('dash-mood-notes').value = '';
        document.querySelectorAll('.mood-btn').forEach(b => b.classList.remove('selected'));
        selectedDashMood = null;
    } catch (err) {
        alert(err.message);
    }
}

// ════════════ CHECK-IN ════════════
let selectedCheckinMood = null;

function selectCheckinMood(btn, mood) {
    document.querySelectorAll('.checkin-mood-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selectedCheckinMood = mood;
    document.getElementById('btn-checkin').disabled = false;
}

async function submitCheckin() {
    if (!selectedCheckinMood || !token) return;
    const btn = document.getElementById('btn-checkin');
    btn.disabled = true;
    btn.textContent = 'Submitting...';

    try {
        const res = await fetch(`${API}/analyze/check-in`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body: JSON.stringify({
                mood_state: selectedCheckinMood,
                optional_notes: document.getElementById('checkin-notes').value || null,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Check-in failed');

        // Show success animation overlay
        showCheckinSuccess(data);
    } catch (err) {
        alert(err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Submit Check-In';
    }
}

function showCheckinSuccess(data) {
    const overlay = document.getElementById('checkin-success-overlay');
    const scoreDisplay = document.getElementById('checkin-score-display');
    
    // Show stress analysis if available
    if (data.stress_category) {
        scoreDisplay.textContent = `Stress analysis: ${data.stress_category} (${data.stress_score}%)`;
    } else {
        scoreDisplay.textContent = 'Your check-in has been recorded.';
    }

    overlay.classList.add('show');

    // Auto-redirect to dashboard after 2.5 seconds
    setTimeout(() => {
        overlay.classList.remove('show');
        navigate('dashboard');
        fetchWellnessScore();
        loadWellnessChart(document.getElementById('chart-range')?.value || 30);
        loadRecommendations();
        loadTriggers();

        // Reset check-in form
        selectedCheckinMood = null;
        document.querySelectorAll('.checkin-mood-btn').forEach(b => b.classList.remove('selected'));
        document.getElementById('checkin-notes').value = '';
        document.getElementById('btn-checkin').disabled = true;
        document.getElementById('checkin-response').classList.remove('show');
    }, 2500);
}

// ════════════ CHAT ════════════
function addMessage(text, type) {
    const area = document.getElementById('messages-area');
    const div = document.createElement('div');
    div.className = `message ${type}`;
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    div.innerHTML = `${text}<div class="msg-time">${time}</div>`;
    area.appendChild(div);
    area.scrollTop = area.scrollHeight;
}

function showTyping(show) {
    document.getElementById('typing-indicator').classList.toggle('show', show);
    if (show) {
        const area = document.getElementById('messages-area');
        area.scrollTop = area.scrollHeight;
    }
}

function updateInsights(emotions) {
    const textEl = document.getElementById('insight-text');
    const voiceEl = document.getElementById('insight-voice');
    const faceEl = document.getElementById('insight-face');

    if (emotions.text) {
        textEl.textContent = emotions.text;
        textEl.className = 'emotion-value ' + getEmotionClass(emotions.text);
    }
    if (emotions.voice) {
        voiceEl.textContent = emotions.voice;
        voiceEl.className = 'emotion-value ' + getEmotionClass(emotions.voice);
    }
    if (emotions.face) {
        faceEl.textContent = emotions.face;
        faceEl.className = 'emotion-value ' + getEmotionClass(emotions.face);
    }
}

function getEmotionClass(label) {
    const positive = ['No Stress', 'Happy', 'Calm', 'Neutral'];
    const warn = ['Low Stress'];
    const negative = ['STRESS DETECTED', 'High Stress', 'Angry', 'Fear', 'Sad', 'Stressed'];
    if (positive.includes(label)) return 'positive';
    if (warn.includes(label)) return 'warn';
    if (negative.includes(label)) return 'negative';
    return 'neutral';
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text && !capturedImageBlob && audioChunks.length === 0) return;
    if (!token) return alert('Please log in first.');

    // Show user message
    if (text) addMessage(text, 'user');
    if (capturedImageBlob) addMessage('📷 Image sent for analysis', 'user');
    input.value = '';

    showTyping(true);

    // Ensure we have a session ID for context windowing
    if (!chatSessionId) {
        chatSessionId = crypto.randomUUID ? crypto.randomUUID() : 
            'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
                const r = Math.random() * 16 | 0;
                return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
            });
    }

    try {
        // Build multipart form
        const formData = new FormData();
        if (text) formData.append('text', text);
        formData.append('session_id', chatSessionId);

        // Attach audio if recorded
        if (audioChunks.length > 0) {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            formData.append('audio', audioBlob, 'voice.webm');
            audioChunks = [];
            addMessage('🎤 Voice sent for analysis', 'user');
        }

        // Attach image if captured
        if (capturedImageBlob) {
            formData.append('image', capturedImageBlob, 'face.jpg');
            capturedImageBlob = null;
            document.getElementById('btn-camera').style.color = '';
        }

        const res = await fetch(`${API}/chat/message`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${token}` },
            body: formData,
        });

        const data = await res.json();
        showTyping(false);

        if (!res.ok) throw new Error(data.detail || 'Chat failed');

        addMessage(data.ai_response, 'ai');
        updateInsights(data.detected_emotions);

        // Store the session_id from backend for context continuity
        if (data.session_id) chatSessionId = data.session_id;

        sessionCount++;
        localStorage.setItem('sm_sessions', sessionCount);
        document.getElementById('stat-sessions').textContent = sessionCount;

    } catch (err) {
        showTyping(false);
        addMessage("I'm here for you. Let's take a deep breath. My connection is a bit slow right now, but I am listening.", 'ai');
    }
}

// ════════════ VOICE RECORDING ════════════
async function toggleRecording() {
    const btn = document.getElementById('btn-mic');

    if (isRecording) {
        // Stop recording
        mediaRecorder.stop();
        isRecording = false;
        btn.classList.remove('recording');
        btn.innerHTML = '<span class="material-icons-round">mic</span>';
        // Show recording duration feedback
        const duration = ((Date.now() - recordingStartTime) / 1000).toFixed(1);
        addMessage(`🎤 Voice recorded (${duration}s) — press Send to analyze`, 'user');
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        recordingStartTime = Date.now();

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            stream.getTracks().forEach(t => t.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        btn.classList.add('recording');
        btn.innerHTML = '<span class="material-icons-round">stop</span>';
    } catch (err) {
        alert('Microphone access denied. Please allow microphone in your browser settings.');
    }
}

// ════════════ CAMERA CAPTURE ════════════
async function captureImage() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        const video = document.getElementById('hidden-video');
        video.srcObject = stream;

        // Wait for video to be ready
        await new Promise(resolve => {
            video.onloadedmetadata = () => { video.play(); resolve(); };
        });

        // Small delay for camera warmup
        await new Promise(r => setTimeout(r, 500));

        const canvas = document.getElementById('hidden-canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        stream.getTracks().forEach(t => t.stop());

        canvas.toBlob(blob => {
            capturedImageBlob = blob;
            document.getElementById('btn-camera').style.color = 'var(--success)';
            addMessage('📸 Face captured — press Send to analyze', 'user');
        }, 'image/jpeg', 0.8);
    } catch (err) {
        alert('Camera access denied. Please allow camera in your browser settings.');
    }
}

// ════════════ PRIVACY TOGGLES (synced with backend) ════════════
async function togglePrivacy(modality, el) {
    el.classList.toggle('active');
    await syncPrivacyToBackend();
}

async function syncPrivacyToBackend() {
    if (!token) return;
    try {
        const textOn = document.getElementById('toggle-text').classList.contains('active');
        const voiceOn = document.getElementById('toggle-voice').classList.contains('active');
        const cameraOn = document.getElementById('toggle-camera').classList.contains('active');
        const style = document.getElementById('pref-style')?.textContent || 'Gentle & Reassuring';
        await fetch(`${API}/user/preferences`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body: JSON.stringify({
                communication_style: style,
                privacy_settings: { allow_text: textOn, allow_voice: voiceOn, allow_camera: cameraOn },
            }),
        });
    } catch (_) {}
}

// ════════════ PREFERENCES ════════════
async function updateStyle(style) {
    document.getElementById('pref-style').textContent = style;
    await syncPrivacyToBackend();
}

// ════════════ WELLNESS SCORE ════════════
async function fetchWellnessScore() {
    if (!token) return;
    try {
        const res = await fetch(`${API}/wellness/score`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) return;
        const data = await res.json();

        const scoreEl = document.getElementById('score-value');
        const ringEl = document.getElementById('score-ring');
        const labelEl = document.getElementById('score-label');

        const score = data.weekly_average ?? data.today;
        if (score !== null && score !== undefined) {
            scoreEl.textContent = score;
            ringEl.style.setProperty('--score', score);
            labelEl.textContent = `Based on ${data.checkins_today} check-in(s) and ${data.sessions_today} AI session(s) today`;
        } else {
            scoreEl.textContent = '--';
            ringEl.style.setProperty('--score', 0);
            labelEl.textContent = 'Complete a check-in to start tracking';
        }
    } catch (_) {}
}

// ════════════ BREATHING EXERCISE ════════════
let breathingInterval = null;
let breathingCycle = 0;

function startBreathing() {
    const widget = document.getElementById('breathing-widget');
    const circle = document.getElementById('breathing-circle');
    const text = document.getElementById('breathing-text');

    widget.classList.add('show');
    breathingCycle = 0;

    function runCycle() {
        const phase = breathingCycle % 4;
        if (phase === 0) {
            circle.className = 'breathing-circle inhale';
            text.textContent = 'Breathe In... (4s)';
        } else if (phase === 1) {
            circle.className = 'breathing-circle hold';
            text.textContent = 'Hold... (4s)';
        } else if (phase === 2) {
            circle.className = 'breathing-circle exhale';
            text.textContent = 'Breathe Out... (4s)';
        } else {
            circle.className = 'breathing-circle';
            text.textContent = 'Hold... (4s)';
        }
        breathingCycle++;
    }

    runCycle();
    breathingInterval = setInterval(runCycle, 4000);
}

function stopBreathing() {
    clearInterval(breathingInterval);
    breathingInterval = null;
    breathingCycle = 0;
    const widget = document.getElementById('breathing-widget');
    widget.classList.remove('show');
    const circle = document.getElementById('breathing-circle');
    circle.className = 'breathing-circle';
}

// ════════════ SETTINGS MODALS ════════════
function openSettingsModal(id) {
    const overlay = document.getElementById(id);
    if (!overlay) return;

    // Pre-fill personal info form
    if (id === 'personal-info-modal' && currentUser) {
        document.getElementById('edit-name').value = currentUser.name || '';
        document.getElementById('edit-email').value = currentUser.email || '';
        hideModalMsg('profile-msg');
    }

    // Pre-fill privacy toggles from chat panel
    if (id === 'privacy-modal') {
        const textOn = document.getElementById('toggle-text').classList.contains('active');
        const voiceOn = document.getElementById('toggle-voice').classList.contains('active');
        const cameraOn = document.getElementById('toggle-camera').classList.contains('active');
        document.getElementById('priv-toggle-text').classList.toggle('active', textOn);
        document.getElementById('priv-toggle-voice').classList.toggle('active', voiceOn);
        document.getElementById('priv-toggle-camera').classList.toggle('active', cameraOn);
    }

    // Reset password fields
    if (id === 'security-modal') {
        document.getElementById('current-pw').value = '';
        document.getElementById('new-pw').value = '';
        hideModalMsg('password-msg');
    }

    overlay.classList.add('show');
}

function closeSettingsModal(id) {
    document.getElementById(id)?.classList.remove('show');
}

function showModalMsg(elId, msg, type) {
    const el = document.getElementById(elId);
    el.textContent = msg;
    el.className = `modal-msg show ${type}`;
}

function hideModalMsg(elId) {
    const el = document.getElementById(elId);
    if (el) { el.className = 'modal-msg'; el.textContent = ''; }
}

async function saveProfile() {
    const name = document.getElementById('edit-name').value.trim();
    const email = document.getElementById('edit-email').value.trim();

    if (!name && !email) {
        showModalMsg('profile-msg', 'Please enter a name or email to update.', 'error');
        return;
    }

    try {
        const body = {};
        if (name) body.display_name = name;
        if (email) body.email = email;

        const res = await fetch(`${API}/auth/profile`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Update failed');

        showModalMsg('profile-msg', '✅ ' + data.message, 'success');

        // Update local state
        if (name) currentUser.name = name;
        if (email) currentUser.email = email;
        localStorage.setItem('sm_user', JSON.stringify(currentUser));
        updateUserUI();

        setTimeout(() => closeSettingsModal('personal-info-modal'), 1500);
    } catch (err) {
        showModalMsg('profile-msg', err.message, 'error');
    }
}

async function changePassword() {
    const currentPw = document.getElementById('current-pw').value;
    const newPw = document.getElementById('new-pw').value;

    if (!currentPw || !newPw) {
        showModalMsg('password-msg', 'Please fill in both fields.', 'error');
        return;
    }
    if (newPw.length < 6) {
        showModalMsg('password-msg', 'New password must be at least 6 characters.', 'error');
        return;
    }

    try {
        const res = await fetch(`${API}/auth/change-password`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body: JSON.stringify({ current_password: currentPw, new_password: newPw }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Password change failed');

        showModalMsg('password-msg', '✅ ' + data.message, 'success');
        document.getElementById('current-pw').value = '';
        document.getElementById('new-pw').value = '';

        setTimeout(() => closeSettingsModal('security-modal'), 1500);
    } catch (err) {
        showModalMsg('password-msg', err.message, 'error');
    }
}

function toggleSettingsPrivacy(modality, el) {
    el.classList.toggle('active');

    // Sync with chat panel toggles
    const chatToggleId = `toggle-${modality}`;
    const chatToggle = document.getElementById(chatToggleId);
    if (chatToggle) {
        chatToggle.classList.toggle('active', el.classList.contains('active'));
    }

    // Save to backend
    syncPrivacyToBackend();
}

// ════════════ PHASE 6: WELLNESS CHART ════════════
let wellnessChart = null;

async function loadWellnessChart(days) {
    if (!token) return;
    try {
        const res = await fetch(`${API}/wellness/history?days=${days}`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) return;
        const data = await res.json();

        const labels = data.history.map(h => {
            const d = new Date(h.date);
            return d.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
        });
        const scores = data.history.map(h => h.score);

        const canvas = document.getElementById('wellness-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // Destroy previous chart
        if (wellnessChart) wellnessChart.destroy();

        // Gradient fill
        const gradient = ctx.createLinearGradient(0, 0, 0, 220);
        gradient.addColorStop(0, 'rgba(19, 127, 236, 0.3)');
        gradient.addColorStop(1, 'rgba(19, 127, 236, 0.02)');

        wellnessChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Wellness Score',
                    data: scores,
                    borderColor: '#137fec',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    pointRadius: scores.length > 14 ? 0 : 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#137fec',
                    pointBorderColor: '#0a0a1a',
                    pointBorderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    spanGaps: true,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#161637',
                        titleColor: '#e0e0f0',
                        bodyColor: '#b0b0d0',
                        borderColor: '#2a2a5a',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: ctx => ctx.parsed.y !== null ? `Score: ${ctx.parsed.y}` : 'No data',
                        },
                    },
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#6a6a9a', font: { size: 11 } },
                    },
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: '#6a6a9a',
                            font: { size: 10 },
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 10,
                        },
                    },
                },
            },
        });
    } catch (e) {
        console.error('Chart load error:', e);
    }
}

// ════════════ PHASE 6: WEEKLY REPORT ════════════
async function downloadWeeklyReport() {
    if (!token) return;
    try {
        const res = await fetch(`${API}/wellness/report`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error('Could not generate report');
        const data = await res.json();

        // Create and download .txt file
        const blob = new Blob([data.report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `StressMitigate_WeeklyReport_${new Date().toISOString().slice(0,10)}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (e) {
        alert(e.message);
    }
}

// ════════════ PHASE 6: RECOMMENDATIONS ════════════
async function loadRecommendations() {
    if (!token) return;
    const container = document.getElementById('recommendation-content');
    if (!container) return;

    try {
        const res = await fetch(`${API}/wellness/recommendations`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) return;
        const data = await res.json();

        let html = `<span class="rec-badge ${data.urgency}">${data.urgency === 'high' ? '⚠️ High Urgency' : data.urgency === 'moderate' ? 'Moderate' : '✅ Normal'}</span>`;
        html += '<div style="display:flex;flex-direction:column;gap:0.4rem;margin-top:0.5rem;">';
        for (const tip of data.tips) {
            const cls = tip.startsWith('⚠️') ? 'urgency-high' : (data.urgency === 'moderate' ? 'urgency-moderate' : '');
            html += `<div class="rec-tip ${cls}">${tip}</div>`;
        }
        html += '</div>';
        container.innerHTML = html;
    } catch (_) {}
}

// ════════════ PHASE 6: STRESS TRIGGERS ════════════
async function loadTriggers() {
    if (!token) return;
    const container = document.getElementById('triggers-content');
    if (!container) return;

    try {
        const res = await fetch(`${API}/wellness/triggers`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) return;
        const data = await res.json();

        if (!data.has_data) {
            container.innerHTML = `<p class="insight-desc">${data.message}</p>`;
            return;
        }

        let html = '';
        if (data.peak_stress_time) {
            html += `<div class="peak-stress-label">⚡ Peak stress: ${data.peak_stress_time} (${data.peak_stress_ratio}%)</div>`;
        }

        for (const bucket of data.buckets) {
            const barColor = bucket.ratio > 60 ? 'var(--danger)' : bucket.ratio > 30 ? 'var(--warning)' : 'var(--success)';
            html += `
                <div class="trigger-bucket">
                    <span class="bucket-icon">${bucket.icon}</span>
                    <span class="bucket-name">${bucket.name.split(' (')[0]}</span>
                    <div class="bucket-bar-bg"><div class="bucket-bar" style="width:${bucket.ratio}%;background:${barColor};"></div></div>
                    <span class="bucket-pct">${bucket.ratio}%</span>
                </div>`;
        }

        container.innerHTML = html;
    } catch (_) {}
}

// ════════════ INIT ════════════
document.addEventListener('DOMContentLoaded', () => {
    if (token && currentUser) {
        enterApp();
    }
});
