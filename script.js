
const $ = (id) => document.getElementById(id);

const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

let detector;
let stream;
let animationId;
let frontCamera = true;
let paused = false;
let soundEnabled = true;
let reps = 0;
let goodReps = 0;
let formFlags = 0;
let phase = "up";
let invalidRep = false;
let reachedDepth = false;
let lastRepTime = 0;
let sessionStart = 0;
let elapsed = 0;
let running = false;
let sessionSaved = false;

const badgeCounts = {
  "Good Form": 0,
  "Go Lower": 0,
  "Hips Sagging": 0,
  "Hips Piking": 0
};

function setStatus(message, color = "") {
  $("status").textContent = message;
  $("status").className = "badge " + color;
}

function formatTime(seconds) {
  seconds = Math.floor(seconds);
  return `${Math.floor(seconds / 60)}:${String(
    seconds % 60
  ).padStart(2, "0")}`;
}

function updateStats() {
  $("reps").textContent = reps;
  $("good").textContent = goodReps;
  $("flags").textContent = formFlags;

  const currentTime =
    elapsed +
    (running && !paused
      ? (Date.now() - sessionStart) / 1000
      : 0);

  $("duration").textContent = formatTime(currentTime);
}

function angle(a, b, c) {
  const abx = a.x - b.x;
  const aby = a.y - b.y;
  const cbx = c.x - b.x;
  const cby = c.y - b.y;

  const dot = abx * cbx + aby * cby;
  const denominator =
    Math.hypot(abx, aby) * Math.hypot(cbx, cby);

  if (!denominator) return 0;

  const cosine = Math.max(
    -1,
    Math.min(1, dot / denominator)
  );

  return Math.acos(cosine) * (180 / Math.PI);
}

function visible(point) {
  return point && (point.visibility ?? 1) > 0.5;
}

function beep() {
  if (!soundEnabled) return;

  try {
    const audio = new AudioContext();
    const oscillator = audio.createOscillator();
    const gain = audio.createGain();

    oscillator.frequency.value = 880;
    gain.gain.value = 0.05;

    oscillator.connect(gain);
    gain.connect(audio.destination);

    oscillator.start();
    oscillator.stop(audio.currentTime + 0.12);
    oscillator.onended = () => audio.close();
  } catch (error) {
    console.warn("Audio unavailable", error);
  }
}

function drawPose(points) {
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  const connections = [
    [11, 12], [11, 13], [13, 15],
    [12, 14], [14, 16], [11, 23],
    [12, 24], [23, 24], [23, 25],
    [25, 27], [24, 26], [26, 28]
  ];

  ctx.lineWidth = 4;
  ctx.lineCap = "round";

  for (const [a, b] of connections) {
    if (!visible(points[a]) || !visible(points[b])) {
      continue;
    }

    ctx.strokeStyle =
      a === 23 || a === 24 || b === 23 || b === 24
        ? "#22c55e"
        : "#38bdf8";

    ctx.beginPath();
    ctx.moveTo(
      points[a].x * width,
      points[a].y * height
    );
    ctx.lineTo(
      points[b].x * width,
      points[b].y * height
    );
    ctx.stroke();
  }

  for (const point of points) {
    if (!visible(point)) continue;

    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(
      point.x * width,
      point.y * height,
      4,
      0,
      Math.PI * 2
    );
    ctx.fill();
  }
}

function analyzePose(points) {
  const required = [11, 12, 13, 14, 15, 16, 23, 24, 27, 28];

  if (!required.every((index) => visible(points[index]))) {
    setStatus("Move full body into frame", "red");
    $("hint").textContent =
      "Keep your shoulders, elbows, hips and ankles visible.";
    return;
  }

  const shoulderSpan = Math.abs(points[11].x - points[12].x);
  const hipSpan = Math.abs(points[23].x - points[24].x);

  const torsoHeight =
    (
      Math.abs(points[11].y - points[23].y) +
      Math.abs(points[12].y - points[24].y)
    ) / 2;

  const profileRatio =
    (shoulderSpan + hipSpan) /
    (2 * Math.max(0.01, torsoHeight));

  if (profileRatio >= 0.45) {
    setStatus("Turn Sideways", "red");
    $("hint").textContent =
      "Turn sideways so the camera can see your body profile.";
    return;
  }

  // Select the more visible side.
  const side =
    (points[11].visibility ?? 1) >=
    (points[12].visibility ?? 1)
      ? [11, 13, 15, 23, 27]
      : [12, 14, 16, 24, 28];

  const [shoulderIndex, elbowIndex, wristIndex, hipIndex, ankleIndex] = side;

  const shoulder = points[shoulderIndex];
  const elbow = points[elbowIndex];
  const wrist = points[wristIndex];
  const hip = points[hipIndex];
  const ankle = points[ankleIndex];

  const elbowAngle = angle(shoulder, elbow, wrist);
  const spineAngle = angle(shoulder, hip, ankle);

  $("angle").textContent = `${Math.round(spineAngle)}°`;

  // Approximate hip position relative to shoulder-ankle line.
  const dx = ankle.x - shoulder.x;
  const dy = ankle.y - shoulder.y;
  const lineY =
    shoulder.y +
    (Math.abs(dx) > 0.0001
      ? ((hip.x - shoulder.x) / dx) * dy
      : 0);

  const offset = hip.y - lineY;

  let form = "Good Form";
  let color = "green";
  let badForm = false;

  if (spineAngle < 170) {
    if (offset > 0.015) {
      form = "Hips Sagging";
      badForm = true;
    } else if (offset < -0.015) {
      form = "Hips Piking";
      badForm = true;
    } else {
      form = "Check plank";
      badForm = true;
    }

    color = "red";
  }

  if (phase === "up" && elbowAngle > 150) {
    reachedDepth = false;
  }

  if (elbowAngle < 90 && phase === "up") {
    phase = "down";
    reachedDepth = true;
    invalidRep = badForm;
  }

  if (phase === "down" && badForm) {
    invalidRep = true;
  }

  if (phase === "down" && elbowAngle > 160) {
    const now = Date.now();

    if (now - lastRepTime > 500) {
      if (!reachedDepth) {
        badgeCounts["Go Lower"]++;
        formFlags++;
        form = "Go Lower";
        color = "red";
      } else if (invalidRep) {
        formFlags++;
        if (badgeCounts[form] !== undefined) {
          badgeCounts[form]++;
        }
      } else {
        reps++;
        goodReps++;
        badgeCounts["Good Form"]++;
        beep();
      }

      phase = "up";
      reachedDepth = false;
      invalidRep = false;
      lastRepTime = now;
    }
  }

  setStatus(form, color);

  $("hint").textContent =
    badForm
      ? "Keep your shoulders, hips and ankles aligned."
      : "Lower with control and maintain a straight plank.";
}

async function loadDetector() {
  if (detector) return;

  $("engine").textContent = "Loading pose detector...";

  const vision = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs"
  );

  const files = await vision.FilesetResolver.forVisionTasks(
    WASM_URL
  );

  detector = await vision.PoseLandmarker.createFromOptions(
    files,
    {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numPoses: 1
    }
  );

  $("engine").textContent = "Pose detector ready";
}

async function startCamera() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Open this website over HTTPS in Chrome.");
    }

    $("startBtn").disabled = true;

    await loadDetector();

    if (!stream) {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: frontCamera ? "user" : "environment",
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      });

      video.srcObject = stream;
      await video.play();
    }

    $("placeholder").style.display = "none";

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    running = true;
    paused = false;
    sessionSaved = false;
    sessionStart = Date.now();

    $("pauseBtn").disabled = false;
    $("pauseBtn").textContent = "Pause";

    setStatus("Position sideways");
    requestAnimationFrame(trackFrame);
  } catch (error) {
    console.error(error);
    setStatus("Camera unavailable", "red");
    $("hint").textContent =
      "Open your published HTTPS website in Chrome, allow camera access, and check your internet connection.";
    $("engine").textContent = error.message;
  } finally {
    $("startBtn").disabled = false;
  }
}

let lastFrame = 0;

function trackFrame(now) {
  cancelAnimationFrame(animationId);

  if (!running || paused) return;

  animationId = requestAnimationFrame(trackFrame);

  if (!detector || video.readyState < 2 || now - lastFrame < 70) {
    return;
  }

  lastFrame = now;

  try {
    const result = detector.detectForVideo(video, now);

    if (result.landmarks?.length) {
      drawPose(result.landmarks[0]);
      analyzePose(result.landmarks[0]);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setStatus("No person detected", "red");
    }

    updateStats();
  } catch (error) {
    console.error("Pose tracking error", error);
  }
}

function togglePause() {
  if (!running) return;

  paused = !paused;

  if (paused) {
    elapsed += (Date.now() - sessionStart) / 1000;
    $("pauseBtn").textContent = "Resume";
    setStatus("Paused");
  } else {
    sessionStart = Date.now();
    $("pauseBtn").textContent = "Pause";
    requestAnimationFrame(trackFrame);
  }

  updateStats();
}

async function flipCamera() {
  frontCamera = !frontCamera;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  running = false;
  paused = false;

  await startCamera();
}

function resetWorkout() {
  cancelAnimationFrame(animationId);

  reps = goodReps = formFlags = elapsed = 0;
  phase = "up";
  invalidRep = reachedDepth = false;
  running = false;
  paused = false;
  sessionSaved = false;

  $("pauseBtn").disabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const key in badgeCounts) badgeCounts[key] = 0;

  updateStats();
  setStatus("Ready to train");
}

function saveWorkout() {
  if (sessionSaved) {
    setStatus("Session already saved");
    return;
  }

  const duration =
    elapsed +
    (running && !paused
      ? (Date.now() - sessionStart) / 1000
      : 0);

  let history = [];

  try {
    history = JSON.parse(
      localStorage.getItem("ragebit-history") || "[]"
    );
  } catch {}

  history.unshift({
    id: Date.now().toString(),
    startedAt: new Date(Date.now() - duration * 1000).toISOString(),
    endedAt: new Date().toISOString(),
    reps,
    duration,
    good: goodReps,
    flags: formFlags,
    badges: { ...badgeCounts }
  });

  localStorage.setItem(
    "ragebit-history",
    JSON.stringify(history.slice(0, 200))
  );

  sessionSaved = true;
  renderHistory();
  setStatus("Workout saved", "green");
}

function renderHistory() {
  let history = [];

  try {
    history = JSON.parse(
      localStorage.getItem("ragebit-history") || "[]"
    );
  } catch {}

  $("totalSessions").textContent = history.length;
  $("totalReps").textContent = history.reduce(
    (sum, item) => sum + (item.reps || 0), 0
  );
  $("best").textContent = history.reduce(
    (best, item) => Math.max(best, item.reps || 0), 0
  );
  $("totalFlags").textContent = history.reduce(
    (sum, item) => sum + (item.flags || 0), 0
  );

  const recent = history.slice(0, 14).reverse();
  const max = Math.max(1, ...recent.map((item) => item.reps || 0));

  $("chart").innerHTML = recent.map((item) => `
    <div class="barcol">
      <div class="bar" style="height:${Math.max(3, (item.reps || 0) / max * 90)}px"></div>
      ${item.reps || 0}
    </div>
  `).join("");

  $("historyList").innerHTML = history.map((item) => `
    <div class="history-item">
      <strong>${new Date(item.endedAt).toLocaleString()}</strong>
      <p>${item.reps} reps · ${formatTime(item.duration || 0)} · ${item.flags || 0} flags</p>
      <button data-delete="${item.id}">Delete</button>
    </div>
  `).join("") || '<p class="muted">No saved workouts yet.</p>';

  $("historyList").querySelectorAll("[data-delete]").forEach((button) => {
    button.onclick = () => {
      const updated = history.filter(
        (item) => item.id !== button.dataset.delete
      );

      localStorage.setItem("ragebit-history", JSON.stringify(updated));
      renderHistory();
    };
  });
}

function showPage(id) {
  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("active", page.id === id);
  });

  if (id === "history") renderHistory();
}

function toggleSound() {
  soundEnabled = !soundEnabled;
  $("soundBtn").textContent = "Sound: " + (soundEnabled ? "On" : "Off");
}

$("chatForm").addEventListener("submit", (event) => {
  event.preventDefault();

  const input = $("chatInput");
  const question = input.value.trim();

  if (!question) return;

  const answer = /pain|injury/i.test(question)
    ? "Stop exercising if you feel pain. Consider advice from a qualified healthcare professional."
    : /beginner|start/i.test(question)
    ? "Start with 3 comfortable sets, rest between sets, and try incline push-ups if needed."
    : "Keep your core engaged, move with control, and increase training volume gradually.";

  const box = document.createElement("div");
  box.className = "history-item";

  const title = document.createElement("strong");
  title.className = "cyan";
  title.textContent = "RageBit Coach";

  const paragraph = document.createElement("p");
  paragraph.textContent = answer;

  box.append(title, paragraph);
  $("chatMessages").append(box);
  input.value = "";
});

window.startCamera = startCamera;
window.togglePause = togglePause;
window.flipCamera = flipCamera;
window.resetWorkout = resetWorkout;
window.saveWorkout = saveWorkout;
window.toggleSound = toggleSound;
window.showPage = showPage;

renderHistory();
updateStats();
r
