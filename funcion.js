// funcion.js - versión final corregida para KNN actual
const classifier = knnClassifier.create();
let isTraining = false;
let currentGesture = null;
let trainingStartTime = 0;
let lastDetection = '';
let lastDetectionTime = 0;
const detectionCooldownMs = 800;

const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('overlay');
const ctx = canvasElement.getContext('2d');
const progressBar = document.getElementById('progress-bar');
const progressContainer = document.getElementById('progress');
const translationsEl = document.getElementById('translations');

const hands = new Hands({
  locateFile: (file) => `libs/${file}`
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.8,
  minTrackingConfidence: 0.7,
});

hands.onResults((results) => {
  resizeCanvas();
  ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (videoElement.readyState >= 2) ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks) {
    results.multiHandLandmarks.forEach((landmarks, i) => {
      const handLabel = results.multiHandedness[i].label.toLowerCase();
      window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS, {
        color: handLabel === 'left' ? '#00AAFF' : '#00FF00',
        lineWidth: 2,
      });
      window.drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 1 });

      const wrist = landmarks[0];
      const label = handLabel === 'left' ? 'L' : 'R';
      const x = wrist.x * canvasElement.width + 10;
      const y = wrist.y * canvasElement.height + 5;
      ctx.fillStyle = 'black';
      ctx.font = '20px Arial';
      ctx.fillText(label, x, y);

      processGesture(landmarks, handLabel);
    });
  }
});

let camera = null;
let facingMode = 'user';
let resolution = { width: 640, height: 480 };
const isDetecting = true;

function resizeCanvas() {
  if (!videoElement.videoWidth) return;
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
}

videoElement.addEventListener('loadedmetadata', resizeCanvas);
window.addEventListener('resize', resizeCanvas);

function startCamera() {
  if (camera) camera.stop();
  camera = new Camera(videoElement, {
    onFrame: async () => await hands.send({ image: videoElement }),
    facingMode: facingMode,
    width: resolution.width,
    height: resolution.height,
  });
  camera.start();
}

startCamera();

(async function startup() {
  await loadPretrainedGestures();
  await loadModel();
})();

async function processGesture(landmarks, handLabel) {
  const features = getNormalizedFeatures(landmarks);
  if (!features || features.length !== 63) return;

  if (isDetecting) {
    try {
      const input = tf.tensor2d(features, [1, 63]);
      const result = await classifier.predictClass(input);
      input.dispose();

      if (result && result.label && result.confidences[result.label] !== undefined) {
        const conf = result.confidences[result.label];
        const now = Date.now();
        if (conf > 0.9 && (now - lastDetectionTime) > detectionCooldownMs) {
          handleDetection(result.label);
          lastDetectionTime = now;
        }
      }
    } catch {}
  }

  if (isTraining) {
    const elapsed = Date.now() - trainingStartTime;
    const progress = Math.min((elapsed / 3000) * 100, 100);
    if (progressBar) progressBar.style.width = `${progress}%`;

    const labelWithHand = `${currentGesture}_${handLabel}`;
    classifier.addExample(tf.tensor2d(features, [1, 63]), labelWithHand);

    if (elapsed >= 3000) {
      isTraining = false;
      if (progressContainer) progressContainer.style.display = 'none';
      const btn = document.getElementById('trainBtn');
      if (btn) btn.disabled = false;
      if (translationsEl) translationsEl.innerHTML += `<div>✅ ${currentGesture} entrenado (izq./der.)</div>`;
      saveModel();
    }
  }
}

function getNormalizedFeatures(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;
  const wrist = landmarks[0];
  const relative = landmarks.map(l => ({ x: l.x - wrist.x, y: l.y - wrist.y, z: l.z - wrist.z }));
  const maxVal = Math.max(...relative.flatMap(l => [Math.abs(l.x), Math.abs(l.y), Math.abs(l.z)]), 1e-6);
  return relative.flatMap(l => [l.x / maxVal, l.y / maxVal, l.z / maxVal]);
}

function startTraining() {
  currentGesture = document.getElementById('gestureName').value.trim();
  if (!currentGesture) return alert('Ingrese nombre del gesto');

  const btn = document.getElementById('trainBtn');
  if (btn) btn.disabled = true;
  const input = document.getElementById('gestureName');
  if (input) input.value = '';

  if (progressContainer) progressContainer.style.display = 'block';
  if (progressBar) progressBar.style.width = '0%';

  isTraining = true;
  trainingStartTime = Date.now();
}

function handleDetection(gesture) {
  if (gesture !== lastDetection) {
    if (translationsEl) {
      translationsEl.innerHTML += `<div>${gesture}</div>`;
      translationsEl.scrollTop = translationsEl.scrollHeight;
    }
    lastDetection = gesture;
  }
}

function clearTranslations() {
  if (translationsEl) translationsEl.innerHTML = '';
  lastDetection = '';
}

function toggleCamera() {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  startCamera();
}

async function saveModel() {
  try {
    const dataset = classifier.getClassifierDataset();
    const datasetObj = {};
    Object.keys(dataset).forEach(key => {
      datasetObj[key] = Array.from(dataset[key].dataSync());
    });
    localStorage.setItem('gestureModel', JSON.stringify(datasetObj));
  } catch (e) {
    console.error('saveModel error', e);
  }
}

async function loadModel() {
  try {
    const savedModel = localStorage.getItem('gestureModel');
    if (!savedModel) return;
    const dataset = JSON.parse(savedModel);
    Object.keys(dataset).forEach(label => {
      const arr = dataset[label];
      if (!arr || arr.length % 63 !== 0) return;
      const numExamples = arr.length / 63;
      for (let i = 0; i < numExamples; i++) {
        const tensor = tf.tensor2d([arr.slice(i*63, (i+1)*63)]);
        classifier.addExample(tensor, label);
        tensor.dispose();
      }
    });
    console.log('Local model loaded');
  } catch (e) {
    console.error('Error loading model', e);
  }
}

async function loadPretrainedGestures() {
  const paths = ['./gestos_entrenados.json', './libs/gestos_entrenados.json'];
  let data = null;
  for (const path of paths) {
    try {
      const resp = await fetch(path);
      if (!resp.ok) continue;
      data = await resp.json();
      console.log('Pretrained gestures loaded from', path);
      break;
    } catch {}
  }
  if (!data) return;

  data.forEach(item => {
    if (!item.nombre || !item.features || item.features.length !== 63) return;
    const tensor = tf.tensor2d([item.features]);
    classifier.addExample(tensor, item.nombre);
    tensor.dispose();
  });
  console.log('Pretrained gestures loaded');
}
