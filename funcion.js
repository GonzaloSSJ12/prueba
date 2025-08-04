const classifier = knnClassifier.create();
let isTraining = false;
let isDetecting = false;
let currentGesture = null;
let trainingStartTime = 0;
let lastDetection = '';

const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('overlay');
const ctx = canvasElement.getContext('2d');
const progressBar = document.getElementById('progress-bar');
const progressContainer = document.getElementById('progress');
const translationsEl = document.getElementById('translations');

let camera = null;
let facingMode = 'user'; // 'user' (frontal) o 'environment' (trasera)

function resizeCanvas() {
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
}

videoElement.addEventListener('loadedmetadata', resizeCanvas);
window.addEventListener('resize', resizeCanvas);

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
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
  ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

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

      if (isTraining || isDetecting) {
        processGesture(landmarks, handLabel);
      }
    });
  }
});

function startCamera() {
  if (camera) camera.stop();

  camera = new Camera(videoElement, {
    onFrame: async () => {
      await hands.send({ image: videoElement });
    },
    facingMode: facingMode,
    width: 640,
    height: 480,
  });

  camera.start();
}

function toggleCamera() {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  startCamera();
}

startCamera();

async function processGesture(landmarks, handLabel) {
  const features = getNormalizedFeatures(landmarks);
  const labelWithHand = `${currentGesture || ''}_${handLabel}`;

  if (isTraining) {
    const elapsed = Date.now() - trainingStartTime;
    const progress = Math.min((elapsed / 3000) * 100, 100);
    progressBar.style.width = `${progress}%`;

    if (elapsed <= 3000) {
      classifier.addExample(tf.tensor2d(features, [1, 63]), labelWithHand);
    }
  }

  if (isDetecting) {
    const result = await classifier.predictClass(tf.tensor2d(features, [1, 63]));
    if (result.confidences[result.label] > 0.9) {
      handleDetection(result.label);
    }
  }
}

function getNormalizedFeatures(landmarks) {
  const wrist = landmarks[0];
  const relativeLandmarks = landmarks.map((l) => ({
    x: l.x - wrist.x,
    y: l.y - wrist.y,
    z: l.z - wrist.z,
  }));

  const maxVal = Math.max(
    ...relativeLandmarks.flatMap((l) => [Math.abs(l.x), Math.abs(l.y), Math.abs(l.z)])
  );

  return relativeLandmarks.flatMap((l) => [l.x / maxVal, l.y / maxVal, l.z / maxVal]);
}

function startTraining() {
  currentGesture = document.getElementById('gestureName').value.trim();
  if (!currentGesture) return alert('Ingrese nombre del gesto');

  const btn = document.getElementById('trainBtn');
  btn.disabled = true;
  document.getElementById('gestureName').value = '';

  progressContainer.style.display = 'block';
  progressBar.style.width = '0%';
  isTraining = true;
  trainingStartTime = Date.now();

  setTimeout(() => {
    isTraining = false;
    progressContainer.style.display = 'none';
    btn.disabled = false;
    translationsEl.innerHTML += `<div>✅ ${currentGesture} entrenado (izq./der.)</div>`;
    saveModel();
  }, 3000);
}

function handleDetection(gesture) {
  if (gesture !== lastDetection) {
    translationsEl.innerHTML += `<div>${gesture}</div>`;
    lastDetection = gesture;
    translationsEl.scrollTop = translationsEl.scrollHeight;
  }
}

function toggleDetection() {
  isDetecting = !isDetecting;
  document.getElementById('detectBtn').textContent = isDetecting ? 'Detener Detección' : 'Iniciar Detección';
}

function clearTranslations() {
  translationsEl.innerHTML = '';
  lastDetection = '';
}

async function saveModel() {
  const dataset = classifier.getClassifierDataset();
  const datasetObj = {};
  Object.keys(dataset).forEach((key) => {
    datasetObj[key] = Array.from(dataset[key].dataSync());
  });
  localStorage.setItem('gestureModel', JSON.stringify(datasetObj));
}

async function loadModel() {
  const savedModel = localStorage.getItem('gestureModel');
  if (savedModel) {
    const dataset = JSON.parse(savedModel);
    Object.keys(dataset).forEach((key) => {
      const tensor = tf.tensor2d(dataset[key], [dataset[key].length / 63, 63]);
      classifier.addDataset(tensor, key);
    });
  }
}

loadModel();
