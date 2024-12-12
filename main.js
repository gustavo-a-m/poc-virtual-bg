import * as constants from './constants.js';

// TFLite
let tflite;
// Model
let model;

// DIV: wrapper
const wrapperDivEle = document.getElementById('wrapper');
// Imagem original
const originalImgEle = document.getElementById('original-img');
// Imagem que será usada como background
const backgroundImageEle = new Image();
backgroundImageEle.src = './assets/background-image.jpg';
backgroundImageEle.crossOrigin = 'anonymous';
// Botão segmentar
const segmentButtonEle = document.getElementById('btn-segment');

// Canvas de saída
const outputCanvasEle = document.getElementById('output-canvas');
const outputCanvasCtx = outputCanvasEle.getContext('2d');
outputCanvasEle.width = originalImgEle.width;
outputCanvasEle.height = originalImgEle.height;
// Canvas de entrada
const inputCanvasEle = document.createElement('canvas');
const inputCanvasCtx = inputCanvasEle.getContext('2d');
// Canvas de inferência
const inferenceCanvasEle =
  typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(1, 1)
    : document.createElement('canvas');
const inferenceCanvasCtx = inferenceCanvasEle.getContext('2d', {
  willReadFrequently: true,
});
// Canvas da máscara
// const maskCanvasEle =
//   typeof OffscreenCanvas !== 'undefined'
//     ? new OffscreenCanvas(1, 1)
//     : document.createElement('canvas');
// const maskCanvasCtx = maskCanvasEle.getContext('2d');
const maskCanvasEle = document.createElement('canvas');
const maskCanvasCtx = maskCanvasEle.getContext('2d');
// Cache para a máscara atual
let currentMask;

// Input Buffer
let inputBuffer;

// Helper para evitar carregar scripts duplicados
const loadedScripts = new Set();
// Flag para indicar se está utilizando o SIMD
let isSimdEnabled;
const fitType = 'fill';

// Função para carregar scripts
async function loadScript(url) {
  return new Promise((resolve, reject) => {
    if (loadedScripts.has(url)) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = url;
    script.onload = () => {
      loadedScripts.add(url);
      resolve();
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

// Função para criar o módulo TFLite
async function loadWasmModule() {
  try {
    await loadScript(`assets/${constants.TFLITE_SIMD_LOADER_NAME}`);
    tflite = await createTFLiteSIMDModule();
    isSimdEnabled = true;
  } catch {
    await loadScript(`assets/${constants.TFLITE_LOADER_NAME}`);
    tflite = await createTFLiteModule();
    isSimdEnabled = false;
  }
}

// Função para inicializar o tflite e carregar o modelo
(async function initialize() {
  const [, segmentationModelResponse] = await Promise.all([
    loadWasmModule(),
    fetch(`assets/${constants.MODEL_NAME}`),
  ]);

  model = model || (await segmentationModelResponse.arrayBuffer());
  const modelBufferOffset = tflite._getModelBufferMemoryOffset();
  tflite.HEAPU8.set(new Uint8Array(model), modelBufferOffset);
  tflite._loadModel(model.byteLength);
})();

// Carrega o buffer de entrada na memória
function loadInputBuffer(_inputBuffer) {
  const height = tflite._getInputHeight();
  const width = tflite._getInputWidth();
  const pixels = width * height;
  const tfliteInputMemoryOffset = tflite._getInputMemoryOffset() / 4;

  for (let i = 0; i < pixels; i++) {
    const curTFLiteOffset = tfliteInputMemoryOffset + i * 3;
    const curImageBufferOffset = i * 4;
    tflite.HEAPF32[curTFLiteOffset] = _inputBuffer[curImageBufferOffset] / 255;
    tflite.HEAPF32[curTFLiteOffset + 1] =
      _inputBuffer[curImageBufferOffset + 1] / 255;
    tflite.HEAPF32[curTFLiteOffset + 2] =
      _inputBuffer[curImageBufferOffset + 2] / 255;
  }
  inputBuffer = _inputBuffer;
}

// Função para rodar a inferência
function runInference() {
  const height = tflite._getInputHeight();
  const width = tflite._getInputWidth();
  const pixels = width * height;
  const tfliteOutputMemoryOffset = tflite._getOutputMemoryOffset() / 4;

  tflite._runInference();

  const outputBuffer = inputBuffer || new Uint8ClampedArray(pixels * 4);
  for (let i = 0; i < pixels; i++) {
    outputBuffer[i * 4 + 3] = Math.round(
      tflite.HEAPF32[tfliteOutputMemoryOffset + i] * 255
    );
  }
  return outputBuffer;
}

// Infere a máscara
function generateMask(inputFrame) {
  const shouldDebounce = !!currentMask;
  const { width: inferenceWidth, height: inferenceHeight } =
    constants.WASM_INFERENCE_DIMENSIONS;

  const inferenceFnc = shouldDebounce
    ? () => currentMask.data
    : () => runInference();
  const resizeFnc = shouldDebounce
    ? () => {}
    : () => {
        inferenceCanvasCtx.drawImage(
          inputFrame,
          0,
          0,
          inferenceWidth,
          inferenceHeight
        );
        const imageData = inferenceCanvasCtx.getImageData(
          0,
          0,
          inferenceWidth,
          inferenceHeight
        );
        loadInputBuffer(imageData.data);
      };

  resizeFnc();
  const mask = inferenceFnc();

  return currentMask || new ImageData(mask, inferenceWidth, inferenceHeight);
}

// Função para segmentar a imagem
async function segmentImage(inputFrameBuffer, outputFrameBuffer) {
  if (!tflite) {
    return;
  }

  // Pega as dimensões da inferência, da entrada e da saída
  const { width: inferenceWidth, height: inferenceHeight } =
    constants.WASM_INFERENCE_DIMENSIONS;
  const { width: inputWidth, height: inputHeight } = inputFrameBuffer;

  // Atualiza as dimensões dos canvas de entrada se necessário
  if (inputCanvasEle.width !== inputWidth) {
    inputCanvasEle.width = inputWidth;
  }
  if (inputCanvasEle.height !== inputHeight) {
    inputCanvasEle.height = inputHeight;
  }

  // Atualiza as dimensões dos canvas de inferência e máscara se necessário
  if (inferenceCanvasEle.width !== inferenceWidth) {
    inferenceCanvasEle.width = inferenceWidth;
    maskCanvasEle.width = inferenceWidth;
  }
  if (inferenceCanvasEle.height !== inferenceHeight) {
    inferenceCanvasEle.height = inferenceHeight;
    maskCanvasEle.height = inferenceHeight;
  }

  let inputFrame = inputFrameBuffer;

  // Processa a máscara de segmentação
  const mask = generateMask(inputFrame);

  // Salva a máscara atual no cache para evitar processamento duplicado
  currentMask = currentMask === mask ? null : mask;

  // Desenha a máscara no canvas
  if (currentMask) {
    maskCanvasCtx.putImageData(mask, 0, 0);
  }

  // Desenha a máscara junto com a imagem original
  const ctx = outputCanvasCtx;
  const { height: outputHeight, width: outputWidth } = outputCanvasEle;
  ctx.save();
  ctx.filter = `blur(2px)`;
  ctx.globalCompositeOperation = 'copy';
  ctx.drawImage(maskCanvasEle, 0, 0, outputWidth, outputHeight);
  ctx.filter = 'none';
  ctx.globalCompositeOperation = 'source-in';
  ctx.drawImage(inputFrameBuffer, 0, 0, outputWidth, outputHeight);
  ctx.globalCompositeOperation = 'destination-over';

  // Desenha a imagem de background
  const img = backgroundImageEle;
  const imageWidth = img.naturalWidth;
  const imageHeight = img.naturalHeight;

  ctx.drawImage(img, 0, 0, outputWidth, outputHeight);

  ctx.restore();
}

// Adiciona o listener para o botão de segmentar
segmentButtonEle.addEventListener('click', async () => {
  // Como a imagem não é um canvas é necessário realizar o draw
  inputCanvasEle.width = originalImgEle.width;
  inputCanvasEle.height = originalImgEle.height;
  inputCanvasCtx.drawImage(
    originalImgEle,
    0,
    0,
    inputCanvasEle.width,
    inputCanvasEle.height
  );

  await segmentImage(inputCanvasEle, outputCanvasEle);
});
