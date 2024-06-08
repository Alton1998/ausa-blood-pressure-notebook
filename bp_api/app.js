const tf = require('@tensorflow/tfjs'); // Core TensorFlow.js library

async function loadModel() {
  const modelPath = 'https://raw.githubusercontent.com/Alton1998/ausa-blood-pressure-notebook/main/bp_api/bp/content/bp/model.json'; // Replace with actual path

  try {
    const model = await tf.loadLayersModel(modelPath); // Uses FileSystem IOHandler
    console.log('Model loaded successfully!');
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    throw error; // Re-throw to handle in your application
  }
}

loadModel()