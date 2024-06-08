const tf = require('@tensorflow/tfjs'); // Core TensorFlow.js library

async function loadModel() {
  const modelPath = 'C:\Users\dsouz\University Notes\Projects\ausa-blood-pressure-notebook\bp_api\bp\content\bp\model.json'; // Replace with actual path

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