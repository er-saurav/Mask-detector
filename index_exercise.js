let model;
const webcam = new Webcam(document.getElementById('wc'));
let isPredicting = false;

async function loadModel() {
  const maskModel = await tf.loadLayersModel('http://127.0.0.1:8887/model.json');
  return tf.model({inputs: maskModel.inputs, outputs: maskModel.outputs});
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const predictions = maskModel.predict(img);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "Unmasked";
			break;
		case 1:
			predictionText = "Masked";
			break;    
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	maskModel = await loadModel();
	tf.tidy(() => maskModel.predict(webcam.capture()));
		
}


init();