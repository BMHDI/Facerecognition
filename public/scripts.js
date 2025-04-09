let faceMatcher; // Declare faceMatcher in the global scope

const run = async () => {
    // Loading the models is going to use await
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
    });
    const videoFeedEl = document.getElementById('video-feed');
    videoFeedEl.srcObject = stream;

    // We need to load our models
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
        faceapi.nets.ageGenderNet.loadFromUri('./models'),
        faceapi.nets.faceExpressionNet.loadFromUri('./models'),
        faceapi.nets.mtcnn.loadFromUri('./models'),
    ]);

    // Make the canvas the same size and in the same location as our video feed
    const canvas = document.getElementById('canvas');
    canvas.style.left = videoFeedEl.offsetLeft;
    canvas.style.top = videoFeedEl.offsetTop;
    canvas.height = videoFeedEl.height;
    canvas.width = videoFeedEl.width;

    ///// OUR FACIAL RECOGNITION DATA
    //upload name img of a person

    document.getElementById('image-upload').addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (file) {
            // Load the image from the file
            const img = await faceapi.bufferToImage(file);
            // Detect faces and get descriptors for the uploaded image
            const refFaceAiData = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
            faceMatcher = new faceapi.FaceMatcher(refFaceAiData); // Initialize faceMatcher here
            console.log('Image uploaded and processed for comparison');
        }
    });

    // Facial detection with points
    setInterval(async () => {
        // Get the video feed and hand it to detectAllFaces method
        let faceAIData = await faceapi.detectAllFaces(videoFeedEl).withFaceLandmarks().withFaceDescriptors().withAgeAndGender().withFaceExpressions();

        // Draw on our face/canvas
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
        faceAIData = faceapi.resizeResults(faceAIData, videoFeedEl);
        faceapi.draw.drawDetections(canvas, faceAIData);
        faceapi.draw.drawFaceLandmarks(canvas, faceAIData);
        faceapi.draw.drawFaceExpressions(canvas, faceAIData);

        // Ask AI to guess age and gender with confidence level
        faceAIData.forEach(face => {
            const { age, gender, genderProbability, detection, descriptor } = face;
            const score = detection?.score; // grab score from detection
            const genderText = `${gender}  ${Math.round(genderProbability * 100) / 100}%`;
            const ageText = `${Math.round(age)} years`;
            const textField = new faceapi.draw.DrawTextField([genderText, ageText], face.detection.box.topRight);
            textField.draw(canvas);
  
            // Ensure faceMatcher is initialized before accessing
            if (faceMatcher) {
                let personName = document.getElementById('person-name').value;
                let label = faceMatcher.findBestMatch(descriptor).toString();
                let options = { label:`${personName} ${Math.round(score * 100) }%` };
                if (label.includes("unknown")) {
                    options = { label: "Unknown..." };
                }
                const drawBox = new faceapi.draw.DrawBox(detection.box, options);
                drawBox.draw(canvas);
            }
        });

    }, 200);
};

run();
