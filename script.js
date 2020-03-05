var labeledFaceDescriptors

async function prepFaces() {
  const labels = ['Vasily', 'Palina', 'Alexey', 'Antoine', 'Dmitry', 'Maria', 'Petr', 'Vladimir']
  labeledFaceDescriptors = await Promise.all(
    labels.map(async label => {
      // fetch image data from urls and convert blob to HTMLImage element
      const imgUrl = `./${label}.jpg`
      const img = await faceapi.fetchImage(imgUrl)

      // detect the face with the highest score in the image and compute it's landmarks and face descriptor
      const fullFaceDescription = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor()
      
      if (!fullFaceDescription) {
        throw new Error(`no faces detected for ${label}`)
      }
      
      const faceDescriptors = [fullFaceDescription.descriptor]
      return new faceapi.LabeledFaceDescriptors(label, faceDescriptors)
    })
  )
}

const video = document.getElementById('video')

function startVideo() {
  const constraints = { video: true, audio: false }
  navigator.mediaDevices.getUserMedia(constraints)
    .then(stream => video.srcObject = stream)
    .catch(err => {
      //document.getElementById("error").textContent = err
      video.src = "./video2.mp4"
    })
}

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('./models')
  //, faceapi.nets.ssdMobilenetv1.loadFromUri('./models')
  , faceapi.nets.faceLandmark68Net.loadFromUri('./models')
  , faceapi.nets.faceRecognitionNet.loadFromUri('./models')
  , faceapi.nets.faceExpressionNet.loadFromUri('./models')
  , faceapi.nets.ageGenderNet.loadFromUri('./models')
])
.then(prepFaces)
.then(startVideo)

video.addEventListener('play', () => {
  const canvas = faceapi.createCanvasFromMedia(video)
  document.body.append(canvas)
  
  var faceDetectionOptions = new faceapi.TinyFaceDetectorOptions()

  setInterval(async () => {
    const displaySize = { width: video.videoWidth, height: video.videoHeight }
    faceapi.matchDimensions(canvas, displaySize)

    const detections = await faceapi.detectAllFaces(video, faceDetectionOptions).withFaceLandmarks().withAgeAndGender().withFaceDescriptors()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)

    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    //faceapi.draw.drawDetections(canvas, resizedDetections)
    //faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)    

    resizedDetections.forEach(({detection,descriptor,age,gender}) => {
      const maxDescriptorDistance = 0.6
      const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance)
      const name = faceMatcher.findBestMatch(descriptor).toString()

      const label = name + ' / ' + gender + ' / age: ' + Math.round(age)
      const options = { label }
      const drawBox = new faceapi.draw.DrawBox(detection.box, options)
      drawBox.draw(canvas)
    })
  }, 100)
})
