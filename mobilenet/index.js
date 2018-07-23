const MODEL_URL =
  "https://raw.githubusercontent.com/shivangidas/image-classifier/master/modelv1/tensorflowjs_model.pb";
const WEIGHTS_URL =
  "https://raw.githubusercontent.com/shivangidas/image-classifier/master/modelv1/weights_manifest.json";
let model;
let IMAGENET_CLASSES = [];
let offset = tf.scalar(128);
async function loadModelAndClasses() {
  $.getJSON(
    "https://raw.githubusercontent.com/shivangidas/image-classifier/master/mobilenet/imagenet_classes.json",
    function(data) {
      $.each(data, function(key, val) {
        IMAGENET_CLASSES.push(val);
      });
    }
  );
  model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
  //console.log("After model is loaded: " + tf.memory().numTensors);
  $(".loadingDiv").hide();
  $("#inputImage").attr("disabled", false);
}
loadModelAndClasses();
function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function(e) {
      $("#imageSrc")
        .attr("src", e.target.result)
        .width(224)
        .height(224);
    };

    reader.readAsDataURL(input.files[0]);
    //console.log("After image is loaded: " + tf.memory().numTensors);

    reader.onloadend = async function() {
      console.log("Before predictions: " + tf.memory().numTensors);

      let imageData = document.getElementById("imageSrc");

      //console.log("After offset: " + tf.memory().numTensors);
      let pixels1 = tf.fromPixels(imageData);
      let pixel2 = pixels1.resizeNearestNeighbor([224, 224]);
      let pixel3 = pixel2.toFloat();
      console.log("After pixels are formed: " + tf.memory().numTensors);

      let pixels = pixel3.sub(offset);
      let pixels4 = pixels.div(offset);
      let pixels5 = pixels4.expandDims();
      console.log("After pre-processing: " + tf.memory().numTensors);

      const output = await model.predict(pixels5);
      console.log("After output: " + tf.memory().numTensors);
      const predictions = Array.from(output.dataSync())
        .map(function(p, i) {
          return {
            probabilty: p,
            classname: IMAGENET_CLASSES[i]
          };
        })
        .sort((a, b) => b.probabilty - a.probabilty)
        .slice(0, 10);

      //console.log(predictions);
      var html = "";
      for (let i = 0; i < 10; i++) {
        html += "<li>" + predictions[i].classname + "</li>";
      }
      $(".predictionList").html(html);
      console.log("After predictions: " + tf.memory().numTensors);

      pixels.dispose();
      pixels1.dispose();
      pixel2.dispose();
      pixel3.dispose();
      pixels4.dispose();
      pixels5.dispose();
      output.dispose();
      console.log("After dispose: " + tf.memory().numTensors);
    };
  }
}
