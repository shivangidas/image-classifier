const MODEL_URL =
  "https://raw.githubusercontent.com/shivangidas/image-classifier/master/modelv1/tensorflowjs_model.pb";
const WEIGHTS_URL =
  "https://raw.githubusercontent.com/shivangidas/image-classifier/master/modelv1/weights_manifest.json";
let model;
let IMAGENET_CLASSES = [];
(async () => {
  $.getJSON(
    "https://raw.githubusercontent.com/shivangidas/image-classifier/master/mobilenet/imagenet_classes.json",
    function(data) {
      $.each(data, function(key, val) {
        IMAGENET_CLASSES.push(val);
      });
    }
  );

  model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
})();
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
    reader.onloadend = async function() {
      let imageData = document.getElementById("imageSrc");
      let pixels = tf
        .fromPixels(imageData)
        .resizeNearestNeighbor([224, 224])
        .toFloat();
      let offset = tf.scalar(128);
      pixels = pixels
        .sub(offset)
        .div(offset)
        .expandDims();
      const output = await model.predict(pixels).data();

      const predictions = Array.from(output)
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
    };
  }
}
