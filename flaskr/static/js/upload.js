// Set the default threshold
var threshold = 100;
var thresholdText = document.getElementById("threshold");
thresholdText.value = threshold

// Setup Event Listeners
thresholdText.addEventListener("blur", function () {
    threshold = parseInt(thresholdText.value, 10)
    thresholdText.value = threshold
})
document.getElementById("inc").addEventListener("click", function () {
    editThreshold(1);
});
document.getElementById("dec").addEventListener("click", function () {
    editThreshold(-1);
});
document.getElementById("run-sift").addEventListener("click", function () {
    uploadFiles();
});

function uploadFiles() {
    var file1 = document.getElementById("img1").files[0];
    var file2 = document.getElementById("img2").files[0];
    
    var formData = new FormData();
    formData.append("img1", file1);
    formData.append("img2", file2);
    
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
          /*
          TODO When the FLASK server returns the images that 
          has been run through the SIFT algorithm, display 
          them on the screen
          */
        }
      };
    xhr.open("POST", "/upload");
    xhr.send(formData);
}

// Edits the threshold by adding the given difference and updates the text
function editThreshold(difference) {
    threshold += difference
    thresholdText.value = threshold
}