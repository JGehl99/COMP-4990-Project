function uploadFiles() {
    let file1 = document.getElementById("img1").files[0];
    let file2 = document.getElementById("img2").files[0];
    
    let formData = new FormData();
    formData.append("img1", file1);
    formData.append("img2", file2);
    
    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState === 4 && this.status === 200) {
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

function onChangeSlider(){
    let slider = document.getElementById("tol");
    let sliderVal = document.getElementById("sliderValue");

    sliderVal.innerText = slider.value;
}

