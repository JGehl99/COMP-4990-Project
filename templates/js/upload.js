function uploadFiles() {
    var file1 = document.getElementById("img1").files[0];
    var file2 = document.getElementById("img2").files[0];
    
    var formData = new FormData();
    formData.append("img1", file1);
    formData.append("img2", file2);
    
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload");
    xhr.send(formData);
  }