function onChangeSlider(){
    let slider = document.getElementById("tol");
    let sliderVal = document.getElementById("sliderValue");

    sliderVal.innerText = slider.value + "%";
}