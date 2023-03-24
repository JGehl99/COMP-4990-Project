// Set Slider value label on change
function onChangeSlider(){
    let slider = document.getElementById("kp");
    let sliderVal = document.getElementById("sliderValue");

    sliderVal.innerText = slider.value;
}