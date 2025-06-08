window.addEventListener("resize", function () {
  var z = document.getElementById("myTopnav");
  var x = window.innerWidth;
  var y = document.getElementById("overlay");
  if (x >= 1000) {
    y.style.display = "none";
    z.className = "list-item";
  }
  if (x < 1000 && z.className === " list-item responsive") {
    y.style.display = "none";
  }
});
function narBar() {
  var x = document.getElementById("myTopnav");
  var y = document.getElementById("overlay");
  if (x.className === "list-item") {
    x.className += " responsive";
    y.style.display = "block";
  } else {
    x.className = "list-item";
    y.style.display = "none";
  }
}
function mylogin() {
  var x = document.getElementById("login-up");

  if (x.className === "iframe-dangnhap") {
    x.className += " block-hide";
  } else {
    x.className = "iframe-dangnhap";
  }
}
function formhide() {
  var x = document.getElementById("form-block");

  if (x.className === "form-none") {
    x.className += " block-form";
  } else {
    x.className = "form-none";
  }
}
