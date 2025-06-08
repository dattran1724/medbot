$(document).ready(function () {
  $(".chuyenkhoa-slide").slick({
    infinite: true,
    slidesToShow: 5,
    slidesToScroll: 1,
    prevArrow:
      "<button type='button' class='slick-prev slide-arrow'><i class='fa fa-angle-left' aria-hidden='true'></i></button>",
    nextArrow:
      "<button type='button' class='slick-next slide-arrow'><i class='fa fa-angle-right' aria-hidden='true'></i></button>",
    responsive: [
      {
        breakpoint: 1400,
        settings: {
          slidesToShow: 4,
          slidesToScroll: 1,
        },
      },
      {
        breakpoint: 1012,
        settings: {
          slidesToShow: 3,
          slidesToScroll: 1,
        },
      },
      {
        breakpoint: 767,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 1,
        },
      },
    ],
  });
});
$(document).ready(function () {
  $(".trainghiem-list").slick({
    infinite: true,
    slidesToShow: 3,
    slidesToScroll: 1,
    prevArrow:
      "<button type='button' class='slick-prev slide-arrow'><i class='fa fa-angle-left' aria-hidden='true'></i></button>",
    nextArrow:
      "<button type='button' class='slick-next slide-arrow'><i class='fa fa-angle-right' aria-hidden='true'></i></button>",
    responsive: [
      {
        breakpoint: 1012,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 1,
        },
      },
      {
        breakpoint: 767,
        settings: {
          slidesToShow: 1,
          slidesToScroll: 1,
        },
      },
    ],
  });
});

let n = 0;
inputplaceholder();
function inputplaceholder() {
  n = n + 1;
  var placeHolder = new Array(
    "Tìm bác sĩ",
    "Tìm dịch vụ",
    "Tìm chuyên khoa",
    "Tìm gói khám"
  );
  if (n >= placeHolder.length) {
    n = 0;
  }
  document.getElementsByName("search")[0].placeholder = placeHolder[n];
  setTimeout(inputplaceholder, 1500);
}
