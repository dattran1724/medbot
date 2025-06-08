$(document).ready(function () {
  $(".nav-links").each(function () {
    const navSlides = $(this).find(".choose-service-tab");
    const contentSlides = $(this).find(".service-all-list");

    contentSlides.slick({
      slidesToShow: 1,
      slidesToScroll: 1,
      arrows: false,
      dots: false,
      infinite: false,
      asNavFor: navSlides,
      mobileFirst: true,
      // prevArrow:"<button type='button' class='slick-prev slide-arrow'><i class='fa-solid fa-angles-left' aria-hidden='true'></i></button>",
      // nextArrow:"<button type='button' class='slick-next slide-arrow'><i class='fa-solid fa-angles-right' aria-hidden='true'></i></button>"
    });

    navSlides.slick({
      slidesToShow: 2,
      // slidesToScroll: 1,
      asNavFor: contentSlides,
      dots: false,
      arrows: false,
      focusOnSelect: true,
    });
  });
});
window.addEventListener("resize", function () {
  $(".slick-slider").slick("refresh");
});
