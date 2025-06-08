$(document).ready(function () {
  $(".nav-links").each(function () {
    const navSlides = $(this).find(".nav-tabs--slider");
    const contentSlides = $(this).find(".tab-content--slider");

    contentSlides.slick({
      slidesToShow: 1,
      slidesToScroll: 1,
      // arrows: true,
      dots: false,
      infinite: false,
      asNavFor: navSlides,
      // infinite: true,
      mobileFirst: true,

      // prevArrow:"<button type='button' class='slick-prev slide-arrow'><i class='fa-solid fa-angles-left' aria-hidden='true'></i></button>",
      // nextArrow:"<button type='button' class='slick-next slide-arrow'><i class='fa-solid fa-angles-right' aria-hidden='true'></i></button>"
    });

    navSlides.slick({
      slidesToShow: 24,
      asNavFor: contentSlides,
      dots: false,
      arrows: false,
      focusOnSelect: true,
      responsive: [
        {
          breakpoint: 767,
          settings: {
            slidesToShow: 10,
          },
        },
        {
          breakpoint: 700,
          settings: {
            slidesToShow: 6,
          },
        },
      ],
    });
  });
});
window.onload = function() {
  var dateSelector = document.getElementById("dateSelector");
  var closestOption = findClosestOption(new Date());
  // dateSelector.value = closestOption;
  dateSelector.value = dateSelector.options[0].value;
  
  toggleAppointmentInfo();
};
// jQuery(document).ready(function () {
//   jQuery(".limit_more").click(function () {
//       if (jQuery(this).parent('.view').hasClass('active')) {
//           jQuery(this).html(' Xem thêm <i aria-hidden="true" class="fa fa-angle-double-down"></i>');
//       } else {
//           jQuery(this).html(' Ẩn nội dung <i aria-hidden="true" class="fa fa-angle-double-up"></i>');
//       }
//       jQuery(this).parent('.view').toggleClass('active');
//       jQuery(this).parent('.view_limit').addClass('active');
      

//   });
//   jQuery('.cl_head[data-toggle="collapse"]').click(function () {
//       if (jQuery(this).find('.fa').hasClass('fa-minus-circle')) {
//           jQuery(this).find('.fa-minus-circle').removeClass('fa-minus-circle').addClass('fa-plus-circle');
//       } else {
//           jQuery(this).find('.fa-plus-circle').removeClass('.fa-plus-circle').addClass('fa-minus-circle');
//       }
//   });
// });