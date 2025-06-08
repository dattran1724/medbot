$(document).ready(function () {
  $(".nav-links").each(function () {
    const navSlides = $(this).find(".nav-tabs--slider");
    const contentSlides = $(this).find(".tab-content--slider");

    contentSlides.slick({
      slidesToShow: 1,
      slidesToScroll: 1,
      dots: false,
      infinite: false,
      asNavFor: navSlides,
      mobileFirst: true,
      draggable: false,
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
var citis = document.getElementById("city");
var districts = document.getElementById("district");
var wards = document.getElementById("ward");
var Parameter = {
url: "https://raw.githubusercontent.com/kenzouno1/DiaGioiHanhChinhVN/master/data.json", 
method: "GET", 
responseType: "application/json", 
};
var promise = axios(Parameter);
promise.then(function (result) {
renderCity(result.data);
});

function renderCity(data) {
  
for (const x of data) {
  citis.options[citis.options.length] = new Option(x.Name, x.Name);

}
citis.onchange = function () {
  district.length = 1;
  ward.length = 1;
  if(this.value != ""){
    const result = data.filter(n => n.Name === this.value);

    for (const k of result[0].Districts) {
      district.options[district.options.length] = new Option(k.Name, k.Name);
    }
  }
};
district.onchange = function () {
  ward.length = 1;
  const dataCity = data.filter((n) => n.Name === citis.value);
  if (this.value != "") {
    const dataWards = dataCity[0].Districts.filter(n => n.Name === this.value)[0].Wards;

    for (const w of dataWards) {
      wards.options[wards.options.length] = new Option(w.Name, w.Name);
    }
  }
};
}

jQuery(document).ready(function () {
    jQuery(".click_cgia").click(function () {
        jQuery('.click_item').addClass('hidden');
        var cgia_value = jQuery(this).data('cgia');
        jQuery("." + cgia_value).removeClass('hidden');
        jQuery(".div_board .click_cgia").removeClass('active');
        jQuery(".div_board #" + cgia_value).addClass('active');
        jQuery('html, body').animate({ scrollTop: jQuery(".div_section").offset().top - 1 }, 1000);
        //return false;
    });
    jQuery(".click_ptrang_tinlienquan").click(function () {
        var ptrang = jQuery(this).data('ptrang');
        var cgia = jQuery(this).data('cgia');
        var cattc = jQuery(this).data('cattc');
        jQuery(this).addClass('load');
        jQuery.ajax({
            type: "POST",
            url: 'https://tamanhhospital.vn/wp-content/themes/wg/ajax_bvcgia.php',
            data: { cur_trang: ptrang, cur_cgia: cgia, cur_limit: 7, cur_catc: cattc },
            dataType: "html",
            success: function (html) {
                if (cattc == 1) {
                    jQuery('.list_cathanhcong').html(html);
                    jQuery('#collapsecathanhcong .click_ptrang_tinlienquan.active').removeClass('active');
                    jQuery('#collapsecathanhcong .lienquan_' + ptrang).addClass('active');
                } else {
                    jQuery('.list_bvlq').html(html);
                    jQuery('#collapsetitle_lienquan .click_ptrang_tinlienquan.active').removeClass('active');
                    jQuery('#collapsetitle_lienquan .lienquan_' + ptrang).addClass('active');
                }
                jQuery('.load').removeClass('load');
            }
        });
    });
    var curl = jQuery(location).attr('href');
    var vtri = curl.lastIndexOf('#');
    if (vtri > 0) {
        var taburl = curl.substr(vtri, 200);
        jQuery(taburl).trigger('click');
    }

});
jQuery(document).ready(function () {
    jQuery(".limit_more").click(function () {
        if (jQuery(this).parent('.view_limit').hasClass('active')) {
            jQuery(this).html(' Xem thêm <i aria-hidden="true" class="fa fa-angle-double-down"></i>');
        } else {
            jQuery(this).html(' Ẩn nội dung <i aria-hidden="true" class="fa fa-angle-double-up"></i>');
        }
        jQuery(this).parent('.view_limit').toggleClass('active');
        

    });
    jQuery('.cl_head[data-toggle="collapse"]').click(function () {
        if (jQuery(this).find('.fa').hasClass('fa-minus-circle')) {
            jQuery(this).find('.fa-minus-circle').removeClass('fa-minus-circle').addClass('fa-plus-circle');
        } else {
            jQuery(this).find('.fa-plus-circle').removeClass('.fa-plus-circle').addClass('fa-minus-circle');
        }
    });
});

document.querySelector('#toggleLink').addEventListener('click', function() {
  document.querySelector('#toggleLink').style.display = 'none';
});

// Sự kiện khi nhấn "Ẩn"
document.querySelector('#toggleLinkHide').addEventListener('click', function() {
  document.querySelector('#toggleLink').style.display = 'inline-block';
}); 

function toggleAppointmentInfo() {
  var selectedDate = document.getElementById("dateSelector").value;
  var appointmentDivs = document.querySelectorAll('.mot-bs-ngay');
  appointmentDivs.forEach(function(div) {
    div.style.display = "none";
  });
  var selectedAppointmentDiv = document.querySelector('.mot-bs-ngay[data-date="' + selectedDate + '"]');
  if (selectedAppointmentDiv) {
    selectedAppointmentDiv.style.display = "block";
  }
}

toggleAppointmentInfo();

window.onload = function() {
  var dateSelector = document.getElementById("dateSelector");
  var closestOption = findClosestOption(new Date());
  // dateSelector.value = closestOption;
  dateSelector.value = dateSelector.options[0].value;
  
  toggleAppointmentInfo();
};

function findClosestOption(currentDate) {
  var options = Array.from(document.getElementById("dateSelector").options);
  var closestOption = options.reduce(function(prev, curr) {
    var prevDate = new Date(prev.value);
    var currDate = new Date(curr.value);
    return Math.abs(currDate - currentDate) < Math.abs(prevDate - currentDate) ? curr : prev;
  });
  return closestOption.value;
}
