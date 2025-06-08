$(document).ready(function () {
  $(".num-in span").click(function () {
    var $numBlock = $(this).parents(".num-block");
    var $input = $numBlock.find("input.in-num");

    if ($(this).hasClass("minus")) {
      var count = parseFloat($input.val()) - 1;
      count = count < 1 ? 1 : count;
      if (count < 2) {
        $(this).addClass("dis");
      } else {
        $(this).removeClass("dis");
      }
      $input.val(count);
    } else {
      var count = parseFloat($input.val()) + 1;
      $input.val(count);
      if (count > 1) {
        $numBlock.find(".minus").removeClass("dis");
      }
    }

    // Enable the input if it was initially disabled
    $input.prop("disabled", false);

    $input.change();
    return false;
  });
});
