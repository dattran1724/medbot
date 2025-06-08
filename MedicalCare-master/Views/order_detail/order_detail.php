<div class="search-top">
  <div class="search text-center">
    <div class="input-group my-3">
      <span class="input-group-text span-icon" style="height: 100%"><i class="fa-solid fa-magnifying-glass"></i></span>
      <input type="text" class="form-control" name="search" placeholder="Tìm kiếm tên, nhãn hiệu, loại thuốc" />
      <span class="input-group-text span-text"><a href="">Search</a></span>
    </div>
  </div>
  <div style="display: flex;align-items: center;flex-wrap: wrap;justify-content: center;">
    <div class="cart-wrapper me-5 ms-4 my-2">
      <a class="btn" href="?act=product-detail&process=list_order">
        <i class="fa-solid fa-cart-plus"></i>
      </a>
    </div>
  </div>
</div>
<div class="container mt-5">
  <div class="card shadow mb-5">
    <div class="card-header py-3">
      <h4 class="m-0 font-weight-bold text-primary">Đặt hàng</h4>
    </div>
    <div class="card-body">
      <?php if (isset($_SESSION['isLogin']) && $_SESSION['isLogin'] == true) { ?>
        <?php if (isset($_SESSION['cart']) && !empty($_SESSION['cart'])) {
          $totalAmount = 0.000;
          foreach ($_SESSION['cart'] as $productId => $info) : ?>

            <div class="card mb-3" style="height: 400px;">
              <div class="row g-0">
                <div class="col-md-3 col-sm-0">
                  <img src="./public/img/images/drug/<?php echo $info['image'] ?>" style="width: 400px; height: 400px; object-fit: fill;" class="img-thumbnail rounded-start" alt="<?php echo $info['name'] ?>">
                </div>
                <div class="col-md-9 col-sm-12">
                  <div class="card-body">
                    <h5 class="card-title h2 mb-3"><?php echo $info['name'] ?></h5>
                    <div class="row">
                      <div class="col-2 h4">Số lượng:</div>
                      <div class="col-3 h4">
                        <div class="num-block skin-2 mb-2">
                          <div class="num-in">
                            <span class="minus dis"></span>
                            <form action="?act=product-detail&process=update" method="post">
                              <input type="hidden" name="id" value="<?php echo $productId ?>">
                              <input name="quantity" type="text" class="in-num" value="<?php echo $info['quantity'] ?>" onchange="this.form.submit()">
                            </form>
                            <span class="plus"></span>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div class="h3">
                      <span>Giá:</span>
                      <span class="text-warning"><?php echo $info['price'] ?> đ</span>
                    </div>
                    <div class="col-2">
                      <a class="btn btn-md btn-danger" href="?act=product-detail&process=delete&id=<?php echo $productId ?>">Xoá</a>
                      <a class="btn btn-md btn-primary" href="?act=product-detail&drug_id=<?php echo $productId ?>">Xem chi tiết</a>

                    </div>
                  </div>
                </div>
              </div>
            </div>
          <?php $totalAmount += $info['price'] * $info['quantity'];
          endforeach;
          $totalAmount = number_format($totalAmount, 3, '.', ','); ?>
          <div class="card-body">
            <div class="card-title alert">
              <p class="h3"><span>Tổng tiền:</span>
                <span class="text-warning"><?php echo $totalAmount; ?> đ</span>
            </div>
          </div>
          <div class="card-footer text-center">
            <!-- Modal -->

            <form action="?act=product-detail&process=make_order" method="post">
              <div class="row mb-2">
                <select class="form-select mx-1 col-md-3" id="city" name="city" required>
                  <option value="" selected>Chọn tỉnh thành </option>
                </select>
                <select class="form-select mx-1 col-md-3" id="district" name="district" required>
                  <option value="" selected>Chọn quận huyện</option>
                </select>
                <select class="form-select mx-1 col-md-3" id="ward" name="ward" required>
                  <option value="" selected>Chọn phường xã</option>
                </select>
              </div>
              <div class="form-group mb-2">
                <span>Địa chỉ cụ thể:</span>
                <input name="home" type="text" class="form-control" placeholder="Số nhà,Tên đường,...">
              </div>
              <div class="row text-center">
                <button type="submit" class="btn btn-primary">Đặt hàng</button>
              </div>
            </form>

          </div>
        <?php } else { ?>
          <div class="h3 text-uppercase">Giỏ hàng đang trống!</div>
          <a href="?act=pharmacy" class="card-link h4">Tới trang mua hàng ^^</a>
        <?php }
      } elseif (!isset($_SESSION['isLogin'])) { ?>
        <div class="h3">Đăng nhập để đặt hàng!</div>
        <a href="?act=account&process=login">Tới trang đăng nhập</a>
      <?php } ?>
    </div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/axios/0.21.1/axios.min.js"></script>
<script>
  const host = "https://provinces.open-api.vn/api/";
  var callAPI = (api) => {
    return axios.get(api)
      .then((response) => {
        renderData(response.data, "city");
      });
  }
  callAPI('https://provinces.open-api.vn/api/?depth=1');
  var callApiDistrict = (api) => {
    return axios.get(api)
      .then((response) => {
        renderData(response.data.districts, "district");
      });
  }
  var callApiWard = (api) => {
    return axios.get(api)
      .then((response) => {
        renderData(response.data.wards, "ward");
      });
  }

  var renderData = (array, select) => {
    let row = ' <option disable value="">Chọn</option>';
    array.forEach(element => {
      row += `<option data-id="${element.code}" value="${element.name}">${element.name}</option>`
    });
    document.querySelector("#" + select).innerHTML = row
  }

  $("#city").change(() => {
    callApiDistrict(host + "p/" + $("#city").find(':selected').data('id') + "?depth=2");
    printResult();
  });
  $("#district").change(() => {
    callApiWard(host + "d/" + $("#district").find(':selected').data('id') + "?depth=2");
    printResult();
  });
  $("#ward").change(() => {
    printResult();
  })

  var printResult = () => {
    if ($("#district").find(':selected').data('id') != "" && $("#city").find(':selected').data('id') != "" &&
      $("#ward").find(':selected').data('id') != "") {
      let result = $("#city option:selected").text() +
        " | " + $("#district option:selected").text() + " | " +
        $("#ward option:selected").text();
      $("#result").text(result)
    }

  }
</script>