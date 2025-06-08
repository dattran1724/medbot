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
<div class="container-fluid mt-3">
  <div class="card shadow mb-4">
    <div class="card-header py-3">
      <h4 class="m-0 font-weight-bold text-primary">Đặt hàng</h4>
    </div>
    <div class="card-body">
      <?php if (isset($_SESSION['isLogin']) && $_SESSION['isLogin'] == true) { ?>
        <?php if (isset($orderDetails) && !empty($orderDetails)) {
          $totalAmount = 0.00;
          foreach ($orderDetails as $detail) : ?>
            <div class="card mb-3" style="height: 400px;">
              <div class="row g-0">
                <div class="col-md-3 col-sm-0">
                  <img src="./public/img/images/drug/<?php echo $detail['Image'] ?>" style="width: 400px; height: 400px; object-fit: fill;" class="img-thumbnail rounded-start" alt="<?php echo $detail['Name'] ?>">
                </div>
                <div class="col-md-9 col-sm-12">
                  <div class="card-body">
                    <h5 class="card-title h2 mb-3"><?php echo $detail['Name'] ?></h5>
                    <div class="row h4 mx-2">
                      <span>Số lượng:</span>
                      <div class="text-info h3"><?php echo $detail['Quantity'] ?></div>
                    </div>
                    <div class="row h4 mx-2">
                      <span>Giá: </span>
                      <div class="text-warning h3"><?php echo $detail['Price'] ?> đ</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          <?php $totalAmount += $detail['Price'] * $detail['Quantity'];
          endforeach;
          $totalAmount = number_format($totalAmount, 3, '.', ','); ?>
          <div class="card-body">
            <div class="text-uppercase h3">Tổng tiền: <span class="text-warning h1"> <?php echo $totalAmount ?> đ</span>
            </div>
          <div class="card-footer">
            <a href="?act=product-detail&process=view_order" class="btn btn-primary">Quay lại</a>
          </div>
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