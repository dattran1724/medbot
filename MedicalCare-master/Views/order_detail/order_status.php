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
  <div class="card shadow mb-4">
    <div class="card-header py-3">
      <h4 class="m-0 font-weight-bold text-primary">Trạng thái đơn hàng</h4>
    </div>
    <div class="card-body">
      <?php foreach ($user_orders as $user_order) : ?>
        <div class="card mb-3">
          <div class="row g-0">
            <div class="card-body">
              <h5 class="card-text h2 mb-3">Mã đơn hàng: <?php echo $user_order['OrderID'] ?></h5>
              <div class="row">
                <div class="col-2 h4">Tổng tiền: <?php echo $user_order['TotalPrice']; ?> đ</div>
                <div class="col-2">
                  <?php if ($user_order['Status'] == 0) { ?>
                    <div class=" alert alert-sm alert-info text-center">Trạng thái: Đang xử lí</div>
                  <?php } else { ?>
                    <div class="alert alert-sm alert-success text-center">Trạng thái: Đã nhận</div>
                  <?php } ?>
                </div>
              </div>
              <div class="row">Địa chỉ giao hàng:<?php echo $user_order['DeliveryAddress'] ?></div>
              <a class="btn btn-md btn-primary" href="?act=product-detail&process=order_detail&id=<?php echo $user_order['OrderID'] ?>">Xem chi tiết</a>
              <?php if ($user_order['Status'] == 0) { ?>
                <a class="btn btn-md btn-danger" href="?act=product-detail&process=order_delete&id=<?php echo $user_order['OrderID'] ?>">Huỷ đơn hàng</a>
              <?php } ?>
            </div>
          </div>
        </div>
        <?php endforeach; ?>
      </div>
  </div>
</div>