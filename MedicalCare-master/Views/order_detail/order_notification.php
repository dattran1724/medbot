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
      <h4 class="m-0 font-weight-bold text-primary">Trạng thái đơn hàng</h4>
    </div>
    <div class="card-body">
      <?php if (isset($_SESSION['msg']) && $_SESSION['msg'] == true) { ?>
        <div class="alert alert-success">
          Đặt hàng thành công!
        </div>
        <a href="?act=product-detail&process=list_order" class="card-link">Quay lại trang đặt hàng</a>

      <?php
        unset($_SESSION['cart']);
      } else if (isset($_SESSION['msg']) && $_SESSION['msg'] == false) { ?>
        <div class="alert alert-danger">
          Đặt hàng thất bại!
        </div>
        <a href="?act=product-detail&process=list_order" class="card-link">Quay lại trang đặt hàng</a>
      <?php }
      unset($_SESSION['msg']); ?>
    </div>
  </div>
</div>