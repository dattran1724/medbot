<header class="header">
  <div class="nav-bar">
    <div class="logo">
      <img src="./public/img/images/logo.png" alt="" />
    </div>
    <div class="list-item" id="myTopnav">
      <div class="item">
        <a href="?act=home">Trang chủ</a>
      </div>
      <div class="item"><a href="?act=news">Tin tức</a></div>
      <div class="item"><a href="?act=chuyenkhoa">Chuyên khoa</a></div>

      <div class="item"><a href="?act=chuyengia">Chuyên gia-Bác sĩ</a></div>
      <div class="item"><a href="?act=pharmacy">Nhà thuốc</a></div>

      <?php if (isset($_SESSION['isLogin']) && $_SESSION['isLogin'] == true) { ?>
        <div class="item">
          <div class="my-1 dropdown">
            <a class=" my-2" href="#" role="button" data-bs-toggle="dropdown"
              aria-expanded="false">
              <img src="https://pharmacy.jiohealth.com/icon-user.6a23f1fc34989959ada9.svg" alt="" />
              <i class="icon-user ng-star-inserted"></i>
              <?php echo (isset($_SESSION['user']) ? $_SESSION['user']['name'] : "") ?>
            </a>
            <ul class="dropdown-menu">
              <?php
              if (isset($_SESSION['isLogin']['admin']) && $_SESSION['isLogin']['admin'] == true) { ?>
                <li><a class="dropdown-item" href="?act=admin">Quản lí</a></li>
              <?php } elseif (isset($_SESSION['isLogin']['doctor']) && $_SESSION['isLogin']['doctor'] == true) { ?>
                <!-- <li><a class="dropdown-item" href="?act=admin">Lịch bác sĩ</a></li> -->
                <li><a class="dropdown-item" href="./admin/index.php?act=appointment&id_user=<?php echo $_SESSION['user']['userID']?>">Lịch bác sĩ</a></li>
                <li><a class="dropdown-item" href="?act=account&process=account">Tài khoản của tôi</a></li>
              <?php } elseif (isset($_SESSION['isLogin']['customer']) && $_SESSION['isLogin']['customer'] == true) { ?>
                <a class="dropdown-item" href="?act=account&process=account">Tài Khoản của tôi</a></li>
              <?php } ?>
              <li><a class="dropdown-item text-info" href="?act=xemlichhen">Xem lịch hẹn  </a>
              </li>
              <li><a class="dropdown-item text-info" href="?act=product-detail&process=view_order">Xem đơn hàng</a>
              </li>
              <li><a class="dropdown-item text-danger" href="?act=account&process=logout">Đăng xuất</a></li>
            </ul>
          </div>
        </div>
      <?php } else { ?>

        <div class="item"><a href="?act=account&process=login">Đăng nhập</a></div>
        <div class="item"><a href="?act=account&process=signup">Đăng ký</a></div>
      <?php } ?>

    </div>
    <!-- </div> -->
    <div class="nav-overlay" id="overlay"></div>
    <a href="javascript:void(0);" class="icon" onclick="narBar()">
      <i class="fa-solid fa-bars"></i>
    </a>



  </div>

</header>