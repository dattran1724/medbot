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
<div class="container-xl px-4 mt-4">
  <hr class="mt-0 mb-4">
  <div class="row">
    <!-- Account details card-->
    <div class="card mb-4">
      <div class="card-header">Thông tin cá nhân</div>
      <div class="card-body">
      <?php if (isset($_SESSION['msg']) && $_SESSION['msg'] != null) { ?>
        <?php switch ($_SESSION['msg']) {
          case 'del_true': ?>
            <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
              Delete successful!
            </div>
          <?php
            break;
          case 'del_false': ?>
            <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
              Delete fail!
            </div>
          <?php
            break;
          case 'update_true': ?>
            <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
              Update successful!
            </div>
          <?php
            break;
          case 'update_false': ?>
            <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
              Update fail!
            </div>
          <?php
            break;
          case 'add_true': ?>
            <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
              Add successful!
            </div>
          <?php
            break;
          case 'add_false': ?>
            <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
              Add fail!
            </div>
      <?php break;
        }
      }
      unset($_SESSION['msg']); ?>
        <form action="?act=account&process=update" method="post">
          <input type="hidden" value="<?php echo $data['userID']; ?>" name="id">
          <!-- Form Row-->
          <div class="mb-3">
            <label class="small mb-1" for="inputUsername">Họ tên</label>
            <input name="name" class="form-control" value="<?php echo $_SESSION['user']['name']; ?>">
          </div>
          <!-- Form Group (username)-->
          <div class="mb-3">
            <label class="small mb-1" for="inputUsername">Tên đăng nhập</label>
            <input name="username" class="form-control" id="inputUsername" type="text" value="<?php echo $_SESSION['user']['username']; ?>">
          </div>
          <!-- Form Row-->
          <div class="mb-3">
            <label class="small mb-1" for="inputUsername">Mật khẩu</label>
            <input name="password" class="form-control" type="text" value="<?php echo $_SESSION['user']['password']; ?>">
          </div>
          <!-- Form Group (email address)-->
          <div class="mb-3">
            <label class="small mb-1" for="inputEmailAddress">Địa chỉ email</label>
            <input name="email" class="form-control" id="inputEmailAddress" type="text" value="<?php echo $_SESSION['user']['email']; ?>">
          </div>
          <div class="mb-3">
            <label class="small mb-1" for="genderSelect">Giới tính</label>
            <select class="form-select mb-3" name="gender">
              <?php if ($_SESSION['user']['gender'] == 'Nam') { ?>
                <option value="Nam" selected>Nam</option>
                <option value="Nữ">Nữ</option>
              <?php  } else { ?>
                <option value="Nam">Nam</option>
                <option value="Nữ" selected>Nữ</option>
              <?php } ?>
            </select>
          </div>
      </div>
      <!-- Save changes button-->
      <div class="text-center mb-3">
        <button class="btn btn-primary" type="submit">Lưu</button>
        <a href="?act=pharmacy" class="btn btn-secondary">Huỷ</a>
      </div>
      </form>
    </div>
  </div>

</div>
</div>