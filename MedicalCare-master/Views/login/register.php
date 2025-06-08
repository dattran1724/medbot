<div class="container">
  <div class="card o-hidden border-0 shadow-lg my-5">
    <div class="card-body p-0">
      <!-- Nested Row within Card Body -->
      <div class="row">
        <div class="col-lg-5 d-none d-lg-block bg-register-image">
          <img src="./public/img/images/login/signup.jpg" alt="Signup" style=" width: 500px; height: 500px; object-fit: contain;">
        </div>
        <div class="col-lg-7">
          <div class="p-5">
            <div class="text-center">
              <h1 class="h3 text-gray-900 mb-4 text-uppercase">Tạo tài khoản</h1>
            </div>
            <?php if (isset($_SESSION['errors']) && !empty($_SESSION['errors'])) {
              $errors = $_SESSION['errors']; ?>
              <div class="alert alert-danger">
                <strong>Thông báo: </strong>
                <br>
                <?php foreach ($errors as $error) {
                  echo $error;
                ?>
                  <br>
              <?php
                }
                unset($_SESSION['errors']);
              } ?>
              </div>
              <?php if (isset($_SESSION['signup_check']) && $_SESSION['signup_check'] == true) { ?>
                <div class="alert alert-success">
                  Tạo tài khoản thành công!
                </div>
              <?php unset($_SESSION['signup_check']);
              } ?>
              <form class="user mx-5" action="?act=account&process=signup_act" method="post">
                <div class="form-group mb-3">
                  <input type="text" name="name" class="form-control form-control-user" id="exampleFirstName" placeholder="Tên">
                </div>
                <div class="form-group mb-3">
                  <input type="text" name="username" class="form-control form-control-user" id="inputUsername" placeholder="Tài khoản">
                </div>
                <div class="form-group row mb-3">
                  <div class="col-sm-6 mb-3 mb-sm-0">
                    <input type="password" name="password" class="form-control form-control-user" id="exampleInputPassword" placeholder="Mật khẩu">
                  </div>
                  <div class="col-sm-6">
                    <input type="password" name="repassword" class="form-control form-control-user" id="exampleRepeatPassword" placeholder="Nhập lại mật khẩu">
                  </div>
                </div>
                <div class="form-group mb-3">
                  <select class="form-select" name="gender">
                    <option value="" disabled selected>Giới tính</option>
                    <option value="Nam">Nam</option>
                    <option value="Nữ">Nữ</option>
                  </select>
                </div>
                <div class="form-group mb-3">
                  <input type="text" name="email" class="form-control form-control-user" id="inputEmail" placeholder="Email">
                </div>
                <div class="text-center">
                  <button type="submit" class="btn btn-primary btn-user btn-block">
                    Đăng ký tài khoản
                  </button>
                </div>
              </form>
              <hr>
              
              <div class="text-center">
                <span>Đã có tài khoản? </span>
                <a class="small" href="?act=account&process=login">Đăng nhập!</a>
              </div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>