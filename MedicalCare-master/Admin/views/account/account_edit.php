<body id="page-top">

  <!-- Page Wrapper -->
  <div id="wrapper">
    <?php require_once('./views/inc/navbar.php'); ?>
    <!-- Content Wrapper -->
    <div id="content-wrapper" class="d-flex flex-column">

      <!-- Main Content -->
      <div id="content">

        <!-- Topbar -->
        <?php require_once('./views/inc/topbar.php'); ?>
        <!-- End of Topbar -->

        <!-- Begin Page Content -->
        <div class="container-fluid">
          <div class="card shadow mb-4">
            <div class="card-header py-3">
              <h6 class="m-0 font-weight-bold text-primary">Edit Account</h6>
            </div>
            <div class="card-body">
              <form action="?act=account&process=update" method="post">
                <div class="input-group mb-3">
                  <span class="input-group-text">ID: <?php echo $account['userID'] ?></span>
                  <input type="hidden" name="id" class="form-control" value="<?php echo $account['userID'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span class="">Name:</span>
                  <input type="text" name="name" class="form-control" placeholder="Name" value="<?php echo $account['name'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span class="">Email:</span>
                  <input type="text" name="email" class="form-control" placeholder="Email" value="<?php echo $account['email'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span class="">Username:</span>
                  <input type="text" name="username" class="form-control" placeholder="Username" value="<?php echo $account['username'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span class="">Password:</span>
                  <input type="text" name="password" class="form-control" placeholder="Password" value="<?php echo $account['password'] ?>">
                </div>
                <div class="form-group mb-3">
                  <span class="">Gender:</span>
                  <select class="form-select mb-3" name="gender">
                    <?php if ($account['gender'] == 'Nam') { ?>
                      <option value="Nam" selected>Nam</option>
                      <option value="Nữ">Nữ</option>
                    <?php  } else { ?>
                      <option value="Nam">Nam</option>
                      <option value="Nữ" selected>Nữ</option>
                    <?php } ?>
                  </select>
                </div>
                <div class="form-group mb-3">
                  <span class="">Role:</span>
                  <select class="form-select mb-3" name="roleID">
                    <?php if ($account['roleID'] == 1) { ?>
                      <option value="1" selected>Customer</option>
                      <option value="2">Doctor</option>
                      <option value="3">Admin</option>
                    <?php  } elseif ($account['roleID'] == 2) { ?>
                      <option value="1" selected>Customer</option>
                      <option value="2">Doctor</option>
                      <option value="3">Admin</option>
                    <?php } else { ?>
                      <option value="1">Customer</option>
                      <option value="2">Doctor</option>
                      <option value="3" selected>Admin</option>
                    <?php } ?>
                  </select>
                </div>
                <div class="form-group mb-3 text-center">
                  <button class="btn btn-lg btn-primary" type="submit">Update</button>
                  <a class="btn btn-lg btn-secondary" href="?act=account&role=<?php switch ($account['roleID']) {
                                                                                case '1':
                                                                                  echo 'customer';
                                                                                  break;
                                                                                case '2':
                                                                                  echo 'doctor';
                                                                                  break;
                                                                                case '3':
                                                                                  echo 'admin';
                                                                                  break;
                                                                                default:
                                                                                  echo 'customer';
                                                                                  break;
                                                                              } ?>">Cancel</a>

                </div>
              </form>
            </div>
          </div>
        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>