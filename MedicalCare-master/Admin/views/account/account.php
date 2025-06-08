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
          <!-- DataTales Example -->
          <div class="card shadow mb-4">
            <div class="card-header py-3">
              <h6 class="m-0 font-weight-bold text-primary">Account</h6>
            </div>
            <div class="card-body">
              <div class="my-2">
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
                <!-- Button trigger modal -->
                <button type="button" class="btn btn-primary mx-1 my-1" data-bs-toggle="modal" data-bs-target="#addAccountModal">
                  Add Account
                </button>
              </div>
              <!-- Modal -->
              <div class="modal fade" id="addAccountModal" tabindex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-lg">
                  <div class="modal-content">
                    <div class="modal-header">
                      <h1 class="modal-title fs-5" id="exampleModalLabel">ACCOUNT</h1>
                      <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                      <form class="user mx-5" action="?act=account&process=add" method="post">
                        <div class="form-group mb-3">
                          <input type="text" name="name" class="form-control" placeholder="Tên">
                        </div>
                        <div class="form-group mb-3">
                          <input type="text" name="username" class="form-control" placeholder="Tài khoản">
                        </div>
                        <div class="form-group mb-3">
                          <input type="password" name="password" class="form-control" placeholder="Mật khẩu">
                        </div>
                        <div class="form-group mb-3">
                          <select class="form-select" name="roleID">
                            <option value="" disabled selected>Role</option>
                            <option value="1">Customer</option>
                            <option value="2">Doctor</option>
                            <option value="3">Admin</option>
                          </select>
                        </div>
                        <div class="form-group mb-3">
                          <select class="form-select" name="gender">
                            <option value="" disabled selected>Giới tính</option>
                            <option value="Nam">Nam</option>
                            <option value="Nữ">Nữ</option>
                          </select>
                        </div>
                        <div class="form-group mb-3">
                          <input type="email" name="email" class="form-control" placeholder="Email">
                        </div>
                    </div>
                    <div class="modal-footer">
                      <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                      <button type="submit" class="btn btn-primary btn-user">Add Account</button>
                    </div>
                    </form>
                  </div>
                </div>
              </div>
              <div class="table-responsive">
                <table class="table table-bordered" id="dataTable" width="100%" cellspacing="0">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Name</th>
                      <th>Username</th>
                      <th>Email</th>
                      <th>Gender</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    <?php foreach ($accounts as $account) : ?>
                      <tr>
                        <td><?php echo $account['userID'] ?></td>
                        <td><?php echo $account['name'] ?></td>
                        <td><?php echo $account['username'] ?></td>
                        <td><?php echo $account['email'] ?></td>
                        <td><?php echo $account['gender'] ?></td>
                        <td>
                          <div class="row mx-1 my-0">
                            <a class="col-2 col-sm btn btn-sm btn-warning my-1 mx-1" href="?act=account&process=edit&id=<?php echo $account['userID']; ?>">Edit</a>
                            <a class="col-2 col-sm btn btn-sm btn-danger my-1 mx-1" href="?act=account&process=delete&id=<?php echo $account['userID']; ?>">Delete</a>
                          </div>
                        </td>
                      </tr>
                    <?php endforeach; ?>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>