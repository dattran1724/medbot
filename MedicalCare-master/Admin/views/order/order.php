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
              <h6 class="m-0 font-weight-bold text-primary">Drug</h6>
            </div>
            <div class="card-body">
              <?php if (isset($_SESSION['msg']) && $_SESSION['msg'] != null) { ?>
                <?php switch ($_SESSION['msg']) {
                  case true: ?>
                    <div class="row mb-3 alert alert-success mx-auto" style="max-width: 12rem;">
                      Proceed successful!
                    </div>
                  <?php
                    break;
                  case false: ?>
                    <div class="row mb-3 alert alert-danger mx-auto" style="max-width: 12rem;">
                      Proceed fail!
                    </div>
              <?php
                    break;
                }
                unset($_SESSION['msg']);
              }
              unset($_SESSION['msg']); ?>
              <div class="row">
                <div class="table-responsive">
                  <table class="table table-bordered text-center" id="dataTable" width="100%" cellspacing="0">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Customer name</th>
                        <th>Order date</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      <?php foreach ($orders as $order) : ?>
                        <tr>
                          <td><?php echo $order['OrderID'] ?></td>
                          <td><?php echo $order['Username'] ?></td>
                          <td><?php echo $order['OrderDate'] ?></td>
                          <td><?php if ($order['Status'] == 0) { ?>
                              <div class="alert alert-info">Pending</div>
                            <?php } else { ?>
                              <div class="alert alert-success">Accepted</div>
                            <?php } ?>
                          </td>
                          <td>
                            <div class="row mx-1 my-0">
                              <a class="col-2 col-sm btn btn-sm btn-info my-1 mx-1" href="?act=order&process=detail&id=<?php echo $order['OrderID'] ?>">Detail</a>
                              <?php if ($order['Status'] == 0) : ?>
                                <a class="col-2 col-sm btn btn-sm btn-success my-1 mx-1" href="?act=order&process=accept&id=<?php echo $order['OrderID'] ?>">Accept</a>
                              <?php endif; ?>
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

        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>