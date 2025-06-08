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
              <?php $totalAmount = 0.00;
              foreach ($orderDetails as $detail) : ?>
                <div class="card mb-3" style="height: 300px;">
                  <div class="row g-0">
                    <div class="col-md-3 col-sm-0">
                      <img src="../public/img/images/drug/<?php echo $detail['Image'] ?>" style="width: 300px; height: 300px; object-fit: contain;" class="img-thumbnail rounded-start" alt="<?php echo $detail['Name'] ?>">
                    </div>
                    <div class="col-md-9 col-sm-12">
                      <div class="card-body">
                        <h5 class="card-title h2 mb-3"><?php echo $detail['Name'] ?></h5>
                        <div class="row h4 mx-2">
                          <span>Quantity:</span>
                          <div class="text-info h3"><?php echo $detail['Quantity'] ?></div>
                        </div>
                        <div class="row h4 mx-2">
                          <span>Price: </span>
                          <div class="text-warning h3"><?php echo $detail['Price'] ?></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              <?php $totalAmount += $detail['Price'] * $detail['Quantity'];
              endforeach; ?>
              <div class="card-body">
                <div class="text-uppercase h3">Total Price: <span class="text-warning"> <?php echo $totalAmount ?></span>
                </div>
              </div>
              <div class="card-footer">
                <a href="?act=order" class="btn btn-primary">Go back</a>
              </div>
            </div>
          </div>

        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>