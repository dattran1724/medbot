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
              <h6 class="m-0 font-weight-bold text-primary">Detail Appoiment</h6>
            </div>
            <?php
            if (isset($data) and $data != NULL) {
              foreach ($data as $value) {
                ?>
                <div class="card-body">
                  
                    <div class="form-group mb-3">
                      <span class="">ID:</span>
                      <input type="text" name="" class="form-control" value="<?php echo $value['id_lh'] ?>" disabled>
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Name customer:</span>
                      <input type="text" name="name" class="form-control" 
                        value="<?php echo $value['name_bn'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Name Doctor:</span>
                      <input type="text" name="name" class="form-control" 
                        value="<?php echo $value['name_bs'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Date:</span>
                      <input type="text" name="name" class="form-control" 
                        value="<?php echo $value['ngay'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Time:</span>
                      <input type="text" name="name" class="form-control" 
                        value="<?php echo $value['gio'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Reason:</span>
                      <input type="text" name="name" class="form-control" 
                        value="<?php echo $value['lydo'] ?>">
                    </div>
                    

                </div>
              <?php }
            } else {
              echo '<p> KHÔNG CÓ DỮ LIỆU </p>';
            }
            ?>
          </div>
        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>