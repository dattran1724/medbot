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
              <h6 class="m-0 font-weight-bold text-primary">Edit Patient</h6>
            </div>
            <?php
            if (isset($data) and $data != NULL) {
              foreach ($data as $value) {
                ?>
                <div class="card-body">
                  <form method="POST" id="edit" action="?act=patient&process=update&id=<?php echo $value['id_bn']; ?>">
                    <div class="form-group mb-3">
                      <span class="">ID:</span>
                      <input type="text" name="" class="form-control" value="<?php echo $value['id_bn'] ?>" disabled>
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Name:</span>
                      <input type="text" name="name" class="form-control" 
                        value="<?php echo $value['name'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Name:</span>
                      <input type="text" name="name" class="form-control" 
                        value="<?php echo $value['name'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Number:</span>
                      <input type="text" name="sdt" class="form-control" 
                        value="<?php echo $value['sdt'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Gender:</span>
                      <input type="text" name="gioitinh" class="form-control" 
                        value="<?php echo $value['gioitinh'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Birthday:</span>
                      <input type="text" name="ngaysinh" class="form-control" 
                        value="<?php echo $value['ngaysinh'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Conscious:</span>
                      <input type="text" name="tinh" class="form-control" 
                        value="<?php echo $value['tinh'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">District:</span>
                      <input type="text" name="huyen" class="form-control" 
                        value="<?php echo $value['huyen'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Commune:</span>
                      <input type="text" name="xa" class="form-control" 
                        value="<?php echo $value['xa'] ?>">
                    </div>
                    <div class="form-group mb-3">
                      <span class="">Address:</span>
                      <input type="text" name="diachi" class="form-control" 
                        value="<?php echo $value['diachi'] ?>">
                    </div>

                    
                    <!-- <div class=" mx-auto justify-content-center" style="width: 200px;">
                      <button class="col-2 col-sm btn btn-lg btn-primary my-1 mx-1" type="submit"
                        form="edit">Confirm</button>
                    </div> -->
                    <script>
                      ClassicEditor
                        .create(document.querySelector('#mota'))
                        .catch(error => {
                          console.error(error);
                        });

                    </script>
                  </form>
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