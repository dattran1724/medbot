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
              <h6 class="m-0 font-weight-bold text-primary">Add Doctor</h6>
            </div>
            <?php
            // if (isset($data) and $data != NULL) {
            //   foreach ($data as $value) {
            //     ?>
            <div class="card-body">
              <form method="POST" id="add" action="?act=doctor&process=store">
                <div class="form-group mb-3">
                  <span class="">Name:</span>
                  <input type="text" name="name" class="form-control" value="">
                </div>
                <div class="form-group mb-3">
                  <span class="">UserID</span>
                  <input type="text" name="userID" class="form-control" value="">
                </div>
                <select class="form-select mb-3" aria-label="Default select example" name="id_ck">
                  <?php foreach ($data_ck as $ck) { ?>
                    <?php if ($ck['id_ck'] == $value['id_ck']) { ?>
                      <option value="<?php echo $ck['id_ck'] ?>" selected>
                        <?php echo $ck['name'] ?>
                      </option>
                    <?php } else { ?>
                      <option value="<?php echo $ck['id_ck'] ?>">
                        <?php echo $ck['name'] ?>
                      </option>
                    <?php }
                  } ?>
                </select>
                <div class="form-group mb-3">
                  <span class="">Position:</span>
                  <input type="text" name="chucvu" class="form-control" value="">
                </div>
                <div class="form-group mb-3">
                  <span class="">Working:</span>
                  <input type="text" name="lamviec" class="form-control" value="">
                </div>
                <div class="form-group mb-3">
                  <span class="">Areas of Expertise:</span>
                  <textarea id="chuyenmon" type="text" name="chuyenmon" class="form-control">

                          </textarea>
                </div>
                <div class="form-group mb-3">
                  <span class="">Academic rank:</span>
                  <input type="text" name="hocham" class="form-control" value="">
                </div>
                <div class="form-group mb-3">
                  <span class="">Degree:</span>
                  <input type="text" name="hocvi" class="form-control" value="">
                </div>
                <div class="form-group mb-3">
                  <span class="">Member:</span>
                  <textarea id="thanhvien" type="text" name="thanhvien" class="form-control" ></textarea>
                </div>
                <div class="form-group mb-3">
                  <span class="">Training process:</span>
                  <textarea id="daotao" type="text" name="daotao" class="form-control">

                          </textarea>
                </div>
                <div class="form-group mb-3">
                  <span class="">Experience:</span>
                  <textarea id="kinhnghiem" type="text" name="kinhnghiem" class="form-control">

                          </textarea>
                </div>
                <div class="form-group mb-3">
                  <span class="">Examination price:</span>
                  <input type="text" name="giakham" class="form-control" value="">
                </div>
                <div class="form-group mb-3">
                  <span>Picture:</span>
                  <input type="file" class="form-control" id="" placeholder="" name="hinhanh" value="">
                </div>
                <div class="form-group mb-3">
                  <span class="">Description:</span>
                  <textarea id="mota" type="text" name="mota" class="form-control">

                          </textarea>
                </div>
                <div class=" mx-auto justify-content-center" style="width: 200px;">
                  <button class="col-2 col-sm btn btn-lg btn-primary my-1 mx-1" type="submit" form="add">Store</button>
                </div>
                <script>
                  ClassicEditor
                    .create(document.querySelector('#mota'))
                    .catch(error => {
                      console.error(error);
                    });

                  ClassicEditor
                    .create(document.querySelector('#daotao'))
                    .catch(error => {
                      console.error(error);
                    });

                  ClassicEditor
                    .create(document.querySelector('#kinhnghiem'))
                    .catch(error => {
                      console.error(error);
                    });
                    ClassicEditor
                    .create(document.querySelector('#thanhvien'))
                    .catch(error => {
                      console.error(error);
                    });
                    ClassicEditor
                    .create(document.querySelector('#chuyenmon'))
                    .catch(error => {
                      console.error(error);
                    });

                </script>
              </form>
            </div>

          </div>
        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>