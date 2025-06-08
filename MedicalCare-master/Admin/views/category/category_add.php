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
              <h6 class="m-0 font-weight-bold text-primary">Add Category</h6>
            </div>
            <div class="card-body">
              <form action="?act=category&process=store" method="post" enctype="multipart/form-data">
                <div class="form-group mb-3">
                  <span class="">Name:</span>
                  <input type="text" name="name" class="form-control" placeholder="Name">
                </div>
                <div class="form-group mb-3">
                  <span>Icon:</span>
                  <input type="file" class="form-control" name="icon">
                </div>
                <div class="form-group mb-3">
                  <span>Image:</span>
                  <input type="file" class="form-control" name="image">
                </div>
                <div class="form-group mb-3 text-center">
                  <button class="btn btn-lg btn-primary" type="submit">Add</button>
                </div>
              </form>
            </div>
          </div>
        </div>
        <!-- /.container-fluid -->

      </div>
      <!-- End of Main Content -->

</body>